from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from spikingjelly.activation_based import functional

from .config import BaselineNMNISTConfig
from .data import build_nmnist_loaders
from .hessian import (
    _collect_weight_layers,
    _iter_with_limit,
    _load_checkpoint,
    _try_plot_sensitivity,
    _weighted_average_bits,
    _write_csv,
    _apply_layer_weight_quantization_,
    estimate_layer_sensitivity,
    finetune_mixed_precision_model,
)
from .metrics import parameter_count, sop_proxy, synapse_count_proxy
from .model import build_model
from .quantization import clone_and_quantize_model_weights, parse_bits_list
from .unquant_runner import direct_encode, evaluate, set_global_seed, write_epoch_metrics_csv


@torch.no_grad()
def estimate_layer_state_cost(
    model: nn.Module,
    loader,
    device: str,
    t_steps: int,
    max_batches: Optional[int],
) -> Dict[str, float]:
    weight_layers = _collect_weight_layers(model)
    if not weight_layers:
        raise RuntimeError("No quantizable Conv/Linear layers were found.")

    state_acc: Dict[str, float] = {name: 0.0 for name, _ in weight_layers}
    state_count: Dict[str, int] = {name: 0 for name, _ in weight_layers}
    handles: List[torch.utils.hooks.RemovableHandle] = []

    def make_hook(layer_name: str):
        def hook(_module: nn.Module, _inputs, output) -> None:
            out = output[0] if isinstance(output, (tuple, list)) else output
            if not torch.is_tensor(out) or out.ndim < 2:
                return
            batch_size = int(out.shape[1]) if out.ndim >= 3 else int(out.shape[0])
            if batch_size <= 0:
                return
            state_acc[layer_name] += float(out.numel()) / float(batch_size)
            state_count[layer_name] += 1

        return hook

    for name, module in weight_layers:
        handles.append(module.register_forward_hook(make_hook(name)))

    try:
        model.eval()
        batch_count = 0
        for x, _y in _iter_with_limit(loader, max_batches):
            x = x.to(device)
            x_seq = direct_encode(x, t_steps)
            model(x_seq)
            batch_count += 1
            functional.reset_net(model)
        if batch_count == 0:
            raise RuntimeError("No batches were processed for state-cost estimation.")
    finally:
        for handle in handles:
            handle.remove()

    return {
        name: state_acc[name] / float(max(state_count[name], 1))
        for name, _module in weight_layers
    }


def _normalize_by_max(rows: List[dict], key: str, output_key: str) -> None:
    max_value = max((abs(float(row.get(key, 0.0))) for row in rows), default=0.0)
    for row in rows:
        value = abs(float(row.get(key, 0.0)))
        row[output_key] = value / max_value if max_value > 0.0 else 0.0


def assign_bits_by_state_aware_score(
    layer_rows: List[dict],
    bits_list: Iterable[int],
    alpha: float = 0.75,
    policy: str = "rank-map",
) -> Tuple[Dict[str, int], float]:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("state-aware alpha must be in [0, 1].")
    bits = parse_bits_list(bits_list)
    if policy not in {"rank-map", "tiered"}:
        raise ValueError("allocation policy must be either 'rank-map' or 'tiered'.")
    if not layer_rows:
        raise RuntimeError("No layer sensitivity rows were provided.")

    _normalize_by_max(layer_rows, "hessian_trace", "hessian_norm")
    _normalize_by_max(layer_rows, "state_cost_proxy", "state_cost_norm")
    for row in layer_rows:
        row["state_aware_score"] = (
            alpha * float(row["hessian_norm"])
            + (1.0 - alpha) * float(row["state_cost_norm"])
        )

    rows = sorted(layer_rows, key=lambda item: item["state_aware_score"], reverse=True)
    layer_count = len(rows)
    layer_bits: Dict[str, int] = {}

    if policy == "rank-map":
        for idx, row in enumerate(rows):
            bit_idx = min((idx * len(bits)) // layer_count, len(bits) - 1)
            layer_bits[row["layer_name"]] = bits[bit_idx]
    else:
        for idx, row in enumerate(rows):
            percentile = idx / max(layer_count - 1, 1)
            if percentile <= 1.0 / 3.0:
                bit = bits[0]
            elif percentile >= 2.0 / 3.0:
                bit = bits[-1]
            else:
                bit = bits[len(bits) // 2]
            layer_bits[row["layer_name"]] = bit

    for rank, row in enumerate(rows, start=1):
        row["state_aware_rank"] = rank
        row["assigned_bits"] = int(layer_bits[row["layer_name"]])
    return layer_bits, float(_weighted_average_bits(layer_rows, layer_bits))


def run_state_aware_hessian_analysis(
    cfg: BaselineNMNISTConfig,
    checkpoint_path: str,
    bits_list: Iterable[int],
    target_avg_bits: Optional[float] = None,
    output_dir: str = "outputs/baseline_nmnist_state_aware_hessian",
    max_hessian_batches: Optional[int] = None,
    max_test_batches: Optional[int] = None,
    trace_probes: int = 1,
    quant_epochs: int = 1,
    quant_lr: float = 1e-4,
    quant_weight_decay: float = 5e-4,
    allocation_policy: str = "rank-map",
    state_aware_alpha: float = 0.75,
) -> dict:
    set_global_seed(cfg.seed, cfg.deterministic)
    device = cfg.resolve_device()
    train_loader, test_loader = build_nmnist_loaders(cfg, device=device)

    fp32_model = build_model(num_classes=10).to(device)
    _load_checkpoint(fp32_model, checkpoint_path=checkpoint_path)

    layer_rows = estimate_layer_sensitivity(
        model=fp32_model,
        loader=train_loader,
        device=device,
        t_steps=cfg.t_steps,
        max_batches=max_hessian_batches,
        trace_probes=trace_probes,
    )

    state_cost = estimate_layer_state_cost(
        model=fp32_model,
        loader=train_loader,
        device=device,
        t_steps=cfg.t_steps,
        max_batches=max_hessian_batches,
    )
    for row in layer_rows:
        row["state_cost_proxy"] = float(state_cost.get(row["layer_name"], 0.0))

    layer_bits, achieved_avg_bits = assign_bits_by_state_aware_score(
        layer_rows=layer_rows,
        bits_list=bits_list,
        alpha=state_aware_alpha,
        policy=allocation_policy,
    )

    criterion = nn.CrossEntropyLoss()
    eval_limit = max_test_batches if max_test_batches is not None else cfg.max_test_batches

    fp32_metrics = evaluate(
        model=fp32_model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        t_steps=cfg.t_steps,
        max_batches=eval_limit,
    )

    bits_sorted = parse_bits_list(bits_list)
    uniform_target = target_avg_bits if target_avg_bits is not None else achieved_avg_bits
    uniform_ref_bits = min(bits_sorted, key=lambda bit: abs(bit - uniform_target))
    uniform_model, _ = clone_and_quantize_model_weights(fp32_model, bits=uniform_ref_bits)
    uniform_model = uniform_model.to(device)
    uniform_metrics = evaluate(
        model=uniform_model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        t_steps=cfg.t_steps,
        max_batches=eval_limit,
    )

    synapse_count = synapse_count_proxy(fp32_model)
    params = parameter_count(fp32_model)

    mixed_model = copy.deepcopy(fp32_model).to(device)
    _apply_layer_weight_quantization_(mixed_model, layer_bits)
    epoch_rows = finetune_mixed_precision_model(
        model=mixed_model,
        train_loader=train_loader,
        test_loader=test_loader,
        layer_bits=layer_bits,
        criterion=criterion,
        device=device,
        t_steps=cfg.t_steps,
        epochs=quant_epochs,
        lr=quant_lr,
        weight_decay=quant_weight_decay,
        max_train_batches=cfg.max_train_batches,
        max_test_batches=eval_limit,
        synapse_count=synapse_count,
    )
    _apply_layer_weight_quantization_(mixed_model, layer_bits)
    mixed_metrics = evaluate(
        model=mixed_model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        t_steps=cfg.t_steps,
        max_batches=eval_limit,
    )

    comparison_rows = [
        {
            "setting": "FP32",
            "test_acc": fp32_metrics["test_acc"],
            "spike_rate": fp32_metrics["spike_rate"],
            "avg_batch_infer_ms": fp32_metrics["avg_batch_infer_ms"],
            "sop_proxy": sop_proxy(fp32_metrics["spike_rate"], synapse_count, cfg.t_steps),
            "avg_weight_bits": 32.0,
        },
        {
            "setting": f"UniformW{uniform_ref_bits}",
            "test_acc": uniform_metrics["test_acc"],
            "spike_rate": uniform_metrics["spike_rate"],
            "avg_batch_infer_ms": uniform_metrics["avg_batch_infer_ms"],
            "sop_proxy": sop_proxy(uniform_metrics["spike_rate"], synapse_count, cfg.t_steps),
            "avg_weight_bits": float(uniform_ref_bits),
        },
        {
            "setting": f"StateAwareHessianMixed_a{state_aware_alpha:g}",
            "test_acc": mixed_metrics["test_acc"],
            "spike_rate": mixed_metrics["spike_rate"],
            "avg_batch_infer_ms": mixed_metrics["avg_batch_infer_ms"],
            "sop_proxy": sop_proxy(mixed_metrics["spike_rate"], synapse_count, cfg.t_steps),
            "avg_weight_bits": achieved_avg_bits,
        },
    ]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    sensitivity_csv = output_path / "layer_sensitivity.csv"
    allocation_csv = output_path / "state_aware_bit_allocation.csv"
    comparison_csv = output_path / "comparison.csv"
    quant_epoch_csv = output_path / "state_aware_quant_finetune_epoch_metrics.csv"
    summary_json = output_path / "summary.json"
    ranking_figure = output_path / "sensitivity_ranking.png"

    _write_csv(sensitivity_csv, layer_rows)
    _write_csv(
        allocation_csv,
        [
            {
                "layer_name": row["layer_name"],
                "assigned_bits": row["assigned_bits"],
                "params": row["params"],
                "hessian_trace": row["hessian_trace"],
                "trace_density": row["trace_density"],
                "rank": row["rank"],
                "state_cost_proxy": row["state_cost_proxy"],
                "hessian_norm": row["hessian_norm"],
                "state_cost_norm": row["state_cost_norm"],
                "state_aware_score": row["state_aware_score"],
                "state_aware_rank": row["state_aware_rank"],
                "state_aware_alpha": state_aware_alpha,
            }
            for row in sorted(layer_rows, key=lambda item: item["state_aware_rank"])
        ],
    )
    _write_csv(comparison_csv, comparison_rows)
    write_epoch_metrics_csv(quant_epoch_csv, epoch_rows)
    figure_path = _try_plot_sensitivity(layer_rows, ranking_figure)

    summary = {
        "method": "state_aware_hutchinson_hessian_trace_mixed_precision_finetune",
        "device": device,
        "checkpoint_path": checkpoint_path,
        "params": params,
        "target_avg_bits": target_avg_bits,
        "achieved_avg_bits": achieved_avg_bits,
        "bits_list": bits_sorted,
        "allocation_policy": allocation_policy,
        "state_aware_alpha": state_aware_alpha,
        "assigned_bits": layer_bits,
        "uniform_reference_bits": uniform_ref_bits,
        "trace_probes": trace_probes,
        "max_hessian_batches": max_hessian_batches,
        "max_test_batches": eval_limit,
        "quant_epochs": quant_epochs,
        "quant_lr": quant_lr,
        "quant_weight_decay": quant_weight_decay,
        "final_mixed_precision_acc": mixed_metrics["test_acc"],
        "outputs": {
            "layer_sensitivity_csv": str(sensitivity_csv),
            "state_aware_bit_allocation_csv": str(allocation_csv),
            "comparison_csv": str(comparison_csv),
            "state_aware_quant_finetune_epoch_metrics_csv": str(quant_epoch_csv),
            "ranking_figure": figure_path,
            "summary_json": str(summary_json),
        },
        "best_setting_by_acc": max(comparison_rows, key=lambda x: x["test_acc"])["setting"],
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
