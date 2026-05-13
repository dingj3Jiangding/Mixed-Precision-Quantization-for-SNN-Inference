from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from spikingjelly.activation_based import functional, neuron

from .config import BaselineFullConfig
from .data import build_cifar10_loaders
from .hessian import (
    _collect_weight_layers,
    _iter_with_limit,
    _load_checkpoint,
    _quantize_tensor_symmetric_per_tensor,
    _try_plot_sensitivity,
    _weighted_average_bits,
    _write_csv,
    _apply_layer_weight_quantization_,
    estimate_layer_sensitivity,
)
from .metrics import parameter_count, sop_proxy, synapse_count_proxy
from .model import build_model
from .quantization import clone_and_quantize_model_weights, parse_bits_list
from .unquant_runner import direct_encode, evaluate, set_global_seed, write_epoch_metrics_csv


def _collect_state_layers(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, neuron.BaseNode):
            layers.append((name, module))
    return layers


@torch.no_grad()
def estimate_state_cost_proxy(
    model: nn.Module,
    loader,
    device: str,
    t_steps: int,
    max_batches: Optional[int],
) -> List[dict]:
    state_layers = _collect_state_layers(model)
    if not state_layers:
        raise RuntimeError("No state layers were found.")

    state_acc: Dict[str, float] = {name: 0.0 for name, _ in state_layers}
    state_count: Dict[str, int] = {name: 0 for name, _ in state_layers}
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

    for name, module in state_layers:
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

    rows: List[dict] = []
    for name, _module in state_layers:
        proxy = state_acc[name] / float(max(state_count[name], 1))
        rows.append(
            {
                "state_layer_name": name,
                "state_cost_proxy": proxy,
            }
        )
    rows.sort(key=lambda item: item["state_cost_proxy"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["state_rank"] = rank
    return rows


def assign_state_bits_by_cost(
    state_rows: List[dict],
    bits_list: Iterable[int],
    policy: str = "rank-map",
) -> Tuple[Dict[str, int], float]:
    bits = parse_bits_list(bits_list)
    if policy not in {"rank-map", "tiered"}:
        raise ValueError("allocation policy must be either 'rank-map' or 'tiered'.")
    if not state_rows:
        raise RuntimeError("No state rows were provided.")

    rows = sorted(state_rows, key=lambda item: item["state_cost_proxy"], reverse=True)
    layer_count = len(rows)
    state_bits: Dict[str, int] = {}
    bits_low_to_high = sorted(bits)

    if policy == "rank-map":
        for idx, row in enumerate(rows):
            bit_idx = min((idx * len(bits_low_to_high)) // layer_count, len(bits_low_to_high) - 1)
            state_bits[row["state_layer_name"]] = bits_low_to_high[bit_idx]
    else:
        for idx, row in enumerate(rows):
            percentile = idx / max(layer_count - 1, 1)
            if percentile <= 1.0 / 3.0:
                bit = bits_low_to_high[0]
            elif percentile >= 2.0 / 3.0:
                bit = bits_low_to_high[-1]
            else:
                bit = bits_low_to_high[len(bits_low_to_high) // 2]
            state_bits[row["state_layer_name"]] = bit

    avg_state_bits = sum(int(state_bits[row["state_layer_name"]]) for row in rows) / float(layer_count)
    for row in state_rows:
        row["assigned_state_bits"] = int(state_bits[row["state_layer_name"]])
    return state_bits, avg_state_bits


class _StateInputQuantizer:
    def __init__(self, model: nn.Module, state_bits: Dict[str, int]) -> None:
        self._state_bits = state_bits
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        for name, module in model.named_modules():
            if name not in state_bits:
                continue
            if not isinstance(module, neuron.BaseNode):
                continue
            self._handles.append(module.register_forward_pre_hook(self._make_pre_hook(name)))

    def _make_pre_hook(self, layer_name: str):
        bits = int(self._state_bits[layer_name])

        def hook(_module: nn.Module, inputs):
            if not inputs:
                return inputs
            x = inputs[0]
            if not torch.is_tensor(x):
                return inputs
            q_x = _quantize_tensor_symmetric_per_tensor(x, bits)
            if len(inputs) == 1:
                return (q_x,)
            return (q_x,) + tuple(inputs[1:])

        return hook

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def finetune_weight_state_mixed_model(
    model: nn.Module,
    train_loader,
    test_loader,
    weight_bits: Dict[str, int],
    state_bits: Dict[str, int],
    criterion: nn.Module,
    device: str,
    t_steps: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    max_train_batches: Optional[int],
    max_test_batches: Optional[int],
    synapse_count: int,
) -> List[dict]:
    if epochs < 0:
        raise ValueError("quantization fine-tuning epochs must be >= 0.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    quantizer = _StateInputQuantizer(model, state_bits)
    rows: List[dict] = []

    try:
        for epoch in range(1, epochs + 1):
            model.train()
            loss_sum = 0.0
            correct = 0
            total = 0

            for x, y in _iter_with_limit(train_loader, max_train_batches):
                x = x.to(device)
                y = y.to(device)
                x_seq = direct_encode(x, t_steps)

                optimizer.zero_grad(set_to_none=True)
                _apply_layer_weight_quantization_(model, weight_bits)
                logits_seq = model(x_seq)
                logits = logits_seq.mean(dim=0)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

                batch_size = y.shape[0]
                loss_sum += float(loss.item()) * batch_size
                pred = logits.argmax(dim=1)
                correct += int((pred == y).sum().item())
                total += batch_size

                functional.reset_net(model)

            _apply_layer_weight_quantization_(model, weight_bits)
            test_metrics = evaluate(
                model=model,
                loader=test_loader,
                criterion=criterion,
                device=device,
                t_steps=t_steps,
                max_batches=max_test_batches,
            )
            row = {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": loss_sum / max(total, 1),
                "train_acc": correct / max(total, 1),
                **test_metrics,
                "sop_proxy": sop_proxy(test_metrics["spike_rate"], synapse_count, t_steps),
            }
            rows.append(row)
    finally:
        quantizer.close()
    return rows


def evaluate_weight_state_mixed(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: str,
    t_steps: int,
    max_batches: Optional[int],
    weight_bits: Dict[str, int],
    state_bits: Dict[str, int],
) -> dict:
    quantizer = _StateInputQuantizer(model, state_bits)
    try:
        _apply_layer_weight_quantization_(model, weight_bits)
        return evaluate(
            model=model,
            loader=loader,
            criterion=criterion,
            device=device,
            t_steps=t_steps,
            max_batches=max_batches,
        )
    finally:
        quantizer.close()


def run_weight_state_mixed_analysis(
    cfg: BaselineFullConfig,
    checkpoint_path: str,
    weight_bits_list: Iterable[int],
    state_bits_list: Optional[Iterable[int]] = None,
    target_avg_weight_bits: Optional[float] = None,
    output_dir: str = "outputs/baseline_full_weight_state_mixed",
    max_hessian_batches: Optional[int] = None,
    max_test_batches: Optional[int] = None,
    trace_probes: int = 1,
    quant_epochs: int = 1,
    quant_lr: float = 1e-4,
    quant_weight_decay: float = 5e-4,
    allocation_policy: str = "rank-map",
) -> dict:
    set_global_seed(cfg.seed, cfg.deterministic)
    device = cfg.resolve_device()
    train_loader, test_loader = build_cifar10_loaders(cfg, device=device)

    if state_bits_list is None:
        state_bits_list = weight_bits_list

    fp32_model = build_model(num_classes=10).to(device)
    _load_checkpoint(fp32_model, checkpoint_path=checkpoint_path)

    weight_rows = estimate_layer_sensitivity(
        model=fp32_model,
        loader=train_loader,
        device=device,
        t_steps=cfg.t_steps,
        max_batches=max_hessian_batches,
        trace_probes=trace_probes,
    )

    from .hessian import assign_bits_by_sensitivity_rank

    weight_bits, achieved_avg_weight_bits = assign_bits_by_sensitivity_rank(
        layer_rows=weight_rows,
        bits_list=weight_bits_list,
        policy=allocation_policy,
    )

    state_rows = estimate_state_cost_proxy(
        model=fp32_model,
        loader=train_loader,
        device=device,
        t_steps=cfg.t_steps,
        max_batches=max_hessian_batches,
    )
    state_bits, achieved_avg_state_bits = assign_state_bits_by_cost(
        state_rows=state_rows,
        bits_list=state_bits_list,
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

    weight_bits_sorted = parse_bits_list(weight_bits_list)
    uniform_target = (
        target_avg_weight_bits if target_avg_weight_bits is not None else achieved_avg_weight_bits
    )
    uniform_ref_bits = min(weight_bits_sorted, key=lambda bit: abs(bit - uniform_target))
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

    joint_model = copy.deepcopy(fp32_model).to(device)
    epoch_rows = finetune_weight_state_mixed_model(
        model=joint_model,
        train_loader=train_loader,
        test_loader=test_loader,
        weight_bits=weight_bits,
        state_bits=state_bits,
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
    mixed_metrics = evaluate_weight_state_mixed(
        model=joint_model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        t_steps=cfg.t_steps,
        max_batches=eval_limit,
        weight_bits=weight_bits,
        state_bits=state_bits,
    )

    comparison_rows = [
        {
            "setting": "FP32",
            "test_acc": fp32_metrics["test_acc"],
            "spike_rate": fp32_metrics["spike_rate"],
            "avg_batch_infer_ms": fp32_metrics["avg_batch_infer_ms"],
            "sop_proxy": sop_proxy(fp32_metrics["spike_rate"], synapse_count, cfg.t_steps),
            "avg_weight_bits": 32.0,
            "avg_state_bits": 32.0,
        },
        {
            "setting": f"UniformW{uniform_ref_bits}",
            "test_acc": uniform_metrics["test_acc"],
            "spike_rate": uniform_metrics["spike_rate"],
            "avg_batch_infer_ms": uniform_metrics["avg_batch_infer_ms"],
            "sop_proxy": sop_proxy(uniform_metrics["spike_rate"], synapse_count, cfg.t_steps),
            "avg_weight_bits": float(uniform_ref_bits),
            "avg_state_bits": 32.0,
        },
        {
            "setting": "WeightStateMixed",
            "test_acc": mixed_metrics["test_acc"],
            "spike_rate": mixed_metrics["spike_rate"],
            "avg_batch_infer_ms": mixed_metrics["avg_batch_infer_ms"],
            "sop_proxy": sop_proxy(mixed_metrics["spike_rate"], synapse_count, cfg.t_steps),
            "avg_weight_bits": achieved_avg_weight_bits,
            "avg_state_bits": achieved_avg_state_bits,
        },
    ]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    weight_csv = output_path / "weight_bit_allocation.csv"
    state_csv = output_path / "state_bit_allocation.csv"
    comparison_csv = output_path / "comparison.csv"
    quant_epoch_csv = output_path / "weight_state_quant_finetune_epoch_metrics.csv"
    summary_json = output_path / "summary.json"
    ranking_figure = output_path / "weight_sensitivity_ranking.png"

    _write_csv(
        weight_csv,
        [
            {
                "layer_name": row["layer_name"],
                "assigned_weight_bits": row["assigned_bits"],
                "params": row["params"],
                "hessian_trace": row["hessian_trace"],
                "trace_density": row["trace_density"],
                "weight_rank": row["rank"],
            }
            for row in weight_rows
        ],
    )
    _write_csv(state_csv, state_rows)
    _write_csv(comparison_csv, comparison_rows)
    write_epoch_metrics_csv(quant_epoch_csv, epoch_rows)
    figure_path = _try_plot_sensitivity(weight_rows, ranking_figure)

    summary = {
        "method": "weight_state_mixed_precision_proxy_experiment",
        "device": device,
        "checkpoint_path": checkpoint_path,
        "params": params,
        "achieved_avg_weight_bits": achieved_avg_weight_bits,
        "achieved_avg_state_bits": achieved_avg_state_bits,
        "weight_bits_list": parse_bits_list(weight_bits_list),
        "state_bits_list": parse_bits_list(state_bits_list),
        "allocation_policy": allocation_policy,
        "assigned_weight_bits": weight_bits,
        "assigned_state_bits": state_bits,
        "uniform_reference_bits": uniform_ref_bits,
        "trace_probes": trace_probes,
        "max_hessian_batches": max_hessian_batches,
        "max_test_batches": eval_limit,
        "quant_epochs": quant_epochs,
        "quant_lr": quant_lr,
        "quant_weight_decay": quant_weight_decay,
        "final_joint_mixed_precision_acc": mixed_metrics["test_acc"],
        "state_quantization_note": (
            "This experiment approximates state precision by quantizing continuous inputs "
            "to each LIF node, rather than modifying the neuron's internal membrane update."
        ),
        "outputs": {
            "weight_bit_allocation_csv": str(weight_csv),
            "state_bit_allocation_csv": str(state_csv),
            "comparison_csv": str(comparison_csv),
            "weight_state_quant_finetune_epoch_metrics_csv": str(quant_epoch_csv),
            "weight_sensitivity_ranking_figure": figure_path,
            "summary_json": str(summary_json),
        },
        "best_setting_by_acc": max(comparison_rows, key=lambda x: x["test_acc"])["setting"],
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
