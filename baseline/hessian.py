from __future__ import annotations

import copy
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from spikingjelly.activation_based import functional

from .config import BaselineConfig
from .data import build_cifar10_loaders
from .metrics import parameter_count, sop_proxy, synapse_count_proxy
from .model import build_model
from .quantization import clone_and_quantize_model_weights, parse_bits_list
from .unquant_runner import direct_encode, evaluate, set_global_seed


def _quantize_tensor_symmetric_per_tensor(tensor: torch.Tensor, bits: int) -> torch.Tensor:
    if bits < 2:
        raise ValueError("bits must be >= 2.")
    if not torch.is_floating_point(tensor):
        return tensor
    max_abs = tensor.detach().abs().max()
    if float(max_abs.item()) == 0.0:
        return tensor.clone()
    qmax = (2 ** (bits - 1)) - 1
    scale = max_abs / qmax
    quantized = torch.round(tensor / scale).clamp(-qmax, qmax)
    return quantized * scale


def _load_checkpoint(model: nn.Module, checkpoint_path: str) -> None:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)


def _iter_with_limit(loader, max_batches: Optional[int]):
    if max_batches is None:
        yield from loader
        return
    for idx, batch in enumerate(loader):
        if idx >= max_batches:
            break
        yield batch


def _collect_weight_layers(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            weight = getattr(module, "weight", None)
            if weight is not None:
                layers.append((name, module))
    return layers


def estimate_layer_sensitivity(
    model: nn.Module,
    loader,
    device: str,
    t_steps: int,
    max_batches: Optional[int],
) -> List[dict]:
    criterion = nn.CrossEntropyLoss()
    model.train()

    weight_layers = _collect_weight_layers(model)
    grad2_acc: Dict[str, torch.Tensor] = {}
    batch_count = 0

    for name, module in weight_layers:
        grad2_acc[name] = torch.zeros_like(module.weight, device=module.weight.device)

    for x, y in _iter_with_limit(loader, max_batches):
        x = x.to(device)
        y = y.to(device)
        x_seq = direct_encode(x, t_steps)

        model.zero_grad(set_to_none=True)
        logits_seq = model(x_seq)
        logits = logits_seq.mean(dim=0)
        loss = criterion(logits, y)
        loss.backward()

        for name, module in weight_layers:
            grad = module.weight.grad
            if grad is None:
                continue
            grad2_acc[name] += grad.detach().pow(2)

        batch_count += 1
        functional.reset_net(model)

    if batch_count == 0:
        raise RuntimeError("No batches were processed for Hessian sensitivity estimation.")

    rows: List[dict] = []
    for name, module in weight_layers:
        grad2_mean = grad2_acc[name] / float(batch_count)
        weight = module.weight.detach()
        score_tensor = grad2_mean * weight.pow(2)
        score = float(score_tensor.sum().item())
        params = int(weight.numel())
        rows.append(
            {
                "layer_name": name,
                "params": params,
                "avg_grad2": float(grad2_mean.mean().item()),
                "sensitivity_score": score,
                "score_density": score / max(params, 1),
            }
        )
    rows.sort(key=lambda item: item["sensitivity_score"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def allocate_bits_by_budget(
    layer_rows: List[dict],
    bits_list: Iterable[int],
    target_avg_bits: float,
) -> Tuple[Dict[str, int], float]:
    bits = sorted(parse_bits_list(bits_list))
    min_bits = bits[0]
    total_params = sum(int(row["params"]) for row in layer_rows)
    if total_params <= 0:
        raise RuntimeError("No quantizable layer parameters found.")

    bit_budget = target_avg_bits * total_params
    min_budget = min_bits * total_params
    if bit_budget < min_budget:
        raise ValueError(
            "target_avg_bits is smaller than min bits in bits_list. "
            f"target_avg_bits={target_avg_bits}, min_bits={min_bits}"
        )

    current_bits: Dict[str, int] = {row["layer_name"]: min_bits for row in layer_rows}
    current_budget = float(min_budget)
    bit_to_index = {bit: idx for idx, bit in enumerate(bits)}

    while True:
        best_choice = None
        best_efficiency = -math.inf

        for row in layer_rows:
            layer_name = row["layer_name"]
            now_bit = current_bits[layer_name]
            now_idx = bit_to_index[now_bit]
            if now_idx >= len(bits) - 1:
                continue
            next_bit = bits[now_idx + 1]
            delta_bits = float((next_bit - now_bit) * int(row["params"]))
            if current_budget + delta_bits > bit_budget + 1e-8:
                continue

            score = float(row["sensitivity_score"])
            gain = score * float(next_bit - now_bit)
            efficiency = gain / max(delta_bits, 1e-12)
            if efficiency > best_efficiency:
                best_efficiency = efficiency
                best_choice = (layer_name, next_bit, delta_bits)

        if best_choice is None:
            break

        layer_name, next_bit, delta_bits = best_choice
        current_bits[layer_name] = next_bit
        current_budget += delta_bits

    achieved_avg_bits = current_budget / float(total_params)
    return current_bits, float(achieved_avg_bits)


def _quantize_model_by_layer_bits(model: nn.Module, layer_bits: Dict[str, int]) -> nn.Module:
    quant_model = copy.deepcopy(model)
    for name, module in quant_model.named_modules():
        if name not in layer_bits:
            continue
        weight = getattr(module, "weight", None)
        if weight is None:
            continue
        with torch.no_grad():
            q_weight = _quantize_tensor_symmetric_per_tensor(weight.data, layer_bits[name])
            weight.data.copy_(q_weight)
    return quant_model


def _write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _try_plot_sensitivity(rows: List[dict], figure_path: Path, top_k: int = 12) -> Optional[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    top_rows = rows[:top_k]
    labels = [row["layer_name"] for row in top_rows]
    scores = [row["sensitivity_score"] for row in top_rows]

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(labels)), scores)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.ylabel("Sensitivity Score")
    plt.title(f"Top-{len(top_rows)} Layer Sensitivity (Hessian Approx)")
    plt.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure_path, dpi=200)
    plt.close()
    return str(figure_path)


def run_hessian_sensitivity_analysis(
    cfg: BaselineConfig,
    checkpoint_path: str,
    bits_list: Iterable[int],
    target_avg_bits: float = 4.0,
    output_dir: str = "outputs/hessian_sensitivity",
    max_hessian_batches: Optional[int] = None,
    max_test_batches: Optional[int] = None,
) -> dict:
    set_global_seed(cfg.seed, cfg.deterministic)
    device = cfg.resolve_device()
    train_loader, test_loader = build_cifar10_loaders(cfg, device=device)

    fp32_model = build_model(num_classes=10).to(device)
    _load_checkpoint(fp32_model, checkpoint_path=checkpoint_path)

    layer_rows = estimate_layer_sensitivity(
        model=fp32_model,
        loader=train_loader,
        device=device,
        t_steps=cfg.t_steps,
        max_batches=max_hessian_batches,
    )
    layer_bits, achieved_avg_bits = allocate_bits_by_budget(
        layer_rows=layer_rows,
        bits_list=bits_list,
        target_avg_bits=target_avg_bits,
    )

    for row in layer_rows:
        row["assigned_bits"] = int(layer_bits[row["layer_name"]])

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

    mixed_model = _quantize_model_by_layer_bits(fp32_model, layer_bits=layer_bits).to(device)
    mixed_metrics = evaluate(
        model=mixed_model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        t_steps=cfg.t_steps,
        max_batches=eval_limit,
    )

    bits_sorted = parse_bits_list(bits_list)
    uniform_ref_bits = min(bits_sorted, key=lambda bit: abs(bit - target_avg_bits))
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
    comparison_rows = [
        {
            "setting": "FP32",
            "test_acc": fp32_metrics["test_acc"],
            "spike_rate": fp32_metrics["spike_rate"],
            "avg_batch_infer_ms": fp32_metrics["avg_batch_infer_ms"],
            "sop_proxy": sop_proxy(fp32_metrics["spike_rate"], synapse_count, cfg.t_steps),
        },
        {
            "setting": f"UniformW{uniform_ref_bits}",
            "test_acc": uniform_metrics["test_acc"],
            "spike_rate": uniform_metrics["spike_rate"],
            "avg_batch_infer_ms": uniform_metrics["avg_batch_infer_ms"],
            "sop_proxy": sop_proxy(uniform_metrics["spike_rate"], synapse_count, cfg.t_steps),
        },
        {
            "setting": "HessianMixed",
            "test_acc": mixed_metrics["test_acc"],
            "spike_rate": mixed_metrics["spike_rate"],
            "avg_batch_infer_ms": mixed_metrics["avg_batch_infer_ms"],
            "sop_proxy": sop_proxy(mixed_metrics["spike_rate"], synapse_count, cfg.t_steps),
        },
    ]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    sensitivity_csv = output_path / "layer_sensitivity.csv"
    allocation_csv = output_path / "bit_allocation.csv"
    comparison_csv = output_path / "comparison.csv"
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
                "sensitivity_score": row["sensitivity_score"],
            }
            for row in layer_rows
        ],
    )
    _write_csv(comparison_csv, comparison_rows)
    figure_path = _try_plot_sensitivity(layer_rows, ranking_figure)

    summary = {
        "device": device,
        "checkpoint_path": checkpoint_path,
        "params": params,
        "target_avg_bits": target_avg_bits,
        "achieved_avg_bits": achieved_avg_bits,
        "bits_list": bits_sorted,
        "uniform_reference_bits": uniform_ref_bits,
        "max_hessian_batches": max_hessian_batches,
        "max_test_batches": eval_limit,
        "outputs": {
            "layer_sensitivity_csv": str(sensitivity_csv),
            "bit_allocation_csv": str(allocation_csv),
            "comparison_csv": str(comparison_csv),
            "ranking_figure": figure_path,
            "summary_json": str(summary_json),
        },
        "best_setting_by_acc": max(comparison_rows, key=lambda x: x["test_acc"])["setting"],
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
