from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from spikingjelly.activation_based import functional

from .config import BaselineNMNISTConfig
from .data import build_nmnist_loaders
from .metrics import parameter_count, sop_proxy, synapse_count_proxy
from .model import build_model
from .quantization import clone_and_quantize_model_weights, parse_bits_list
from .unquant_runner import direct_encode, evaluate, set_global_seed, write_epoch_metrics_csv


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
    trace_probes: int = 1,
) -> List[dict]:
    if trace_probes < 1:
        raise ValueError("trace_probes must be >= 1.")

    criterion = nn.CrossEntropyLoss()
    model.train()

    weight_layers = _collect_weight_layers(model)
    if not weight_layers:
        raise RuntimeError("No quantizable Conv/Linear layers were found.")

    weights = [module.weight for _, module in weight_layers]
    trace_acc: Dict[str, float] = {name: 0.0 for name, _ in weight_layers}
    batch_count = 0

    for x, y in _iter_with_limit(loader, max_batches):
        x = x.to(device)
        y = y.to(device)
        x_seq = direct_encode(x, t_steps)

        model.zero_grad(set_to_none=True)
        logits_seq = model(x_seq)
        logits = logits_seq.mean(dim=0)
        loss = criterion(logits, y)
        grads = torch.autograd.grad(
            loss,
            weights,
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )

        for (name, module), weight, grad in zip(weight_layers, weights, grads):
            if grad is None:
                continue
            for _ in range(trace_probes):
                probe = torch.empty_like(weight).bernoulli_(0.5).mul_(2.0).sub_(1.0)
                probe = probe / probe.norm().clamp_min(1e-12)
                grad_probe = (grad * probe).sum()
                hvp = torch.autograd.grad(
                    grad_probe,
                    weight,
                    retain_graph=True,
                    allow_unused=True,
                )[0]
                if hvp is None:
                    continue
                trace_acc[name] += float((probe * hvp).sum().detach().item()) * int(weight.numel())

        batch_count += 1
        functional.reset_net(model)
        model.zero_grad(set_to_none=True)

    if batch_count == 0:
        raise RuntimeError("No batches were processed for Hessian sensitivity estimation.")

    rows: List[dict] = []
    for name, module in weight_layers:
        weight = module.weight.detach()
        params = int(weight.numel())
        trace = trace_acc[name] / float(batch_count * trace_probes)
        rows.append(
            {
                "layer_name": name,
                "params": params,
                "hessian_trace": trace,
                "trace_density": trace / max(params, 1),
            }
        )
    rows.sort(key=lambda item: item["hessian_trace"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def _weighted_average_bits(layer_rows: List[dict], layer_bits: Dict[str, int]) -> float:
    total_params = sum(int(row["params"]) for row in layer_rows)
    if total_params <= 0:
        raise RuntimeError("No quantizable layer parameters found.")
    return sum(
        int(row["params"]) * int(layer_bits[row["layer_name"]]) for row in layer_rows
    ) / float(total_params)


def assign_bits_by_sensitivity_rank(
    layer_rows: List[dict],
    bits_list: Iterable[int],
    policy: str = "rank-map",
) -> Tuple[Dict[str, int], float]:
    bits = parse_bits_list(bits_list)
    if policy not in {"rank-map", "tiered"}:
        raise ValueError("allocation policy must be either 'rank-map' or 'tiered'.")
    if not layer_rows:
        raise RuntimeError("No layer sensitivity rows were provided.")

    rows = sorted(layer_rows, key=lambda item: item["hessian_trace"], reverse=True)
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

    achieved_avg_bits = _weighted_average_bits(layer_rows, layer_bits)
    for row in layer_rows:
        row["assigned_bits"] = int(layer_bits[row["layer_name"]])
    return layer_bits, float(achieved_avg_bits)


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


def _apply_layer_weight_quantization_(model: nn.Module, layer_bits: Dict[str, int]) -> None:
    for name, module in model.named_modules():
        if name not in layer_bits:
            continue
        weight = getattr(module, "weight", None)
        if weight is None:
            continue
        with torch.no_grad():
            q_weight = _quantize_tensor_symmetric_per_tensor(weight.data, layer_bits[name])
            weight.data.copy_(q_weight)


def _quantize_layer_weights_for_forward_(
    model: nn.Module, layer_bits: Dict[str, int]
) -> Dict[str, torch.Tensor]:
    originals: Dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if name not in layer_bits:
            continue
        weight = getattr(module, "weight", None)
        if weight is None:
            continue
        with torch.no_grad():
            originals[name] = weight.data.detach().clone()
            q_weight = _quantize_tensor_symmetric_per_tensor(weight.data, layer_bits[name])
            weight.data.copy_(q_weight)
    return originals


def _restore_layer_weights_(model: nn.Module, originals: Dict[str, torch.Tensor]) -> None:
    if not originals:
        return
    modules = dict(model.named_modules())
    for name, original in originals.items():
        weight = getattr(modules[name], "weight", None)
        if weight is None:
            continue
        with torch.no_grad():
            weight.data.copy_(original)


def finetune_mixed_precision_model(
    model: nn.Module,
    train_loader,
    test_loader,
    layer_bits: Dict[str, int],
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
    rows: List[dict] = []

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
            original_weights = _quantize_layer_weights_for_forward_(model, layer_bits)
            logits_seq = model(x_seq)
            logits = logits_seq.mean(dim=0)
            loss = criterion(logits, y)
            loss.backward()
            _restore_layer_weights_(model, original_weights)
            optimizer.step()

            batch_size = y.shape[0]
            loss_sum += float(loss.item()) * batch_size
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += batch_size

            functional.reset_net(model)

        _apply_layer_weight_quantization_(model, layer_bits)
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
    return rows


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
    scores = [row["hessian_trace"] for row in top_rows]

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(labels)), scores)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.ylabel("Hessian Trace")
    plt.title(f"Top-{len(top_rows)} Layer Sensitivity (Hutchinson Trace)")
    plt.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure_path, dpi=200)
    plt.close()
    return str(figure_path)


def run_hessian_sensitivity_analysis(
    cfg: BaselineNMNISTConfig,
    checkpoint_path: str,
    bits_list: Iterable[int],
    target_avg_bits: Optional[float] = None,
    output_dir: str = "outputs/baseline_nmnist_hessian_sensitivity",
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
    layer_bits, achieved_avg_bits = assign_bits_by_sensitivity_rank(
        layer_rows=layer_rows,
        bits_list=bits_list,
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
            "setting": "HessianMixed",
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
    allocation_csv = output_path / "bit_allocation.csv"
    comparison_csv = output_path / "comparison.csv"
    quant_epoch_csv = output_path / "quant_finetune_epoch_metrics.csv"
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
            }
            for row in layer_rows
        ],
    )
    _write_csv(comparison_csv, comparison_rows)
    write_epoch_metrics_csv(quant_epoch_csv, epoch_rows)
    figure_path = _try_plot_sensitivity(layer_rows, ranking_figure)

    summary = {
        "method": "hutchinson_hessian_trace_mixed_precision_finetune",
        "device": device,
        "checkpoint_path": checkpoint_path,
        "params": params,
        "target_avg_bits": target_avg_bits,
        "achieved_avg_bits": achieved_avg_bits,
        "bits_list": bits_sorted,
        "allocation_policy": allocation_policy,
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
            "bit_allocation_csv": str(allocation_csv),
            "comparison_csv": str(comparison_csv),
            "quant_finetune_epoch_metrics_csv": str(quant_epoch_csv),
            "ranking_figure": figure_path,
            "summary_json": str(summary_json),
        },
        "best_setting_by_acc": max(comparison_rows, key=lambda x: x["test_acc"])["setting"],
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
