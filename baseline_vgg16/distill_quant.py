from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import functional

from .config import BaselineVGG16Config
from .data import build_cifar10_loaders
from .hessian import (
    _apply_layer_weight_quantization_,
    _collect_weight_layers,
    _iter_with_limit,
    _load_checkpoint,
    _quantize_layer_weights_for_forward_,
    _restore_layer_weights_,
    _try_plot_sensitivity,
    _weighted_average_bits,
    _write_csv,
    assign_bits_by_sensitivity_rank,
    estimate_layer_sensitivity,
)
from .metrics import parameter_count, sop_proxy, synapse_count_proxy
from .model import build_model
from .quantization import clone_and_quantize_model_weights, parse_bits_list
from .unquant_runner import direct_encode, evaluate, set_global_seed, write_epoch_metrics_csv
from .weight_state_mixed import load_weight_allocation_rows


def _soft_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if temperature <= 0.0:
        raise ValueError("temperature must be > 0.")
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)


def finetune_mixed_precision_with_distillation(
    student_model: nn.Module,
    teacher_model: nn.Module,
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
    distill_alpha: float,
    distill_temperature: float,
) -> List[dict]:
    if epochs < 0:
        raise ValueError("quantization fine-tuning epochs must be >= 0.")
    if not 0.0 <= distill_alpha <= 1.0:
        raise ValueError("distill_alpha must be in [0, 1].")

    optimizer = torch.optim.AdamW(student_model.parameters(), lr=lr, weight_decay=weight_decay)
    rows: List[dict] = []
    teacher_model.eval()

    for epoch in range(1, epochs + 1):
        student_model.train()
        loss_sum = 0.0
        ce_loss_sum = 0.0
        kd_loss_sum = 0.0
        correct = 0
        total = 0

        for x, y in _iter_with_limit(train_loader, max_train_batches):
            x = x.to(device)
            y = y.to(device)
            x_seq = direct_encode(x, t_steps)

            with torch.no_grad():
                teacher_logits_seq = teacher_model(x_seq)
                teacher_logits = teacher_logits_seq.mean(dim=0)

            optimizer.zero_grad(set_to_none=True)
            original_weights = _quantize_layer_weights_for_forward_(student_model, layer_bits)
            student_logits_seq = student_model(x_seq)
            student_logits = student_logits_seq.mean(dim=0)

            ce_loss = criterion(student_logits, y)
            kd_loss = _soft_kd_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                temperature=distill_temperature,
            )
            loss = (1.0 - distill_alpha) * ce_loss + distill_alpha * kd_loss
            loss.backward()
            _restore_layer_weights_(student_model, original_weights)
            optimizer.step()

            batch_size = y.shape[0]
            loss_sum += float(loss.item()) * batch_size
            ce_loss_sum += float(ce_loss.item()) * batch_size
            kd_loss_sum += float(kd_loss.item()) * batch_size
            pred = student_logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += batch_size

            functional.reset_net(student_model)
            functional.reset_net(teacher_model)

        _apply_layer_weight_quantization_(student_model, layer_bits)
        test_metrics = evaluate(
            model=student_model,
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
            "train_ce_loss": ce_loss_sum / max(total, 1),
            "train_kd_loss": kd_loss_sum / max(total, 1),
            "train_acc": correct / max(total, 1),
            **test_metrics,
            "sop_proxy": sop_proxy(test_metrics["spike_rate"], synapse_count, t_steps),
        }
        rows.append(row)
    return rows


def run_distillation_quantization_analysis(
    cfg: BaselineVGG16Config,
    checkpoint_path: str,
    bits_list: Iterable[int],
    target_avg_bits: Optional[float] = None,
    output_dir: str = "outputs/baseline_vgg16_distill_quant",
    weight_allocation_csv: Optional[str] = None,
    max_hessian_batches: Optional[int] = None,
    max_test_batches: Optional[int] = None,
    trace_probes: int = 1,
    quant_epochs: int = 5,
    quant_lr: float = 1e-4,
    quant_weight_decay: float = 5e-4,
    allocation_policy: str = "rank-map",
    distill_alpha: float = 0.5,
    distill_temperature: float = 2.0,
) -> dict:
    set_global_seed(cfg.seed, cfg.deterministic)
    device = cfg.resolve_device()
    train_loader, test_loader = build_cifar10_loaders(cfg, device=device)

    fp32_model = build_model(num_classes=10).to(device)
    _load_checkpoint(fp32_model, checkpoint_path=checkpoint_path)

    if weight_allocation_csv is not None:
        layer_rows = load_weight_allocation_rows(weight_allocation_csv)
        layer_bits = {row["layer_name"]: int(row["assigned_bits"]) for row in layer_rows}
        achieved_avg_bits = _weighted_average_bits(layer_rows, layer_bits)
    else:
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

    teacher_model = copy.deepcopy(fp32_model).to(device)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad_(False)

    student_model = copy.deepcopy(fp32_model).to(device)
    _apply_layer_weight_quantization_(student_model, layer_bits)
    epoch_rows = finetune_mixed_precision_with_distillation(
        student_model=student_model,
        teacher_model=teacher_model,
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
        distill_alpha=distill_alpha,
        distill_temperature=distill_temperature,
    )
    _apply_layer_weight_quantization_(student_model, layer_bits)
    distilled_metrics = evaluate(
        model=student_model,
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
            "setting": f"HessianKD_T{distill_temperature:g}_a{distill_alpha:g}",
            "test_acc": distilled_metrics["test_acc"],
            "spike_rate": distilled_metrics["spike_rate"],
            "avg_batch_infer_ms": distilled_metrics["avg_batch_infer_ms"],
            "sop_proxy": sop_proxy(distilled_metrics["spike_rate"], synapse_count, cfg.t_steps),
            "avg_weight_bits": achieved_avg_bits,
        },
    ]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    sensitivity_csv = output_path / "layer_sensitivity.csv"
    allocation_csv = output_path / "bit_allocation.csv"
    comparison_csv = output_path / "comparison.csv"
    quant_epoch_csv = output_path / "distill_quant_epoch_metrics.csv"
    summary_json = output_path / "summary.json"
    ranking_figure = output_path / "sensitivity_ranking.png"

    _write_csv(sensitivity_csv, layer_rows)
    _write_csv(
        allocation_csv,
        [
            {
                "layer_name": row["layer_name"],
                "assigned_bits": int(layer_bits[row["layer_name"]]),
                "params": row["params"],
                "hessian_trace": row.get("hessian_trace", 0.0),
                "trace_density": row.get("trace_density", 0.0),
                "rank": row.get("rank", 0),
            }
            for row in layer_rows
        ],
    )
    _write_csv(comparison_csv, comparison_rows)
    write_epoch_metrics_csv(quant_epoch_csv, epoch_rows)
    figure_path = _try_plot_sensitivity(layer_rows, ranking_figure)

    summary = {
        "method": "hessian_guided_distillation_assisted_quantization",
        "device": device,
        "checkpoint_path": checkpoint_path,
        "params": params,
        "target_avg_bits": target_avg_bits,
        "achieved_avg_bits": achieved_avg_bits,
        "bits_list": bits_sorted,
        "allocation_policy": allocation_policy,
        "weight_allocation_csv": weight_allocation_csv,
        "assigned_bits": layer_bits,
        "uniform_reference_bits": uniform_ref_bits,
        "trace_probes": trace_probes,
        "max_hessian_batches": max_hessian_batches,
        "max_test_batches": eval_limit,
        "quant_epochs": quant_epochs,
        "quant_lr": quant_lr,
        "quant_weight_decay": quant_weight_decay,
        "distill_alpha": distill_alpha,
        "distill_temperature": distill_temperature,
        "final_distilled_mixed_precision_acc": distilled_metrics["test_acc"],
        "outputs": {
            "layer_sensitivity_csv": str(sensitivity_csv),
            "bit_allocation_csv": str(allocation_csv),
            "comparison_csv": str(comparison_csv),
            "distill_quant_epoch_metrics_csv": str(quant_epoch_csv),
            "ranking_figure": figure_path,
            "summary_json": str(summary_json),
        },
        "best_setting_by_acc": max(comparison_rows, key=lambda x: x["test_acc"])["setting"],
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
