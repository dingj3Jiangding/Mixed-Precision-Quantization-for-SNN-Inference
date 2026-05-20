from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import functional

from baseline_nmnist.config import BaselineNMNISTConfig
from baseline_nmnist.data import build_nmnist_loaders
from baseline_nmnist.hessian import (
    _apply_layer_weight_quantization_,
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
from baseline_nmnist.metrics import parameter_count, sop_proxy, synapse_count_proxy
from baseline_nmnist.model import build_model
from baseline_nmnist.quantization import clone_and_quantize_model_weights, parse_bits_list
from baseline_nmnist.unquant_runner import direct_encode, evaluate, set_global_seed, write_epoch_metrics_csv
from baseline_nmnist.weight_state_mixed import load_weight_allocation_rows


def _soft_kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0.0:
        raise ValueError("temperature must be > 0.")
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)


def _summarize_activation(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 5:
        return x.mean(dim=0).mean(dim=(2, 3))
    if x.ndim == 3:
        return x.mean(dim=0)
    if x.ndim == 2:
        return x
    raise ValueError(f"Unsupported activation shape for distillation: {tuple(x.shape)}")


class _FeatureRecorder:
    def __init__(self, model: nn.Module, layer_names: Iterable[str]) -> None:
        self.features: Dict[str, torch.Tensor] = {}
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        wanted = set(layer_names)
        for name, module in model.named_modules():
            if name not in wanted:
                continue
            self._handles.append(module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, layer_name: str):
        def hook(_module: nn.Module, _inputs, output):
            out = output[0] if isinstance(output, (tuple, list)) else output
            if torch.is_tensor(out):
                self.features[layer_name] = _summarize_activation(out)

        return hook

    def clear(self) -> None:
        self.features.clear()

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.features.clear()


def _build_layer_severity(
    layer_rows: List[dict],
    layer_bits: Dict[str, int],
    bits_sorted: List[int],
    severity_power: float,
) -> Dict[str, float]:
    max_bit = max(bits_sorted)
    min_bit = min(bits_sorted)
    bit_span = max(max_bit - min_bit, 1)
    target_rows = [
        row
        for row in layer_rows
        if row["layer_name"] != "classifier" and int(layer_bits[row["layer_name"]]) < max_bit
    ]
    if not target_rows:
        return {}

    metric_key = "trace_density"
    if not any(abs(float(row.get(metric_key, 0.0))) > 0.0 for row in target_rows):
        metric_key = "hessian_trace"

    metric_by_layer = {
        row["layer_name"]: abs(float(row.get(metric_key, 0.0)))
        for row in target_rows
    }
    max_metric = max(metric_by_layer.values()) or 1.0

    ranked_layers = sorted(
        target_rows,
        key=lambda row: metric_by_layer[row["layer_name"]],
        reverse=True,
    )
    rank_count = max(len(ranked_layers) - 1, 1)
    rank_score_by_layer = {
        row["layer_name"]: 1.0 - (idx / float(rank_count))
        for idx, row in enumerate(ranked_layers)
    }

    severity: Dict[str, float] = {}
    for row in target_rows:
        layer_name = row["layer_name"]
        assigned_bit = int(layer_bits[layer_name])
        bit_pressure = ((max_bit - assigned_bit) / float(bit_span)) ** severity_power
        metric_ratio = metric_by_layer[layer_name] / max_metric
        metric_score = math.sqrt(metric_ratio)
        rank_score = rank_score_by_layer[layer_name]
        raw = bit_pressure * (0.7 * metric_score + 0.3 * rank_score)
        severity[layer_name] = raw

    max_raw = max(severity.values()) or 1.0
    for layer_name, raw in list(severity.items()):
        normalized = raw / max_raw
        severity[layer_name] = 0.1 + 0.9 * normalized
    return severity


def _allocation_aware_feature_loss(
    student_features: Dict[str, torch.Tensor],
    teacher_features: Dict[str, torch.Tensor],
    layer_severity: Dict[str, float],
) -> torch.Tensor:
    losses: List[torch.Tensor] = []
    for layer_name, severity in layer_severity.items():
        student = student_features.get(layer_name)
        teacher = teacher_features.get(layer_name)
        if student is None or teacher is None:
            continue
        student_norm = F.normalize(student, dim=1)
        teacher_norm = F.normalize(teacher.detach(), dim=1)
        losses.append(student_norm.new_tensor(float(severity)) * F.mse_loss(student_norm, teacher_norm))

    if not losses:
        sample = next(iter(student_features.values()), None)
        if sample is None:
            return torch.tensor(0.0)
        return sample.new_zeros(())
    return torch.stack(losses).sum()


def finetune_with_allocation_aware_distillation(
    student_model: nn.Module,
    teacher_model: nn.Module,
    train_loader,
    test_loader,
    layer_bits: Dict[str, int],
    layer_severity: Dict[str, float],
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
    feature_distill_beta: float,
) -> List[dict]:
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=lr, weight_decay=weight_decay)
    rows: List[dict] = []
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad_(False)

    recorder_layers = list(layer_severity.keys())
    teacher_recorder = _FeatureRecorder(teacher_model, recorder_layers)
    student_recorder = _FeatureRecorder(student_model, recorder_layers)

    try:
        for epoch in range(1, epochs + 1):
            student_model.train()
            loss_sum = 0.0
            ce_loss_sum = 0.0
            kd_loss_sum = 0.0
            feature_loss_sum = 0.0
            correct = 0
            total = 0

            for x, y in _iter_with_limit(train_loader, max_train_batches):
                x = x.to(device)
                y = y.to(device)
                x_seq = direct_encode(x, t_steps)

                teacher_recorder.clear()
                student_recorder.clear()
                with torch.no_grad():
                    teacher_logits_seq = teacher_model(x_seq)
                    teacher_logits = teacher_logits_seq.mean(dim=0)

                optimizer.zero_grad(set_to_none=True)
                original_weights = _quantize_layer_weights_for_forward_(student_model, layer_bits)
                student_logits_seq = student_model(x_seq)
                student_logits = student_logits_seq.mean(dim=0)

                ce_loss = criterion(student_logits, y)
                kd_loss = _soft_kd_loss(student_logits, teacher_logits, distill_temperature)
                feature_loss = _allocation_aware_feature_loss(
                    student_features=student_recorder.features,
                    teacher_features=teacher_recorder.features,
                    layer_severity=layer_severity,
                )
                loss = (1.0 - distill_alpha) * ce_loss + distill_alpha * kd_loss + feature_distill_beta * feature_loss
                loss.backward()
                _restore_layer_weights_(student_model, original_weights)
                optimizer.step()

                batch_size = y.shape[0]
                loss_sum += float(loss.item()) * batch_size
                ce_loss_sum += float(ce_loss.item()) * batch_size
                kd_loss_sum += float(kd_loss.item()) * batch_size
                feature_loss_sum += float(feature_loss.item()) * batch_size
                correct += int((student_logits.argmax(dim=1) == y).sum().item())
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
            rows.append(
                {
                    "epoch": epoch,
                    "lr": optimizer.param_groups[0]["lr"],
                    "train_loss": loss_sum / max(total, 1),
                    "train_ce_loss": ce_loss_sum / max(total, 1),
                    "train_kd_loss": kd_loss_sum / max(total, 1),
                    "train_feature_loss": feature_loss_sum / max(total, 1),
                    "train_acc": correct / max(total, 1),
                    **test_metrics,
                    "sop_proxy": sop_proxy(test_metrics["spike_rate"], synapse_count, t_steps),
                }
            )
    finally:
        teacher_recorder.close()
        student_recorder.close()

    return rows


def run_allocation_aware_distillation_analysis(
    cfg: BaselineNMNISTConfig,
    checkpoint_path: str,
    bits_list: Iterable[int],
    target_avg_bits: Optional[float] = None,
    output_dir: str = "outputs/allocation_aware_kd_nmnist",
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
    feature_distill_beta: float = 0.1,
    severity_power: float = 1.0,
) -> dict:
    set_global_seed(cfg.seed, cfg.deterministic)
    device = cfg.resolve_device()
    train_loader, test_loader = build_nmnist_loaders(cfg, device=device)

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

    bits_sorted = parse_bits_list(bits_list)
    layer_severity = _build_layer_severity(layer_rows, layer_bits, bits_sorted, severity_power)
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
    student_model = copy.deepcopy(fp32_model).to(device)
    _apply_layer_weight_quantization_(student_model, layer_bits)

    epoch_rows = finetune_with_allocation_aware_distillation(
        student_model=student_model,
        teacher_model=teacher_model,
        train_loader=train_loader,
        test_loader=test_loader,
        layer_bits=layer_bits,
        layer_severity=layer_severity,
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
        feature_distill_beta=feature_distill_beta,
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
            "setting": f"AllocAwareKD_T{distill_temperature:g}_a{distill_alpha:g}_b{feature_distill_beta:g}",
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
    severity_csv = output_path / "layer_severity.csv"
    comparison_csv = output_path / "comparison.csv"
    quant_epoch_csv = output_path / "allocation_aware_kd_epoch_metrics.csv"
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
    _write_csv(
        severity_csv,
        [
            {
                "layer_name": row["layer_name"],
                "assigned_bits": int(layer_bits[row["layer_name"]]),
                "hessian_trace": row.get("hessian_trace", 0.0),
                "severity": float(layer_severity.get(row["layer_name"], 0.0)),
            }
            for row in layer_rows
            if row["layer_name"] != "classifier"
        ],
    )
    _write_csv(comparison_csv, comparison_rows)
    write_epoch_metrics_csv(quant_epoch_csv, epoch_rows)
    figure_path = _try_plot_sensitivity(layer_rows, ranking_figure)

    summary = {
        "method": "allocation_aware_distillation_quantization",
        "device": device,
        "checkpoint_path": checkpoint_path,
        "params": params,
        "target_avg_bits": target_avg_bits,
        "achieved_avg_bits": achieved_avg_bits,
        "bits_list": bits_sorted,
        "allocation_policy": allocation_policy,
        "weight_allocation_csv": weight_allocation_csv,
        "assigned_bits": layer_bits,
        "layer_severity": layer_severity,
        "uniform_reference_bits": uniform_ref_bits,
        "trace_probes": trace_probes,
        "max_hessian_batches": max_hessian_batches,
        "max_test_batches": eval_limit,
        "quant_epochs": quant_epochs,
        "quant_lr": quant_lr,
        "quant_weight_decay": quant_weight_decay,
        "distill_alpha": distill_alpha,
        "distill_temperature": distill_temperature,
        "feature_distill_beta": feature_distill_beta,
        "severity_power": severity_power,
        "final_allocation_aware_acc": distilled_metrics["test_acc"],
        "outputs": {
            "layer_sensitivity_csv": str(sensitivity_csv),
            "bit_allocation_csv": str(allocation_csv),
            "layer_severity_csv": str(severity_csv),
            "comparison_csv": str(comparison_csv),
            "allocation_aware_kd_epoch_metrics_csv": str(quant_epoch_csv),
            "ranking_figure": figure_path,
            "summary_json": str(summary_json),
        },
        "best_setting_by_acc": max(comparison_rows, key=lambda x: x["test_acc"])["setting"],
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
