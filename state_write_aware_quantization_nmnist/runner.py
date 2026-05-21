from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from spikingjelly.activation_based import functional, neuron

from baseline_nmnist.config import BaselineNMNISTConfig
from baseline_nmnist.data import build_nmnist_loaders
from baseline_nmnist.hessian import _iter_with_limit, _load_checkpoint, _quantize_tensor_symmetric_per_tensor
from baseline_nmnist.metrics import (
    SpikeRateTracker,
    model_size_mb,
    parameter_count,
    sop_proxy,
    synapse_count_proxy,
)
from baseline_nmnist.model import build_model
from baseline_nmnist.unquant_runner import direct_encode, set_global_seed, write_epoch_metrics_csv
from baseline_nmnist.weight_state_mixed import estimate_state_cost_proxy


def _collect_state_layers(model: nn.Module) -> List[str]:
    layer_names: List[str] = []
    for name, module in model.named_modules():
        if isinstance(module, neuron.BaseNode):
            layer_names.append(name)
    return layer_names


class _StateWriteAwareQuantizer:
    def __init__(
        self,
        model: nn.Module,
        state_bits: int,
        relative_delta_threshold: float,
        write_warmup_steps: int,
    ) -> None:
        self.state_bits = int(state_bits)
        self.relative_delta_threshold = float(relative_delta_threshold)
        self.write_warmup_steps = int(write_warmup_steps)
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._cache: Dict[str, torch.Tensor] = {}
        self._current_step = 0
        self._total_writes = 0
        self._total_assignments = 0
        self._layer_writes: Dict[str, int] = {}
        self._layer_assignments: Dict[str, int] = {}

        for name, module in model.named_modules():
            if not isinstance(module, neuron.BaseNode):
                continue
            self._layer_writes[name] = 0
            self._layer_assignments[name] = 0
            self._handles.append(module.register_forward_pre_hook(self._make_pre_hook(name)))

    def reset_batch(self) -> None:
        self._cache.clear()
        self._current_step = 0

    def begin_timestep(self, step_idx: int) -> None:
        self._current_step = int(step_idx)

    def _make_pre_hook(self, layer_name: str):
        def hook(_module: nn.Module, inputs):
            if not inputs:
                return inputs
            x = inputs[0]
            if not torch.is_tensor(x):
                return inputs
            q_x = self._quantize_or_reuse(layer_name, x)
            if len(inputs) == 1:
                return (q_x,)
            return (q_x,) + tuple(inputs[1:])

        return hook

    def _quantize_or_reuse(self, layer_name: str, x: torch.Tensor) -> torch.Tensor:
        q_current = _quantize_tensor_symmetric_per_tensor(x, self.state_bits)
        batch_dim = 1 if x.ndim >= 2 and x.shape[0] == 1 else 0
        batch_size = int(x.shape[batch_dim])

        self._total_assignments += batch_size
        self._layer_assignments[layer_name] += batch_size

        if (
            self._current_step < self.write_warmup_steps
            or layer_name not in self._cache
            or self.relative_delta_threshold < 0.0
        ):
            self._cache[layer_name] = q_current.detach()
            self._total_writes += batch_size
            self._layer_writes[layer_name] += batch_size
            return q_current

        prev = self._cache[layer_name]
        flat_current = q_current.movedim(batch_dim, 0).reshape(batch_size, -1)
        flat_prev = prev.movedim(batch_dim, 0).reshape(batch_size, -1)
        delta = (flat_current - flat_prev).abs().mean(dim=1)
        denom = flat_prev.abs().mean(dim=1).clamp_min(1e-6)
        relative_delta = delta / denom
        write_mask = relative_delta > self.relative_delta_threshold

        if bool(write_mask.all()):
            self._cache[layer_name] = q_current.detach()
            self._total_writes += batch_size
            self._layer_writes[layer_name] += batch_size
            return q_current

        if not bool(write_mask.any()):
            return prev

        out = prev.clone()
        indices = write_mask.nonzero(as_tuple=False).flatten()
        current_subset = q_current.index_select(batch_dim, indices)
        out.index_copy_(batch_dim, indices, current_subset)
        self._cache[layer_name] = out.detach()
        write_count = int(write_mask.sum().item())
        self._total_writes += write_count
        self._layer_writes[layer_name] += write_count
        return out

    def summary(self) -> dict:
        write_ratio = self._total_writes / float(max(self._total_assignments, 1))
        layer_rows: List[dict] = []
        for layer_name in self._layer_assignments:
            assignments = self._layer_assignments[layer_name]
            writes = self._layer_writes[layer_name]
            layer_rows.append(
                {
                    "state_layer_name": layer_name,
                    "writes": writes,
                    "assignments": assignments,
                    "write_ratio": writes / float(max(assignments, 1)),
                }
            )
        return {
            "write_ratio": write_ratio,
            "total_writes": self._total_writes,
            "total_assignments": self._total_assignments,
            "layer_rows": layer_rows,
        }

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._cache.clear()


def _run_sequence(
    model: nn.Module,
    x_seq: torch.Tensor,
    quantizer: _StateWriteAwareQuantizer,
) -> torch.Tensor:
    quantizer.reset_batch()
    logits_steps: List[torch.Tensor] = []
    for step_idx in range(x_seq.shape[0]):
        quantizer.begin_timestep(step_idx)
        logits_step = model(x_seq[step_idx : step_idx + 1])
        if logits_step.ndim == 3 and logits_step.shape[0] == 1:
            logits_step = logits_step.squeeze(0)
        logits_steps.append(logits_step)
    return torch.stack(logits_steps, dim=0)


def _estimate_state_write_bytes_per_sample(
    total_state_cost_proxy: float,
    state_bits: int,
    write_ratio: float,
) -> float:
    return total_state_cost_proxy * float(state_bits) / 8.0 * write_ratio


def evaluate_with_quantizer(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: str,
    t_steps: int,
    max_batches: Optional[int],
    quantizer: _StateWriteAwareQuantizer,
    total_state_cost_proxy: float,
) -> dict:
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    infer_time_sum = 0.0
    batch_count = 0

    spike_tracker = SpikeRateTracker(model)
    spike_tracker.reset()
    try:
        for x, y in _iter_with_limit(loader, max_batches):
            x = x.to(device)
            y = y.to(device)
            x_seq = direct_encode(x, t_steps)

            start = time.perf_counter()
            logits_seq = _run_sequence(model, x_seq, quantizer)
            infer_time_sum += time.perf_counter() - start
            batch_count += 1

            logits = logits_seq.mean(dim=0)
            loss = criterion(logits, y)
            batch_size = y.shape[0]
            loss_sum += float(loss.item()) * batch_size
            correct += int((logits.argmax(dim=1) == y).sum().item())
            total += batch_size

            functional.reset_net(model)
    finally:
        summary = quantizer.summary()
        quantizer.close()

    spike_rate = spike_tracker.rate()
    spike_tracker.close()
    return {
        "test_loss": loss_sum / float(max(total, 1)),
        "test_acc": correct / float(max(total, 1)),
        "spike_rate": spike_rate,
        "avg_batch_infer_ms": infer_time_sum / float(max(batch_count, 1)) * 1000.0,
        "write_ratio": summary["write_ratio"],
        "estimated_state_write_bytes_per_sample": _estimate_state_write_bytes_per_sample(
            total_state_cost_proxy, quantizer.state_bits, summary["write_ratio"]
        ),
        "layer_write_rows": summary["layer_rows"],
    }


def finetune_with_quantizer(
    model: nn.Module,
    train_loader,
    test_loader,
    criterion: nn.Module,
    device: str,
    t_steps: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    max_train_batches: Optional[int],
    max_test_batches: Optional[int],
    synapse_count: int,
    state_bits: int,
    relative_delta_threshold: float,
    write_warmup_steps: int,
    total_state_cost_proxy: float,
) -> List[dict]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    rows: List[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_quantizer = _StateWriteAwareQuantizer(
            model=model,
            state_bits=state_bits,
            relative_delta_threshold=relative_delta_threshold,
            write_warmup_steps=write_warmup_steps,
        )
        loss_sum = 0.0
        correct = 0
        total = 0
        try:
            for x, y in _iter_with_limit(train_loader, max_train_batches):
                x = x.to(device)
                y = y.to(device)
                x_seq = direct_encode(x, t_steps)

                optimizer.zero_grad(set_to_none=True)
                logits_seq = _run_sequence(model, x_seq, train_quantizer)
                logits = logits_seq.mean(dim=0)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

                batch_size = y.shape[0]
                loss_sum += float(loss.item()) * batch_size
                correct += int((logits.argmax(dim=1) == y).sum().item())
                total += batch_size

                functional.reset_net(model)
        finally:
            train_summary = train_quantizer.summary()
            train_quantizer.close()

        test_quantizer = _StateWriteAwareQuantizer(
            model=model,
            state_bits=state_bits,
            relative_delta_threshold=relative_delta_threshold,
            write_warmup_steps=write_warmup_steps,
        )
        test_metrics = evaluate_with_quantizer(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            t_steps=t_steps,
            max_batches=max_test_batches,
            quantizer=test_quantizer,
            total_state_cost_proxy=total_state_cost_proxy,
        )
        rows.append(
            {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": loss_sum / float(max(total, 1)),
                "train_acc": correct / float(max(total, 1)),
                "train_write_ratio": train_summary["write_ratio"],
                "train_estimated_state_write_bytes_per_sample": _estimate_state_write_bytes_per_sample(
                    total_state_cost_proxy, state_bits, train_summary["write_ratio"]
                ),
                **{k: v for k, v in test_metrics.items() if k != "layer_write_rows"},
                "sop_proxy": sop_proxy(test_metrics["spike_rate"], synapse_count, t_steps),
            }
        )
    return rows


def run_state_write_aware_quantization_analysis(
    cfg: BaselineNMNISTConfig,
    checkpoint_path: str,
    state_bits: int,
    relative_delta_threshold: float = 0.05,
    write_warmup_steps: int = 4,
    output_dir: str = "outputs/state_write_aware_quantization_nmnist",
    max_hessian_batches: Optional[int] = None,
    max_test_batches: Optional[int] = None,
    quant_epochs: int = 1,
    quant_lr: float = 1e-4,
    quant_weight_decay: float = 5e-4,
) -> dict:
    if state_bits < 2:
        raise ValueError("state_bits must be >= 2.")
    if write_warmup_steps < 0:
        raise ValueError("write_warmup_steps must be >= 0.")

    set_global_seed(cfg.seed, cfg.deterministic)
    device = cfg.resolve_device()
    train_loader, test_loader = build_nmnist_loaders(cfg, device=device)

    fp32_model = build_model(num_classes=10).to(device)
    _load_checkpoint(fp32_model, checkpoint_path=checkpoint_path)

    state_rows = estimate_state_cost_proxy(
        model=fp32_model,
        loader=train_loader,
        device=device,
        t_steps=cfg.t_steps,
        max_batches=max_hessian_batches,
    )
    total_state_cost_proxy = sum(float(row["state_cost_proxy"]) for row in state_rows)

    criterion = nn.CrossEntropyLoss()
    eval_limit = max_test_batches if max_test_batches is not None else cfg.max_test_batches
    params = parameter_count(fp32_model)
    model_size = model_size_mb(fp32_model)
    synapse_count = synapse_count_proxy(fp32_model)

    fp32_quantizer = _StateWriteAwareQuantizer(
        model=fp32_model,
        state_bits=32,
        relative_delta_threshold=-1.0,
        write_warmup_steps=cfg.t_steps,
    )
    fp32_metrics = evaluate_with_quantizer(
        model=fp32_model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        t_steps=cfg.t_steps,
        max_batches=eval_limit,
        quantizer=fp32_quantizer,
        total_state_cost_proxy=total_state_cost_proxy,
    )

    fixed_model = copy.deepcopy(fp32_model).to(device)
    fixed_rows = finetune_with_quantizer(
        model=fixed_model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
        t_steps=cfg.t_steps,
        epochs=quant_epochs,
        lr=quant_lr,
        weight_decay=quant_weight_decay,
        max_train_batches=cfg.max_train_batches,
        max_test_batches=eval_limit,
        synapse_count=synapse_count,
        state_bits=state_bits,
        relative_delta_threshold=-1.0,
        write_warmup_steps=cfg.t_steps,
        total_state_cost_proxy=total_state_cost_proxy,
    )
    fixed_quantizer = _StateWriteAwareQuantizer(
        model=fixed_model,
        state_bits=state_bits,
        relative_delta_threshold=-1.0,
        write_warmup_steps=cfg.t_steps,
    )
    fixed_metrics = evaluate_with_quantizer(
        model=fixed_model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        t_steps=cfg.t_steps,
        max_batches=eval_limit,
        quantizer=fixed_quantizer,
        total_state_cost_proxy=total_state_cost_proxy,
    )

    write_aware_model = copy.deepcopy(fp32_model).to(device)
    write_aware_rows = finetune_with_quantizer(
        model=write_aware_model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
        t_steps=cfg.t_steps,
        epochs=quant_epochs,
        lr=quant_lr,
        weight_decay=quant_weight_decay,
        max_train_batches=cfg.max_train_batches,
        max_test_batches=eval_limit,
        synapse_count=synapse_count,
        state_bits=state_bits,
        relative_delta_threshold=relative_delta_threshold,
        write_warmup_steps=write_warmup_steps,
        total_state_cost_proxy=total_state_cost_proxy,
    )
    write_aware_quantizer = _StateWriteAwareQuantizer(
        model=write_aware_model,
        state_bits=state_bits,
        relative_delta_threshold=relative_delta_threshold,
        write_warmup_steps=write_warmup_steps,
    )
    write_aware_metrics = evaluate_with_quantizer(
        model=write_aware_model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        t_steps=cfg.t_steps,
        max_batches=eval_limit,
        quantizer=write_aware_quantizer,
        total_state_cost_proxy=total_state_cost_proxy,
    )

    comparison_rows = [
        {
            "setting": "FP32State",
            "test_acc": fp32_metrics["test_acc"],
            "spike_rate": fp32_metrics["spike_rate"],
            "avg_batch_infer_ms": fp32_metrics["avg_batch_infer_ms"],
            "sop_proxy": sop_proxy(fp32_metrics["spike_rate"], synapse_count, cfg.t_steps),
            "state_bits": 32.0,
            "write_ratio": fp32_metrics["write_ratio"],
            "estimated_state_write_bytes_per_sample": fp32_metrics["estimated_state_write_bytes_per_sample"],
        },
        {
            "setting": f"FixedStateB{state_bits}",
            "test_acc": fixed_metrics["test_acc"],
            "spike_rate": fixed_metrics["spike_rate"],
            "avg_batch_infer_ms": fixed_metrics["avg_batch_infer_ms"],
            "sop_proxy": sop_proxy(fixed_metrics["spike_rate"], synapse_count, cfg.t_steps),
            "state_bits": float(state_bits),
            "write_ratio": fixed_metrics["write_ratio"],
            "estimated_state_write_bytes_per_sample": fixed_metrics["estimated_state_write_bytes_per_sample"],
        },
        {
            "setting": f"StateWriteAwareB{state_bits}",
            "test_acc": write_aware_metrics["test_acc"],
            "spike_rate": write_aware_metrics["spike_rate"],
            "avg_batch_infer_ms": write_aware_metrics["avg_batch_infer_ms"],
            "sop_proxy": sop_proxy(write_aware_metrics["spike_rate"], synapse_count, cfg.t_steps),
            "state_bits": float(state_bits),
            "write_ratio": write_aware_metrics["write_ratio"],
            "estimated_state_write_bytes_per_sample": write_aware_metrics["estimated_state_write_bytes_per_sample"],
        },
    ]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    state_proxy_csv = output_path / "state_cost_proxy.csv"
    layer_write_csv = output_path / "layer_write_ratio.csv"
    comparison_csv = output_path / "comparison.csv"
    fixed_epoch_csv = output_path / "fixed_state_epoch_metrics.csv"
    write_aware_epoch_csv = output_path / "state_write_aware_epoch_metrics.csv"
    summary_json = output_path / "summary.json"

    with state_proxy_csv.open("w", encoding="utf-8") as f:
        f.write("state_layer_name,state_cost_proxy,state_rank\n")
        for row in state_rows:
            f.write(f"{row['state_layer_name']},{row['state_cost_proxy']},{row['state_rank']}\n")
    with layer_write_csv.open("w", encoding="utf-8") as f:
        f.write("state_layer_name,writes,assignments,write_ratio\n")
        for row in write_aware_metrics["layer_write_rows"]:
            f.write(f"{row['state_layer_name']},{row['writes']},{row['assignments']},{row['write_ratio']}\n")
    with comparison_csv.open("w", encoding="utf-8") as f:
        headers = list(comparison_rows[0].keys())
        f.write(",".join(headers) + "\n")
        for row in comparison_rows:
            f.write(",".join(str(row[h]) for h in headers) + "\n")
    write_epoch_metrics_csv(fixed_epoch_csv, fixed_rows)
    write_epoch_metrics_csv(write_aware_epoch_csv, write_aware_rows)

    summary = {
        "method": "state_write_aware_quantization_proxy_experiment",
        "device": device,
        "checkpoint_path": checkpoint_path,
        "params": params,
        "model_size_mb": model_size,
        "state_bits": state_bits,
        "relative_delta_threshold": relative_delta_threshold,
        "write_warmup_steps": write_warmup_steps,
        "total_state_cost_proxy": total_state_cost_proxy,
        "max_hessian_batches": max_hessian_batches,
        "max_test_batches": eval_limit,
        "quant_epochs": quant_epochs,
        "quant_lr": quant_lr,
        "quant_weight_decay": quant_weight_decay,
        "final_fixed_state_acc": fixed_metrics["test_acc"],
        "final_state_write_aware_acc": write_aware_metrics["test_acc"],
        "state_write_aware_write_ratio": write_aware_metrics["write_ratio"],
        "state_quantization_note": (
            "This experiment approximates state-write savings by quantizing continuous inputs "
            "to each LIF node and conditionally reusing the previous quantized value, "
            "rather than modifying the neuron's internal membrane update."
        ),
        "outputs": {
            "state_cost_proxy_csv": str(state_proxy_csv),
            "layer_write_ratio_csv": str(layer_write_csv),
            "comparison_csv": str(comparison_csv),
            "fixed_state_epoch_metrics_csv": str(fixed_epoch_csv),
            "state_write_aware_epoch_metrics_csv": str(write_aware_epoch_csv),
            "summary_json": str(summary_json),
        },
        "best_setting_by_acc": max(comparison_rows, key=lambda item: item["test_acc"])["setting"],
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
