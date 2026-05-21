from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
from spikingjelly.activation_based import functional, neuron

from baseline_vgg16.config import BaselineVGG16Config
from baseline_vgg16.data import build_cifar10_loaders
from baseline_vgg16.hessian import _iter_with_limit, _load_checkpoint, _quantize_tensor_symmetric_per_tensor
from baseline_vgg16.metrics import (
    SpikeRateTracker,
    model_size_mb,
    parameter_count,
    sop_proxy,
    synapse_count_proxy,
)
from baseline_vgg16.model import build_model
from baseline_vgg16.unquant_runner import direct_encode, set_global_seed, write_epoch_metrics_csv
from baseline_vgg16.weight_state_mixed import estimate_state_cost_proxy


def _estimate_state_bytes_per_sample(total_state_cost_proxy: float, avg_state_bits_used: float) -> float:
    return total_state_cost_proxy * avg_state_bits_used / 8.0


def _collect_state_layers(model: nn.Module) -> List[str]:
    layer_names: List[str] = []
    for name, module in model.named_modules():
        if isinstance(module, neuron.BaseNode):
            layer_names.append(name)
    return layer_names


class _DynamicStateInputQuantizer:
    def __init__(self, model: nn.Module, state_layer_names: List[str]) -> None:
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._bits_per_sample: Optional[torch.Tensor] = None
        wanted = set(state_layer_names)
        for name, module in model.named_modules():
            if name not in wanted:
                continue
            self._handles.append(module.register_forward_pre_hook(self._make_pre_hook()))

    def set_bits_per_sample(self, bits_per_sample: torch.Tensor) -> None:
        self._bits_per_sample = bits_per_sample

    def clear_bits_per_sample(self) -> None:
        self._bits_per_sample = None

    def _make_pre_hook(self):
        def hook(_module: nn.Module, inputs):
            if not inputs:
                return inputs
            x = inputs[0]
            if not torch.is_tensor(x) or self._bits_per_sample is None:
                return inputs
            q_x = self._quantize_by_sample_bits(x, self._bits_per_sample)
            if len(inputs) == 1:
                return (q_x,)
            return (q_x,) + tuple(inputs[1:])

        return hook

    @staticmethod
    def _quantize_by_sample_bits(x: torch.Tensor, bits_per_sample: torch.Tensor) -> torch.Tensor:
        if bits_per_sample.ndim != 1:
            raise ValueError("bits_per_sample must be a 1D tensor.")
        batch_dim = 1 if x.ndim >= 2 and x.shape[0] == 1 else 0
        if x.shape[batch_dim] != bits_per_sample.shape[0]:
            raise ValueError("bits_per_sample shape does not match the current batch size.")

        q_x = x.clone()
        unique_bits = torch.unique(bits_per_sample.detach().to(dtype=torch.int64, device="cpu"))
        for bit_value in unique_bits.tolist():
            if int(bit_value) >= 32:
                continue
            mask = bits_per_sample == int(bit_value)
            if not bool(mask.any()):
                continue
            indices = mask.nonzero(as_tuple=False).flatten()
            subset = x.index_select(batch_dim, indices)
            q_subset = _quantize_tensor_symmetric_per_tensor(subset, int(bit_value))
            q_x.index_copy_(batch_dim, indices, q_subset)
        return q_x

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._bits_per_sample = None


class _FixedStatePolicy:
    def __init__(self, bits: int) -> None:
        self.bits = int(bits)
        self._total_bits = 0.0
        self._total_assignments = 0

    def start_batch(self, batch_size: int, device: torch.device | str) -> None:
        self._current_bits = torch.full((batch_size,), self.bits, dtype=torch.int64, device=device)

    def bits_for_step(self, step_idx: int) -> torch.Tensor:
        self._total_bits += float(self._current_bits.sum().item())
        self._total_assignments += int(self._current_bits.numel())
        return self._current_bits

    def observe_logits(self, logits: torch.Tensor, step_idx: int) -> None:
        return None

    def summary(self, high_bits: int) -> dict:
        avg_bits = self._total_bits / float(max(self._total_assignments, 1))
        return {
            "avg_state_bits_used": avg_bits,
            "state_bit_ratio_vs_high": avg_bits / float(max(high_bits, 1)),
            "easy_fraction_post_warmup": 0.0,
        }


class _DifficultyAdaptiveStatePolicy:
    def __init__(
        self,
        low_bits: int,
        high_bits: int,
        warmup_steps: int,
        difficulty_metric: str,
        easy_threshold: float,
    ) -> None:
        self.low_bits = int(low_bits)
        self.high_bits = int(high_bits)
        self.warmup_steps = int(warmup_steps)
        self.difficulty_metric = difficulty_metric
        self.easy_threshold = float(easy_threshold)
        self._total_bits = 0.0
        self._total_assignments = 0
        self._total_post_warmup_assignments = 0
        self._total_easy_assignments = 0

    def start_batch(self, batch_size: int, device: torch.device | str) -> None:
        self._cumulative_logits = None
        self._easy_mask: Optional[torch.Tensor] = None
        self._current_bits = torch.full((batch_size,), self.high_bits, dtype=torch.int64, device=device)

    def bits_for_step(self, step_idx: int) -> torch.Tensor:
        if self._easy_mask is None or step_idx < self.warmup_steps:
            self._current_bits.fill_(self.high_bits)
        else:
            self._current_bits = torch.where(
                self._easy_mask,
                torch.full_like(self._current_bits, self.low_bits),
                torch.full_like(self._current_bits, self.high_bits),
            )
            self._total_post_warmup_assignments += int(self._current_bits.numel())
            self._total_easy_assignments += int(self._easy_mask.sum().item())

        self._total_bits += float(self._current_bits.sum().item())
        self._total_assignments += int(self._current_bits.numel())
        return self._current_bits

    def observe_logits(self, logits: torch.Tensor, step_idx: int) -> None:
        self._cumulative_logits = logits if self._cumulative_logits is None else self._cumulative_logits + logits
        if step_idx + 1 != self.warmup_steps:
            return

        mean_logits = self._cumulative_logits / float(step_idx + 1)
        probs = torch.softmax(mean_logits, dim=1)
        if self.difficulty_metric == "confidence":
            easy_score = probs.max(dim=1).values
        elif self.difficulty_metric == "margin":
            top2 = probs.topk(k=2, dim=1).values
            easy_score = top2[:, 0] - top2[:, 1]
        else:
            raise ValueError(f"Unsupported difficulty metric: {self.difficulty_metric}")
        self._easy_mask = easy_score >= self.easy_threshold

    def summary(self, high_bits: int) -> dict:
        avg_bits = self._total_bits / float(max(self._total_assignments, 1))
        easy_fraction = self._total_easy_assignments / float(max(self._total_post_warmup_assignments, 1))
        return {
            "avg_state_bits_used": avg_bits,
            "state_bit_ratio_vs_high": avg_bits / float(max(high_bits, 1)),
            "easy_fraction_post_warmup": easy_fraction,
        }


def _run_sequence_with_policy(
    model: nn.Module,
    x_seq: torch.Tensor,
    quantizer: _DynamicStateInputQuantizer,
    policy,
) -> torch.Tensor:
    policy.start_batch(batch_size=x_seq.shape[1], device=x_seq.device)
    logits_steps: List[torch.Tensor] = []
    for step_idx in range(x_seq.shape[0]):
        quantizer.set_bits_per_sample(policy.bits_for_step(step_idx))
        logits_step = model(x_seq[step_idx : step_idx + 1])
        if logits_step.ndim == 3 and logits_step.shape[0] == 1:
            logits_step = logits_step.squeeze(0)
        logits_steps.append(logits_step)
        policy.observe_logits(logits_step, step_idx)
    quantizer.clear_bits_per_sample()
    return torch.stack(logits_steps, dim=0)


def evaluate_with_state_policy(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: str,
    t_steps: int,
    max_batches: Optional[int],
    policy_factory: Callable[[], object],
    high_state_bits: int,
) -> dict:
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    infer_time_sum = 0.0
    batch_count = 0

    state_layer_names = _collect_state_layers(model)
    quantizer = _DynamicStateInputQuantizer(model, state_layer_names)
    policy = policy_factory()

    spike_tracker = SpikeRateTracker(model)
    spike_tracker.reset()
    try:
        for x, y in _iter_with_limit(loader, max_batches):
            x = x.to(device)
            y = y.to(device)
            x_seq = direct_encode(x, t_steps)

            start_time = time.perf_counter()
            logits_seq = _run_sequence_with_policy(model, x_seq, quantizer, policy)
            infer_time_sum += time.perf_counter() - start_time
            batch_count += 1

            logits = logits_seq.mean(dim=0)
            loss = criterion(logits, y)

            batch_size = y.shape[0]
            loss_sum += float(loss.item()) * batch_size
            correct += int((logits.argmax(dim=1) == y).sum().item())
            total += batch_size

            functional.reset_net(model)
    finally:
        quantizer.close()

    avg_loss = loss_sum / float(max(total, 1))
    spike_rate = spike_tracker.rate()
    spike_tracker.close()
    return {
        "test_loss": avg_loss,
        "test_acc": correct / float(max(total, 1)),
        "spike_rate": spike_rate,
        "avg_batch_infer_ms": (infer_time_sum / float(max(batch_count, 1))) * 1000.0,
        **policy.summary(high_state_bits),
    }


def finetune_with_state_policy(
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
    policy_factory: Callable[[], object],
    high_state_bits: int,
) -> List[dict]:
    if epochs < 0:
        raise ValueError("quantization fine-tuning epochs must be >= 0.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    state_layer_names = _collect_state_layers(model)
    quantizer = _DynamicStateInputQuantizer(model, state_layer_names)
    rows: List[dict] = []

    try:
        for epoch in range(1, epochs + 1):
            model.train()
            loss_sum = 0.0
            correct = 0
            total = 0
            train_policy = policy_factory()

            for x, y in _iter_with_limit(train_loader, max_train_batches):
                x = x.to(device)
                y = y.to(device)
                x_seq = direct_encode(x, t_steps)

                optimizer.zero_grad(set_to_none=True)
                logits_seq = _run_sequence_with_policy(model, x_seq, quantizer, train_policy)
                logits = logits_seq.mean(dim=0)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

                batch_size = y.shape[0]
                loss_sum += float(loss.item()) * batch_size
                correct += int((logits.argmax(dim=1) == y).sum().item())
                total += batch_size

                functional.reset_net(model)

            test_metrics = evaluate_with_state_policy(
                model=model,
                loader=test_loader,
                criterion=criterion,
                device=device,
                t_steps=t_steps,
                max_batches=max_test_batches,
                policy_factory=policy_factory,
                high_state_bits=high_state_bits,
            )
            rows.append(
                {
                    "epoch": epoch,
                    "lr": optimizer.param_groups[0]["lr"],
                    "train_loss": loss_sum / float(max(total, 1)),
                    "train_acc": correct / float(max(total, 1)),
                    "train_avg_state_bits_used": train_policy.summary(high_state_bits)["avg_state_bits_used"],
                    **test_metrics,
                    "sop_proxy": sop_proxy(test_metrics["spike_rate"], synapse_count, t_steps),
                }
            )
    finally:
        quantizer.close()

    return rows


def run_difficulty_adaptive_state_precision_analysis(
    cfg: BaselineVGG16Config,
    checkpoint_path: str,
    low_state_bits: int,
    high_state_bits: int,
    warmup_steps: int = 4,
    difficulty_metric: str = "confidence",
    easy_threshold: float = 0.7,
    output_dir: str = "outputs/difficulty_adaptive_state_precision_vgg16",
    max_hessian_batches: Optional[int] = None,
    max_test_batches: Optional[int] = None,
    quant_epochs: int = 1,
    quant_lr: float = 1e-4,
    quant_weight_decay: float = 5e-4,
) -> dict:
    if low_state_bits < 2 or high_state_bits < 2:
        raise ValueError("state bits must be >= 2.")
    if low_state_bits > high_state_bits:
        raise ValueError("low_state_bits must be <= high_state_bits.")
    if warmup_steps < 1:
        raise ValueError("warmup_steps must be >= 1.")

    set_global_seed(cfg.seed, cfg.deterministic)
    device = cfg.resolve_device()
    train_loader, test_loader = build_cifar10_loaders(cfg, device=device)

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
    synapse_count = synapse_count_proxy(fp32_model)
    params = parameter_count(fp32_model)
    model_size = model_size_mb(fp32_model)

    fp32_metrics = evaluate_with_state_policy(
        model=fp32_model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        t_steps=cfg.t_steps,
        max_batches=eval_limit,
        policy_factory=lambda: _FixedStatePolicy(bits=32),
        high_state_bits=32,
    )

    fixed_high_metrics = evaluate_with_state_policy(
        model=copy.deepcopy(fp32_model).to(device),
        loader=test_loader,
        criterion=criterion,
        device=device,
        t_steps=cfg.t_steps,
        max_batches=eval_limit,
        policy_factory=lambda: _FixedStatePolicy(bits=high_state_bits),
        high_state_bits=high_state_bits,
    )

    fixed_low_model = copy.deepcopy(fp32_model).to(device)
    fixed_low_rows = finetune_with_state_policy(
        model=fixed_low_model,
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
        policy_factory=lambda: _FixedStatePolicy(bits=low_state_bits),
        high_state_bits=high_state_bits,
    )
    fixed_low_metrics = evaluate_with_state_policy(
        model=fixed_low_model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        t_steps=cfg.t_steps,
        max_batches=eval_limit,
        policy_factory=lambda: _FixedStatePolicy(bits=low_state_bits),
        high_state_bits=high_state_bits,
    )

    adaptive_model = copy.deepcopy(fp32_model).to(device)
    adaptive_factory = lambda: _DifficultyAdaptiveStatePolicy(
        low_bits=low_state_bits,
        high_bits=high_state_bits,
        warmup_steps=warmup_steps,
        difficulty_metric=difficulty_metric,
        easy_threshold=easy_threshold,
    )
    adaptive_rows = finetune_with_state_policy(
        model=adaptive_model,
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
        policy_factory=adaptive_factory,
        high_state_bits=high_state_bits,
    )
    adaptive_metrics = evaluate_with_state_policy(
        model=adaptive_model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        t_steps=cfg.t_steps,
        max_batches=eval_limit,
        policy_factory=adaptive_factory,
        high_state_bits=high_state_bits,
    )

    comparison_rows = [
        {
            "setting": "FP32State",
            "test_acc": fp32_metrics["test_acc"],
            "spike_rate": fp32_metrics["spike_rate"],
            "avg_batch_infer_ms": fp32_metrics["avg_batch_infer_ms"],
            "sop_proxy": sop_proxy(fp32_metrics["spike_rate"], synapse_count, cfg.t_steps),
            "avg_state_bits_used": fp32_metrics["avg_state_bits_used"],
            "estimated_state_bytes_per_sample": _estimate_state_bytes_per_sample(
                total_state_cost_proxy, fp32_metrics["avg_state_bits_used"]
            ),
            "easy_fraction_post_warmup": fp32_metrics["easy_fraction_post_warmup"],
        },
        {
            "setting": f"FixedStateHighB{high_state_bits}",
            "test_acc": fixed_high_metrics["test_acc"],
            "spike_rate": fixed_high_metrics["spike_rate"],
            "avg_batch_infer_ms": fixed_high_metrics["avg_batch_infer_ms"],
            "sop_proxy": sop_proxy(fixed_high_metrics["spike_rate"], synapse_count, cfg.t_steps),
            "avg_state_bits_used": fixed_high_metrics["avg_state_bits_used"],
            "estimated_state_bytes_per_sample": _estimate_state_bytes_per_sample(
                total_state_cost_proxy, fixed_high_metrics["avg_state_bits_used"]
            ),
            "easy_fraction_post_warmup": fixed_high_metrics["easy_fraction_post_warmup"],
        },
        {
            "setting": f"FixedStateLowB{low_state_bits}",
            "test_acc": fixed_low_metrics["test_acc"],
            "spike_rate": fixed_low_metrics["spike_rate"],
            "avg_batch_infer_ms": fixed_low_metrics["avg_batch_infer_ms"],
            "sop_proxy": sop_proxy(fixed_low_metrics["spike_rate"], synapse_count, cfg.t_steps),
            "avg_state_bits_used": fixed_low_metrics["avg_state_bits_used"],
            "estimated_state_bytes_per_sample": _estimate_state_bytes_per_sample(
                total_state_cost_proxy, fixed_low_metrics["avg_state_bits_used"]
            ),
            "easy_fraction_post_warmup": fixed_low_metrics["easy_fraction_post_warmup"],
        },
        {
            "setting": f"AdaptiveStateB{low_state_bits}to{high_state_bits}",
            "test_acc": adaptive_metrics["test_acc"],
            "spike_rate": adaptive_metrics["spike_rate"],
            "avg_batch_infer_ms": adaptive_metrics["avg_batch_infer_ms"],
            "sop_proxy": sop_proxy(adaptive_metrics["spike_rate"], synapse_count, cfg.t_steps),
            "avg_state_bits_used": adaptive_metrics["avg_state_bits_used"],
            "estimated_state_bytes_per_sample": _estimate_state_bytes_per_sample(
                total_state_cost_proxy, adaptive_metrics["avg_state_bits_used"]
            ),
            "easy_fraction_post_warmup": adaptive_metrics["easy_fraction_post_warmup"],
        },
    ]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    state_proxy_csv = output_path / "state_cost_proxy.csv"
    comparison_csv = output_path / "comparison.csv"
    fixed_low_epoch_csv = output_path / "fixed_low_state_epoch_metrics.csv"
    adaptive_epoch_csv = output_path / "adaptive_state_epoch_metrics.csv"
    summary_json = output_path / "summary.json"

    with state_proxy_csv.open("w", encoding="utf-8") as f:
        f.write("state_layer_name,state_cost_proxy,state_rank\n")
        for row in state_rows:
            f.write(f"{row['state_layer_name']},{row['state_cost_proxy']},{row['state_rank']}\n")
    with comparison_csv.open("w", encoding="utf-8") as f:
        headers = list(comparison_rows[0].keys())
        f.write(",".join(headers) + "\n")
        for row in comparison_rows:
            f.write(",".join(str(row[h]) for h in headers) + "\n")
    write_epoch_metrics_csv(fixed_low_epoch_csv, fixed_low_rows)
    write_epoch_metrics_csv(adaptive_epoch_csv, adaptive_rows)

    summary = {
        "method": "difficulty_adaptive_state_precision_proxy_experiment",
        "device": device,
        "checkpoint_path": checkpoint_path,
        "params": params,
        "model_size_mb": model_size,
        "low_state_bits": low_state_bits,
        "high_state_bits": high_state_bits,
        "warmup_steps": warmup_steps,
        "difficulty_metric": difficulty_metric,
        "easy_threshold": easy_threshold,
        "total_state_cost_proxy": total_state_cost_proxy,
        "max_hessian_batches": max_hessian_batches,
        "max_test_batches": eval_limit,
        "quant_epochs": quant_epochs,
        "quant_lr": quant_lr,
        "quant_weight_decay": quant_weight_decay,
        "final_fixed_low_acc": fixed_low_metrics["test_acc"],
        "final_adaptive_acc": adaptive_metrics["test_acc"],
        "adaptive_avg_state_bits_used": adaptive_metrics["avg_state_bits_used"],
        "adaptive_easy_fraction_post_warmup": adaptive_metrics["easy_fraction_post_warmup"],
        "state_quantization_note": (
            "This experiment approximates state precision by quantizing continuous inputs "
            "to each LIF node, rather than modifying the neuron's internal membrane update."
        ),
        "outputs": {
            "state_cost_proxy_csv": str(state_proxy_csv),
            "comparison_csv": str(comparison_csv),
            "fixed_low_state_epoch_metrics_csv": str(fixed_low_epoch_csv),
            "adaptive_state_epoch_metrics_csv": str(adaptive_epoch_csv),
            "summary_json": str(summary_json),
        },
        "best_setting_by_acc": max(comparison_rows, key=lambda item: item["test_acc"])["setting"],
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
