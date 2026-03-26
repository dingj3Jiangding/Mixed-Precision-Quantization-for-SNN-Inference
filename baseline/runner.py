from __future__ import annotations

import csv
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from spikingjelly.activation_based import functional

from .config import BaselineConfig
from .data import build_cifar10_loaders
from .metrics import (
    SpikeRateTracker,
    model_size_mb,
    parameter_count,
    sop_proxy,
    synapse_count_proxy,
)
from .model import build_model


def set_global_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def direct_encode(x: torch.Tensor, t_steps: int) -> torch.Tensor:
    return x.unsqueeze(0).repeat(t_steps, 1, 1, 1, 1)


def _iter_with_limit(loader, max_batches: Optional[int]):
    if max_batches is None:
        yield from loader
        return
    for idx, batch in enumerate(loader):
        if idx >= max_batches:
            break
        yield batch


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    t_steps: int,
    max_batches: Optional[int],
) -> dict[str, float]:
    model.train()
    loss_sum = 0.0
    correct = 0
    total = 0

    for x, y in _iter_with_limit(loader, max_batches):
        x = x.to(device)
        y = y.to(device)

        x_seq = direct_encode(x, t_steps)
        optimizer.zero_grad(set_to_none=True)
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

    avg_loss = loss_sum / max(total, 1)
    acc = correct / max(total, 1)
    return {"train_loss": avg_loss, "train_acc": acc}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: str,
    t_steps: int,
    max_batches: Optional[int],
) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    infer_time_sum = 0.0
    batch_count = 0

    spike_tracker = SpikeRateTracker(model)
    spike_tracker.reset()

    for x, y in _iter_with_limit(loader, max_batches):
        x = x.to(device)
        y = y.to(device)
        x_seq = direct_encode(x, t_steps)

        start_time = time.perf_counter()
        logits_seq = model(x_seq)
        infer_time_sum += time.perf_counter() - start_time
        batch_count += 1

        logits = logits_seq.mean(dim=0)
        loss = criterion(logits, y)

        batch_size = y.shape[0]
        loss_sum += float(loss.item()) * batch_size
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += batch_size

        functional.reset_net(model)

    avg_loss = loss_sum / max(total, 1)
    acc = correct / max(total, 1)
    spike_rate = spike_tracker.rate()
    spike_tracker.close()
    avg_batch_infer_ms = (infer_time_sum / max(batch_count, 1)) * 1000.0

    return {
        "test_loss": avg_loss,
        "test_acc": acc,
        "spike_rate": spike_rate,
        "avg_batch_infer_ms": avg_batch_infer_ms,
    }


def write_epoch_metrics_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_baseline(cfg: BaselineConfig) -> dict:
    set_global_seed(cfg.seed, cfg.deterministic)
    device = cfg.resolve_device()

    train_loader, test_loader = build_cifar10_loaders(cfg, device=device)
    model = build_model(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    synapse_count = synapse_count_proxy(model)
    param_num = parameter_count(model)
    model_size = model_size_mb(model)

    epoch_rows: list[dict] = []
    for epoch in range(1, cfg.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            t_steps=cfg.t_steps,
            max_batches=cfg.max_train_batches,
        )
        test_metrics = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            t_steps=cfg.t_steps,
            max_batches=cfg.max_test_batches,
        )

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            **train_metrics,
            **test_metrics,
            "params": param_num,
            "model_size_mb": model_size,
            "synapse_count_proxy": synapse_count,
            "sop_proxy": sop_proxy(test_metrics["spike_rate"], synapse_count, cfg.t_steps),
        }
        epoch_rows.append(row)
        print(
            f"[Epoch {epoch:02d}] "
            f"train_acc={row['train_acc']:.4f} "
            f"test_acc={row['test_acc']:.4f} "
            f"spike_rate={row['spike_rate']:.4f}"
        )

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = output_dir / "epoch_metrics.csv"
    summary_json = output_dir / "summary.json"
    write_epoch_metrics_csv(metrics_csv, epoch_rows)

    best_test_acc = max((row["test_acc"] for row in epoch_rows), default=0.0)
    summary = {
        "config": asdict(cfg),
        "device": device,
        "best_test_acc": best_test_acc,
        "final_epoch": epoch_rows[-1] if epoch_rows else None,
        "metrics_csv": str(metrics_csv),
        "summary_json": str(summary_json),
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
