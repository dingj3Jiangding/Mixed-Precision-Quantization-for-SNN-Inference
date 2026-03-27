from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Optional

import torch
import torch.nn as nn

from .config import BaselineConfig
from .data import build_cifar10_loaders
from .metrics import parameter_count, sop_proxy, synapse_count_proxy
from .model import build_model
from .quantization import (
    clone_and_quantize_model_weights,
    estimate_quantized_model_size_mb,
    parse_bits_list,
)
from .unquant_runner import evaluate, set_global_seed


def _load_checkpoint(model: nn.Module, checkpoint_path: str) -> None:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. "
            "Please run baseline training first to produce fp32_last.pt."
        )

    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_uniform_quant_comparison(
    cfg: BaselineConfig,
    checkpoint_path: str,
    bits_list: Iterable[int],
    output_dir: str = "outputs/uniform_quant",
    max_test_batches: Optional[int] = None,
) -> dict:
    set_global_seed(cfg.seed, cfg.deterministic)
    device = cfg.resolve_device()
    _, test_loader = build_cifar10_loaders(cfg, device=device)
    criterion = nn.CrossEntropyLoss()

    fp32_model = build_model(num_classes=10).to(device)
    _load_checkpoint(fp32_model, checkpoint_path)

    synapse_count = synapse_count_proxy(fp32_model)
    params = parameter_count(fp32_model)

    rows: list[dict] = []
    fp32_metrics = evaluate(
        model=fp32_model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        t_steps=cfg.t_steps,
        max_batches=max_test_batches if max_test_batches is not None else cfg.max_test_batches,
    )
    fp32_row = {
        "setting": "FP32",
        "weight_bits": 32,
        "params": params,
        "quantized_weight_params": 0,
        "model_size_mb_proxy": estimate_quantized_model_size_mb(fp32_model, bits=32, quantized_weight_count=0),
        "synapse_count_proxy": synapse_count,
        **fp32_metrics,
        "sop_proxy": sop_proxy(fp32_metrics["spike_rate"], synapse_count, cfg.t_steps),
    }
    rows.append(fp32_row)
    print(
        f"[FP32] test_acc={fp32_row['test_acc']:.4f} "
        f"spike_rate={fp32_row['spike_rate']:.4f}"
    )

    for bits in parse_bits_list(bits_list):
        quant_model, quantized_count = clone_and_quantize_model_weights(fp32_model, bits=bits)
        quant_model = quant_model.to(device)
        quant_metrics = evaluate(
            model=quant_model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            t_steps=cfg.t_steps,
            max_batches=max_test_batches if max_test_batches is not None else cfg.max_test_batches,
        )
        row = {
            "setting": f"W{bits}",
            "weight_bits": bits,
            "params": params,
            "quantized_weight_params": quantized_count,
            "model_size_mb_proxy": estimate_quantized_model_size_mb(
                quant_model, bits=bits, quantized_weight_count=quantized_count
            ),
            "synapse_count_proxy": synapse_count,
            **quant_metrics,
            "sop_proxy": sop_proxy(quant_metrics["spike_rate"], synapse_count, cfg.t_steps),
        }
        rows.append(row)
        print(
            f"[W{bits}] test_acc={row['test_acc']:.4f} "
            f"spike_rate={row['spike_rate']:.4f}"
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    comparison_csv = output_path / "uniform_comparison.csv"
    summary_json = output_path / "summary.json"
    _write_csv(comparison_csv, rows)

    summary = {
        "device": device,
        "checkpoint_path": checkpoint_path,
        "bits_list": parse_bits_list(bits_list),
        "config": {
            "t_steps": cfg.t_steps,
            "seed": cfg.seed,
            "max_test_batches": max_test_batches if max_test_batches is not None else cfg.max_test_batches,
        },
        "comparison_csv": str(comparison_csv),
        "summary_json": str(summary_json),
        "best_setting": max(rows, key=lambda item: item["test_acc"])["setting"] if rows else None,
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
