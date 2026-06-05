from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


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
