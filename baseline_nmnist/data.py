from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import BaselineNMNISTConfig


def _ensure_tensor_frames(frames) -> torch.Tensor:
    frames = torch.as_tensor(frames, dtype=torch.float32)
    if frames.ndim != 4:
        raise ValueError(
            "Expected N-MNIST frame sample shape [T, C, H, W], "
            f"but got {tuple(frames.shape)}."
        )
    return frames


def _collate_frames(batch):
    frames = torch.stack([_ensure_tensor_frames(x) for x, _y in batch], dim=0)
    labels = torch.as_tensor([int(y) for _x, y in batch], dtype=torch.long)
    return frames, labels


def build_nmnist_loaders(cfg: BaselineNMNISTConfig, device: str):
    try:
        from spikingjelly.datasets.n_mnist import NMNIST
    except Exception as exc:
        raise RuntimeError(
            "spikingjelly with N-MNIST dataset support is required to run baseline_nmnist."
        ) from exc

    data_root = Path(cfg.data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    dataset_kwargs = dict(
        root=str(data_root),
        data_type="frame",
        frames_number=cfg.t_steps,
        split_by="number",
    )

    train_set = NMNIST(train=True, **dataset_kwargs)
    test_set = NMNIST(train=False, **dataset_kwargs)

    pin_memory = device == "cuda"
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size_train,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=_collate_frames,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.batch_size_test,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=_collate_frames,
    )
    return train_loader, test_loader
