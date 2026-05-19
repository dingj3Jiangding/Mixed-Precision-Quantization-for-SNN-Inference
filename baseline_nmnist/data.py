from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

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


class _FrameFromEventDataset(Dataset):
    def __init__(self, event_dataset, frames_number: int, split_by: str = "number") -> None:
        self.event_dataset = event_dataset
        self.frames_number = int(frames_number)
        self.split_by = split_by

        try:
            from spikingjelly.datasets import integrate_events_by_fixed_frames_number
        except Exception as exc:
            raise RuntimeError(
                "spikingjelly.datasets.integrate_events_by_fixed_frames_number is required."
            ) from exc

        self._integrate = integrate_events_by_fixed_frames_number
        self._H, self._W = self.event_dataset.get_H_W()

    def __len__(self) -> int:
        return len(self.event_dataset)

    def __getitem__(self, index: int):
        events, label = self.event_dataset[index]
        frames = self._integrate(
            events=events,
            split_by=self.split_by,
            frames_num=self.frames_number,
            H=self._H,
            W=self._W,
        )
        return _ensure_tensor_frames(frames), int(label)


def build_nmnist_loaders(cfg: BaselineNMNISTConfig, device: str):
    try:
        from spikingjelly.datasets.n_mnist import NMNIST
    except Exception as exc:
        raise RuntimeError(
            "spikingjelly with N-MNIST dataset support is required to run baseline_nmnist."
        ) from exc

    data_root = Path(cfg.data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    # Use event mode and integrate each sample to fixed frame counts on the fly.
    # This avoids relying on the global pre-generated frame cache path, which can
    # be left in a partially generated state when preprocessing is interrupted.
    train_event_set = NMNIST(root=str(data_root), train=True, data_type="event")
    test_event_set = NMNIST(root=str(data_root), train=False, data_type="event")

    train_set = _FrameFromEventDataset(train_event_set, frames_number=cfg.t_steps, split_by="number")
    test_set = _FrameFromEventDataset(test_event_set, frames_number=cfg.t_steps, split_by="number")

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
