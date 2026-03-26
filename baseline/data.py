from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from .config import BaselineConfig

_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed + worker_id)
    np.random.seed(worker_seed + worker_id)


def build_cifar10_loaders(cfg: BaselineConfig, device: str) -> tuple[DataLoader, DataLoader]:
    data_root = Path(cfg.data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
        ]
    )

    train_set = torchvision.datasets.CIFAR10(
        root=str(data_root),
        train=True,
        download=cfg.dataset_download,
        transform=train_transform,
    )
    test_set = torchvision.datasets.CIFAR10(
        root=str(data_root),
        train=False,
        download=cfg.dataset_download,
        transform=test_transform,
    )

    pin_memory = device == "cuda"
    generator = torch.Generator().manual_seed(cfg.seed)
    persistent_workers = cfg.num_workers > 0

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size_train,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_seed_worker,
        generator=generator,
        persistent_workers=persistent_workers,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.batch_size_test,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_seed_worker,
        generator=generator,
        persistent_workers=persistent_workers,
    )
    return train_loader, test_loader
