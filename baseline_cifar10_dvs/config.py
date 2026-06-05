from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class BaselineCIFAR10DVSConfig:
    data_root: str = "baseline_cifar10_dvs/data"
    output_dir: str = "outputs/baseline_cifar10_dvs"
    dataset_download: bool = True
    batch_size_train: int = 16
    batch_size_test: int = 32
    num_workers: int = 4
    epochs: int = 64
    t_steps: int = 16
    lr: float = 1e-3
    weight_decay: float = 5e-4
    seed: int = 42
    deterministic: bool = True
    device: str = "auto"
    max_train_batches: Optional[int] = None
    max_test_batches: Optional[int] = None
    frame_split_by: str = "number"

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
