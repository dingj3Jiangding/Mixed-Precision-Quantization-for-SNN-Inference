from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class BaselineNMNISTConfig:
    data_root: str = "baseline_nmnist/data"
    output_dir: str = "outputs/baseline_nmnist"
    dataset_download: bool = True
    batch_size_train: int = 64
    batch_size_test: int = 128
    num_workers: int = 4
    epochs: int = 20
    t_steps: int = 16
    lr: float = 1e-3
    weight_decay: float = 5e-4
    seed: int = 42
    deterministic: bool = True
    device: str = "auto"
    max_train_batches: Optional[int] = None
    max_test_batches: Optional[int] = None

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
