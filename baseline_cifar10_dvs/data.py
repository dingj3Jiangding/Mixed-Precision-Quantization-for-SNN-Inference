from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Tuple
import urllib.error
import urllib.request

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from .config import BaselineCIFAR10DVSConfig


_CIFAR10_DVS_RESOURCES = [
    ("airplane.zip", "7712788", "0afd5c4bf9ae06af762a77b180354fdd"),
    ("automobile.zip", "7712791", "8438dfeba3bc970c94962d995b1b9bdd"),
    ("bird.zip", "7712794", "a9c207c91c55b9dc2002dc21c684d785"),
    ("cat.zip", "7712812", "52c63c677c2b15fa5146a8daf4d56687"),
    ("deer.zip", "7712815", "b6bf21f6c04d21ba4e23fc3e36c8a4a3"),
    ("dog.zip", "7712818", "f379ebdf6703d16e0a690782e62639c3"),
    ("frog.zip", "7712842", "cad6ed91214b1c7388a5f6ee56d08803"),
    ("horse.zip", "7712851", "e7cbbf77bec584ffbf913f00e682782a"),
    ("ship.zip", "7712836", "41c7bd7d6b251be82557c6cce9a7d5c9"),
    ("truck.zip", "7712839", "89f3922fd147d9aeff89e76a2b0b70a7"),
]


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed + worker_id)
    np.random.seed(worker_seed + worker_id)


def _file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_file_with_headers(url: str, path: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
            )
        },
    )
    tmp_path = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(request, timeout=120) as response, tmp_path.open("wb") as f:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    tmp_path.replace(path)


def _ensure_cifar10_dvs_archives(cfg: BaselineCIFAR10DVSConfig) -> None:
    if not cfg.dataset_download:
        return

    download_root = Path(cfg.data_root) / "download"
    download_root.mkdir(parents=True, exist_ok=True)

    for filename, file_id, md5 in _CIFAR10_DVS_RESOURCES:
        path = download_root / filename
        if path.exists() and _file_md5(path) == md5:
            continue
        if path.exists():
            path.unlink()

        urls = [
            f"https://figshare.com/ndownloader/files/{file_id}",
            f"https://ndownloader.figshare.com/files/{file_id}",
        ]
        errors: list[str] = []
        for url in urls:
            try:
                print(f"Download [{filename}] from [{url}] to [{path}]")
                _download_file_with_headers(url, path)
                if _file_md5(path) != md5:
                    errors.append(f"{url}: downloaded file md5 mismatch")
                    path.unlink(missing_ok=True)
                    continue
                break
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
                errors.append(f"{url}: {exc}")
                path.unlink(missing_ok=True)
        else:
            detail = "\n".join(errors)
            raise RuntimeError(
                f"Unable to download CIFAR10-DVS archive {filename}.\n"
                f"Tried:\n{detail}\n"
                "You can also download CIFAR10-DVS manually and place the class zip files "
                f"under {download_root}."
            )


def _ensure_tensor_frames(frames) -> torch.Tensor:
    frames = torch.as_tensor(frames, dtype=torch.float32)
    if frames.ndim != 4:
        raise ValueError(
            "Expected CIFAR10-DVS frame sample shape [T, C, H, W], "
            f"but got {tuple(frames.shape)}."
        )
    return frames


def _collate_frames(batch):
    frames = torch.stack([_ensure_tensor_frames(x) for x, _y in batch], dim=0)
    labels = torch.as_tensor([int(y) for _x, y in batch], dtype=torch.long)
    return frames, labels


class _FrameTensorDataset(Dataset):
    def __init__(self, dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        frames, label = self.dataset[index]
        return _ensure_tensor_frames(frames), int(label)


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
        base_dataset = self.event_dataset.dataset if hasattr(self.event_dataset, "dataset") else self.event_dataset
        self._H, self._W = base_dataset.get_H_W()

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


def _build_tebn_split(cfg: BaselineCIFAR10DVSConfig) -> Tuple[Dataset, Dataset]:
    try:
        from spikingjelly.datasets.cifar10_dvs import CIFAR10DVSTEBNSplit
    except Exception as exc:
        raise RuntimeError(
            "spikingjelly.datasets.cifar10_dvs.CIFAR10DVSTEBNSplit is required "
            "for the default CIFAR10-DVS train/test split."
        ) from exc

    data_root = Path(cfg.data_root)
    train_set = CIFAR10DVSTEBNSplit(
        root=str(data_root),
        train=True,
        data_type="frame",
        frames_number=cfg.t_steps,
        split_by=cfg.frame_split_by,
    )
    test_set = CIFAR10DVSTEBNSplit(
        root=str(data_root),
        train=False,
        data_type="frame",
        frames_number=cfg.t_steps,
        split_by=cfg.frame_split_by,
    )
    return _FrameTensorDataset(train_set), _FrameTensorDataset(test_set)


def _dataset_targets(dataset) -> list[int]:
    if hasattr(dataset, "targets"):
        return [int(target) for target in dataset.targets]
    if hasattr(dataset, "samples"):
        return [int(sample[1]) for sample in dataset.samples]
    raise RuntimeError("Unable to infer CIFAR10-DVS labels from dataset.targets or dataset.samples.")


def _split_first_100_per_class(dataset) -> Tuple[Subset, Subset]:
    targets = _dataset_targets(dataset)
    class_counts: dict[int, int] = {}
    train_indices: list[int] = []
    test_indices: list[int] = []

    for idx, target in enumerate(targets):
        local_idx = class_counts.get(target, 0)
        class_counts[target] = local_idx + 1
        if local_idx < 100:
            test_indices.append(idx)
        else:
            train_indices.append(idx)

    if len(class_counts) != 10:
        raise RuntimeError(f"Expected 10 CIFAR10-DVS classes, but found {len(class_counts)} classes.")
    if any(count < 100 for count in class_counts.values()):
        raise RuntimeError(f"Each CIFAR10-DVS class needs at least 100 samples, got {class_counts}.")

    return Subset(dataset, train_indices), Subset(dataset, test_indices)


def _build_manual_tebn_split(cfg: BaselineCIFAR10DVSConfig) -> Tuple[Dataset, Dataset]:
    try:
        from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
    except Exception as exc:
        raise RuntimeError(
            "spikingjelly.datasets.cifar10_dvs.CIFAR10DVS is required to run CIFAR10-DVS."
        ) from exc

    data_root = Path(cfg.data_root)
    full_set = CIFAR10DVS(
        root=str(data_root),
        data_type="frame",
        frames_number=cfg.t_steps,
        split_by=cfg.frame_split_by,
    )
    train_set, test_set = _split_first_100_per_class(full_set)
    return _FrameTensorDataset(train_set), _FrameTensorDataset(test_set)


def _build_manual_tebn_split_from_events(cfg: BaselineCIFAR10DVSConfig) -> Tuple[Dataset, Dataset]:
    try:
        from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
    except Exception as exc:
        raise RuntimeError(
            "spikingjelly.datasets.cifar10_dvs.CIFAR10DVS is required to run CIFAR10-DVS."
        ) from exc

    data_root = Path(cfg.data_root)
    full_event_set = CIFAR10DVS(root=str(data_root), data_type="event")
    train_event_set, test_event_set = _split_first_100_per_class(full_event_set)
    return (
        _FrameFromEventDataset(train_event_set, frames_number=cfg.t_steps, split_by=cfg.frame_split_by),
        _FrameFromEventDataset(test_event_set, frames_number=cfg.t_steps, split_by=cfg.frame_split_by),
    )


def _build_tebn_split_from_events(cfg: BaselineCIFAR10DVSConfig) -> Tuple[Dataset, Dataset]:
    try:
        from spikingjelly.datasets.cifar10_dvs import CIFAR10DVSTEBNSplit
    except Exception as exc:
        raise RuntimeError(
            "spikingjelly.datasets.cifar10_dvs.CIFAR10DVSTEBNSplit is required "
            "for the default CIFAR10-DVS train/test split."
        ) from exc

    data_root = Path(cfg.data_root)
    train_event_set = CIFAR10DVSTEBNSplit(root=str(data_root), train=True, data_type="event")
    test_event_set = CIFAR10DVSTEBNSplit(root=str(data_root), train=False, data_type="event")
    return (
        _FrameFromEventDataset(train_event_set, frames_number=cfg.t_steps, split_by=cfg.frame_split_by),
        _FrameFromEventDataset(test_event_set, frames_number=cfg.t_steps, split_by=cfg.frame_split_by),
    )


def build_cifar10_dvs_loaders(cfg: BaselineCIFAR10DVSConfig, device: str) -> tuple[DataLoader, DataLoader]:
    data_root = Path(cfg.data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    _ensure_cifar10_dvs_archives(cfg)

    try:
        train_set, test_set = _build_tebn_split(cfg)
    except TypeError:
        # Older SpikingJelly builds may not expose frame conversion arguments on
        # CIFAR10DVSTEBNSplit. Fall back to event mode and integrate on the fly,
        # matching the robust N-MNIST path used elsewhere in this repository.
        train_set, test_set = _build_tebn_split_from_events(cfg)
    except RuntimeError:
        try:
            train_set, test_set = _build_manual_tebn_split(cfg)
        except TypeError:
            train_set, test_set = _build_manual_tebn_split_from_events(cfg)

    pin_memory = device == "cuda"
    generator = torch.Generator().manual_seed(cfg.seed)
    persistent_workers = cfg.num_workers > 0

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size_train,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=_collate_frames,
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
        drop_last=False,
        collate_fn=_collate_frames,
        worker_init_fn=_seed_worker,
        generator=generator,
        persistent_workers=persistent_workers,
    )
    return train_loader, test_loader
