"""Dataset loaders for ISIC2018 and Kvasir."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder

from .transforms import build_transforms


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    num_classes: int
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]


DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    "kvasir": DatasetConfig(
        name="kvasir",
        num_classes=8,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
    "isic2018": DatasetConfig(
        name="isic2018",
        num_classes=7,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
}


def build_dataloaders(
    dataset_root: str,
    dataset_name: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    mean: Tuple[float, float, float] | None = None,
    std: Tuple[float, float, float] | None = None,
    aug_cfg: dict[str, Any] | None = None,
    balance_cfg: dict[str, Any] | None = None,
) -> Tuple[DataLoader, DataLoader, int, torch.Tensor | None]:
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Expected one of: {', '.join(DATASET_CONFIGS.keys())}"
        )

    config = DATASET_CONFIGS[dataset_name]
    use_mean = mean or config.mean
    use_std = std or config.std
    train_transform = build_transforms(
        image_size=image_size,
        train=True,
        mean=use_mean,
        std=use_std,
        aug_cfg=aug_cfg,
    )
    val_transform = build_transforms(
        image_size=image_size, train=False, mean=use_mean, std=use_std
    )

    root = Path(dataset_root)
    train_dir = root / "train"
    val_dir = root / "val"

    if dataset_name == "kvasir" and not train_dir.exists():
        train_dir = root / "train_kvasir"
        val_dir = root / "test_kvasir"

    if dataset_name == "isic2018" and not train_dir.exists():
        train_dir = root / "train_ISIC"
        val_dir = root / "test_ISIC"

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            "Expected dataset folders not found. "
            f"Got train='{train_dir}' and val='{val_dir}'."
        )

    train_ds = ImageFolder(root=str(train_dir), transform=train_transform)
    val_ds = ImageFolder(root=str(val_dir), transform=val_transform)

    class_weights = None
    sampler = None
    if balance_cfg and balance_cfg.get("sampler", False):
        targets = torch.as_tensor(train_ds.targets, dtype=torch.long)
        class_counts = torch.bincount(targets, minlength=config.num_classes).float()
        power = float(balance_cfg.get("weight_power", 1.0))
        class_weights = (class_counts.sum() / (class_counts + 1e-6)) ** power
        sample_weights = class_weights[targets].double()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
    elif balance_cfg and balance_cfg.get("loss_weighted", False):
        targets = torch.as_tensor(train_ds.targets, dtype=torch.long)
        class_counts = torch.bincount(targets, minlength=config.num_classes).float()
        power = float(balance_cfg.get("weight_power", 1.0))
        class_weights = (class_counts.sum() / (class_counts + 1e-6)) ** power

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, config.num_classes, class_weights

