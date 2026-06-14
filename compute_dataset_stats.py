"""Compute per-channel mean/std for a dataset split."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute dataset mean/std")
    parser.add_argument("--root", required=True, help="Dataset root folder")
    parser.add_argument(
        "--train-dir",
        default="train",
        help="Training split folder name (default: train)",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    return parser.parse_args()


def compute_mean_std(root: Path, train_dir_name: str, image_size: int, batch_size: int, workers: int) -> tuple[list[float], list[float], int]:
    train_dir = root / train_dir_name
    if not train_dir.exists():
        raise FileNotFoundError(f"Train folder not found: {train_dir}")

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    dataset = ImageFolder(root=str(train_dir), transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=False,
    )

    channel_sum = torch.zeros(3, dtype=torch.float64)
    channel_sq = torch.zeros(3, dtype=torch.float64)
    pixel_count = 0
    for images, _ in tqdm(loader, desc="Stats", unit="batch"):
        images = images.double()
        channel_sum += images.sum(dim=(0, 2, 3))
        channel_sq += (images**2).sum(dim=(0, 2, 3))
        pixel_count += images.size(0) * images.size(2) * images.size(3)

    mean = channel_sum / pixel_count
    std = torch.sqrt(channel_sq / pixel_count - mean**2)
    return [float(v) for v in mean], [float(v) for v in std], len(dataset)


def main() -> None:
    args = parse_args()
    mean, std, count = compute_mean_std(
        root=Path(args.root),
        train_dir_name=args.train_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    print(f"Samples: {count}")
    print("Mean:", [round(v, 6) for v in mean])
    print("Std:", [round(v, 6) for v in std])


if __name__ == "__main__":
    main()
