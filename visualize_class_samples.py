"""Export sample images per class and build grids."""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export class samples and grids")
    parser.add_argument("--isic-root", default="/data/bisec/bice_net/dataset/ISIC2018")
    parser.add_argument("--isic-train-dir", default="train_ISIC")
    parser.add_argument("--kvasir-root", default="/data/bisec/bice_net/dataset/kvasir")
    parser.add_argument("--kvasir-train-dir", default="train_kvasir")
    parser.add_argument("--out-dir", default="/data/bisec/bice_net/outputs/samples")
    parser.add_argument("--num", type=int, default=30)
    parser.add_argument("--thumb-size", type=int, default=192)
    return parser.parse_args()


def collect_images(class_dir: Path, limit: int) -> list[Path]:
    images = [p for p in class_dir.iterdir() if p.is_file()]
    images.sort()
    return images[:limit]


def save_grid(images: list[Path], out_path: Path, thumb_size: int) -> None:
    if not images:
        return
    cols = 6
    rows = math.ceil(len(images) / cols)
    grid = Image.new("RGB", (cols * thumb_size, rows * thumb_size), color=(0, 0, 0))
    for idx, img_path in enumerate(images):
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            img = img.resize((thumb_size, thumb_size), Image.BILINEAR)
        x = (idx % cols) * thumb_size
        y = (idx // cols) * thumb_size
        grid.paste(img, (x, y))
    grid.save(out_path)


def export_dataset(root: Path, train_dir: str, out_root: Path, num: int, thumb_size: int) -> None:
    split_dir = root / train_dir
    if not split_dir.exists():
        raise FileNotFoundError(f"Split not found: {split_dir}")

    for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        class_name = class_dir.name
        class_out = out_root / class_name
        class_out.mkdir(parents=True, exist_ok=True)

        images = collect_images(class_dir, num)
        for img_path in images:
            shutil.copy2(img_path, class_out / img_path.name)

        grid_path = class_out / f"{class_name}_grid.png"
        save_grid(images, grid_path, thumb_size)


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    export_dataset(
        root=Path(args.isic_root),
        train_dir=args.isic_train_dir,
        out_root=out_root / "ISIC2018",
        num=args.num,
        thumb_size=args.thumb_size,
    )
    export_dataset(
        root=Path(args.kvasir_root),
        train_dir=args.kvasir_train_dir,
        out_root=out_root / "kvasir",
        num=args.num,
        thumb_size=args.thumb_size,
    )


if __name__ == "__main__":
    main()
