"""Image preprocessing and augmentation."""

from __future__ import annotations

from typing import Any, Optional, Tuple

from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(
    image_size: int,
    train: bool,
    mean: Optional[Tuple[float, float, float]] = None,
    std: Optional[Tuple[float, float, float]] = None,
    aug_cfg: Optional[dict[str, Any]] = None,
) -> transforms.Compose:
    mean = mean or IMAGENET_MEAN
    std = std or IMAGENET_STD

    if train:
        cfg = aug_cfg or {}
        crop_scale = tuple(cfg.get("crop_scale", (0.7, 1.0)))
        crop_ratio = tuple(cfg.get("crop_ratio", (0.75, 1.33)))
        jitter_cfg = cfg.get("color_jitter", {})
        erasing_cfg = cfg.get("random_erasing", {})

        ops = [
            transforms.RandomResizedCrop(
                image_size, scale=crop_scale, ratio=crop_ratio
            ),
            transforms.RandomHorizontalFlip(p=cfg.get("hflip", 0.5)),
            transforms.RandomVerticalFlip(p=cfg.get("vflip", 0.5)),
            transforms.RandomRotation(degrees=cfg.get("rotation", 20)),
            transforms.ColorJitter(
                brightness=jitter_cfg.get("brightness", 0.2),
                contrast=jitter_cfg.get("contrast", 0.2),
                saturation=jitter_cfg.get("saturation", 0.2),
                hue=jitter_cfg.get("hue", 0.05),
            ),
        ]
        if cfg.get("grayscale", 0.0) > 0:
            ops.append(transforms.RandomGrayscale(p=cfg.get("grayscale", 0.1)))
        if cfg.get("gaussian_blur", False):
            ops.append(
                transforms.GaussianBlur(
                    kernel_size=cfg.get("blur_kernel", 3),
                    sigma=cfg.get("blur_sigma", (0.1, 2.0)),
                )
            )
        ops.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        if erasing_cfg.get("enabled", True):
            ops.append(
                transforms.RandomErasing(
                    p=erasing_cfg.get("p", 0.25),
                    scale=tuple(erasing_cfg.get("scale", (0.02, 0.2))),
                    ratio=tuple(erasing_cfg.get("ratio", (0.3, 3.3))),
                    value=erasing_cfg.get("value", "random"),
                )
            )
        return transforms.Compose(ops)

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

