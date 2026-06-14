"""Train BICE-Net on ISIC2018 or Kvasir."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import yaml
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data.datasets import DATASET_CONFIGS, build_dataloaders
from models.bice_net import BICENet
from utils.metrics import (
    AverageMeter,
    accuracy_from_confusion,
    compute_confusion_matrix,
    f1_macro_from_confusion,
    precision_recall_macro_from_confusion,
)
from PIL import Image
from torchvision import transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BICE-Net Training")
    parser.add_argument(
        "--config",
        default="/home/test/data/bisec/bice_net/train_config.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument("--device", default=None, help="Override device if set.")
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def compute_dataset_mean_std(
    dataset_root: str,
    dataset_name: str,
    image_size: int,
    num_workers: int,
    max_samples: int | None,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    from torchvision.datasets import ImageFolder

    root = Path(dataset_root)
    train_dir = root / "train"
    if dataset_name == "kvasir" and not train_dir.exists():
        train_dir = root / "train_kvasir"
    if dataset_name == "isic2018" and not train_dir.exists():
        train_dir = root / "train_ISIC"
    if not train_dir.exists():
        raise FileNotFoundError(f"Expected train folder not found: '{train_dir}'.")

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    dataset = ImageFolder(root=str(train_dir), transform=transform)
    if len(dataset) == 0:
        raise RuntimeError("Cannot compute normalization stats: dataset is empty.")

    if max_samples is not None and max_samples > 0 and len(dataset) > max_samples:
        indices = torch.randperm(len(dataset))[:max_samples].tolist()
        dataset = Subset(dataset, indices)

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    channel_sum = torch.zeros(3, dtype=torch.float64)
    channel_sq = torch.zeros(3, dtype=torch.float64)
    pixel_count = 0
    for images, _ in tqdm(loader, desc="Stats", leave=False):
        images = images.double()
        channel_sum += images.sum(dim=(0, 2, 3))
        channel_sq += (images ** 2).sum(dim=(0, 2, 3))
        pixel_count += images.size(0) * images.size(2) * images.size(3)

    mean = channel_sum / pixel_count
    std = torch.sqrt(channel_sq / pixel_count - mean ** 2)
    mean_tuple = (float(mean[0]), float(mean[1]), float(mean[2]))
    std_tuple = (float(std[0]), float(std[1]), float(std[2]))
    return mean_tuple, std_tuple


@torch.no_grad()
def select_best_val_sample(
    model: nn.Module, val_dataset: Any, device: torch.device, max_images: int
) -> int:
    if not getattr(val_dataset, "samples", None):
        raise RuntimeError("Validation dataset has no samples.")
    total_samples = len(val_dataset.samples)
    if total_samples == 0:
        raise RuntimeError("Validation dataset is empty.")

    best_idx = 0
    best_conf = -1.0
    max_images = max(1, min(max_images, total_samples))
    for idx in range(max_images):
        image_tensor, _ = val_dataset[idx]
        image_tensor = image_tensor.unsqueeze(0).to(device)
        logits = model(image_tensor)
        conf = F.softmax(logits, dim=1).max().item()
        if conf > best_conf:
            best_conf = conf
            best_idx = idx
    return best_idx


def _mixup_cutmix(
    images: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    mixup_alpha: float,
    cutmix_alpha: float,
    mixup_prob: float,
    cutmix_prob: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    batch_size = images.size(0)
    perm = torch.randperm(batch_size, device=images.device)
    y_a = targets
    y_b = targets[perm]

    total_prob = mixup_prob + cutmix_prob
    cutmix_threshold = cutmix_prob / total_prob if total_prob > 0 else 0.0
    use_cutmix = torch.rand(1).item() < cutmix_threshold
    if use_cutmix and cutmix_alpha > 0:
        lam = np.random.beta(cutmix_alpha, cutmix_alpha)
        bbx1, bby1, bbx2, bby2 = _rand_bbox(images.size(), lam)
        images[:, :, bby1:bby2, bbx1:bbx2] = images[perm, :, bby1:bby2, bbx1:bbx2]
        lam = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size(-1) * images.size(-2)))
    else:
        if mixup_alpha > 0:
            lam = np.random.beta(mixup_alpha, mixup_alpha)
        else:
            lam = 1.0
        images = lam * images + (1 - lam) * images[perm]

    y_a_onehot = F.one_hot(y_a, num_classes=num_classes).float()
    y_b_onehot = F.one_hot(y_b, num_classes=num_classes).float()
    mixed_targets = lam * y_a_onehot + (1 - lam) * y_b_onehot
    return images, mixed_targets, y_a, lam


def _rand_bbox(size: torch.Size, lam: float) -> Tuple[int, int, int, int]:
    w = size[3]
    h = size[2]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)

    cx = np.random.randint(w)
    cy = np.random.randint(h)

    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)
    return bbx1, bby1, bbx2, bby2


def _compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
) -> torch.Tensor:
    if targets.dtype.is_floating_point and targets.dim() == 2:
        log_probs = F.log_softmax(logits, dim=1)
        return -(targets * log_probs).sum(dim=1).mean()
    return criterion(logits, targets)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    mix_cfg: dict[str, Any] | None = None,
) -> Tuple[float, float, float, float, float]:
    model.train()
    loss_meter = AverageMeter("loss")
    confusion = torch.zeros((num_classes, num_classes), device=device)

    mix_cfg = mix_cfg or {}
    mixup_alpha = float(mix_cfg.get("mixup_alpha", 0.4))
    cutmix_alpha = float(mix_cfg.get("cutmix_alpha", 1.0))
    mixup_prob = float(mix_cfg.get("mixup_prob", 0.4))
    cutmix_prob = float(mix_cfg.get("cutmix_prob", 0.0))

    for images, targets in tqdm(loader, desc="Train", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mix_cfg.get("enabled", False) and torch.rand(1).item() < mixup_prob + cutmix_prob:
            images, mixed_targets, hard_targets, _ = _mixup_cutmix(
                images=images,
                targets=targets,
                num_classes=num_classes,
                mixup_alpha=mixup_alpha,
                cutmix_alpha=cutmix_alpha,
                mixup_prob=mixup_prob,
                cutmix_prob=cutmix_prob,
            )
            logits = model(images)
            loss = _compute_loss(logits, mixed_targets, criterion)
            preds = torch.argmax(logits, dim=1)
            confusion += compute_confusion_matrix(preds, hard_targets, num_classes).to(
                device
            )
        else:
            logits = model(images)
            loss = _compute_loss(logits, targets, criterion)
            preds = torch.argmax(logits, dim=1)
            confusion += compute_confusion_matrix(preds, targets, num_classes).to(device)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), images.size(0))

    accuracy = accuracy_from_confusion(confusion.cpu())
    precision, recall = precision_recall_macro_from_confusion(confusion.cpu())
    f1 = f1_macro_from_confusion(confusion.cpu())
    return loss_meter.average, accuracy, precision, recall, f1


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> Tuple[float, float, float, float, float]:
    model.eval()
    loss_meter = AverageMeter("loss")
    confusion = torch.zeros((num_classes, num_classes), device=device)

    for images, targets in tqdm(loader, desc="Val", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = _compute_loss(logits, targets, criterion)
        preds = torch.argmax(logits, dim=1)
        confusion += compute_confusion_matrix(preds, targets, num_classes).to(device)
        loss_meter.update(loss.item(), images.size(0))

    accuracy = accuracy_from_confusion(confusion.cpu())
    precision, recall = precision_recall_macro_from_confusion(confusion.cpu())
    f1 = f1_macro_from_confusion(confusion.cpu())
    return loss_meter.average, accuracy, precision, recall, f1


@torch.no_grad()
def collect_predictions(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    targets_all = []
    probs_all = []
    for images, targets in tqdm(loader, desc="ROC", leave=False):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        probs_all.append(probs)
        targets_all.append(targets.numpy())
    return np.concatenate(targets_all, axis=0), np.concatenate(probs_all, axis=0)


def plot_roc_curves(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    try:
        from sklearn.metrics import auc, roc_curve
        from sklearn.preprocessing import label_binarize
    except ImportError:
        print("ROC plot skipped: install 'scikit-learn'.")
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    num_classes = y_score.shape[1]
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))

    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(
        y_true_bin.ravel(), y_score.ravel()
    )
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    fig = plt.figure(figsize=(6, 5))
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label=f"micro-average (AUC={roc_auc['micro']:.3f})",
        linewidth=2,
    )
    for i in range(num_classes):
        plt.plot(
            fpr[i], tpr[i], label=f"Class {i} (AUC={roc_auc[i]:.3f})", alpha=0.8
        )
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(fontsize=8, loc="lower right")
    plt.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_training_curves(
    history: Dict[str, list], output_path: Path, best_epoch: int | None = None
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["loss"]) + 1)
    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="C0")
    ax1.plot(epochs, history["loss"], label="Loss", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy", color="C1")
    ax2.plot(epochs, history["accuracy"], label="Accuracy", color="C1")
    ax2.tick_params(axis="y", labelcolor="C1")
    if best_epoch is not None and 1 <= best_epoch <= len(history["accuracy"]):
        best_acc = history["accuracy"][best_epoch - 1]
        ax2.scatter(
            [best_epoch],
            [best_acc],
            color="C1",
            s=120,
            zorder=5,
            edgecolors="black",
            linewidths=2,
            label=f"Best ({best_acc:.3f})",
        )
    ax2.set_ylim(0.0, 1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _best_spatial_layer_name(model: nn.Module) -> str:
    last_conv = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_conv = name
    return last_conv or "fusion (pre-pool spatial features)"


def run_gradcam_visualization(
    model: nn.Module,
    image_path: str,
    image_np: np.ndarray,
    image_size: int,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
    output_dir: Path,
    device: torch.device,
    roi_percentile: float,
) -> None:
    if image_np.size == 0:
        print("Grad-CAM skipped: empty input image.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    target_module = getattr(model, "fusion", None)
    if target_module is None:
        print("Grad-CAM skipped: model has no fusion module.")
        return

    preprocess = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    input_tensor = preprocess(Image.fromarray(image_np)).unsqueeze(0).to(device)

    activations: dict[str, torch.Tensor] = {}

    def forward_hook(_module: nn.Module, _inputs: tuple, output: torch.Tensor) -> None:
        activations["value"] = output
        output.retain_grad()

    hook = target_module.register_forward_hook(forward_hook)
    try:
        logits = model(input_tensor)
        class_idx = int(torch.argmax(logits, dim=1).item())
        score = logits[0, class_idx]
        model.zero_grad(set_to_none=True)
        score.backward()

        feat = activations["value"]
        grads = feat.grad
        if grads is None:
            print("Grad-CAM skipped: no gradients captured.")
            return

        grads2 = grads ** 2
        grads3 = grads ** 3
        denom = 2 * grads2 + (feat * grads3).sum(dim=(2, 3), keepdim=True)
        denom = torch.where(denom != 0, denom, torch.ones_like(denom))
        alpha = grads2 / denom
        weights = (alpha * F.relu(grads)).sum(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * feat).sum(dim=1, keepdim=True))
        cam = cam[0, 0].detach().cpu().numpy()
    finally:
        hook.remove()

    if cam.max() > 0:
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(
        (image_size, image_size), resample=Image.BILINEAR
    )
    cam_arr = np.array(cam_img) / 255.0

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("jet")
    heatmap = cmap(cam_arr)[..., :3]
    image_float = image_np.astype(np.float32) / 255.0
    overlay = (0.45 * heatmap + 0.55 * image_float).clip(0, 1)

    heatmap_path = output_dir / "gradcam_heatmap.png"
    overlay_path = output_dir / "gradcam_overlay.png"
    Image.fromarray((heatmap * 255).astype(np.uint8)).save(heatmap_path)
    Image.fromarray((overlay * 255).astype(np.uint8)).save(overlay_path)

    cutoff = np.percentile(cam_arr, max(0.0, min(roi_percentile, 100.0)))
    roi_mask = cam_arr >= cutoff
    roi_only = np.zeros_like(image_np)
    roi_only[roi_mask] = image_np[roi_mask]
    roi_only_path = output_dir / "gradcam_roi_only.png"
    Image.fromarray(roi_only).save(roi_only_path)

    info_path = output_dir / "visualization_info.txt"
    with open(info_path, "a", encoding="utf-8") as handle:
        handle.write(f"Grad-CAM image: {image_path}\n")
        handle.write(f"Saved Grad-CAM heatmap: {heatmap_path}\n")
        handle.write(f"Saved Grad-CAM overlay: {overlay_path}\n")
        handle.write(f"Saved Grad-CAM ROI-only: {roi_only_path}\n")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    train_cfg = config["train"]
    dataset_cfg = config["dataset"]
    optim_cfg = config["optimizer"]
    sched_cfg = config["lr_scheduler"]

    device_name = args.device or train_cfg.get(
        "device", "cuda" if torch.cuda.is_available() else "cpu"
    )
    device = torch.device(device_name)

    output_dir = Path(train_cfg["save_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_meta = DATASET_CONFIGS.get(dataset_cfg["name"])
    cfg_mean = dataset_cfg.get("mean")
    cfg_std = dataset_cfg.get("std")
    if cfg_mean and cfg_std:
        mean = tuple(cfg_mean)
        std = tuple(cfg_std)
    else:
        mean = dataset_meta.mean if dataset_meta else (0.485, 0.456, 0.406)
        std = dataset_meta.std if dataset_meta else (0.229, 0.224, 0.225)
    if train_cfg.get("use_dataset_stats", False):
        max_samples = train_cfg.get("stats_max_samples", 2000)
        mean, std = compute_dataset_mean_std(
            dataset_root=dataset_cfg["root"],
            dataset_name=dataset_cfg["name"],
            image_size=train_cfg["image_size"],
            num_workers=train_cfg["workers"],
            max_samples=max_samples,
        )
        stats_path = output_dir / "normalization_stats.yaml"
        with open(stats_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(
                {
                    "dataset": dataset_cfg["name"],
                    "mean": [float(v) for v in mean],
                    "std": [float(v) for v in std],
                    "max_samples": max_samples,
                },
                handle,
                default_flow_style=False,
            )

    balance_cfg = train_cfg.get("class_balance", {})
    aug_cfg = train_cfg.get("augmentation", {})
    train_loader, val_loader, num_classes, class_weights = build_dataloaders(
        dataset_root=dataset_cfg["root"],
        dataset_name=dataset_cfg["name"],
        image_size=train_cfg["image_size"],
        batch_size=train_cfg["batch_size"],
        num_workers=train_cfg["workers"],
        mean=mean,
        std=std,
        aug_cfg=aug_cfg,
        balance_cfg=balance_cfg,
    )

    model = BICENet(
        num_classes=num_classes,
        backbone=train_cfg.get("backbone", "resnet101"),
        pretrained=train_cfg.get("pretrained", True),
    ).to(device)
    freeze_epochs = int(train_cfg.get("freeze_backbone_epochs", 0))
    if freeze_epochs > 0:
        for param in model.backbone.parameters():
            param.requires_grad = False
    if balance_cfg.get("loss_weighted", False) and class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=optim_cfg["lr"],
        weight_decay=optim_cfg["weight_decay"],
    )
    scheduler = StepLR(
        optimizer, step_size=sched_cfg["step_size"], gamma=sched_cfg["gamma"]
    )

    best_f1 = 0.0
    best_train_acc = 0.0
    best_epoch_acc = 0
    best_state = None
    history = {
        "loss": [],
        "accuracy": [],
    }
    sep = "-----------------------------------------------------"
    for epoch in range(train_cfg["epochs"]):
        print(sep)
        if freeze_epochs > 0 and epoch == freeze_epochs:
            for param in model.backbone.parameters():
                param.requires_grad = True
        train_loss, train_acc, train_prec, train_rec, train_f1 = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            num_classes,
            mix_cfg=train_cfg.get("mixup", {}),
        )
        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(
            model, val_loader, criterion, device, num_classes
        )
        scheduler.step()

        history["loss"].append(train_loss)
        history["accuracy"].append(train_acc)

        torch.save(
            {
                "model_state": model.state_dict(),
                "epoch": epoch + 1,
                "f1": val_f1,
                "dataset": dataset_cfg["name"],
            },
            output_dir / f"last_{dataset_cfg['name']}.pth",
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch + 1,
                    "f1": best_f1,
                    "dataset": dataset_cfg["name"],
                },
                output_dir / f"best_{dataset_cfg['name']}.pth",
            )

        is_best_acc = train_acc > best_train_acc
        if is_best_acc:
            best_train_acc = train_acc
            best_epoch_acc = epoch + 1
        best_tag = " (best)" if is_best_acc else ""

        print(
            f"Epoch {epoch+1}/{train_cfg['epochs']} | "
            f"loss={train_loss:.4f} | "
            f"accuracy={train_acc:.4f}{best_tag} | "
            f"precision={train_prec:.4f} recall={train_rec:.4f} f1={train_f1:.4f}"
        )
        print(sep)

    if history["loss"]:
        plot_training_curves(
            history,
            output_dir / f"training_curves_{dataset_cfg['name']}.png",
            best_epoch=best_epoch_acc,
        )

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    y_true, y_score = collect_predictions(model, val_loader, device)
    class_names = getattr(val_loader.dataset, "classes", [])
    plot_roc_curves(
        y_true,
        y_score,
        class_names=class_names,
        output_path=output_dir / f"roc_{dataset_cfg['name']}.png",
    )

    gradcam_cfg = train_cfg.get("gradcam", {"enabled": True})
    viz_cfg = train_cfg.get("visualizations", {})
    num_images = int(viz_cfg.get("num_images", 100))

    val_samples = getattr(val_loader.dataset, "samples", [])
    if not val_samples:
        print("Visualization skipped: validation dataset has no samples.")
    else:
        num_images = max(1, min(num_images, len(val_samples)))
        for idx in range(num_images):
            image_path, _ = val_samples[idx]
            image = Image.open(image_path).convert("RGB")
            image = image.resize((train_cfg["image_size"], train_cfg["image_size"]))
            image_np = np.array(image)

            if gradcam_cfg.get("enabled", True):
                run_gradcam_visualization(
                    model=model,
                    image_path=image_path,
                    image_np=image_np,
                    image_size=train_cfg["image_size"],
                    mean=mean,
                    std=std,
                    output_dir=output_dir / "gradcam" / f"sample_{idx:03d}",
                    device=device,
                    roi_percentile=gradcam_cfg.get("roi_percentile", 85.0),
                )


if __name__ == "__main__":
    main()

