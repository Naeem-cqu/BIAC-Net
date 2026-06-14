"""Training metrics utilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class AverageMeter:
    name: str
    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += value * n
        self.count += n

    @property
    def average(self) -> float:
        return self.total / max(self.count, 1)


def compute_confusion_matrix(
    preds: torch.Tensor, targets: torch.Tensor, num_classes: int
) -> torch.Tensor:
    preds = preds.view(-1).to(torch.int64)
    targets = targets.view(-1).to(torch.int64)
    indices = targets * num_classes + preds
    mat = torch.bincount(indices, minlength=num_classes**2)
    return mat.reshape(num_classes, num_classes)


def accuracy_from_confusion(confusion: torch.Tensor) -> float:
    correct = torch.trace(confusion).item()
    total = confusion.sum().item()
    return correct / max(total, 1.0)


def f1_macro_from_confusion(confusion: torch.Tensor, eps: float = 1e-8) -> float:
    true_pos = torch.diag(confusion).float()
    predicted = confusion.sum(dim=0).float()
    actual = confusion.sum(dim=1).float()
    precision = true_pos / (predicted + eps)
    recall = true_pos / (actual + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return torch.mean(f1).item()


def precision_recall_macro_from_confusion(
    confusion: torch.Tensor, eps: float = 1e-8
) -> tuple[float, float]:
    true_pos = torch.diag(confusion).float()
    predicted = confusion.sum(dim=0).float()
    actual = confusion.sum(dim=1).float()
    precision = true_pos / (predicted + eps)
    recall = true_pos / (actual + eps)
    return torch.mean(precision).item(), torch.mean(recall).item()

