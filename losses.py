from __future__ import annotations

import torch
from torch import Tensor, nn


def dice_loss(logits: Tensor, targets: Tensor, eps: float = 1e-6) -> Tensor:
    probabilities = torch.sigmoid(logits)
    targets = targets.float()

    intersection = (probabilities * targets).sum(dim=(1, 2, 3))
    denominator = probabilities.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    score = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - score.mean()


class SegmentationLoss(nn.Module):
    def __init__(self, positive_class_weight: float = 1.0) -> None:
        super().__init__()
        pos_weight = torch.tensor([positive_class_weight], dtype=torch.float32)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        bce = self.bce(logits, targets.float())
        dice = dice_loss(logits, targets)
        return bce + dice


def build_loss(task_type: str, positive_class_weight: float = 1.0) -> nn.Module:
    if task_type == "segmentation":
        return SegmentationLoss(positive_class_weight=positive_class_weight)
    if task_type == "classification":
        pos_weight = torch.tensor([positive_class_weight], dtype=torch.float32)
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    raise ValueError("task_type must be 'segmentation' or 'classification'.")
