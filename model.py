from __future__ import annotations

from dataclasses import dataclass

import torch.nn.functional as F
from torch import Tensor, nn

from satellite_ssmoe import RoutingState, SatelliteSSMoE


@dataclass
class LandslideModelOutput:
    logits: Tensor
    specific_routing: RoutingState
    shared_routing: RoutingState


class LandslideSSMoEModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 10,
        task_type: str = "segmentation",
        dim: int = 128,
        patch_size: int = 8,
        specific_experts: int = 4,
        shared_experts: int = 2,
        top_k: int = 2,
        expert_hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if task_type not in {"segmentation", "classification"}:
            raise ValueError("task_type must be 'segmentation' or 'classification'.")

        self.task_type = task_type
        self.backbone = SatelliteSSMoE(
            in_bands=in_channels,
            dim=dim,
            patch_size=patch_size,
            specific_experts=specific_experts,
            shared_experts=shared_experts,
            top_k=top_k,
            expert_hidden_dim=expert_hidden_dim,
            dropout=dropout,
        )

        if task_type == "segmentation":
            self.head = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(dim, 1, kernel_size=1),
            )
        else:
            self.head = nn.Linear(dim, 1)

    def forward(self, image: Tensor) -> LandslideModelOutput:
        features = self.backbone(image)

        if self.task_type == "segmentation":
            patch_logits = self.head(features.feature_map)
            logits = F.interpolate(
                patch_logits,
                size=image.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        else:
            pooled = features.feature_map.mean(dim=(-2, -1))
            logits = self.head(pooled).squeeze(-1)

        return LandslideModelOutput(
            logits=logits,
            specific_routing=features.specific_routing,
            shared_routing=features.shared_routing,
        )
