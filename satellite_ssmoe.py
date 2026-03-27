from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn


@dataclass
class RoutingState:
    logits: Tensor
    probabilities: Tensor
    topk_indices: Optional[Tensor] = None
    topk_probabilities: Optional[Tensor] = None


@dataclass
class SSMoEOutput:
    tokens: Tensor
    feature_map: Tensor
    specific_routing: RoutingState
    shared_routing: RoutingState


class MLPExpert(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class SpecificMoE(nn.Module):
    """
    Implements the paper's specific-expert branch:
        g_x = W_e x
        p_i(x) = softmax(g_x)_i
        SpecMoE(x) = sum_{i in TopK} p_i(x) * e_i(x)
    """

    def __init__(
        self,
        dim: int,
        num_experts: int,
        hidden_dim: int,
        top_k: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if top_k < 1 or top_k > num_experts:
            raise ValueError("top_k must satisfy 1 <= top_k <= num_experts.")

        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(dim, num_experts)
        self.experts = nn.ModuleList(
            MLPExpert(dim=dim, hidden_dim=hidden_dim, dropout=dropout)
            for _ in range(num_experts)
        )

    def forward(self, x: Tensor) -> tuple[Tensor, RoutingState]:
        logits = self.router(x)
        probabilities = torch.softmax(logits, dim=-1)

        topk_probabilities, topk_indices = probabilities.topk(self.top_k, dim=-1)

        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=2)
        selected_outputs = expert_outputs.gather(
            2,
            topk_indices.unsqueeze(-1).expand(-1, -1, -1, expert_outputs.size(-1)),
        )
        mixed = (topk_probabilities.unsqueeze(-1) * selected_outputs).sum(dim=2)

        return mixed, RoutingState(
            logits=logits,
            probabilities=probabilities,
            topk_indices=topk_indices,
            topk_probabilities=topk_probabilities,
        )


class SharedMoE(nn.Module):
    """
    Implements the paper's shared-expert branch:
        g_x = W_f x
        p_i(x) = softmax(g_x)_i
        ShareMoE(x) = sum_i p_i(x) * f_i(x)
    """

    def __init__(
        self,
        dim: int,
        num_experts: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.router = nn.Linear(dim, num_experts)
        self.experts = nn.ModuleList(
            MLPExpert(dim=dim, hidden_dim=hidden_dim, dropout=dropout)
            for _ in range(num_experts)
        )

    def forward(self, x: Tensor) -> tuple[Tensor, RoutingState]:
        logits = self.router(x)
        probabilities = torch.softmax(logits, dim=-1)

        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=2)
        mixed = (probabilities.unsqueeze(-1) * expert_outputs).sum(dim=2)

        return mixed, RoutingState(logits=logits, probabilities=probabilities)


class TokenwiseSSMoE(nn.Module):
    """
    Equations (1)-(5) from the paper, kept intact but applied to image tokens.
    """

    def __init__(
        self,
        dim: int,
        specific_experts: int = 4,
        shared_experts: int = 2,
        top_k: int = 2,
        expert_hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = expert_hidden_dim or dim * 4
        self.norm = nn.LayerNorm(dim)
        self.specific_moe = SpecificMoE(
            dim=dim,
            num_experts=specific_experts,
            hidden_dim=hidden_dim,
            top_k=top_k,
            dropout=dropout,
        )
        self.shared_moe = SharedMoE(
            dim=dim,
            num_experts=shared_experts,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(self, x: Tensor) -> tuple[Tensor, RoutingState, RoutingState]:
        x_norm = self.norm(x)
        specific_out, specific_routing = self.specific_moe(x_norm)
        shared_out, shared_routing = self.shared_moe(x_norm)
        return specific_out + shared_out, specific_routing, shared_routing


class SatellitePatchEmbedding(nn.Module):
    def __init__(self, in_bands: int = 10, dim: int = 128, patch_size: int = 8) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels=in_bands,
            out_channels=dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: Tensor) -> tuple[Tensor, tuple[int, int]]:
        if x.ndim != 4:
            raise ValueError("Expected input shape [batch, bands, height, width].")
        if x.size(1) != self.proj.in_channels:
            raise ValueError(
                f"Expected {self.proj.in_channels} input bands, got {x.size(1)}."
            )
        if x.size(-2) % self.patch_size != 0 or x.size(-1) % self.patch_size != 0:
            raise ValueError(
                "Height and width must be divisible by patch_size for patch embedding."
            )

        patches = self.proj(x)
        grid_h, grid_w = patches.shape[-2:]
        tokens = patches.flatten(2).transpose(1, 2)
        return tokens, (grid_h, grid_w)


class SatelliteSSMoE(nn.Module):
    """
    Adaptation of the EEGMoE routing logic for 10-band satellite imagery.

    Input:
        x: [batch, 10, height, width]

    Output:
        tokens: [batch, num_patches, dim]
        feature_map: [batch, dim, grid_h, grid_w]
    """

    def __init__(
        self,
        in_bands: int = 10,
        dim: int = 128,
        patch_size: int = 8,
        specific_experts: int = 4,
        shared_experts: int = 2,
        top_k: int = 2,
        expert_hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.patch_embed = SatellitePatchEmbedding(
            in_bands=in_bands,
            dim=dim,
            patch_size=patch_size,
        )
        self.ssmoe = TokenwiseSSMoE(
            dim=dim,
            specific_experts=specific_experts,
            shared_experts=shared_experts,
            top_k=top_k,
            expert_hidden_dim=expert_hidden_dim,
            dropout=dropout,
        )

    def forward(self, x: Tensor) -> SSMoEOutput:
        tokens, (grid_h, grid_w) = self.patch_embed(x)
        mixed_tokens, specific_routing, shared_routing = self.ssmoe(tokens)
        feature_map = mixed_tokens.transpose(1, 2).reshape(
            x.size(0), mixed_tokens.size(-1), grid_h, grid_w
        )
        return SSMoEOutput(
            tokens=mixed_tokens,
            feature_map=feature_map,
            specific_routing=specific_routing,
            shared_routing=shared_routing,
        )


if __name__ == "__main__":
    model = SatelliteSSMoE(
        in_bands=10,
        dim=96,
        patch_size=8,
        specific_experts=4,
        shared_experts=2,
        top_k=2,
    )
    dummy = torch.randn(2, 10, 64, 64)
    output = model(dummy)

    print("tokens:", tuple(output.tokens.shape))
    print("feature_map:", tuple(output.feature_map.shape))
    print("specific topk indices:", tuple(output.specific_routing.topk_indices.shape))
    print("specific probs:", tuple(output.specific_routing.probabilities.shape))
    print("shared probs:", tuple(output.shared_routing.probabilities.shape))
