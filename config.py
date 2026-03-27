from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DataConfig:
    train_manifest: str | None = None
    train_root: str | None = None
    val_manifest: str | None = None
    val_root: str | None = None
    task_type: str = "segmentation"
    in_channels: int = 10
    batch_size: int = 4
    num_workers: int = 0
    pin_memory: bool = False
    augment_flip: bool = False
    augment_rotate: bool = False
    gaussian_noise_std: float = 0.0


@dataclass
class ModelConfig:
    in_channels: int = 10
    task_type: str = "segmentation"
    dim: int = 128
    patch_size: int = 8
    specific_experts: int = 4
    shared_experts: int = 2
    top_k: int = 2
    expert_hidden_dim: int | None = None
    dropout: float = 0.0


@dataclass
class TrainConfig:
    epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    positive_class_weight: float = 1.0
    log_every: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_path: str = "checkpoint.pt"
    report_dir: str = "reports/runs"
    run_name: str | None = None
