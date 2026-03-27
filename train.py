from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader

from config import DataConfig, ModelConfig, TrainConfig
from dataset import LandslideTileDataset
from losses import build_loss
from model import LandslideSSMoEModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a first-pass landslide model built around the SSMoE block."
    )
    train_source = parser.add_mutually_exclusive_group(required=True)
    train_source.add_argument("--train-manifest")
    train_source.add_argument("--train-root")
    val_source = parser.add_mutually_exclusive_group()
    val_source.add_argument("--val-manifest")
    val_source.add_argument("--val-root")
    parser.add_argument("--task-type", choices=["segmentation", "classification"], default="segmentation")
    parser.add_argument("--in-channels", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--positive-class-weight", type=float, default=1.0)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--specific-experts", type=int, default=4)
    parser.add_argument("--shared-experts", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="checkpoint.pt")
    return parser.parse_args()


def build_dataloaders(data_config: DataConfig) -> tuple[DataLoader, DataLoader | None]:
    train_dataset = LandslideTileDataset(
        task_type=data_config.task_type,
        manifest_path=data_config.train_manifest,
        data_root=data_config.train_root,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config.batch_size,
        shuffle=True,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory,
    )

    if not data_config.val_manifest:
        if not data_config.val_root:
            return train_loader, None

    val_dataset = LandslideTileDataset(
        task_type=data_config.task_type,
        manifest_path=data_config.val_manifest,
        data_root=data_config.val_root,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config.batch_size,
        shuffle=False,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory,
    )
    return train_loader, val_loader


def move_batch_to_device(batch: dict[str, object], device: str) -> tuple[Tensor, Tensor]:
    image = batch["image"].to(device)  # type: ignore[assignment]
    target = batch["target"].to(device)  # type: ignore[assignment]
    return image, target


def compute_segmentation_iou(logits: Tensor, targets: Tensor, eps: float = 1e-6) -> float:
    predictions = (torch.sigmoid(logits) >= 0.5).float()
    targets = (targets >= 0.5).float()
    intersection = (predictions * targets).sum(dim=(1, 2, 3))
    union = ((predictions + targets) > 0).float().sum(dim=(1, 2, 3))
    iou = (intersection + eps) / (union + eps)
    return float(iou.mean().item())


def compute_classification_accuracy(logits: Tensor, targets: Tensor) -> float:
    predictions = (torch.sigmoid(logits) >= 0.5).float()
    return float((predictions == targets).float().mean().item())


def train_one_epoch(
    model: LandslideSSMoEModel,
    loader: DataLoader,
    criterion,
    optimizer: AdamW,
    device: str,
    log_every: int,
    task_type: str,
) -> float:
    model.train()
    total_loss = 0.0

    for step, batch in enumerate(loader, start=1):
        image, target = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        output = model(image)
        logits = output.logits

        if task_type == "classification":
            target = target.view_as(logits)

        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())

        if step % log_every == 0:
            print(f"step={step} loss={loss.item():.4f}")

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(
    model: LandslideSSMoEModel,
    loader: DataLoader,
    criterion,
    device: str,
    task_type: str,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_metric = 0.0

    for batch in loader:
        image, target = move_batch_to_device(batch, device)
        output = model(image)
        logits = output.logits

        if task_type == "classification":
            target = target.view_as(logits)
            metric = compute_classification_accuracy(logits, target)
        else:
            metric = compute_segmentation_iou(logits, target)

        loss = criterion(logits, target)
        total_loss += float(loss.item())
        total_metric += metric

    count = max(len(loader), 1)
    return total_loss / count, total_metric / count


def main() -> None:
    args = parse_args()

    data_config = DataConfig(
        train_manifest=args.train_manifest,
        train_root=args.train_root,
        val_manifest=args.val_manifest,
        val_root=args.val_root,
        task_type=args.task_type,
        in_channels=args.in_channels,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    model_config = ModelConfig(
        in_channels=args.in_channels,
        task_type=args.task_type,
        dim=args.dim,
        patch_size=args.patch_size,
        specific_experts=args.specific_experts,
        shared_experts=args.shared_experts,
        top_k=args.top_k,
        dropout=args.dropout,
    )
    train_config = TrainConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        positive_class_weight=args.positive_class_weight,
        log_every=args.log_every,
        device=args.device,
        output_path=args.output,
    )

    train_loader, val_loader = build_dataloaders(data_config)

    model = LandslideSSMoEModel(
        in_channels=model_config.in_channels,
        task_type=model_config.task_type,
        dim=model_config.dim,
        patch_size=model_config.patch_size,
        specific_experts=model_config.specific_experts,
        shared_experts=model_config.shared_experts,
        top_k=model_config.top_k,
        expert_hidden_dim=model_config.expert_hidden_dim,
        dropout=model_config.dropout,
    ).to(train_config.device)

    criterion = build_loss(
        task_type=data_config.task_type,
        positive_class_weight=train_config.positive_class_weight,
    ).to(train_config.device)
    optimizer = AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    best_metric = None
    metric_name = "iou" if data_config.task_type == "segmentation" else "accuracy"

    for epoch in range(1, train_config.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=train_config.device,
            log_every=train_config.log_every,
            task_type=data_config.task_type,
        )
        print(f"epoch={epoch} train_loss={train_loss:.4f}")

        if val_loader is None:
            continue

        val_loss, val_metric = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=train_config.device,
            task_type=data_config.task_type,
        )
        print(f"epoch={epoch} val_loss={val_loss:.4f} val_{metric_name}={val_metric:.4f}")

        is_best = best_metric is None or val_metric > best_metric
        if is_best:
            best_metric = val_metric
            output_path = Path(train_config.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_metric": val_metric,
                    "task_type": data_config.task_type,
                    "in_channels": data_config.in_channels,
                },
                output_path,
            )
            print(f"saved_checkpoint={output_path}")


if __name__ == "__main__":
    main()
