from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

MPL_CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from config import ModelConfig
from dataset import LandslideTileDataset
from model import LandslideSSMoEModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference on processed tiles and save probability maps plus review images."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data-root", help="Processed dataset root such as data/val.")
    source.add_argument("--manifest-path", help="Optional JSONL manifest path.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path from train.py.")
    parser.add_argument("--output-root", default="predictions", help="Folder to save outputs.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--max-visualizations", type=int, default=16)
    parser.add_argument("--task-type", choices=["segmentation", "classification"], default=None)
    parser.add_argument("--in-channels", type=int, default=None)
    parser.add_argument("--dim", type=int, default=ModelConfig.dim)
    parser.add_argument("--patch-size", type=int, default=ModelConfig.patch_size)
    parser.add_argument("--specific-experts", type=int, default=ModelConfig.specific_experts)
    parser.add_argument("--shared-experts", type=int, default=ModelConfig.shared_experts)
    parser.add_argument("--top-k", type=int, default=ModelConfig.top_k)
    parser.add_argument("--dropout", type=float, default=ModelConfig.dropout)
    parser.add_argument("--expert-hidden-dim", type=int, default=None)
    return parser.parse_args()


def _load_metadata(data_root: str | None) -> dict[str, dict[str, Any]]:
    if data_root is None:
        return {}
    metadata_dir = Path(data_root) / "metadata"
    if not metadata_dir.exists():
        return {}
    return {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(metadata_dir.glob("*.json"))
    }


def _build_model_config(checkpoint: dict[str, Any], args: argparse.Namespace) -> ModelConfig:
    saved = checkpoint.get("model_config", {})
    task_type = saved.get("task_type") or checkpoint.get("task_type") or args.task_type or "segmentation"
    in_channels = saved.get("in_channels") or checkpoint.get("in_channels") or args.in_channels
    if in_channels is None:
        raise ValueError(
            "Could not infer in_channels from the checkpoint. Pass --in-channels explicitly."
        )

    return ModelConfig(
        in_channels=int(in_channels),
        task_type=str(task_type),
        dim=int(saved.get("dim", args.dim)),
        patch_size=int(saved.get("patch_size", args.patch_size)),
        specific_experts=int(saved.get("specific_experts", args.specific_experts)),
        shared_experts=int(saved.get("shared_experts", args.shared_experts)),
        top_k=int(saved.get("top_k", args.top_k)),
        expert_hidden_dim=(
            int(saved["expert_hidden_dim"])
            if saved.get("expert_hidden_dim") is not None
            else args.expert_hidden_dim
        ),
        dropout=float(saved.get("dropout", args.dropout)),
    )


def _sample_output_stem(sample_id: str) -> Path:
    sample_path = Path(sample_id)
    return sample_path


def _event_channel_names(metadata: dict[str, dict[str, Any]], event_id: str | None) -> list[str] | None:
    if event_id is None:
        return None
    payload = metadata.get(event_id)
    if not payload:
        return None
    channel_names = payload.get("channel_names")
    return list(channel_names) if isinstance(channel_names, list) else None


def _normalize_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.float32, copy=False)
    normalized = np.zeros_like(rgb, dtype=np.float32)
    for channel in range(rgb.shape[-1]):
        plane = rgb[..., channel]
        lo = float(np.percentile(plane, 2))
        hi = float(np.percentile(plane, 98))
        if hi <= lo:
            normalized[..., channel] = 0.0
        else:
            normalized[..., channel] = np.clip((plane - lo) / (hi - lo), 0.0, 1.0)
    return normalized


def _build_rgb_preview(image: np.ndarray, channel_names: list[str] | None) -> np.ndarray:
    indices = None
    if channel_names:
        lookup = {name.lower(): index for index, name in enumerate(channel_names)}
        wanted = ["sentinel2_b04", "sentinel2_b03", "sentinel2_b02"]
        if all(name in lookup for name in wanted):
            indices = [lookup[name] for name in wanted]

    if indices is None:
        if image.shape[0] >= 3:
            indices = [0, 1, 2]
        elif image.shape[0] == 2:
            indices = [0, 1, 1]
        else:
            indices = [0, 0, 0]

    rgb = np.stack([image[index] for index in indices], axis=-1)
    return _normalize_rgb(rgb)


def _mean_iou(probability: np.ndarray, target: np.ndarray, threshold: float) -> float:
    prediction = probability >= threshold
    target_binary = target >= 0.5
    intersection = float(np.logical_and(prediction, target_binary).sum())
    union = float(np.logical_or(prediction, target_binary).sum())
    return (intersection + 1e-6) / (union + 1e-6)


def _save_segmentation_visualization(
    *,
    image: np.ndarray,
    target: np.ndarray | None,
    probability: np.ndarray,
    binary_prediction: np.ndarray,
    expert_map: np.ndarray | None,
    channel_names: list[str] | None,
    output_path: Path,
    sample_id: str,
    event_id: str | None,
) -> None:
    rgb = _build_rgb_preview(image, channel_names)
    columns = 5 if expert_map is not None else 4
    figure, axes = plt.subplots(1, columns, figsize=(4 * columns, 4), dpi=150)
    if columns == 1:
        axes = [axes]

    axes[0].imshow(rgb)
    axes[0].set_title("RGB Preview")
    axes[0].axis("off")

    if target is not None:
        axes[1].imshow(target, cmap="gray", vmin=0.0, vmax=1.0)
        axes[1].set_title("Target Mask")
    else:
        axes[1].imshow(np.zeros_like(probability), cmap="gray", vmin=0.0, vmax=1.0)
        axes[1].set_title("Target Mask (N/A)")
    axes[1].axis("off")

    axes[2].imshow(probability, cmap="magma", vmin=0.0, vmax=1.0)
    axes[2].set_title("Probability")
    axes[2].axis("off")

    axes[3].imshow(binary_prediction, cmap="gray", vmin=0.0, vmax=1.0)
    axes[3].set_title("Binary Prediction")
    axes[3].axis("off")

    if expert_map is not None:
        axes[4].imshow(expert_map, cmap="tab10")
        axes[4].set_title("Specific Expert")
        axes[4].axis("off")

    figure.suptitle(f"event={event_id or 'unknown'} sample={sample_id}", fontsize=10)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_config = _build_model_config(checkpoint, args)

    dataset = LandslideTileDataset(
        task_type=model_config.task_type,
        manifest_path=args.manifest_path,
        data_root=args.data_root,
    )
    metadata = _load_metadata(args.data_root)

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
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(args.device)
    model.eval()

    output_root = Path(args.output_root)
    summary: dict[str, Any] = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "task_type": model_config.task_type,
        "threshold": args.threshold,
        "sample_count": len(dataset),
        "visualizations_saved": 0,
    }

    if model_config.task_type == "segmentation":
        ious: list[float] = []

    with torch.no_grad():
        for index in range(len(dataset)):
            sample = dataset[index]
            image = sample["image"].unsqueeze(0).to(args.device)
            output = model(image)
            sample_id = str(sample["id"])
            event_id = str(sample["event_id"]) if sample["event_id"] else None
            relative_stem = _sample_output_stem(sample_id)
            event_channel_names = _event_channel_names(metadata, event_id)

            if model_config.task_type == "segmentation":
                probability = torch.sigmoid(output.logits).squeeze().cpu().numpy().astype(np.float32)
                binary_prediction = (probability >= args.threshold).astype(np.float32)

                probability_path = output_root / "probabilities" / relative_stem.parent / f"{relative_stem.name}_prob.npy"
                binary_path = output_root / "binaries" / relative_stem.parent / f"{relative_stem.name}_pred.npy"
                probability_path.parent.mkdir(parents=True, exist_ok=True)
                binary_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(probability_path, probability)
                np.save(binary_path, binary_prediction)

                target_tensor = sample["target"]
                target = target_tensor.squeeze().cpu().numpy().astype(np.float32) if target_tensor is not None else None
                if target is not None:
                    ious.append(_mean_iou(probability, target, args.threshold))

                expert_map = None
                if output.specific_routing.topk_indices is not None:
                    grid_size = (
                        probability.shape[0] // model_config.patch_size,
                        probability.shape[1] // model_config.patch_size,
                    )
                    expert_map = (
                        output.specific_routing.topk_indices[0, :, 0]
                        .reshape(grid_size[0], grid_size[1])
                        .cpu()
                        .numpy()
                    )

                if summary["visualizations_saved"] < args.max_visualizations:
                    visualization_path = output_root / "visualizations" / relative_stem.parent / f"{relative_stem.name}.png"
                    _save_segmentation_visualization(
                        image=sample["image"].cpu().numpy(),
                        target=target,
                        probability=probability,
                        binary_prediction=binary_prediction,
                        expert_map=expert_map,
                        channel_names=event_channel_names,
                        output_path=visualization_path,
                        sample_id=sample_id,
                        event_id=event_id,
                    )
                    summary["visualizations_saved"] += 1
            else:
                probability = float(torch.sigmoid(output.logits).item())
                relative_dir = output_root / "classification_scores" / relative_stem.parent
                relative_dir.mkdir(parents=True, exist_ok=True)
                (relative_dir / f"{relative_stem.name}.json").write_text(
                    json.dumps(
                        {
                            "sample_id": sample_id,
                            "event_id": event_id,
                            "probability": probability,
                            "prediction": float(probability >= args.threshold),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )

    if model_config.task_type == "segmentation":
        summary["mean_iou"] = float(np.mean(ious)) if ious else None

    summary_path = output_root / "summary.json"
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"predicted_samples={summary['sample_count']} "
        f"task_type={summary['task_type']} "
        f"output_root={output_root}"
    )
    if summary.get("mean_iou") is not None:
        print(f"mean_iou={summary['mean_iou']:.4f}")
    print(f"summary_path={summary_path}")


if __name__ == "__main__":
    main()
