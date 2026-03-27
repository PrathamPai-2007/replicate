from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
from rasterio.transform import Affine

from config import ModelConfig
from dataset import LandslideTileDataset
from model import LandslideSSMoEModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stitch tile predictions back into a scene-level probability map and export GeoTIFF outputs."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data-root", help="Processed dataset root such as data_dense/val.")
    source.add_argument("--manifest-path", help="Optional JSONL manifest path.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--event-id", required=True, help="Event id to stitch, for example wayanad_2024_val.")
    parser.add_argument("--output-root", default="scene_exports")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--in-channels", type=int, default=None)
    parser.add_argument("--task-type", choices=["segmentation", "classification"], default=None)
    parser.add_argument("--dim", type=int, default=ModelConfig.dim)
    parser.add_argument("--patch-size", type=int, default=ModelConfig.patch_size)
    parser.add_argument("--specific-experts", type=int, default=ModelConfig.specific_experts)
    parser.add_argument("--shared-experts", type=int, default=ModelConfig.shared_experts)
    parser.add_argument("--top-k", type=int, default=ModelConfig.top_k)
    parser.add_argument("--dropout", type=float, default=ModelConfig.dropout)
    parser.add_argument("--expert-hidden-dim", type=int, default=None)
    return parser.parse_args()


def _build_model_config(checkpoint: dict[str, Any], args: argparse.Namespace) -> ModelConfig:
    saved = checkpoint.get("model_config", {})
    task_type = saved.get("task_type") or checkpoint.get("task_type") or args.task_type or "segmentation"
    in_channels = saved.get("in_channels") or checkpoint.get("in_channels") or args.in_channels
    if in_channels is None:
        raise ValueError("Could not infer in_channels from checkpoint. Pass --in-channels explicitly.")

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


def _load_event_metadata(data_root: str | None, event_id: str) -> dict[str, Any]:
    if data_root is None:
        raise ValueError("Scene stitching requires --data-root so metadata can be loaded.")
    metadata_path = Path(data_root) / "metadata" / f"{event_id}.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found for event '{event_id}': {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _parse_offsets(sample_id: str) -> tuple[int, int]:
    stem = Path(sample_id).name
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"Could not parse tile offsets from sample id: {sample_id}")
    return int(parts[-2]), int(parts[-1])


def _load_reference_grid(metadata: dict[str, Any]) -> tuple[int, int, Affine, Any]:
    sentinel2_paths = metadata.get("sentinel2_paths")
    if not sentinel2_paths:
        raise ValueError("Metadata must include sentinel2_paths for stitched export.")
    reference_path = Path(sentinel2_paths[0])
    with rasterio.open(reference_path) as src:
        return src.height, src.width, src.transform, src.crs


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_config = _build_model_config(checkpoint, args)
    if model_config.task_type != "segmentation":
        raise ValueError("Scene stitching is currently implemented only for segmentation checkpoints.")

    dataset = LandslideTileDataset(
        task_type=model_config.task_type,
        manifest_path=args.manifest_path,
        data_root=args.data_root,
    )
    metadata = _load_event_metadata(args.data_root, args.event_id)
    scene_height, scene_width, transform, crs = _load_reference_grid(metadata)

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

    probability_sum = np.zeros((scene_height, scene_width), dtype=np.float32)
    coverage_count = np.zeros((scene_height, scene_width), dtype=np.float32)
    target_sum = np.zeros((scene_height, scene_width), dtype=np.float32)
    target_count = np.zeros((scene_height, scene_width), dtype=np.float32)
    stitched_samples = 0
    tile_size = int(metadata["tile_size"])

    with torch.no_grad():
        for index in range(len(dataset)):
            sample = dataset[index]
            if sample["event_id"] != args.event_id:
                continue

            image = sample["image"].unsqueeze(0).to(args.device)
            probability = torch.sigmoid(model(image).logits).squeeze().cpu().numpy().astype(np.float32)
            target = sample["target"].squeeze().cpu().numpy().astype(np.float32)
            top, left = _parse_offsets(str(sample["id"]))

            row_slice = slice(top, top + tile_size)
            col_slice = slice(left, left + tile_size)

            probability_sum[row_slice, col_slice] += probability
            coverage_count[row_slice, col_slice] += 1.0
            target_sum[row_slice, col_slice] += target
            target_count[row_slice, col_slice] += 1.0
            stitched_samples += 1

    probability_map = np.divide(
        probability_sum,
        np.maximum(coverage_count, 1.0),
        out=np.zeros_like(probability_sum),
        where=coverage_count > 0,
    )
    target_map = np.divide(
        target_sum,
        np.maximum(target_count, 1.0),
        out=np.zeros_like(target_sum),
        where=target_count > 0,
    )
    binary_map = (probability_map >= args.threshold).astype(np.uint8)
    coverage_mask = (coverage_count > 0).astype(np.uint8)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    probability_tif = output_root / f"{args.event_id}_probability.tif"
    binary_tif = output_root / f"{args.event_id}_binary.tif"
    coverage_tif = output_root / f"{args.event_id}_coverage.tif"
    target_tif = output_root / f"{args.event_id}_target.tif"
    probability_npy = output_root / f"{args.event_id}_probability.npy"
    binary_npy = output_root / f"{args.event_id}_binary.npy"

    profile = {
        "driver": "GTiff",
        "height": scene_height,
        "width": scene_width,
        "count": 1,
        "crs": crs,
        "transform": transform,
        "compress": "lzw",
    }

    with rasterio.open(probability_tif, "w", dtype="float32", **profile) as dst:
        dst.write(probability_map, 1)
    with rasterio.open(binary_tif, "w", dtype="uint8", **profile) as dst:
        dst.write(binary_map, 1)
    with rasterio.open(coverage_tif, "w", dtype="uint8", **profile) as dst:
        dst.write(coverage_mask, 1)
    with rasterio.open(target_tif, "w", dtype="float32", **profile) as dst:
        dst.write(target_map, 1)

    np.save(probability_npy, probability_map)
    np.save(binary_npy, binary_map)

    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "event_id": args.event_id,
        "stitched_samples": stitched_samples,
        "threshold": args.threshold,
        "scene_height": scene_height,
        "scene_width": scene_width,
        "probability_tif": str(probability_tif.resolve()),
        "binary_tif": str(binary_tif.resolve()),
        "coverage_tif": str(coverage_tif.resolve()),
        "target_tif": str(target_tif.resolve()),
    }
    summary_path = output_root / f"{args.event_id}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"stitched_samples={stitched_samples} event_id={args.event_id}")
    print(f"probability_tif={probability_tif}")
    print(f"binary_tif={binary_tif}")
    print(f"coverage_tif={coverage_tif}")
    print(f"target_tif={target_tif}")
    print(f"summary_path={summary_path}")


if __name__ == "__main__":
    main()
