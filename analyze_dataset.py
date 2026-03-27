from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from dataset import LandslideTileDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize processed tiles and highlight segmentation imbalance issues."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data-root", help="Processed dataset root such as data/train or data/val.")
    source.add_argument("--manifest-path", help="Optional JSONL manifest path.")
    parser.add_argument(
        "--task-type",
        choices=["segmentation", "classification"],
        default="segmentation",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on the number of samples to inspect.",
    )
    parser.add_argument(
        "--output-json",
        help="Optional path to save the summary as JSON.",
    )
    return parser.parse_args()


def _load_metadata(data_root: str | None) -> dict[str, dict[str, Any]]:
    if data_root is None:
        return {}

    metadata_dir = Path(data_root) / "metadata"
    if not metadata_dir.exists():
        return {}

    payloads: dict[str, dict[str, Any]] = {}
    for path in sorted(metadata_dir.glob("*.json")):
        payloads[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    return payloads


def _segmentation_summary(dataset: LandslideTileDataset, limit: int) -> dict[str, Any]:
    positive_tiles = 0
    total_positive_pixels = 0.0
    total_pixels = 0
    max_mask_sum = 0.0
    max_mask_sample = None

    for index in range(limit):
        sample = dataset[index]
        target = sample["target"].numpy()
        positive_pixels = float(target.sum())
        total_positive_pixels += positive_pixels
        total_pixels += int(np.prod(target.shape))
        if positive_pixels > 0.0:
            positive_tiles += 1
        if positive_pixels > max_mask_sum:
            max_mask_sum = positive_pixels
            max_mask_sample = str(sample["id"])

    return {
        "positive_tiles": positive_tiles,
        "empty_tiles": limit - positive_tiles,
        "positive_tile_fraction": positive_tiles / max(limit, 1),
        "positive_pixel_fraction": total_positive_pixels / max(total_pixels, 1),
        "max_mask_sum": max_mask_sum,
        "max_mask_sample": max_mask_sample,
    }


def _classification_summary(dataset: LandslideTileDataset, limit: int) -> dict[str, Any]:
    positives = 0
    for index in range(limit):
        sample = dataset[index]
        positives += int(float(sample["target"].item()) >= 0.5)

    return {
        "positive_labels": positives,
        "negative_labels": limit - positives,
        "positive_label_fraction": positives / max(limit, 1),
    }


def main() -> None:
    args = parse_args()
    dataset = LandslideTileDataset(
        task_type=args.task_type,
        manifest_path=args.manifest_path,
        data_root=args.data_root,
    )
    limit = min(len(dataset), args.max_samples) if args.max_samples is not None else len(dataset)
    if limit < 1:
        raise ValueError("No samples available to analyze.")

    first_image = dataset[0]["image"].numpy()
    event_counts = Counter(str(dataset.records[index].event_id or "unknown") for index in range(limit))
    image_shapes = Counter(tuple(dataset[index]["image"].shape) for index in range(min(limit, 32)))

    channel_min = np.full(first_image.shape[0], np.inf, dtype=np.float64)
    channel_max = np.full(first_image.shape[0], -np.inf, dtype=np.float64)
    channel_sum = np.zeros(first_image.shape[0], dtype=np.float64)
    channel_sum_sq = np.zeros(first_image.shape[0], dtype=np.float64)
    channel_pixels = 0

    for index in range(limit):
        image = dataset[index]["image"].numpy().astype(np.float64, copy=False)
        flattened = image.reshape(image.shape[0], -1)
        channel_min = np.minimum(channel_min, flattened.min(axis=1))
        channel_max = np.maximum(channel_max, flattened.max(axis=1))
        channel_sum += flattened.sum(axis=1)
        channel_sum_sq += np.square(flattened).sum(axis=1)
        channel_pixels += flattened.shape[1]

    channel_mean = channel_sum / max(channel_pixels, 1)
    channel_var = np.maximum(channel_sum_sq / max(channel_pixels, 1) - np.square(channel_mean), 0.0)
    channel_std = np.sqrt(channel_var)

    metadata = _load_metadata(args.data_root)
    channel_schemas = {
        event_id: payload.get("channel_names", [])
        for event_id, payload in metadata.items()
    }

    summary: dict[str, Any] = {
        "task_type": args.task_type,
        "sample_count": limit,
        "dataset_length": len(dataset),
        "event_counts": dict(event_counts),
        "image_shapes_preview": {str(key): value for key, value in image_shapes.items()},
        "channel_count": int(first_image.shape[0]),
        "channel_min": channel_min.tolist(),
        "channel_max": channel_max.tolist(),
        "channel_mean": channel_mean.tolist(),
        "channel_std": channel_std.tolist(),
        "channel_schemas": channel_schemas,
    }

    if args.task_type == "segmentation":
        summary.update(_segmentation_summary(dataset, limit))
    else:
        summary.update(_classification_summary(dataset, limit))

    print(f"samples={summary['sample_count']} task_type={summary['task_type']}")
    print(f"channel_count={summary['channel_count']}")
    print(f"events={summary['event_counts']}")
    if args.task_type == "segmentation":
        print(
            "positive_tiles="
            f"{summary['positive_tiles']} empty_tiles={summary['empty_tiles']} "
            f"positive_tile_fraction={summary['positive_tile_fraction']:.4f} "
            f"positive_pixel_fraction={summary['positive_pixel_fraction']:.6f}"
        )
        print(
            f"max_mask_sum={summary['max_mask_sum']:.1f} "
            f"max_mask_sample={summary['max_mask_sample']}"
        )
    else:
        print(
            "positive_labels="
            f"{summary['positive_labels']} negative_labels={summary['negative_labels']} "
            f"positive_label_fraction={summary['positive_label_fraction']:.4f}"
        )

    if channel_schemas:
        first_event = next(iter(channel_schemas))
        print(f"channel_names[{first_event}]={channel_schemas[first_event]}")

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"wrote_summary={output_path}")


if __name__ == "__main__":
    main()
