from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass
class SampleRecord:
    sample_id: str
    image_path: str
    target_path: str | None = None
    label: float | None = None
    event_id: str | None = None


VALID_TENSOR_SUFFIXES = {".pt", ".pth", ".npy", ".npz"}
NAME_SUFFIXES = ("_image", "_img", "_mask", "_target", "_label")


def _strip_name_suffixes(stem: str) -> str:
    normalized = stem
    changed = True
    while changed:
        changed = False
        for suffix in NAME_SUFFIXES:
            if normalized.lower().endswith(suffix):
                normalized = normalized[: -len(suffix)]
                changed = True
    return normalized


def _sample_key(path: Path, base_dir: Path) -> str:
    relative = path.relative_to(base_dir)
    stem = _strip_name_suffixes(relative.stem)
    parent = relative.parent.as_posix()
    return stem if parent == "." else f"{parent}/{stem}"


def _derive_event_id(path: Path, base_dir: Path) -> str | None:
    parent = path.relative_to(base_dir).parent.as_posix()
    return None if parent == "." else parent


def _iter_tensor_files(root: str | Path) -> list[Path]:
    root_path = Path(root)
    return sorted(
        path
        for path in root_path.rglob("*")
        if path.is_file() and path.suffix.lower() in VALID_TENSOR_SUFFIXES
    )


def _load_tensor(path: str | Path) -> Tensor:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in {".pt", ".pth"}:
        tensor = torch.load(path, map_location="cpu")
        if isinstance(tensor, dict):
            if "tensor" in tensor:
                tensor = tensor["tensor"]
            else:
                raise ValueError(f"Tensor dict at {path} must contain a 'tensor' key.")
        return torch.as_tensor(tensor)

    if suffix == ".npy":
        return torch.from_numpy(np.load(path))

    if suffix == ".npz":
        archive = np.load(path)
        keys = list(archive.keys())
        if len(keys) != 1:
            raise ValueError(
                f"NPZ archive at {path} must contain exactly one array, found {keys}."
            )
        return torch.from_numpy(archive[keys[0]])

    raise ValueError(f"Unsupported tensor file format: {path.suffix}")


def discover_segmentation_records(data_root: str | Path) -> list[SampleRecord]:
    root = Path(data_root)
    images_dir = root / "images"
    targets_dir = root / "targets"
    if not targets_dir.exists():
        targets_dir = root / "masks"

    if not images_dir.exists():
        raise ValueError(f"Expected images directory at {images_dir}.")
    if not targets_dir.exists():
        raise ValueError(f"Expected targets/ or masks/ directory under {root}.")

    image_files = _iter_tensor_files(images_dir)
    target_files = _iter_tensor_files(targets_dir)
    if not image_files:
        raise ValueError(f"No image tensors found under {images_dir}.")
    if not target_files:
        raise ValueError(f"No target tensors found under {targets_dir}.")

    target_index: dict[str, Path] = {}
    for target_path in target_files:
        key = _sample_key(target_path, targets_dir)
        if key in target_index:
            raise ValueError(f"Duplicate target key '{key}' found in {targets_dir}.")
        target_index[key] = target_path

    records: list[SampleRecord] = []
    missing_targets: list[str] = []

    for image_path in image_files:
        key = _sample_key(image_path, images_dir)
        target_path = target_index.get(key)
        if target_path is None:
            missing_targets.append(key)
            continue

        records.append(
            SampleRecord(
                sample_id=key,
                image_path=str(image_path),
                target_path=str(target_path),
                event_id=_derive_event_id(image_path, images_dir),
            )
        )

    if missing_targets:
        preview = ", ".join(missing_targets[:5])
        raise ValueError(
            f"Missing matching targets for {len(missing_targets)} image files under {images_dir}. "
            f"Examples: {preview}"
        )

    image_keys = {_sample_key(path, images_dir) for path in image_files}
    extra_targets = sorted(set(target_index) - image_keys)
    if extra_targets:
        preview = ", ".join(extra_targets[:5])
        raise ValueError(
            f"Found {len(extra_targets)} target files without matching images under {targets_dir}. "
            f"Examples: {preview}"
        )

    return records


def _load_classification_labels(labels_path: str | Path) -> dict[str, tuple[float, str | None]]:
    path = Path(labels_path)
    suffix = path.suffix.lower()
    label_map: dict[str, tuple[float, str | None]] = {}

    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row_number, row in enumerate(reader, start=2):
                if not row.get("id"):
                    raise ValueError(f"labels.csv row {row_number} is missing 'id'.")
                if row.get("label") is None:
                    raise ValueError(f"labels.csv row {row_number} is missing 'label'.")
                label_map[str(row["id"])] = (
                    float(row["label"]),
                    str(row["event_id"]) if row.get("event_id") else None,
                )
        return label_map

    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row: dict[str, Any] = json.loads(line)
                if "id" not in row:
                    raise ValueError(f"labels.jsonl line {line_number} is missing 'id'.")
                if "label" not in row:
                    raise ValueError(f"labels.jsonl line {line_number} is missing 'label'.")
                label_map[str(row["id"])] = (
                    float(row["label"]),
                    str(row["event_id"]) if row.get("event_id") else None,
                )
        return label_map

    raise ValueError("Classification roots require labels.csv or labels.jsonl.")


def discover_classification_records(data_root: str | Path) -> list[SampleRecord]:
    root = Path(data_root)
    images_dir = root / "images"
    labels_csv = root / "labels.csv"
    labels_jsonl = root / "labels.jsonl"

    if not images_dir.exists():
        raise ValueError(f"Expected images directory at {images_dir}.")

    labels_path = labels_csv if labels_csv.exists() else labels_jsonl
    if not labels_path.exists():
        raise ValueError(f"Expected labels.csv or labels.jsonl under {root}.")

    image_files = _iter_tensor_files(images_dir)
    if not image_files:
        raise ValueError(f"No image tensors found under {images_dir}.")

    label_map = _load_classification_labels(labels_path)
    records: list[SampleRecord] = []
    missing_labels: list[str] = []

    for image_path in image_files:
        key = _sample_key(image_path, images_dir)
        label_info = label_map.get(key)
        if label_info is None:
            missing_labels.append(key)
            continue

        label_value, event_id = label_info
        records.append(
            SampleRecord(
                sample_id=key,
                image_path=str(image_path),
                label=label_value,
                event_id=event_id or _derive_event_id(image_path, images_dir),
            )
        )

    if missing_labels:
        preview = ", ".join(missing_labels[:5])
        raise ValueError(
            f"Missing labels for {len(missing_labels)} image files under {images_dir}. "
            f"Examples: {preview}"
        )

    image_keys = {_sample_key(path, images_dir) for path in image_files}
    extra_labels = sorted(set(label_map) - image_keys)
    if extra_labels:
        preview = ", ".join(extra_labels[:5])
        raise ValueError(
            f"Found {len(extra_labels)} labels without matching image tensors under {images_dir}. "
            f"Examples: {preview}"
        )

    return records


def load_manifest(path: str | Path) -> list[SampleRecord]:
    records: list[SampleRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            raw: dict[str, Any] = json.loads(line)
            if "id" not in raw:
                raise ValueError(f"Manifest line {line_number} is missing 'id'.")
            if "image" not in raw:
                raise ValueError(f"Manifest line {line_number} is missing 'image'.")

            records.append(
                SampleRecord(
                    sample_id=str(raw["id"]),
                    image_path=str(raw["image"]),
                    target_path=(
                        str(raw["target"]) if raw.get("target") is not None else None
                    ),
                    label=float(raw["label"]) if raw.get("label") is not None else None,
                    event_id=str(raw["event_id"]) if raw.get("event_id") else None,
                )
            )

    if not records:
        raise ValueError(f"Manifest at {path} is empty.")

    return records


def load_records(
    *,
    task_type: str,
    manifest_path: str | Path | None = None,
    data_root: str | Path | None = None,
) -> list[SampleRecord]:
    if (manifest_path is None) == (data_root is None):
        raise ValueError("Provide exactly one of manifest_path or data_root.")

    if manifest_path is not None:
        return load_manifest(manifest_path)

    if task_type == "segmentation":
        return discover_segmentation_records(data_root)
    if task_type == "classification":
        return discover_classification_records(data_root)
    raise ValueError("task_type must be 'segmentation' or 'classification'.")


class LandslideTileDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        *,
        task_type: str = "segmentation",
        manifest_path: str | Path | None = None,
        data_root: str | Path | None = None,
    ) -> None:
        super().__init__()
        if task_type not in {"segmentation", "classification"}:
            raise ValueError("task_type must be 'segmentation' or 'classification'.")

        self.task_type = task_type
        self.records = load_records(
            task_type=task_type,
            manifest_path=manifest_path,
            data_root=data_root,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image = _load_tensor(record.image_path).float()

        if image.ndim != 3:
            raise ValueError(
                f"Expected image tensor [channels, height, width], got {tuple(image.shape)} "
                f"for sample {record.sample_id}."
            )

        if self.task_type == "segmentation":
            if not record.target_path:
                raise ValueError(
                    f"Sample {record.sample_id} requires a 'target' path for segmentation."
                )
            target = _load_tensor(record.target_path).float()
            if target.ndim == 2:
                target = target.unsqueeze(0)
            if target.ndim != 3:
                raise ValueError(
                    f"Expected target tensor [1, height, width] or [height, width], "
                    f"got {tuple(target.shape)} for sample {record.sample_id}."
                )
        else:
            if record.label is None:
                raise ValueError(
                    f"Sample {record.sample_id} requires a scalar 'label' for classification."
                )
            target = torch.tensor(record.label, dtype=torch.float32)

        return {
            "id": record.sample_id,
            "image": image,
            "target": target,
            "event_id": record.event_id,
        }
