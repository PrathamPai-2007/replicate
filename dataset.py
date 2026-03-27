from __future__ import annotations

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


class LandslideTileDataset(Dataset[dict[str, Any]]):
    def __init__(self, manifest_path: str | Path, task_type: str = "segmentation") -> None:
        super().__init__()
        if task_type not in {"segmentation", "classification"}:
            raise ValueError("task_type must be 'segmentation' or 'classification'.")

        self.task_type = task_type
        self.records = load_manifest(manifest_path)

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
