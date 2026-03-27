from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dataset import SampleRecord, load_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a JSONL manifest from an easy folder-based dataset layout."
    )
    parser.add_argument("--data-root", required=True, help="Dataset root containing images/ and targets/ or labels.csv.")
    parser.add_argument("--output", required=True, help="Path to write the JSONL manifest.")
    parser.add_argument(
        "--task-type",
        choices=["segmentation", "classification"],
        default="segmentation",
    )
    parser.add_argument(
        "--relative-paths",
        action="store_true",
        help="Write paths relative to the manifest file instead of absolute paths.",
    )
    return parser.parse_args()


def record_to_json(record: SampleRecord, manifest_dir: Path, relative_paths: bool) -> dict[str, object]:
    def normalize(path: str | None) -> str | None:
        if path is None:
            return None
        resolved = Path(path).resolve()
        if relative_paths:
            return str(Path(os.path.relpath(resolved, manifest_dir.resolve())).as_posix())
        return str(resolved)

    payload: dict[str, object] = {
        "id": record.sample_id,
        "image": normalize(record.image_path),
    }
    if record.target_path is not None:
        payload["target"] = normalize(record.target_path)
    if record.label is not None:
        payload["label"] = record.label
    if record.event_id is not None:
        payload["event_id"] = record.event_id
    return payload


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = load_records(task_type=args.task_type, data_root=args.data_root)
    manifest_dir = output_path.parent

    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            payload = record_to_json(
                record=record,
                manifest_dir=manifest_dir,
                relative_paths=args.relative_paths,
            )
            handle.write(json.dumps(payload) + "\n")

    print(f"wrote_manifest={output_path} samples={len(records)}")


if __name__ == "__main__":
    main()
