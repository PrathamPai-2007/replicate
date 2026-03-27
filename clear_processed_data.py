from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Safely clear processed training data for a full reset or a selective new attempt."
    )
    parser.add_argument(
        "--root",
        default="data",
        help="Processed data root to clean, for example data or data_dense.",
    )
    parser.add_argument(
        "--split",
        action="append",
        choices=["train", "val", "test"],
        help="Optional split(s) to clean. Repeat the flag to clean multiple splits.",
    )
    parser.add_argument(
        "--event-id",
        action="append",
        default=[],
        help="Optional event id(s) to clean inside the chosen splits. Repeat the flag for multiple events.",
    )
    parser.add_argument(
        "--include-cache",
        action="store_true",
        help="Also delete matching .cache/<event_id> folders or the whole .cache folder during a full reset.",
    )
    parser.add_argument(
        "--remove-empty-parents",
        action="store_true",
        help="Remove empty images/targets/metadata directories after cleanup.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without removing anything.",
    )
    return parser.parse_args()


def _iter_split_targets(root: Path, split: str, event_ids: list[str]) -> list[Path]:
    split_root = root / split
    targets: list[Path] = []

    if not split_root.exists():
        return targets

    if not event_ids:
        targets.append(split_root)
        return targets

    for event_id in event_ids:
        for folder_name in ("images", "targets", "masks"):
            path = split_root / folder_name / event_id
            if path.exists():
                targets.append(path)

        metadata_path = split_root / "metadata" / f"{event_id}.json"
        if metadata_path.exists():
            targets.append(metadata_path)

    return targets


def _iter_cache_targets(root: Path, event_ids: list[str]) -> list[Path]:
    cache_root = root / ".cache"
    if not cache_root.exists():
        return []

    if not event_ids:
        return [cache_root]

    return [cache_root / event_id for event_id in event_ids if (cache_root / event_id).exists()]


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    unique: list[Path] = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def _prune_empty_dirs(root: Path, splits: list[str]) -> list[Path]:
    removed: list[Path] = []
    for split in splits:
        split_root = root / split
        if not split_root.exists():
            continue

        for folder_name in ("images", "targets", "masks", "metadata"):
            folder = split_root / folder_name
            if folder.exists() and not any(folder.iterdir()):
                folder.rmdir()
                removed.append(folder)

        if split_root.exists() and not any(split_root.iterdir()):
            split_root.rmdir()
            removed.append(split_root)
    return removed


def _delete_path(path: Path, dry_run: bool) -> None:
    if dry_run:
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    if not root.exists():
        raise ValueError(f"Processed root does not exist: {root}")

    selected_splits = args.split or ["train", "val", "test"]
    targets: list[Path] = []

    for split in selected_splits:
        targets.extend(_iter_split_targets(root, split, args.event_id))

    if args.include_cache:
        targets.extend(_iter_cache_targets(root, args.event_id))

    targets = _dedupe_paths([path for path in targets if path.exists()])
    if not targets:
        print("nothing_to_delete")
        return

    for path in targets:
        print(f"{'would_delete' if args.dry_run else 'deleting'}={path}")
        _delete_path(path, dry_run=args.dry_run)

    if args.remove_empty_parents and not args.dry_run:
        removed = _prune_empty_dirs(root, selected_splits)
        for path in removed:
            print(f"removed_empty={path}")

    print(f"deleted_count={len(targets)} dry_run={args.dry_run}")


if __name__ == "__main__":
    main()
