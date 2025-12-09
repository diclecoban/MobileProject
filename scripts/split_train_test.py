#!/usr/bin/env python3
"""
Split a class-folder dataset into train/test subfolders (80/20 by default).

Original layout (per-emotion folders directly under Data/):
Data/
  Angry/
  Happy/
  ...

After running:
Data/
  train/Angry
  train/Happy
  test/Angry
  test/Happy
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import List, Sequence, Tuple

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
RESERVED = {"train", "test"}


def find_class_dirs(root: Path) -> List[Path]:
    dirs = [
        p for p in root.iterdir() if p.is_dir() and p.name not in RESERVED and not p.name.startswith(".")
    ]
    if not dirs:
        raise SystemExit(f"No class folders found under {root}")
    return sorted(dirs, key=lambda p: p.name.lower())


def collect_images(class_dir: Path) -> List[Path]:
    return sorted(
        [
            p
            for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ]
    )


def split_indices(total: int, test_ratio: float) -> Tuple[int, int]:
    if total == 0:
        return 0, 0
    raw = int(round(total * test_ratio))
    test_count = max(1 if test_ratio > 0 else 0, raw)
    test_count = min(test_count, total)
    if total > 1 and test_count == total:
        test_count = total - 1
    return total - test_count, test_count


def move_files(files: Sequence[Path], dst_dir: Path, copy: bool, dry_run: bool) -> None:
    if not dry_run:
        dst_dir.mkdir(parents=True, exist_ok=True)
    for src in files:
        dst = dst_dir / src.name
        if dst.exists():
            raise FileExistsError(f"Destination already has {dst}")
        action = "COPY" if copy else "MOVE"
        if dry_run:
            print(f"[DRY] {action} {src} -> {dst}")
            continue
        if copy:
            shutil.copy2(src, dst)
        else:
            shutil.move(src, dst)


def cleanup_empty(dir_path: Path) -> None:
    try:
        next(dir_path.iterdir())
    except StopIteration:
        dir_path.rmdir()
    except FileNotFoundError:
        pass


def run_split(root: Path, test_ratio: float, seed: int, copy: bool, dry_run: bool) -> None:
    train_root = root / "train"
    test_root = root / "test"

    if not dry_run:
        train_root.mkdir(exist_ok=True)
        test_root.mkdir(exist_ok=True)
    else:
        print("[INFO] Dry run – no files will be moved.")

    class_dirs = find_class_dirs(root)
    total_train = total_test = 0

    for class_dir in class_dirs:
        files = collect_images(class_dir)
        if not files:
            print(f"[WARN] Skipping '{class_dir.name}': no images.")
            continue

        rng = random.Random(f"{seed}_{class_dir.name}")
        rng.shuffle(files)
        train_count, test_count = split_indices(len(files), test_ratio)

        test_files = files[:test_count]
        train_files = files[test_count:]

        print(
            f"[INFO] {class_dir.name}: {len(files)} files -> train {train_count}, test {test_count}"
        )

        move_files(train_files, train_root / class_dir.name, copy, dry_run)
        move_files(test_files, test_root / class_dir.name, copy, dry_run)

        total_train += train_count
        total_test += test_count

        if not copy and not dry_run:
            cleanup_empty(class_dir)

    print(f"[DONE] Train: {total_train} | Test: {total_test}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split per-emotion folders into Data/train & Data/test."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("Data"),
        help="Root folder containing emotion subfolders.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Fraction of files to move into test split (per class).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed controlling the shuffle per class.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of moving them (original structure stays intact).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned moves without touching files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.root.exists():
        raise SystemExit(f"Root folder '{args.root}' does not exist.")
    run_split(
        root=args.root,
        test_ratio=args.test_ratio,
        seed=args.seed,
        copy=args.copy,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
