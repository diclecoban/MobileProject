from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

from .config import DatasetConfig

Image.MAX_IMAGE_PIXELS = None  # allow large files


@dataclass
class YOLOSample:
    image_path: Path
    label: int
    bbox: Tuple[float, float, float, float]  # (cx, cy, w, h) normalized


class YOLOFaceDataset(Dataset):
    """Turns YOLO annotations into classification samples by cropping faces."""

    def __init__(
        self,
        cfg: DatasetConfig,
        split: str,
        transform: Optional[Callable] = None,
    ):
        if cfg.yolo is None:
            raise ValueError("YOLO settings are missing from dataset config.")
        self.cfg = cfg
        self.split = split
        self.transform = transform
        self.expand = float(cfg.yolo.bbox_expand)
        self.min_pixels = int(cfg.yolo.min_pixels)

        split_root = cfg.split_dir(split)
        self.image_dir = split_root / cfg.yolo.image_dir
        self.label_dir = split_root / cfg.yolo.label_dir

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.label_dir.exists():
            raise FileNotFoundError(f"Label directory not found: {self.label_dir}")

        self.samples = self._load_samples()

    def _load_samples(self) -> List[YOLOSample]:
        image_lookup: Dict[str, Path] = {}
        for img_path in self.image_dir.iterdir():
            if img_path.is_file() and img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                image_lookup[img_path.stem] = img_path

        samples: List[YOLOSample] = []
        for lbl_path in sorted(self.label_dir.glob("*.txt")):
            stem = lbl_path.stem
            img_path = image_lookup.get(stem)
            if img_path is None:
                continue
            with lbl_path.open("r", encoding="utf-8") as fp:
                for line in fp:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls_id = int(float(parts[0]))
                    bbox = tuple(float(x) for x in parts[1:])
                    samples.append(YOLOSample(image_path=img_path, label=cls_id, bbox=bbox))  # type: ignore[arg-type]
        if not samples:
            raise RuntimeError(f"No samples found for split '{self.split}' in {self.label_dir}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _crop(self, img: Image.Image, bbox: Tuple[float, float, float, float]) -> Image.Image:
        cx, cy, bw, bh = bbox
        w, h = img.size
        box_w = max(bw * w, self.min_pixels)
        box_h = max(bh * h, self.min_pixels)
        pad = self.expand * max(box_w, box_h)
        half_w = box_w / 2 + pad
        half_h = box_h / 2 + pad

        center_x = cx * w
        center_y = cy * h
        x0 = max(0, int(center_x - half_w))
        y0 = max(0, int(center_y - half_h))
        x1 = min(w, int(center_x + half_w))
        y1 = min(h, int(center_y + half_h))
        if x1 <= x0 or y1 <= y0:
            return img
        return img.crop((x0, y0, x1, y1))

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        with Image.open(sample.image_path) as img:
            img = img.convert("RGB")
            face = self._crop(img, sample.bbox)
        if self.transform:
            face = self.transform(face)
        return face, sample.label


def _clone_imagefolder_subset(
    base_dataset: datasets.ImageFolder,
    indices: List[int],
    transform,
) -> datasets.ImageFolder:
    subset = datasets.ImageFolder(base_dataset.root, transform=transform)
    subset.classes = base_dataset.classes
    subset.class_to_idx = base_dataset.class_to_idx
    subset.samples = [base_dataset.samples[i] for i in indices]
    subset.targets = [base_dataset.targets[i] for i in indices]
    subset.imgs = subset.samples
    return subset


def _allocate_counts(total: int, ratios: List[float]) -> List[int]:
    raw = [ratio * total for ratio in ratios]
    floors = [int(math.floor(value)) for value in raw]
    remainder = total - sum(floors)
    frac = [value - math.floor(value) for value in raw]
    order = sorted(range(len(ratios)), key=lambda idx: frac[idx], reverse=True)
    for idx in order:
        if remainder <= 0:
            break
        floors[idx] += 1
        remainder -= 1
    if remainder > 0:
        floors[-1] += remainder
    return floors


def _stratified_split_indices(
    base_dataset: datasets.ImageFolder,
    ratios: Dict[str, float],
    split_order: List[str],
    seed: int,
) -> Dict[str, List[int]]:
    rng = random.Random(seed)
    class_indices: Dict[int, List[int]] = defaultdict(list)
    for idx, (_, label) in enumerate(base_dataset.samples):
        class_indices[label].append(idx)

    split_indices = {split: [] for split in split_order}
    ratio_values = [ratios[split] for split in split_order]

    for indices in class_indices.values():
        rng.shuffle(indices)
        alloc = _allocate_counts(len(indices), ratio_values)
        start = 0
        for split, count in zip(split_order, alloc):
            if count <= 0:
                continue
            split_indices[split].extend(indices[start : start + count])
            start += count
    return split_indices


def _plain_split_indices(
    total_count: int,
    ratios: Dict[str, float],
    split_order: List[str],
    seed: int,
) -> Dict[str, List[int]]:
    rng = random.Random(seed)
    all_indices = list(range(total_count))
    rng.shuffle(all_indices)
    ratio_values = [ratios[split] for split in split_order]
    alloc = _allocate_counts(total_count, ratio_values)
    split_indices = {split: [] for split in split_order}
    start = 0
    for split, count in zip(split_order, alloc):
        if count <= 0:
            continue
        split_indices[split] = all_indices[start : start + count]
        start += count
    return split_indices


def _build_autosplit_imagefolder(
    cfg: DatasetConfig,
    batch_size: int,
    num_workers: int,
    train_transform,
    eval_transform,
    val_split: str,
    test_split: Optional[str],
    pin_memory: bool,
) -> Dict[str, DataLoader]:
    if cfg.auto_split is None:
        raise ValueError("auto_split settings are required for automatic ImageFolder splits.")

    auto_cfg = cfg.auto_split
    ratios = auto_cfg.ratios
    required_splits = ["train", val_split]
    for split in required_splits:
        ratio = ratios.get(split)
        if ratio is None or ratio <= 0:
            raise ValueError(f"auto_split.ratios must define a positive entry for '{split}'.")
    split_order = ["train", val_split]
    include_test = test_split and ratios.get(test_split, 0.0) > 0
    if include_test:
        split_order.append(test_split)  # type: ignore[arg-type]

    total_ratio = sum(ratios.get(split, 0.0) for split in split_order)
    if not math.isclose(total_ratio, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(
            f"auto_split ratios for dataset '{cfg.name}' must sum to 1.0 for splits {split_order} "
            f"(got {total_ratio:.4f})."
        )

    base_dataset = datasets.ImageFolder(cfg.root)
    if len(base_dataset) == 0:
        raise RuntimeError(f"No samples found under '{cfg.root}'.")

    if auto_cfg.stratified:
        split_indices = _stratified_split_indices(base_dataset, ratios, split_order, auto_cfg.seed)
    else:
        split_indices = _plain_split_indices(len(base_dataset), ratios, split_order, auto_cfg.seed)

    datasets_by_split: Dict[str, Dataset] = {}
    for split in split_order:
        indices = split_indices.get(split, [])
        if not indices:
            raise RuntimeError(
                f"Split '{split}' is empty after automatic split. "
                "Check the ratios or class sample counts."
            )
        transform = train_transform if split == "train" else eval_transform
        datasets_by_split[split] = _clone_imagefolder_subset(base_dataset, indices, transform)

    loaders: Dict[str, DataLoader] = {}
    train_dataset = datasets_by_split["train"]
    loaders["train"] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_dataset = datasets_by_split[val_split]
    loaders["val"] = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    if include_test and test_split:
        test_dataset = datasets_by_split[test_split]
        loaders["test"] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    return loaders


def build_dataset(cfg: DatasetConfig, split: str, transform=None) -> Dataset:
    dataset_type = cfg.dataset_type.lower()
    if dataset_type == "imagefolder":
        split_dir = cfg.split_dir(split)
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        return datasets.ImageFolder(split_dir, transform=transform)
    if dataset_type == "yolo_faces":
        return YOLOFaceDataset(cfg, split, transform=transform)
    raise ValueError(f"Unsupported dataset type: {cfg.dataset_type}")


def build_dataloaders(
    cfg: DatasetConfig,
    batch_size: int,
    num_workers: int,
    train_transform,
    eval_transform,
    val_split: str = "val",
    test_split: Optional[str] = "test",
    pin_memory: bool = True,
) -> Dict[str, DataLoader]:
    loaders: Dict[str, DataLoader] = {}

    if cfg.dataset_type.lower() == "imagefolder" and cfg.auto_split:
        return _build_autosplit_imagefolder(
            cfg,
            batch_size,
            num_workers,
            train_transform,
            eval_transform,
            val_split,
            test_split,
            pin_memory,
        )

    train_ds = build_dataset(cfg, "train", transform=train_transform)
    val_ds = build_dataset(cfg, val_split, transform=eval_transform)

    loaders["train"] = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    loaders["val"] = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    if test_split and test_split in cfg.splits:
        test_ds = build_dataset(cfg, test_split, transform=eval_transform)
        loaders["test"] = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    return loaders


def iter_class_counts(dataset: Dataset) -> List[int]:
    counts: Dict[int, int] = {}
    for _, label in dataset:
        counts[label] = counts.get(label, 0) + 1
    max_idx = max(counts.keys())
    return [counts.get(i, 0) for i in range(max_idx + 1)]
