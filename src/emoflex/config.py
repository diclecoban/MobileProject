from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_CATALOG = PROJECT_ROOT / "configs" / "datasets.yaml"


@dataclass(frozen=True)
class YOLOSettings:
    """Configuration for YOLO-style datasets."""

    image_dir: str = "images"
    label_dir: str = "labels"
    bbox_expand: float = 0.0
    min_pixels: int = 1


@dataclass(frozen=True)
class DatasetConfig:
    """Normalized representation of a dataset entry."""

    name: str
    dataset_type: str
    root: Path
    splits: Dict[str, str]
    classes: List[str]
    input_size: Tuple[int, int]
    normalization_mean: Tuple[float, float, float]
    normalization_std: Tuple[float, float, float]
    force_grayscale: bool = False
    yolo: Optional[YOLOSettings] = None

    def split_dir(self, split: str) -> Path:
        if split not in self.splits:
            raise KeyError(f"Split '{split}' not defined for dataset '{self.name}'.")
        return self.root / self.splits[split]

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "type": self.dataset_type,
            "root": str(self.root),
            "splits": self.splits,
            "classes": self.classes,
            "input_size": self.input_size,
            "normalization_mean": self.normalization_mean,
            "normalization_std": self.normalization_std,
            "force_grayscale": self.force_grayscale,
            "yolo": None
            if self.yolo is None
            else {
                "image_dir": self.yolo.image_dir,
                "label_dir": self.yolo.label_dir,
                "bbox_expand": self.yolo.bbox_expand,
                "min_pixels": self.yolo.min_pixels,
            },
        }


def _make_dataset_config(name: str, payload: Dict) -> DatasetConfig:
    normalization = payload.get("normalization", {})
    mean = tuple(normalization.get("mean", [0.485, 0.456, 0.406]))
    std = tuple(normalization.get("std", [0.229, 0.224, 0.225]))
    yolo_cfg = payload.get("yolo")
    yolo = None
    if yolo_cfg:
        yolo = YOLOSettings(
            image_dir=yolo_cfg.get("image_dir", "images"),
            label_dir=yolo_cfg.get("label_dir", "labels"),
            bbox_expand=float(yolo_cfg.get("bbox_expand", 0.0)),
            min_pixels=int(yolo_cfg.get("min_pixels", 1)),
        )

    return DatasetConfig(
        name=name,
        dataset_type=payload["type"],
        root=(PROJECT_ROOT / payload["root"]).resolve(),
        splits=payload.get("splits", {}),
        classes=payload.get("classes", []),
        input_size=tuple(payload.get("input_size", [224, 224])),
        normalization_mean=mean,
        normalization_std=std,
        force_grayscale=bool(payload.get("force_grayscale", False)),
        yolo=yolo,
    )


def load_dataset_catalog(path: Path = DEFAULT_DATASET_CATALOG) -> Dict[str, DatasetConfig]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset catalog not found at {path}")
    with path.open("r", encoding="utf-8") as fp:
        payload = yaml.safe_load(fp) or {}
    datasets = payload.get("datasets", {})
    catalog: Dict[str, DatasetConfig] = {}
    for name, cfg in datasets.items():
        catalog[name] = _make_dataset_config(name, cfg)
    return catalog


def resolve_dataset(name: str, catalog: Optional[Dict[str, DatasetConfig]] = None) -> DatasetConfig:
    catalog = catalog or load_dataset_catalog()
    try:
        return catalog[name]
    except KeyError as exc:
        raise KeyError(f"Dataset '{name}' is not defined in the catalog.") from exc
