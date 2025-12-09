#!/usr/bin/env python3
"""
Lightweight CLI to run inference with `models/emotion.onnx` on images or folders.

It mirrors the preprocessing that was used during training:
- optional grayscale conversion (driven by the dataset catalog)
- resize to dataset input_size
- normalization with dataset mean/std
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

import cv2 as cv
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.is_dir() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DEFAULT_LABELS = ["Angry", "Fear", "Happy", "Sad", "Surprise"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

try:
    from emoflex.config import DatasetConfig, load_dataset_catalog, resolve_dataset
except Exception as exc:  # pragma: no cover - import guard for missing PYTHONPATH
    raise SystemExit(
        "Failed to import emoflex. Make sure PYTHONPATH includes 'src' "
        f"(original error: {exc})"
    ) from exc


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-8)


def normalize_percent(p: np.ndarray) -> np.ndarray:
    p = p.astype(np.float32)
    total = p.sum()
    if total <= 0:
        return p
    return (p / total) * 100.0


class EmotionONNXRunner:
    def __init__(
        self,
        onnx_path: Path,
        input_size: Sequence[int],
        mean: Sequence[float],
        std: Sequence[float],
        force_grayscale: bool,
    ) -> None:
        import onnxruntime as ort

        self.session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        self.size = tuple(int(v) for v in input_size)
        self.mean = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.asarray(std, dtype=np.float32).reshape(3, 1, 1)
        self.force_grayscale = bool(force_grayscale)

    def _prepare(self, bgr_img: np.ndarray) -> np.ndarray:
        if self.force_grayscale:
            img = cv.cvtColor(bgr_img, cv.COLOR_BGR2GRAY)
            img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
        else:
            img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)

        img = cv.resize(img, self.size, interpolation=cv.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # CHW
        img = (img - self.mean) / self.std
        return np.expand_dims(img, 0)  # NCHW

    def predict(self, bgr_img: np.ndarray) -> np.ndarray:
        inputs = self._prepare(bgr_img)
        logits = self.session.run([self.output_name], {self.input_name: inputs})[0][0]
        probs = softmax(logits)
        return normalize_percent(probs)


def resolve_dataset_config(dataset_name: str) -> DatasetConfig:
    catalog = load_dataset_catalog()
    return resolve_dataset(dataset_name, catalog)


def build_runner(onnx_path: Path, cfg: DatasetConfig) -> EmotionONNXRunner:
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found at {onnx_path}")
    return EmotionONNXRunner(
        onnx_path=onnx_path,
        input_size=cfg.input_size,
        mean=cfg.normalization_mean,
        std=cfg.normalization_std,
        force_grayscale=cfg.force_grayscale,
    )


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def expand_inputs(paths: Iterable[str], recursive: bool) -> List[Path]:
    collected: List[Path] = []
    for raw in paths:
        path = Path(raw).expanduser()
        if is_image_file(path):
            collected.append(path)
            continue
        if path.is_dir():
            iterator = path.rglob("*") if recursive else path.glob("*")
            for candidate in iterator:
                if is_image_file(candidate):
                    collected.append(candidate)
            continue
        print(f"[WARN] Skipping '{raw}': not a readable image or directory.")
    return sorted(collected)


def fmt_topk(percentages: np.ndarray, labels: Sequence[str], topk: int) -> str:
    idx = np.argsort(percentages)[::-1]
    lines = []
    for i in idx[:topk]:
        label = labels[i] if i < len(labels) else f"class_{i}"
        lines.append(f"    {label:<12} {percentages[i]:5.1f}%")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference with emotion.onnx on images or folders."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Image file(s) or directories containing images.",
    )
    parser.add_argument(
        "--onnx",
        default=os.environ.get("EMOFLEX_ONNX", "models/emotion.onnx"),
        help="Path to the exported ONNX model.",
    )
    parser.add_argument(
        "--dataset",
        default=os.environ.get("EMOFLEX_DATASET", "data_faces"),
        help="Dataset name from configs/datasets.yaml (used for labels & preprocessing).",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=3,
        help="How many labels to show per image.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search directories recursively for images.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional cap on the number of images to process.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_cfg = resolve_dataset_config(args.dataset)
    runner = build_runner(Path(args.onnx), dataset_cfg)
    if dataset_cfg.classes:
        labels = list(dataset_cfg.classes)
    else:
        print("[WARN] Dataset has no class list; falling back to default labels.")
        labels = DEFAULT_LABELS

    images = expand_inputs(args.paths, args.recursive)
    if not images:
        raise SystemExit("No images found. Provide valid file paths or directories.")

    if args.max_files is not None:
        images = images[: args.max_files]

    print(
        f"[INFO] Using ONNX: {args.onnx} | dataset: {args.dataset} | files: {len(images)}"
    )

    for path in images:
        img = cv.imread(str(path))
        if img is None:
            print(f"[WARN] Could not read image: {path}")
            continue
        perc = runner.predict(img)
        top_line = fmt_topk(perc, labels, args.topk)
        best_idx = int(np.argmax(perc))
        best_label = (
            labels[best_idx] if best_idx < len(labels) else f"class_{best_idx}"
        )
        print(f"\n[{path}] -> {best_label} ({perc[best_idx]:.1f}%)")
        print(top_line)


if __name__ == "__main__":
    main()
