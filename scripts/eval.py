#!/usr/bin/env python3
"""Evaluate a trained emotion model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from emoflex.config import load_dataset_catalog, resolve_dataset
from emoflex.data import build_dataset
from emoflex.evaluation import evaluate_model
from emoflex.models import create_model
from emoflex.transforms import build_eval_transform


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on a dataset split.")
    parser.add_argument("--dataset", default="facedata", help="Dataset name from configs/datasets.yaml.")
    parser.add_argument("--split", default="test", help="Dataset split to evaluate.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path (defaults to artifacts/<dataset>/emotion_model_best.pth).")
    parser.add_argument("--model", default="mobilenet_v3_small")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def resolve_checkpoint(dataset: str, checkpoint: str | None) -> Path:
    if checkpoint:
        return Path(checkpoint)
    return PROJECT_ROOT / "artifacts" / dataset / "emotion_model_best.pth"


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    catalog = load_dataset_catalog()
    dataset_cfg = resolve_dataset(args.dataset, catalog)
    transform = build_eval_transform(dataset_cfg)
    dataset = build_dataset(dataset_cfg, args.split, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    checkpoint = resolve_checkpoint(dataset_cfg.name, args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    model = create_model(args.model, dataset_cfg.num_classes, pretrained=False)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device)

    result = evaluate_model(model, loader, device, dataset_cfg.classes)

    print(f"Accuracy: {result.accuracy*100:.2f}%")
    print(f"Macro Precision: {result.macro_precision*100:.2f}% | Macro Recall: {result.macro_recall*100:.2f}%")
    print(f"Macro F1: {result.macro_f1*100:.2f}%")
    print("\nPer-class metrics:")
    for idx, cls in enumerate(dataset_cfg.classes):
        print(
            f"{cls:>10s} | P: {result.per_class_precision[idx]*100:6.2f}% "
            f"R: {result.per_class_recall[idx]*100:6.2f}% "
            f"F1: {result.per_class_f1[idx]*100:6.2f}% "
            f"N: {result.support[idx]:4d}"
        )
    print("\nConfusion Matrix:")
    print(result.confusion)
    print("\nClassification report:")
    print(result.report)


if __name__ == "__main__":
    main()
