#!/usr/bin/env python3
"""Flexible emotion model training script."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from emoflex.config import load_dataset_catalog, resolve_dataset
from emoflex.data import build_dataloaders
from emoflex.evaluation import evaluate_model
from emoflex.models import create_model, trainable_parameters
from emoflex.trainer import Trainer
from emoflex.transforms import build_transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an emotion recognition model.")
    parser.add_argument("--dataset", default="facedata", help="Dataset name defined in configs/datasets.yaml.")
    parser.add_argument("--model", default="mobilenet_v3_small", help="Backbone architecture.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience.")
    parser.add_argument("--lr-patience", type=int, default=3, help="LR scheduler patience.")
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--grad-clip", type=float, default=None, help="Gradient clipping norm.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for checkpoints/history (defaults to artifacts/<dataset>).",
    )
    parser.add_argument("--checkpoint-prefix", default="emotion_model")
    parser.add_argument("--checkpoint-freq", type=int, default=5)
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--freeze-backbone", action="store_true", help="Train only the classifier head.")
    parser.add_argument("--no-pretrained", action="store_true", help="Disable ImageNet weights.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_output_dir(dataset: str, output_dir: str | None) -> Path:
    if output_dir:
        return Path(output_dir)
    return PROJECT_ROOT / "artifacts" / dataset


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    catalog = load_dataset_catalog()
    dataset_cfg = resolve_dataset(args.dataset, catalog)
    print(f"Dataset: {dataset_cfg.name} ({dataset_cfg.dataset_type})")

    output_dir = resolve_output_dir(dataset_cfg.name, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Artifacts will be saved under: {output_dir}")

    train_tfms, eval_tfms = build_transforms(dataset_cfg)
    loaders = build_dataloaders(
        dataset_cfg,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_transform=train_tfms,
        eval_transform=eval_tfms,
        val_split=args.val_split,
        test_split=args.test_split,
    )

    model = create_model(
        args.model,
        num_classes=dataset_cfg.num_classes,
        pretrained=not args.no_pretrained,
        head_only=args.freeze_backbone,
    )
    optimizer = optim.AdamW(
        trainable_parameters(model),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=args.lr_patience,
        min_lr=args.min_lr,
    )
    criterion = nn.CrossEntropyLoss()
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        grad_clip=args.grad_clip,
    )

    summary = trainer.fit(
        loaders,
        epochs=args.epochs,
        patience=args.patience,
        output_dir=output_dir,
        checkpoint_prefix=args.checkpoint_prefix,
        checkpoint_freq=args.checkpoint_freq,
    )

    if summary.best_checkpoint:
        print(f"Best checkpoint: {summary.best_checkpoint}")
    if summary.last_checkpoint:
        print(f"Last checkpoint: {summary.last_checkpoint}")

    if "test" in loaders and summary.best_checkpoint:
        print("Evaluating best checkpoint on test split...")
        state = torch.load(summary.best_checkpoint, map_location=device)
        model.load_state_dict(state)
        test_result = evaluate_model(model, loaders["test"], device, dataset_cfg.classes)
        print(f"Test accuracy: {test_result.accuracy*100:.2f}%")
        print("Macro F1: {:.2f}%".format(test_result.macro_f1 * 100))
        print(test_result.report)


if __name__ == "__main__":
    main()
