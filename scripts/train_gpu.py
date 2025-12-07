#!/usr/bin/env python3
"""Convenience entry point to train the emotion model on a GPU."""

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
    parser = argparse.ArgumentParser(
        description="Train an emotion recognition backbone on a CUDA-capable GPU.",
    )
    parser.add_argument("--dataset", default="data_faces", help="Dataset name from configs/datasets.yaml.")
    parser.add_argument("--model", default="mobilenet_v3_small", help="Model architecture to use.")
    parser.add_argument("--epochs", type=int, default=25, help="Number of full training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size.")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Base learning rate for AdamW.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--patience", type=int, default=6, help="Early stopping patience on validation accuracy.")
    parser.add_argument("--lr-patience", type=int, default=3, help="LR scheduler patience.")
    parser.add_argument("--min-lr", type=float, default=1e-6, help="Lower bound for the LR scheduler.")
    parser.add_argument("--grad-clip", type=float, default=5.0, help="Gradient clipping L2 norm. Use 0 to disable.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of dataloader workers.")
    parser.add_argument(
        "--experiment",
        default=None,
        help="Optional experiment name used to group artifacts. Defaults to <model>.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override the artifacts directory. Defaults to artifacts/<dataset>/<experiment>.",
    )
    parser.add_argument("--checkpoint-prefix", default="gpu_model", help="Checkpoint filename prefix.")
    parser.add_argument("--checkpoint-freq", type=int, default=5, help="Create intermediate checkpoints every N epochs.")
    parser.add_argument("--freeze-backbone", action="store_true", help="Update classifier head only.")
    parser.add_argument("--no-pretrained", action="store_true", help="Disable ImageNet initialization.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index to target.")
    parser.add_argument("--eval-test", action="store_true", help="Evaluate the best checkpoint on the test split.")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_output_dir(dataset_name: str, experiment: str | None, output_override: str | None) -> Path:
    if output_override:
        return Path(output_override)
    exp_name = experiment or "default"
    return PROJECT_ROOT / "artifacts" / dataset_name / exp_name


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device not found. Please ensure a GPU-capable PyTorch install is active.")

    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    torch.backends.cudnn.benchmark = True
    set_seed(args.seed)

    catalog = load_dataset_catalog()
    dataset_cfg = resolve_dataset(args.dataset, catalog)
    experiment_name = args.experiment or args.model
    output_dir = resolve_output_dir(dataset_cfg.name, experiment_name, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {torch.cuda.get_device_name(device)}")
    print(f"Dataset: {dataset_cfg.name} ({dataset_cfg.dataset_type})")
    print(f"Artifacts directory: {output_dir}")

    train_tfms, eval_tfms = build_transforms(dataset_cfg)
    loaders = build_dataloaders(
        dataset_cfg,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_transform=train_tfms,
        eval_transform=eval_tfms,
    )

    model = create_model(
        args.model,
        num_classes=dataset_cfg.num_classes,
        pretrained=not args.no_pretrained,
        head_only=args.freeze_backbone,
    ).to(device)

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
    grad_clip = args.grad_clip
    if grad_clip is not None and grad_clip <= 0:
        grad_clip = None
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=nn.CrossEntropyLoss(),
        device=device,
        scheduler=scheduler,
        grad_clip=grad_clip,
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
    print(f"Best validation accuracy: {summary.best_metrics.get('val_acc', 0)*100:.2f}% at epoch {summary.best_epoch}")

    if args.eval_test and "test" in loaders and summary.best_checkpoint:
        print("Evaluating on test split with the best checkpoint...")
        state = torch.load(summary.best_checkpoint, map_location=device)
        model.load_state_dict(state)
        model = model.to(device)
        results = evaluate_model(model, loaders["test"], device, dataset_cfg.classes)
        print(f"Test accuracy: {results.accuracy*100:.2f}%")
        print(f"Macro F1: {results.macro_f1*100:.2f}%")
        print(results.report)


if __name__ == "__main__":
    main()
