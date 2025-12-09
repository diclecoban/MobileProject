#!/usr/bin/env python3
"""Single entry point to train the Data/ image folders on a CUDA GPU."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch import amp
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_DIR = PROJECT_ROOT / "Data" / "train"
DEFAULT_VAL_DIR = PROJECT_ROOT / "Data" / "test"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "gpu_training"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
AVAILABLE_MODELS = (
    "mobilenet_v3_small",
    "mobilenet_v3_large",
    "resnet18",
    "resnet34",
    "efficientnet_b0",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a torchvision classifier using the folders under Data/train and Data/test. "
            "The script requires a CUDA-capable GPU."
        )
    )
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR, help="Path to Data/train.")
    parser.add_argument("--val-dir", type=Path, default=DEFAULT_VAL_DIR, help="Path to Data/test (validation).")
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=None,
        help="Optional separate test folder. Defaults to --val-dir when omitted.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Where checkpoints/logs are saved.")
    parser.add_argument("--model", choices=AVAILABLE_MODELS, default="mobilenet_v3_small")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-patience", type=int, default=3, help="LR scheduler patience (ReduceLROnPlateau).")
    parser.add_argument("--min-lr", type=float, default=1e-6, help="Lower bound for scheduler.")
    parser.add_argument("--patience", type=int, default=7, help="Stop if validation accuracy stalls for N epochs.")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--grad-clip", type=float, default=5.0, help="Set <=0 to disable gradient clipping.")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true", help="Disable ImageNet initialization.")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze backbone, train classifier head only.")
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="Disable mixed precision.")
    parser.set_defaults(amp=True)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms(img_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    train_tfms = transforms.Compose(
        [
            transforms.Resize(img_size + 32),
            transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_tfms = transforms.Compose(
        [
            transforms.Resize(img_size + 32),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_tfms, eval_tfms


def build_model(name: str, num_classes: int, pretrained: bool, freeze_backbone: bool) -> nn.Module:
    name = name.lower()
    model: nn.Module
    head_attr = "classifier"

    if name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
    elif name == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
    elif name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        head_attr = "fc"
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif name == "resnet34":
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        model = models.resnet34(weights=weights)
        head_attr = "fc"
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {name}")

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        head = getattr(model, head_attr, None)
        if isinstance(head, nn.Module):
            for param in head.parameters():
                param.requires_grad = True

    return model


def make_dataloader(
    dataset: datasets.ImageFolder,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
    grad_clip: Optional[float],
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    autocast_kwargs = {
        "device_type": device.type,
        "dtype": torch.float16 if device.type == "cuda" else torch.bfloat16,
        "enabled": amp_enabled,
    }

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with amp.autocast(**autocast_kwargs):
            outputs = model(images)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        if grad_clip:
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        preds = outputs.argmax(dim=1)
        total_loss += loss.item() * targets.size(0)
        total_correct += (preds == targets).sum().item()
        total_samples += targets.size(0)

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, targets)
        preds = outputs.argmax(dim=1)
        total_loss += loss.item() * targets.size(0)
        total_correct += (preds == targets).sum().item()
        total_samples += targets.size(0)

    return total_loss / total_samples, total_correct / total_samples


def ensure_dir(path: Path, description: str):
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{description} is not a directory: {path}")


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script. No GPU detected by PyTorch.")

    torch.cuda.set_device(args.gpu_index)
    device = torch.device(f"cuda:{args.gpu_index}")
    torch.backends.cudnn.benchmark = True
    set_seed(args.seed)

    train_dir = args.train_dir.resolve()
    val_dir = args.val_dir.resolve()
    test_dir = args.test_dir.resolve() if args.test_dir else val_dir
    ensure_dir(train_dir, "Train directory")
    ensure_dir(val_dir, "Validation directory")
    ensure_dir(test_dir, "Test directory")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "training_history.json"
    best_ckpt = output_dir / "emotion_model_best.pth"
    last_ckpt = output_dir / "emotion_model.pth"

    train_tfms, eval_tfms = build_transforms(args.img_size)
    train_dataset = datasets.ImageFolder(str(train_dir), transform=train_tfms)
    val_dataset = datasets.ImageFolder(str(val_dir), transform=eval_tfms)
    test_dataset = datasets.ImageFolder(str(test_dir), transform=eval_tfms)

    if train_dataset.classes != val_dataset.classes or train_dataset.classes != test_dataset.classes:
        raise ValueError("Train/validation/test folders must contain identical class sub-folders.")

    class_names = train_dataset.classes
    print(f"Detected {len(class_names)} classes: {class_names}")
    print(f"Train images: {len(train_dataset)}, Val images: {len(val_dataset)}, Test images: {len(test_dataset)}")

    pin_memory = device.type == "cuda"
    train_loader = make_dataloader(train_dataset, args.batch_size, True, args.num_workers, pin_memory)
    val_loader = make_dataloader(val_dataset, args.batch_size, False, args.num_workers, pin_memory)
    test_loader = make_dataloader(test_dataset, args.batch_size, False, args.num_workers, pin_memory)

    model = build_model(
        args.model,
        num_classes=len(class_names),
        pretrained=not args.no_pretrained,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=args.lr_patience,
        factor=0.5,
        min_lr=args.min_lr,
    )
    grad_clip = args.grad_clip if args.grad_clip and args.grad_clip > 0 else None
    scaler = amp.GradScaler(device=device.type, enabled=args.amp)
    history: list[Dict[str, float]] = []
    best_val_acc = 0.0
    epochs_without_improve = 0

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            amp_enabled=args.amp,
            grad_clip=grad_clip,
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_acc)

        epoch_time = time.time() - start
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "train_acc": round(train_acc, 6),
                "val_loss": round(val_loss, 6),
                "val_acc": round(val_acc, 6),
                "lr": optimizer.param_groups[0]["lr"],
                "epoch_seconds": round(epoch_time, 2),
            }
        )
        torch.save(model.state_dict(), last_ckpt)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improve = 0
            torch.save(model.state_dict(), best_ckpt)
            print(f"[Epoch {epoch:03d}] val_acc improved to {val_acc:.4f}. Checkpoint saved.")
        else:
            epochs_without_improve += 1
            print(f"[Epoch {epoch:03d}] no improvement. Patience {epochs_without_improve}/{args.patience}.")

        print(
            f"Epoch {epoch:03d}/{args.epochs} - "
            f"train_loss: {train_loss:.4f}, train_acc: {train_acc*100:.2f}% - "
            f"val_loss: {val_loss:.4f}, val_acc: {val_acc*100:.2f}% - "
            f"time: {epoch_time:.1f}s"
        )

        if args.patience > 0 and epochs_without_improve >= args.patience:
            print("Early stopping triggered.")
            break

    with history_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "history": history,
                "class_names": class_names,
                "best_val_acc": best_val_acc,
                "epochs_trained": len(history),
            },
            f,
            indent=2,
        )

    best_state = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(best_state)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Best validation accuracy: {best_val_acc*100:.2f}%")
    print(f"Test accuracy (@best): {test_acc*100:.2f}% | loss: {test_loss:.4f}")
    print(f"Artifacts saved in: {output_dir}")


if __name__ == "__main__":
    main()
