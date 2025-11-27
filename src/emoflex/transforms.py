from __future__ import annotations

from typing import Tuple

from torchvision import transforms

from .config import DatasetConfig


def _maybe_grayscale(cfg: DatasetConfig):
    if cfg.force_grayscale:
        return [transforms.Grayscale(num_output_channels=3)]
    return []


def build_train_transform(cfg: DatasetConfig) -> transforms.Compose:
    aug: Tuple[transforms.RandomApply, ...] = (
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15)],
            p=0.7,
        ),
    )
    ops = [
        *_maybe_grayscale(cfg),
        transforms.RandomResizedCrop(cfg.input_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomApply(
            [transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))],
            p=0.4,
        ),
    ]
    if not cfg.force_grayscale:
        ops.extend(aug)
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(cfg.normalization_mean, cfg.normalization_std),
        ]
    )
    return transforms.Compose(ops)


def build_eval_transform(cfg: DatasetConfig) -> transforms.Compose:
    ops = [
        *_maybe_grayscale(cfg),
        transforms.Resize(cfg.input_size),
        transforms.ToTensor(),
        transforms.Normalize(cfg.normalization_mean, cfg.normalization_std),
    ]
    return transforms.Compose(ops)


def build_transforms(cfg: DatasetConfig):
    return build_train_transform(cfg), build_eval_transform(cfg)
