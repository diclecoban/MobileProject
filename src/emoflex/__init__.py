"""Core helpers for training and deploying emotion models."""

from .config import DatasetConfig, load_dataset_catalog, resolve_dataset
from .data import build_dataset, build_dataloaders
from .models import create_model
from .trainer import Trainer, TrainingSummary
from .evaluation import evaluate_model, EvaluationResult

__all__ = [
    "DatasetConfig",
    "load_dataset_catalog",
    "resolve_dataset",
    "build_dataset",
    "build_dataloaders",
    "create_model",
    "Trainer",
    "TrainingSummary",
    "evaluate_model",
    "EvaluationResult",
]
