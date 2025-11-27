from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader


@dataclass
class EvaluationResult:
    accuracy: float
    per_class_precision: Sequence[float]
    per_class_recall: Sequence[float]
    per_class_f1: Sequence[float]
    support: Sequence[int]
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_precision: float
    weighted_recall: float
    weighted_f1: float
    confusion: np.ndarray
    report: str


def evaluate_model(model, loader: DataLoader, device: torch.device, class_names: List[str]) -> EvaluationResult:
    model.eval()
    preds: List[int] = []
    labels: List[int] = []

    with torch.no_grad():
        for images, target in loader:
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            outputs = model(images)
            pred = outputs.argmax(dim=1)
            preds.extend(pred.cpu().tolist())
            labels.extend(target.cpu().tolist())

    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, labels=list(range(len(class_names))), average=None, zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )

    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    report = classification_report(labels, preds, target_names=class_names, digits=3, zero_division=0)

    return EvaluationResult(
        accuracy=accuracy,
        per_class_precision=precision,
        per_class_recall=recall,
        per_class_f1=f1,
        support=support,
        macro_precision=macro_p,
        macro_recall=macro_r,
        macro_f1=macro_f1,
        weighted_precision=weighted_p,
        weighted_recall=weighted_r,
        weighted_f1=weighted_f1,
        confusion=cm,
        report=report,
    )
