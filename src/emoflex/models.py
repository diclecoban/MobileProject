from __future__ import annotations

from typing import Iterable

import torch.nn as nn
from torchvision import models
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
)


def create_model(
    arch: str,
    num_classes: int,
    pretrained: bool = True,
    head_only: bool = False,
) -> nn.Module:
    """Create a classifier backbone with the requested head."""
    arch = arch.lower()
    if arch == "mobilenet_v3_small":
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
        head = model.classifier
    elif arch == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        head = model.fc
    elif arch == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        head = model.classifier[1]
    else:
        raise ValueError(f"Unsupported architecture '{arch}'.")

    if head_only:
        for param in model.parameters():
            param.requires_grad = False
        for param in head.parameters():
            param.requires_grad = True

    return model


def trainable_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    return (p for p in model.parameters() if p.requires_grad)
