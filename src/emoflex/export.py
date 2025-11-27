from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch

from .models import create_model


def export_checkpoint_to_onnx(
    checkpoint_path: Path,
    export_path: Path,
    arch: str,
    num_classes: int,
    input_size: Tuple[int, int] = (224, 224),
    opset: int = 13,
) -> Path:
    checkpoint_path = checkpoint_path.resolve()
    export_path = export_path.resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = create_model(arch, num_classes=num_classes, pretrained=False)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    dummy = torch.randn(1, 3, input_size[0], input_size[1])
    export_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        export_path,
        input_names=["input"],
        output_names=["logits"],
        opset_version=opset,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )
    return export_path
