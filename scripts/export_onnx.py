#!/usr/bin/env python3
"""Convert a trained checkpoint to ONNX."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from emoflex.config import load_dataset_catalog, resolve_dataset
from emoflex.export import export_checkpoint_to_onnx


def parse_args():
    parser = argparse.ArgumentParser(description="Export PyTorch checkpoint to ONNX.")
    parser.add_argument("--dataset", default="facedata")
    parser.add_argument("--model", default="mobilenet_v3_small")
    parser.add_argument("--checkpoint", default=None, help="Defaults to artifacts/<dataset>/emotion_model_best.pth.")
    parser.add_argument("--output", default=None, help="Defaults to models/exported/<dataset>_<model>.onnx.")
    parser.add_argument("--opset", type=int, default=13)
    return parser.parse_args()


def resolve_paths(dataset: str, model: str, checkpoint: str | None, output: str | None):
    ckpt = Path(checkpoint) if checkpoint else PROJECT_ROOT / "artifacts" / dataset / "emotion_model_best.pth"
    out_dir = PROJECT_ROOT / "models" / "exported"
    if output:
        out_path = Path(output)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{dataset}_{model}.onnx"
    return ckpt, out_path


def main():
    args = parse_args()
    catalog = load_dataset_catalog()
    dataset_cfg = resolve_dataset(args.dataset, catalog)
    checkpoint, out_path = resolve_paths(dataset_cfg.name, args.model, args.checkpoint, args.output)

    output_path = export_checkpoint_to_onnx(
        checkpoint_path=checkpoint,
        export_path=out_path,
        arch=args.model,
        num_classes=dataset_cfg.num_classes,
        input_size=dataset_cfg.input_size,
        opset=args.opset,
    )
    print(f"Exported to {output_path}")


if __name__ == "__main__":
    main()
