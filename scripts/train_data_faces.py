#!/usr/bin/env python3
"""Convenience launcher that trains the Data/ faces dataset with GPU support."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train.py"


def _append_pythonpath(env: dict):
    src_path = str(PROJECT_ROOT / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src_path if not existing else f"{src_path}{os.pathsep}{existing}"
    return env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Mobilenet on the Data/ face dataset with the configured GPU environment."
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--model", default="mobilenet_v3_small")
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "artifacts" / "data_faces_mobilenet_v3"),
    )
    parser.add_argument("--checkpoint-prefix", default="data_faces_mnv3")
    parser.add_argument(
        "remainder",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded to scripts/train.py (prefix with -- before the extras).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--dataset",
        "data_faces",
        "--model",
        args.model,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--num-workers",
        str(args.num_workers),
        "--output-dir",
        args.output_dir,
        "--checkpoint-prefix",
        args.checkpoint_prefix,
    ]
    if "--" in args.remainder:
        # drop separating token so downstream receives only meaningful options
        extras = [tok for tok in args.remainder if tok != "--"]
    else:
        extras = args.remainder
    cmd.extend(extras)

    env = _append_pythonpath(os.environ.copy())
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
