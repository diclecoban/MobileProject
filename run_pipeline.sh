#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

DATASET="${1:-facedata}"
MODEL="${2:-mobilenet_v3_small}"
ARTIFACT_DIR="${ROOT_DIR}/artifacts/${DATASET}"

echo ">>> Training ${MODEL} on ${DATASET}"
python3 "${ROOT_DIR}/scripts/train.py" \
  --dataset "${DATASET}" \
  --model "${MODEL}" \
  --output-dir "${ARTIFACT_DIR}"

echo ">>> Evaluating best checkpoint on ${DATASET}/${MODEL}"
python3 "${ROOT_DIR}/scripts/eval.py" \
  --dataset "${DATASET}" \
  --checkpoint "${ARTIFACT_DIR}/emotion_model_best.pth" \
  --model "${MODEL}"

echo ">>> Exporting ONNX model"
python3 "${ROOT_DIR}/scripts/export_onnx.py" \
  --dataset "${DATASET}" \
  --model "${MODEL}" \
  --checkpoint "${ARTIFACT_DIR}/emotion_model_best.pth" \
  --output "${ROOT_DIR}/models/exported/${DATASET}_${MODEL}.onnx"

echo "Pipeline finished. Artifacts in: ${ARTIFACT_DIR}"
