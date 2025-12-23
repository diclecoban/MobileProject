#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_onnx.sh [--onnx PATH] [--dataset NAME] [--topk N] [--max-files N] [--recursive] [IMAGE_OR_DIR ...]

Simple wrapper around scripts/run_emotion_onnx.py so you can run an exported ONNX model
without touching the training code. All arguments are optional; by default the script
loads models/exported/facedata_mnv3.onnx and runs it on Data/test using the data_faces
preprocessing profile.

Examples:
  ./run_onnx.sh
  ./run_onnx.sh --onnx models/exported/data_faces_efficientnet_b0.onnx ./Data/test
  ./run_onnx.sh --recursive path/to/images

Options:
  --onnx PATH       Path to the .onnx file (default: models/exported/facedata_mnv3.onnx)
  --dataset NAME    Dataset key from configs/datasets.yaml (default: data_faces)
  --topk N          Number of labels to display per image (default: 3)
  --max-files N     Process at most N files
  --recursive       Recurse into directories when expanding images
  -h, --help        Show this message
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

default_onnx="${ROOT_DIR}/models/exported/facedata_mnv3.onnx"
dataset="data_faces"
topk="3"
max_files=""
recursive_flag=""
declare -a inputs=()

is_absolute() {
  case "$1" in
    /*|[A-Za-z]:*) return 0 ;;
    *) return 1 ;;
  esac
}

to_abs() {
  local path="$1"
  if is_absolute "$path"; then
    printf '%s\n' "$path"
  else
    printf '%s\n' "${ROOT_DIR}/${path}"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --onnx)
      [[ $# -lt 2 ]] && { echo "ERROR: --onnx requires a path" >&2; exit 1; }
      default_onnx="$(to_abs "$2")"
      shift 2
      ;;
    --dataset)
      [[ $# -lt 2 ]] && { echo "ERROR: --dataset requires a name" >&2; exit 1; }
      dataset="$2"
      shift 2
      ;;
    --topk)
      [[ $# -lt 2 ]] && { echo "ERROR: --topk requires a number" >&2; exit 1; }
      topk="$2"
      shift 2
      ;;
    --max-files)
      [[ $# -lt 2 ]] && { echo "ERROR: --max-files requires a number" >&2; exit 1; }
      max_files="$2"
      shift 2
      ;;
    --recursive)
      recursive_flag="--recursive"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        inputs+=("$(to_abs "$1")")
        shift
      done
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
    *)
      inputs+=("$(to_abs "$1")")
      shift
      ;;
  esac
done

if [[ ${#inputs[@]} -eq 0 ]]; then
  inputs+=("${ROOT_DIR}/Data/test")
fi

for target in "${inputs[@]}"; do
  if [[ ! -e "$target" ]]; then
    echo "ERROR: Input path not found: $target" >&2
    exit 1
  fi
done

if [[ ! -f "$default_onnx" ]]; then
  echo "ERROR: ONNX model not found at $default_onnx" >&2
  exit 1
fi

cmd=(
  python3 "${ROOT_DIR}/scripts/run_emotion_onnx.py"
  --onnx "$default_onnx"
  --dataset "$dataset"
  --topk "$topk"
)

[[ -n "$recursive_flag" ]] && cmd+=("$recursive_flag")
[[ -n "$max_files" ]] && cmd+=(--max-files "$max_files")
cmd+=("${inputs[@]}")

echo ">>> Running ${cmd[*]}"
"${cmd[@]}"
