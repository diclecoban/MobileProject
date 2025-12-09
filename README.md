# Emotion Project

Flexible training and deployment toolkit for facial emotion recognition.  
The new structure keeps datasets, models, and scripts decoupled so you can switch between FER2013 and the new YOLO-style `facedata` set without touching code.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src:$PYTHONPATH  # needed so scripts can import emoflex
```

## Dataset catalog

`configs/datasets.yaml` centralizes every dataset definition:

- `fer2013`: classic ImageFolder layout (7 classes, grayscale).
- `facedata`: YOLO annotations (9 classes, crops are extracted on-the-fly).

Update this file to register new datasets or tweak transforms (input size, normalization, bbox expansion, etc.).

## Training

Use `scripts/train_gpu.py` to fine-tune a classifier directly from the `Data/train` and `Data/test` folders.  
Each class must live in its own sub-folder (PyTorch `ImageFolder` layout). A CUDA-capable GPU is required.

```bash
# Default MobileNetV3-Small run
python scripts/train_gpu.py \
  --epochs 30 \
  --batch-size 96 \
  --output-dir artifacts/default_run

# ResNet34 with a larger input size and frozen backbone
python scripts/train_gpu.py \
  --model resnet34 \
  --img-size 256 \
  --freeze-backbone \
  --output-dir artifacts/resnet34_freeze
```

Key arguments:

- `--train-dir` / `--val-dir` / `--test-dir`: point to the prepared folders (defaults to `Data/train` and `Data/test`).
- `--model`: `mobilenet_v3_small`, `mobilenet_v3_large`, `resnet18`, `resnet34`, `efficientnet_b0`.
- `--epochs`, `--batch-size`, `--learning-rate`, `--weight-decay`.
- `--no-pretrained` or `--freeze-backbone` for transfer-learning tweaks.
- `--amp/--no-amp`, `--grad-clip`, `--patience` for GPU/memory control.

Outputs saved under `--output-dir`:

- `emotion_model_best.pth` – checkpoint with the best validation accuracy.
- `emotion_model.pth` – latest epoch weights.
- `training_history.json` – per-epoch metrics + class list.

After training, the script automatically reloads the best checkpoint and reports accuracy on the provided test folder.

## Evaluation

```bash
python scripts/eval.py \
  --dataset facedata \
  --split test \
  --checkpoint artifacts/emotion_model_best.pth
```

Outputs overall/macro metrics, per-class scores, confusion matrix, and a full classification report.

## ONNX export

```bash
python scripts/export_onnx.py \
  --dataset facedata \
  --checkpoint artifacts/emotion_model_best.pth \
  --output models/emotion.onnx
```

## Real-time demo

`emotion_rt.py` mirrors the preprocessing from the dataset catalog (or `training_history.json`) so the webcam overlay stays consistent with the trained checkpoint. Run it by pointing to your exported ONNX file:

```bash
export PYTHONPATH=src:$PYTHONPATH
python emotion_rt.py \
  --onnx models/exported/data_faces_mobilenet_v3_small.onnx \
  --dataset data_faces \
  --history artifacts/gpu_training/training_history.json
```

Environment variables such as `EMOFLEX_ONNX`, `EMOFLEX_DATASET`, `EMOFLEX_HISTORY`, `EMOFLEX_YUNET`, or camera overrides still work, but every option is also exposed via CLI flags (`python emotion_rt.py --help`).

## Automated pipeline

To train → evaluate → export in one go, use the helper script:

```bash
./run_pipeline.sh facedata mobilenet_v3_small
```

Arguments default to `facedata` and `mobilenet_v3_small`.

## Project layout

```
configs/            # dataset catalog
src/emoflex/        # reusable library (config, data, models, trainer, eval, export)
scripts/train_gpu.py # GPU training CLI (ImageFolder)
scripts/eval.py     # evaluator
scripts/export_onnx.py  # exporter
emotion_rt.py               # realtime demo (YuNet + ONNX classifier)
artifacts/<dataset>/        # checkpoints + history
models/exported/            # exported ONNX models
run_pipeline.sh             # shells together train+eval+export
```

Extendability tips:

1. Register new datasets in `configs/datasets.yaml`.
2. Add transforms/model options inside `src/emoflex`.
3. Reuse the same training/eval/export scripts without touching the data pipeline.
