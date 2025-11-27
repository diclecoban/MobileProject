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

`scripts/train.py` is now the configurable entry point.

```bash
# Train on facedata with default settings
python scripts/train.py \
  --dataset facedata \
  --model mobilenet_v3_small \
  --epochs 25 \
  --output-dir artifacts

# Fine-tune FER2013 with ResNet18 head-only training
python scripts/train.py \
  --dataset fer2013 \
  --model resnet18 \
  --freeze-backbone \
  --batch-size 128
```

Key arguments:

- `--dataset`: name from `configs/datasets.yaml`
- `--model`: `mobilenet_v3_small`, `resnet18`, `efficientnet_b0`
- `--freeze-backbone`: lock backbone, train only classifier
- `--output-dir`: directory for checkpoints/history (default `artifacts`)

The trainer saves:

- `emotion_model_best.pth` (highest validation accuracy)
- `emotion_model.pth` (latest epoch)
- `training_history.json` (per-epoch metrics)

If a test split exists in the catalog, the script automatically evaluates the best checkpoint and prints detailed metrics.

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

`emotion_rt.py` now reads labels from the dataset catalog so overlays stay in sync with the trained checkpoint.  
Customize via environment variables:

```bash
export PYTHONPATH=src:$PYTHONPATH
export EMOFLEX_DATASET=facedata
export EMOFLEX_ONNX=models/emotion.onnx
python emotion_rt.py
```

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
scripts/train.py    # trainer CLI
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
