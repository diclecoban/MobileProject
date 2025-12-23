# EmoFlex - Facial Emotion Toolkit

GPU-ready training, evaluation, and deployment scripts for facial emotion recognition.
Everything under `src/emoflex` centralizes dataset metadata, transforms, model heads, evaluation, export,
and real-time helpers so the CLI utilities stay focused on their tasks.

## Requirements

- Python 3.11
- CUDA-capable GPU (PyTorch 2.6.0+cu124 and TorchVision 0.21.0+cu124)
- ONNX Runtime, OpenCV (YuNet lives in `models/face_detection_yunet.onnx`), scikit-learn, PyYAML

Install the dependencies and expose the package path once per shell:

```bash
python -m venv venv
source venv/bin/activate            # use venv\Scripts\activate on Windows
pip install -r requirements.txt

# Linux / macOS
export PYTHONPATH=src:$PYTHONPATH

# PowerShell
$env:PYTHONPATH="src;$env:PYTHONPATH"
```

## Data preparation

Images live under `Data/` in the standard `ImageFolder` layout (`Data/<split>/<class>/*.jpg`).
If you only have raw class folders (no `train/` or `test/` directories yet), run:

```bash
python scripts/split_train_test.py --root Data --test-ratio 0.2
```

The script shuffles each class independently, then moves (or copies with `--copy`) files into
`Data/train/<class>` and `Data/test/<class>`.

## Dataset catalog (`configs/datasets.yaml`)

`emoflex` reads dataset metadata from YAML. The current project ships the `data_faces` entry:

```yaml
datasets:
  data_faces:
    type: imagefolder
    root: ./Data
    splits:
      train: train
      val: test
      test: test
    classes: [Angry, Fear, Happy, Sad, Suprise]
    input_size: [224, 224]
    normalization:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
    force_grayscale: true
    auto_split:
      seed: 1337
      ratios: { train: 0.7, val: 0.2, test: 0.1 }
```

Register more datasets here or tweak the transforms (input size, grayscale toggle, normalization,
YOLO crop settings, auto splits, etc.). Any script that resolves metadata expects the dataset key
(`data_faces` above) as an argument.

## Training (`scripts/train_gpu.py`)

Fine-tune a TorchVision backbone on the prepared folders:

```bash
python scripts/train_gpu.py \
  --model mobilenet_v3_small \
  --epochs 30 \
  --batch-size 96 \
  --output-dir artifacts/gpu_training
```

Highlights:

- Models: `mobilenet_v3_small`, `mobilenet_v3_large`, `resnet18`, `resnet34`, `efficientnet_b0`
- Data parameters: `--train-dir`, `--val-dir`, `--test-dir`, `--img-size`, `--batch-size`
- Optimization: `--epochs`, `--learning-rate`, `--weight-decay`, `--grad-clip`, `--amp/--no-amp`
- Transfer tweaks: `--freeze-backbone`, `--no-pretrained`

Outputs inside `--output-dir`:

- `emotion_model_best.pth` - best validation checkpoint
- `emotion_model.pth` - last epoch weights
- `training_history.json` - per-epoch metrics and the discovered class names (consumed by demos)

## Evaluation (`scripts/eval.py`)

```bash
python scripts/eval.py \
  --dataset data_faces \
  --split test \
  --model mobilenet_v3_small \
  --checkpoint artifacts/gpu_training/emotion_model_best.pth
```

You will get overall accuracy, macro precision/recall/F1, per-class metrics, a confusion matrix,
and the full scikit-learn classification report.

## Export to ONNX (`scripts/export_onnx.py`)

```bash
python scripts/export_onnx.py \
  --dataset data_faces \
  --model mobilenet_v3_small \
  --checkpoint artifacts/gpu_training/emotion_model_best.pth \
  --output models/exported/data_faces_mobilenet_v3_small.onnx
```

The exporter rebuilds the classifier with the correct head, loads the checkpoint, and writes a
dynamic-batch ONNX graph (opset 13 by default).

## Batch ONNX inference (`scripts/run_emotion_onnx.py`)

```bash
python scripts/run_emotion_onnx.py \
  --onnx models/exported/data_faces_mobilenet_v3_small.onnx \
  --dataset data_faces \
  --recursive \
  sample_images/
```

The CLI mirrors the preprocessing stored in the dataset catalog (grayscale flag, resize, normalization)
and reports the top-K labels per image. `--max-files`, `--topk`, and env vars such as `EMOFLEX_ONNX`
or `EMOFLEX_DATASET` are also supported.

## Real-time webcam demo (`emotion_rt.py`)

```bash
python emotion_rt.py \
  --onnx models/exported/facedata_mnv3.onnx \
  --dataset data_faces \
  --history artifacts/gpu_training/training_history.json \
  --detector models/face_detection_yunet.onnx
```

Features:

- YuNet face detection with `models/face_detection_yunet.onnx`
- Emotion smoothing (EMA + sliding window) and HUD overlay with stacked bars
- Labels resolved from `--labels`, the training history, or the dataset catalog
- Camera overrides via CLI flags or `EMOFLEX_CAM_*` environment variables

## Automated pipeline (`run_pipeline.sh`)

Use the helper script to train, evaluate, and export in one run (requires Bash or WSL):

```bash
./run_pipeline.sh data_faces mobilenet_v3_small
```

Explicitly pass the dataset key because the legacy default (`facedata`) no longer exists in the catalog.
Artifacts end up in `artifacts/<dataset>/`, and the ONNX model is written to
`models/exported/<dataset>_<model>.onnx`.

## Project layout

```
configs/                # dataset catalog (YAML)
Data/                   # emotion images in ImageFolder format
models/                 # YuNet detector + exported ONNX classifiers
artifacts/              # training outputs, checkpoints, and history logs
scripts/                # training, eval, export, dataset prep, and ONNX helpers
src/emoflex/            # reusable library (config, data, transforms, trainer, eval, export)
emotion_rt.py           # real-time webcam demo
run_pipeline.sh         # train + eval + export helper
```

## Tips

- Always activate the virtual environment before running scripts so Torch, ONNX, and OpenCV share the same install.
- When adding a new dataset entry, verify the `classes` order; it feeds evaluation reports, ONNX inference, and the webcam overlay.
- `scripts/run_emotion_onnx.py --help`, `scripts/train_gpu.py --help`, and `emotion_rt.py --help` list the rest of the CLI switches and environment overrides.
