#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-time Facial Emotion Recognition (Laptop webcam)
- Face detection: YuNet (OpenCV FaceDetectorYN, ONNX)
- Emotion classifier: ONNX Runtime (.onnx)
- Stabilization: EMA + window mean + hysteresis
- Overlay: clear text with background + stacked bars + big dominant label
"""

import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2 as cv
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.is_dir() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DEFAULT_LABELS = ["Angry", "Fear", "Happy", "Sad", "Surprise"]
EMO_LABELS: List[str] = DEFAULT_LABELS.copy()
DEFAULT_HISTORY_PATH = PROJECT_ROOT / "artifacts" / "gpu_training" / "training_history.json"
DEFAULT_EXPORTED_DIR = PROJECT_ROOT / "models" / "exported"
DEFAULT_YUNET = PROJECT_ROOT / "models" / "face_detection_yunet.onnx"

try:
    from emoflex.config import load_dataset_catalog, resolve_dataset
except Exception as exc:  # pragma: no cover - optional dependency
    LOAD_CONFIG_ERROR = exc
    load_dataset_catalog = None  # type: ignore
    resolve_dataset = None  # type: ignore
else:
    LOAD_CONFIG_ERROR = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time facial emotion recognition using YuNet + ONNX Runtime.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--onnx",
        default=os.environ.get("EMOFLEX_ONNX"),
        help="Path to the exported emotion classifier ONNX file. "
        "If omitted, defaults to models/exported/<dataset>_<model>.onnx.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("EMOFLEX_MODEL", "mobilenet_v3_small"),
        help="Model name (used only to infer the default ONNX path).",
    )
    parser.add_argument(
        "--dataset",
        default=os.environ.get("EMOFLEX_DATASET", "data_faces"),
        help="Dataset key from configs/datasets.yaml.",
    )
    parser.add_argument(
        "--history",
        default=os.environ.get("EMOFLEX_HISTORY", str(DEFAULT_HISTORY_PATH)),
        help="Path to training_history.json to recover class names (optional).",
    )
    parser.add_argument(
        "--labels",
        help="Comma-separated list of labels to override dataset/history labels.",
    )
    parser.add_argument(
        "--detector",
        default=os.environ.get("EMOFLEX_YUNET", str(DEFAULT_YUNET)),
        help="Path to YuNet ONNX face detector.",
    )
    parser.add_argument("--camera", type=int, default=int(os.environ.get("EMOFLEX_CAM_INDEX", "0")))
    parser.add_argument("--width", type=int, default=int(os.environ.get("EMOFLEX_CAM_WIDTH", "1280")))
    parser.add_argument("--height", type=int, default=int(os.environ.get("EMOFLEX_CAM_HEIGHT", "720")))
    parser.add_argument(
        "--classify-every",
        type=int,
        default=2,
        help="Run the emotion classifier every N frames (higher = faster, more latency).",
    )
    parser.add_argument(
        "--min-face",
        type=int,
        default=100,
        help="Ignore detections smaller than this many pixels.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=None,
        help="Override square input size for the classifier (defaults to dataset/input metadata).",
    )
    parser.add_argument(
        "--force-grayscale",
        dest="force_grayscale",
        action="store_true",
        help="Force grayscale preprocessing regardless of dataset config.",
    )
    parser.add_argument(
        "--color",
        dest="force_grayscale",
        action="store_false",
        help="Disable grayscale even if the dataset config requests it.",
    )
    parser.add_argument(
        "--det-only",
        action="store_true",
        help="Skip loading the emotion classifier (face detection only).",
    )
    parser.set_defaults(force_grayscale=None)
    return parser.parse_args()


def try_load_dataset_config(dataset_name: str | None):
    if not dataset_name or load_dataset_catalog is None or resolve_dataset is None:
        if LOAD_CONFIG_ERROR:
            print(
                f"[WARN] Dataset catalog unavailable ({LOAD_CONFIG_ERROR}). "
                "Ensure PYTHONPATH includes 'src'.",
            )
        return None
    try:
        catalog = load_dataset_catalog()
        return resolve_dataset(dataset_name, catalog)
    except Exception as exc:  # pragma: no cover - runtime path
        print(f"[WARN] Failed to load dataset config '{dataset_name}': {exc}")
        return None


def parse_label_override(raw: str | None) -> List[str] | None:
    if not raw:
        return None
    labels = [chunk.strip() for chunk in raw.split(",") if chunk.strip()]
    return labels or None


def labels_from_history(history_path: Path) -> List[str] | None:
    if not history_path.exists():
        return None
    try:
        data = json.loads(history_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - simple IO
        print(f"[WARN] Failed to read history file '{history_path}': {exc}")
        return None
    labels = data.get("class_names")
    if isinstance(labels, list) and labels:
        return [str(lbl) for lbl in labels]
    return None


def resolve_labels(args: argparse.Namespace, dataset_cfg) -> List[str]:
    override = parse_label_override(args.labels)
    if override:
        print(f"[INFO] Using labels from --labels override: {override}")
        return override

    history_labels = labels_from_history(Path(args.history))
    if history_labels:
        print(f"[INFO] Loaded labels from history file: {history_labels}")
        return history_labels

    if dataset_cfg and dataset_cfg.classes:
        print(f"[INFO] Loaded labels from dataset '{dataset_cfg.name}'.")
        return list(dataset_cfg.classes)

    print("[WARN] Falling back to default label list.")
    return DEFAULT_LABELS.copy()


def resolve_preprocess_config(args: argparse.Namespace, dataset_cfg):
    if args.input_size:
        size = (args.input_size, args.input_size)
    elif dataset_cfg:
        size = tuple(int(v) for v in dataset_cfg.input_size)
    else:
        size = (224, 224)

    if dataset_cfg:
        mean = np.array(dataset_cfg.normalization_mean, dtype=np.float32)
        std = np.array(dataset_cfg.normalization_std, dtype=np.float32)
    else:
        mean = IMNET_MEAN.copy()
        std = IMNET_STD.copy()

    if args.force_grayscale is not None:
        force_gray = args.force_grayscale
    elif dataset_cfg:
        force_gray = dataset_cfg.force_grayscale
    else:
        force_gray = False

    return size, mean, std, force_gray


def resolve_onnx_path(args: argparse.Namespace) -> Path:
    if args.onnx:
        return Path(args.onnx)
    candidate = DEFAULT_EXPORTED_DIR / f"{args.dataset}_{args.model}.onnx"
    return candidate

SCORE_THR = 0.6
NMS_THR   = 0.3
TOPK      = 5000

EMO_COLOR = {
    "happy":    (0, 255,   0),     # Yeşil – pozitif, mutluluk ve canlılık hissi
    "sad":      (255, 0,   0),     # Mavi tonlu kırmızı – hüzün, düşük enerji
    "angry":    (0,   0, 255),     # Kırmızı – öfke, yoğun duygu
    "fear":     (255, 255, 0),     # Sarı – dikkat, tedirginlik
    "surprise": (0, 255, 255),     # Camgöbeği – şaşkınlık, ani değişim
    "neutral":  (160,160,160),     # Gri – tarafsız, sakin, duygu yokluğu
    "disgust":  (0, 128,   0),     # Koyu yeşil – tiksinme, olumsuz tepki
    "contempt": (128, 0, 128),
    "natural":  (128, 128, 255),
    "sleepy":   (255, 165, 0),
}

IMNET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMNET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# run classifier every Nth frame (values overwritten via CLI)
CLASSIFY_EVERY = 2
MIN_FACE = 100   # px

# -------------------- Utils --------------------
def softmax(x: np.ndarray, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-8)

def normalize_to_percentages(p: np.ndarray):
    p = p / (p.sum() + 1e-8)
    return p * 100.0

def draw_text_with_bg(img, text, org, font_scale=0.8, fg=(255,255,255), bg=(0,0,0), thickness=2):
    """readable text: draws filled bg rectangle behind the text"""
    (tw, th), baseline = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x, y = org
    
    overlay = img.copy()
    cv.rectangle(overlay, (x-4, y-th-6), (x+tw+4, y+baseline+4), bg, -1)
    cv.addWeighted(overlay, 0.5, img, 0.5, 0, img)
    cv.putText(img, text, (x, y), cv.FONT_HERSHEY_SIMPLEX, font_scale, fg, thickness, cv.LINE_AA)

def draw_bar(img, x, y, width, label, percent, color=(0,255,255), bar_h=20):
    p = max(0.0, min(100.0, percent)) / 100.0
    filled = int(width * p)
    
    cv.rectangle(img, (x, y), (x+width, y+bar_h), (50,50,50), -1)
    
    cv.rectangle(img, (x, y), (x+filled, y+bar_h), color, -1)
    
    text = f"{label}: {percent:.0f}%"
    (tw, th), base = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    tx = x + 6
    ty = y + bar_h - 5
    cv.putText(img, text, (tx, ty), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv.LINE_AA)

# -------------------- Stabilizer --------------------
class EmotionStabilizer:
    """
    EMA + rolling mean + hysteresis (debounce) for label switching.
    perc is expected in 0..100 (percentages).
    """
    def __init__(self, num_classes, ema_alpha=0.2, win=8, debounce_steps=5, switch_margin=8.0):
        self.alpha = ema_alpha
        self.buf = deque(maxlen=win)
        self.ema = None
        self.cur_label = None
        self.strikes = 0
        self.debounce_steps = debounce_steps
        self.switch_margin = switch_margin
        self.C = num_classes
        self.last_percent = np.zeros(num_classes, dtype=np.float32)

    def update(self, perc: np.ndarray):
        perc = perc.astype(np.float32)
        if self.ema is None:
            self.ema = perc.copy()
        else:
            self.ema = self.alpha * perc + (1 - self.alpha) * self.ema

        self.buf.append(self.ema.copy())
        smoothed = np.mean(self.buf, axis=0)

        top_new = int(np.argmax(smoothed))
        if self.cur_label is None:
            self.cur_label = top_new
            self.strikes = 0
        else:
            if smoothed[top_new] > smoothed[self.cur_label] + self.switch_margin:
                self.strikes += 1
                if self.strikes >= self.debounce_steps:
                    self.cur_label = top_new
                    self.strikes = 0
            else:
                self.strikes = 0

        self.last_percent = smoothed
        return smoothed, self.cur_label

# -------------------- Emotion model --------------------
class EmotionONNX:
    def __init__(
        self,
        onnx_path: Path,
        input_size: Tuple[int, int],
        mean: Sequence[float],
        std: Sequence[float],
        force_grayscale: bool,
    ):
        import onnxruntime as ort

        self.ort_sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.iname = self.ort_sess.get_inputs()[0].name
        self.oname = self.ort_sess.get_outputs()[0].name
        self.apply_softmax = True  # set False if your ONNX already outputs probs
        self.size = tuple(int(v) for v in input_size)
        self.mean = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.asarray(std, dtype=np.float32).reshape(3, 1, 1)
        self.force_grayscale = bool(force_grayscale)

    def preprocess(self, bgr_roi: np.ndarray):
        img = cv.resize(bgr_roi, self.size, interpolation=cv.INTER_AREA)
        if self.force_grayscale:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
        else:
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = (img - self.mean) / self.std
        return np.expand_dims(img, 0)

    def __call__(self, bgr_roi):
        x = self.preprocess(bgr_roi)
        y = self.ort_sess.run([self.oname], {self.iname: x})[0][0]
        probs = softmax(y) if self.apply_softmax else y
        return normalize_to_percentages(probs)

# -------------------- Main --------------------
def main():
    args = parse_args()
    dataset_cfg = try_load_dataset_config(args.dataset)
    size, mean, std, force_gray = resolve_preprocess_config(args, dataset_cfg)
    labels = resolve_labels(args, dataset_cfg)

    global EMO_LABELS, CLASSIFY_EVERY, MIN_FACE
    EMO_LABELS = labels
    CLASSIFY_EVERY = max(1, args.classify_every)
    MIN_FACE = max(1, args.min_face)

    yunet_path = Path(args.detector)
    if not yunet_path.exists():
        raise FileNotFoundError(f"YuNet model not found at {yunet_path}")
    det = cv.FaceDetectorYN_create(
        str(yunet_path),
        "",
        (320, 320),
        SCORE_THR,
        NMS_THR,
        TOPK,
        backend_id=cv.dnn.DNN_BACKEND_OPENCV,
        target_id=cv.dnn.DNN_TARGET_CPU,
    )

    emo = None
    if args.det_only:
        print("[INFO] Detection-only mode enabled (skipping emotion classifier).")
    else:
        onnx_path = resolve_onnx_path(args)
        if not onnx_path.exists():
            raise FileNotFoundError(f"Emotion ONNX not found at {onnx_path}")
        emo = EmotionONNX(onnx_path, size, mean, std, force_gray)
        print(f"[INFO] Loaded emotion model from {onnx_path}")

    cap = cv.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Camera not available. Check permissions.")
    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)

    # per-face stabilizer (idx used as pseudo id)
    stab = {}

    t0, frames = time.time(), 0
    frame_id = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        det.setInputSize((w, h))

        _, faces = det.detect(frame)

        if faces is not None:
            for idx, f in enumerate(faces):
                x, y, ww, hh = map(int, f[:4])
                conf = float(f[-1])

                # bbox & padding to stabilize ROI
                x0, y0 = max(0, x), max(0, y)
                x1, y1 = min(x + ww, w - 1), min(y + hh, h - 1)
                if x1 <= x0 or y1 <= y0:
                    continue

                # skip tiny faces
                if ww < MIN_FACE or hh < MIN_FACE:
                    continue

                # pad ROI by 10% to absorb small jitter
                pad = int(0.1 * max(ww, hh))
                xx0, yy0 = max(0, x0 - pad), max(0, y0 - pad)
                xx1, yy1 = min(w - 1, x1 + pad), min(h - 1, y1 + pad)
                roi = frame[yy0:yy1, xx0:xx1]
                if roi.size == 0:
                    continue

                # draw bbox
                cv.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
                draw_text_with_bg(frame, f"{conf:.2f}", (x0, y0 - 8), font_scale=0.6, fg=(255,255,255), bg=(0,128,0))

                # initialize stabilizer
                if idx not in stab:
                    stab[idx] = EmotionStabilizer(len(EMO_LABELS), ema_alpha=0.2, win=8, debounce_steps=5, switch_margin=8.0)

                # run emotion every Nth frame; otherwise reuse last smoothed
                if emo is not None and (frame_id % CLASSIFY_EVERY == 0):
                    try:
                        perc = emo(roi)  # np.array in %
                        smoothed, top_idx = stab[idx].update(perc)
                    except Exception as e:
                        draw_text_with_bg(frame, f"EMO ERR: {str(e)[:18]}", (x0, y0 - 30), 0.6, (255,255,255), (0,0,255))
                        smoothed, top_idx = stab[idx].last_percent, stab[idx].cur_label
                else:
                    smoothed, top_idx = stab[idx].last_percent, stab[idx].cur_label

                if emo is not None and smoothed is not None and len(smoothed) == len(EMO_LABELS):
                    
                    if top_idx is None:
                        top_idx = int(np.argmax(smoothed))
                    top_idx = max(0, min(top_idx, len(EMO_LABELS) - 1))
                    dom_lab = EMO_LABELS[top_idx]
                    dom_val = float(smoothed[top_idx])
                    dom_col = EMO_COLOR.get(dom_lab, (0,255,255))

                    BAR_W   = max(180, int(ww * 0.65))   
                    BAR_H   = 22                         
                    GAP     = 10                         
                    TOP_PAD = 12                         

                    order = np.argsort(-smoothed)
                    kmax = min(3, len(order))

                    total_bars_h = kmax*BAR_H + (kmax-1)*GAP
                    start_y_above = y0 - TOP_PAD - total_bars_h
                    place_above = start_y_above >= 30
                    if place_above:
                        start_y = start_y_above
                        dom_y   = start_y - 28     
                    else:
                        start_y = y1 + TOP_PAD
                        dom_y   = start_y + total_bars_h + 28

                    cv.putText(frame, f"{dom_lab.upper()} {dom_val:.0f}%",
                    (x0, max(26, dom_y)),
                    cv.FONT_HERSHEY_SIMPLEX, 1.0, dom_col, 3, cv.LINE_AA)

                    for k in range(kmax):
                        lab = EMO_LABELS[order[k]]
                        val = float(smoothed[order[k]])
                        color = EMO_COLOR.get(lab, (0,255,255))
                        yy = start_y + k*(BAR_H + GAP)

                        yy = min(max(yy, 26), h - BAR_H - 10)
                        draw_bar(frame, x0, yy, BAR_W, lab, val, color=color, bar_h=BAR_H)

        # FPS
        frames += 1
        frame_id += 1
        if time.time() - t0 >= 1.0:
            fps = frames / (time.time() - t0)
            frames, t0 = 0, time.time()
            draw_text_with_bg(frame, f"FPS: {fps:.1f}", (10, 30), 0.9, (0,0,0), (0,255,255))

        cv.imshow("Real-time Emotion (YuNet + Stabilized)", frame)
        if cv.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
