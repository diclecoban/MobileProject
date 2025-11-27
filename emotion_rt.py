#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-time Facial Emotion Recognition (Laptop webcam)
- Face detection: YuNet (OpenCV FaceDetectorYN, ONNX)
- Emotion classifier: ONNX Runtime (.onnx)
- Stabilization: EMA + window mean + hysteresis
- Overlay: clear text with background + stacked bars + big dominant label
"""

import os
import sys
import time
from collections import deque
import numpy as np
import cv2 as cv

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if os.path.isdir(SRC_DIR) and SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

DEFAULT_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

try:
    from emoflex.config import load_dataset_catalog, resolve_dataset

    DATASET_NAME = os.environ.get("EMOFLEX_DATASET", "facedata")
    _catalog = load_dataset_catalog()
    _dataset_cfg = resolve_dataset(DATASET_NAME, _catalog)
    EMO_LABELS = _dataset_cfg.classes
    print(f"[INFO] Loaded labels for dataset '{DATASET_NAME}': {EMO_LABELS}")
except Exception as exc:  # pylint: disable=broad-except
    print(f"[WARN] Falling back to default labels ({exc}). Ensure PYTHONPATH includes 'src'.")
    EMO_LABELS = DEFAULT_LABELS

# -------------------- Paths & Config --------------------
YUNET_PATH   = os.environ.get("EMOFLEX_YUNET", "models/face_detection_yunet.onnx")
EMOTION_ONNX = os.environ.get("EMOFLEX_ONNX", "models/emotion.onnx")
USE_EMOTION  = os.path.exists(EMOTION_ONNX)

CAM_INDEX = 0
CAM_W, CAM_H = 1280, 720

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

# run classifier every Nth frame (stability + speed)
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
    def __init__(self, onnx_path):
        import onnxruntime as ort
        self.ort_sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.iname = self.ort_sess.get_inputs()[0].name
        self.oname = self.ort_sess.get_outputs()[0].name
        self.apply_softmax = True  # set False if your ONNX already outputs probs

    def preprocess(self, bgr_roi):
        img = cv.resize(bgr_roi, (224, 224), interpolation=cv.INTER_AREA)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - IMNET_MEAN) / IMNET_STD
        img = np.transpose(img, (2,0,1))
        return np.expand_dims(img, 0)

    def __call__(self, bgr_roi):
        x = self.preprocess(bgr_roi)
        y = self.ort_sess.run([self.oname], {self.iname: x})[0][0]
        probs = softmax(y) if self.apply_softmax else y
        return normalize_to_percentages(probs)

# -------------------- Main --------------------
def main():
    # Detector
    if not os.path.exists(YUNET_PATH):
        raise FileNotFoundError(f"YuNet model not found at {YUNET_PATH}")
    det = cv.FaceDetectorYN_create(
        YUNET_PATH, "", (320, 320), SCORE_THR, NMS_THR, TOPK,
        backend_id=cv.dnn.DNN_BACKEND_OPENCV,
        target_id=cv.dnn.DNN_TARGET_CPU
    )

    # Emotion model
    emo = EmotionONNX(EMOTION_ONNX) if USE_EMOTION else None
    if emo is None:
        print("[Info] Emotion model not found. Running detection-only demo.")

    cap = cv.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise SystemExit("Camera not available. Check permissions.")
    cap.set(cv.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, CAM_H)

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
