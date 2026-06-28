"""
Shared numpy/cv2 preprocessing — used by BOTH compile-time calibration and the
on-board runtime, so the INT8 calibration statistics match what the model sees at
inference EXACTLY. Any mismatch here silently wrecks quantized accuracy.

Torch-free on purpose (the board has no torch). Mirrors the original torchvision
preprocessing from lane_service/inference_runtime.py and ultralytics' letterbox.

Callers must have bbai64-deploy/ on sys.path so `import config` resolves.
"""
from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np

import config as C


# ─────────────────────────────────────────────────────────────
# YOLO — letterbox to square, RGB, /255, NCHW float32
# ─────────────────────────────────────────────────────────────
def letterbox(bgr: np.ndarray, size: int, pad: int = 114) -> Tuple[np.ndarray, float, int, int]:
    h, w = bgr.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), pad, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, r, left, top


def preprocess_yolo(bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """Returns (input[1,3,640,640] float32, letterbox meta to map boxes back)."""
    canvas, r, left, top = letterbox(bgr, C.YOLO.IMGSZ)
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32)
    rgb *= np.float32(C.YOLO.SCALE[0])              # 1/255
    chw = np.transpose(rgb, (2, 0, 1))[None]        # 1,3,H,W
    meta = {"r": r, "left": float(left), "top": float(top),
            "src_w": float(bgr.shape[1]), "src_h": float(bgr.shape[0])}
    return np.ascontiguousarray(chw, dtype=np.float32), meta


# ─────────────────────────────────────────────────────────────
# UFLDv2 — resize to (1600,533), crop bottom 320, RGB, ImageNet norm, NCHW
# Equivalent to: ToTensor (→[0,1]) then Normalize(mean,std), expressed on 0..255.
# MEAN/SCALE come from config (single source) and are applied here on the A72 —
# NOT folded into the TIDL compile (see config.UFLD note).
# ─────────────────────────────────────────────────────────────
_UFLD_MEAN = np.array(C.UFLD.MEAN, dtype=np.float32)
_UFLD_SCALE = np.array(C.UFLD.SCALE, dtype=np.float32)


def preprocess_ufld(bgr: np.ndarray) -> np.ndarray:
    """Returns input[1,3,320,1600] float32 (ImageNet-normalized)."""
    resized = cv2.resize(bgr, (C.UFLD.RESIZE_W, C.UFLD.RESIZE_H),
                         interpolation=cv2.INTER_LINEAR)
    crop = resized[C.UFLD.RESIZE_H - C.UFLD.TRAIN_H:, :, :]   # bottom TRAIN_H rows
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
    rgb = (rgb - _UFLD_MEAN) * _UFLD_SCALE
    chw = np.transpose(rgb, (2, 0, 1))[None]
    return np.ascontiguousarray(chw, dtype=np.float32)


# ─────────────────────────────────────────────────────────────
# TwinLiteNet — plain resize to (TWIN_W × TWIN_H), RGB, /255, NCHW.
# Lane + drivable-area segmentation replacing UFLDv2 (which resets the board with
# its 196 MB FC subgraph). The attention-free conv-only net (twinlite_noattn.onnx)
# offloads 127/127 nodes to the C7x. No ImageNet normalisation — the model was
# trained on a bare /255 scale, matching the BDD100K TwinLite recipe.
# ─────────────────────────────────────────────────────────────
def preprocess_twinlite(bgr: np.ndarray) -> np.ndarray:
    """Returns input[1,3,TWIN_H,TWIN_W] float32 in [0,1] (RGB, NCHW)."""
    resized = cv2.resize(bgr, (C.TWINLITE.TWIN_W, C.TWINLITE.TWIN_H),
                         interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    chw = np.transpose(rgb, (2, 0, 1))[None]
    return np.ascontiguousarray(chw, dtype=np.float32)


# NOTE: there is no preprocess_depth anymore. Per-object distance is closed-form
# monocular geometry on the A72 (runtime/depth_runtime.py): it consumes YOLO boxes
# + camera geometry, not a preprocessed image tensor, so no depth model is
# exported, compiled, calibrated, or run on the C7x.
