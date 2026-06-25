"""
bbai64-deploy — central configuration (single source of truth).

Imported by every stage (export / compile / runtime) so shapes, paths, and
decode parameters cannot drift apart between the PC-side compile and the
on-board runtime. Deliberately torch-free and ultralytics-free: this module is
safe to import on the BeagleBone AI-64, which has neither.

Version pinning rule (see README): the edgeai-tidl-tools used to produce the
TIDL artifacts MUST match the TIDL runtime/firmware on the board. TENSOR_BITS
and the artifact folders below are the on-disk contract between the two.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# ─────────────────────────────────────────────────────────────
# Repo layout
# ─────────────────────────────────────────────────────────────
DEPLOY_ROOT = Path(__file__).resolve().parent            # .../bbai64-deploy
GP_ROOT = DEPLOY_ROOT.parent                             # "F:\Graduation Project"
UFLD_ROOT = GP_ROOT / "Ultra-Fast-Lane-Detection-v2"
OD_ROOT = GP_ROOT / "Object-detection"

# Produced by export/compile, consumed by runtime. Copy this whole tree to the
# board after stage 2.
ARTIFACTS = Path(os.environ.get("BBAI64_ARTIFACTS", DEPLOY_ROOT / "artifacts"))
CALIB_DIR = DEPLOY_ROOT / "calib"

# ─────────────────────────────────────────────────────────────
# Quantization / TIDL
# ─────────────────────────────────────────────────────────────
# 8 = INT8 (fastest, default). Bump to 16 globally, or use per-layer mixed
# precision in the compile scripts, if INT8 accuracy is insufficient.
TENSOR_BITS = int(os.environ.get("BBAI64_TENSOR_BITS", "8"))
# Set on the PC before running stage 2, e.g.
#   export TIDL_TOOLS_PATH=/opt/edgeai-tidl-tools/tidl_tools
TIDL_TOOLS_PATH = os.environ.get("TIDL_TOOLS_PATH", "")
ONNX_OPSET = 11          # TIDL-safe opset; raise only if a needed op requires it

# ─────────────────────────────────────────────────────────────
# Class names — single source of truth = the model's data.yaml
# ─────────────────────────────────────────────────────────────
def load_class_names(yaml_path: Path) -> list[str]:
    """Read YOLO class names from a data.yaml.

    Parsed ONCE at import; the runtime then labels a detection by plain list
    indexing (names[class_id]) — O(1), so this has zero per-frame / FPS cost.
    Swapping models = drop the new best.pt + its data.yaml and repoint
    YOLO.DATA_YAML; names follow automatically and can never drift.

    PyYAML is used when available; falls back to a dependency-free parse of the
    inline `names: [...]` form (what this project's data.yaml uses) so the board
    needs no extra package. Returns [] on failure (runtime then labels by id).
    """
    try:
        text = Path(yaml_path).read_text(encoding="utf-8")
    except OSError as e:  # noqa: BLE001
        print(f"[config] class names: data.yaml unreadable ({e}); labelling by id")
        return []
    try:
        import yaml
        names = (yaml.safe_load(text) or {}).get("names")
        if isinstance(names, dict):              # {0: 'vehicle', 1: 'bike', ...}
            names = [names[k] for k in sorted(names)]
        if names:
            return [str(n) for n in names]
    except Exception:  # noqa: BLE001 — no PyYAML on the board, or odd YAML
        pass
    import ast
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("names:"):
            try:
                return [str(n) for n in ast.literal_eval(s.split("names:", 1)[1].strip())]
            except (ValueError, SyntaxError):
                break
    print(f"[config] class names: could not parse {yaml_path}; labelling by id")
    return []


# ─────────────────────────────────────────────────────────────
# YOLO (object detection) — the fine-tuned 8-class CARLA model
# ─────────────────────────────────────────────────────────────
class YOLO:
    NAME = "yolo26n_carla8"                      # logical id → artifact filenames
    MODEL_DIR = GP_ROOT / "Fined-Tuned-Model"    # swap point: drop best.pt + data.yaml here
    PT = MODEL_DIR / "best.pt"                   # source weights (PC only)
    DATA_YAML = MODEL_DIR / "data.yaml"          # class names live here (single source)
    ONNX = ARTIFACTS / f"{NAME}.onnx"
    TIDL_DIR = ARTIFACTS / "yolo_tidl"           # TIDL artifacts folder
    # Square letterbox input. The fine-tuned model was TRAINED @1280 (that is what
    # made small speed-sign detection work — docs headline), but 1280 is ~4x the
    # C7x cost of 640 on this 8-TOPS edge part, so the board default stays 640.
    # It is a USER CHOICE: set BBAI64_YOLO_IMGSZ (e.g. 960 or 1280) BEFORE export +
    # compile (it sizes the TIDL graph) and keep the same value at runtime — it is
    # read here once so the two stages cannot disagree. Must be a multiple of 32.
    IMGSZ = int(os.environ.get("BBAI64_YOLO_IMGSZ", "640"))
    IN_SHAPE = (1, 3, IMGSZ, IMGSZ)
    IN_NAME = "images"

    # Names read from DATA_YAML at import (no torch / ultralytics needed).
    NAMES = load_class_names(DATA_YAML)          # e.g. ['vehicle','bike',...]
    NC = len(NAMES)

    # Operating point. Per-class confidence (docs/04): pedestrian kept low for
    # recall/safety, the rest at the F1-optimal global. Edit PER_CLASS_CONF /
    # CONF_DEFAULT here, or override at runtime via runtime/config.yaml.
    CONF_DEFAULT = 0.40
    PER_CLASS_CONF = {"pedestrian": 0.15}        # name → keep-threshold
    IOU_THRES = 0.45

    # Normalization is applied in numpy preprocess.py on the A72 as
    # (rgb - MEAN) * SCALE — used IDENTICALLY at calibration and runtime. TIDL is
    # NOT given mean/scale (no folding), so do not add them to the compile config
    # or you double-normalize. For YOLO: just /255.
    MEAN = [0.0, 0.0, 0.0]
    SCALE = [1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0]

    @classmethod
    def conf_by_id(cls, names: list[str] | None = None,
                   per_class: dict | None = None,
                   default: float | None = None) -> np.ndarray:
        """Per-class-id keep-threshold array, resolved by class name. Built once
        by the runtime; indexing it with class_ids is then vectorized + cheap."""
        names = cls.NAMES if names is None else names
        per_class = cls.PER_CLASS_CONF if per_class is None else per_class
        default = cls.CONF_DEFAULT if default is None else default
        return np.array([per_class.get(n, default) for n in names], dtype=np.float32)


# ─────────────────────────────────────────────────────────────
# UFLDv2 (lane detection) — CULane / ResNet-18
# Constants mirror configs/culane_res18.py so the board does not need the
# UFLD repo or its Config loader at runtime.
# ─────────────────────────────────────────────────────────────
class UFLD:
    CONFIG = UFLD_ROOT / "configs" / "culane_res18.py"   # PC export only
    WEIGHTS = UFLD_ROOT / "culane_res18.pth"             # PC export only
    ONNX = ARTIFACTS / "ufld_culane_res18.onnx"
    TIDL_DIR = ARTIFACTS / "ufld_tidl"

    DATASET = "CULane"
    TRAIN_H = 320
    TRAIN_W = 1600
    CROP_RATIO = 0.6
    # PIL Resize target before the bottom-crop to TRAIN_H (see preprocessing).
    RESIZE_H = int(TRAIN_H / CROP_RATIO)                 # 533
    RESIZE_W = TRAIN_W                                   # 1600
    IN_SHAPE = (1, 3, TRAIN_H, TRAIN_W)
    IN_NAME = "input"
    OUT_NAMES = ["loc_row", "loc_col", "exist_row", "exist_col"]

    # CULane visualization space (lane coords are decoded into this frame).
    VIS_W = 1640
    VIS_H = 590

    # Head dimensions (config: num_row/num_col/num_lanes/num_cell_*).
    NUM_ROW = 72          # row anchors (cls_row)
    NUM_COL = 81          # col anchors (cls_col)
    NUM_LANES = 4
    NUM_CELL_ROW = 200    # grid_row
    NUM_CELL_COL = 100    # grid_col
    LOCAL_WIDTH = 1

    # Anchors — identical to ufld_inference.build_cfg() for CULane.
    ROW_ANCHOR = np.linspace(0.42, 1.0, NUM_ROW)
    COL_ANCHOR = np.linspace(0.0, 1.0, NUM_COL)

    # ImageNet normalization, applied in numpy preprocess.py on the A72 as
    # (rgb - MEAN) * SCALE (same at calibration and runtime). NOT folded into the
    # TIDL compile config — see the YOLO note above.
    _IMAGENET_MEAN = [0.485, 0.456, 0.406]
    _IMAGENET_STD = [0.229, 0.224, 0.225]
    MEAN = [m * 255.0 for m in _IMAGENET_MEAN]
    SCALE = [1.0 / (s * 255.0) for s in _IMAGENET_STD]


# ─────────────────────────────────────────────────────────────
# Depth-Anything-V2 (monocular METRIC depth) — per-object distance in metres
# Ported from Integrate-Features/integrate.py. The Metric-Outdoor-Small head is
# VKITTI-trained and returns absolute metres (0..~80 m) — the right scale for
# CARLA driving scenes. On the board it is the THIRD model time-sharing the one
# C7x, so combined latency = yolo + ufld + depth (serial; see README).
#
# ⚠️ Depth-Anything-V2 is a ViT/DINOv2 transformer + DPT head. TIDL transformer
# support is version-dependent and ViTs are quantization-sensitive — this is the
# riskiest model to offload. The compile script will report which layers land on
# the A72; if INT8 accuracy is poor, bump DEPTH.TENSOR_BITS to 16 (per-model, set
# below) or BBAI64_DEPTH_BITS=16. The runtime CPUExecutionProvider catches any
# subgraph TIDL won't take, so it always runs — just possibly slower.
# ─────────────────────────────────────────────────────────────
class DEPTH:
    # On by default (mirrors integrate.py); disable with app.py --no-depth or
    # runtime/config.yaml `depth: false`. Off => YOLO+UFLD only, leaner pipeline.
    ENABLED = os.environ.get("BBAI64_DEPTH", "1") not in ("0", "false", "False")

    # HF checkpoint — used by the PC export step ONLY (torch/transformers there);
    # the board consumes the compiled ONNX/TIDL artifacts, never this id.
    HF_MODEL_ID = os.environ.get(
        "BBAI64_DEPTH_MODEL",
        "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf")
    ONNX = ARTIFACTS / "depth_anything_v2_metric_s.onnx"
    TIDL_DIR = ARTIFACTS / "depth_tidl"

    # DINOv2 ViT-S/14 wants a side that is a multiple of 14; 518 = 14*37 is the
    # checkpoint's native working size. Fixed (static) for a clean TIDL graph.
    INPUT_SIZE = int(os.environ.get("BBAI64_DEPTH_SIZE", "518"))
    IN_SHAPE = (1, 3, INPUT_SIZE, INPUT_SIZE)
    IN_NAME = "pixel_values"          # transformers' depth-estimation input name
    OUT_NAME = "predicted_depth"

    MAX_M = 80.0                      # outdoor head range; clamps label sanity

    # Per-object depth sampling: median of the box's inner SHRINK fraction, so
    # edge/background pixels and depth noise don't skew the reading (integrate.py).
    SAMPLE_SHRINK = 0.5

    # Higher-opset transformer ops (LayerNorm etc.) — keep at the TIDL-safe global
    # default; raise only if torch.onnx.export complains a needed op needs it.
    ONNX_OPSET = ONNX_OPSET

    # ViTs are quantization-sensitive: allow INT16 for depth ALONE without
    # bumping the whole project. Defaults to the global TENSOR_BITS (INT8).
    TENSOR_BITS = int(os.environ.get("BBAI64_DEPTH_BITS", str(TENSOR_BITS)))

    # ImageNet normalization (HF processor does rescale 1/255 then normalize),
    # applied in numpy preprocess.py on the A72 as (rgb - MEAN) * SCALE — same at
    # calibration and runtime. NOT folded into TIDL (see the YOLO/UFLD note).
    _IMAGENET_MEAN = [0.485, 0.456, 0.406]
    _IMAGENET_STD = [0.229, 0.224, 0.225]
    MEAN = [m * 255.0 for m in _IMAGENET_MEAN]
    SCALE = [1.0 / (s * 255.0) for s in _IMAGENET_STD]


# ─────────────────────────────────────────────────────────────
# Runtime I/O (board) — MQTT frames in, ADAS alerts out
# ─────────────────────────────────────────────────────────────
class MQTT:
    BROKER = os.environ.get("BBAI64_MQTT_BROKER", "127.0.0.1")
    PORT = int(os.environ.get("BBAI64_MQTT_PORT", "1883"))
    TOPIC_FRAMES = os.environ.get("BBAI64_TOPIC_FRAMES", "carla/camera/front")
    TOPIC_ADAS = os.environ.get("BBAI64_TOPIC_ADAS", "adas/alerts")
    QOS = 0


# Optional live preview window for the annotated frame on the board.
#   "imshow"  — cv2 window (dev/debug)
#   "none"    — headless
# This only controls the on-screen preview. Saved artifacts are driven by the
# input mode: --source image → annotated image + JSON; --source video → annotated
# video + JSON; --source mqtt → JSON published to Qt over MQTT (+ runtime KPIs).
DISPLAY_SINK = os.environ.get("BBAI64_DISPLAY", "imshow")
