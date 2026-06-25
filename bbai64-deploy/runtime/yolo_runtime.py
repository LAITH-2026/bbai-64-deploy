"""
YOLO object-detection runtime (board) — onnxruntime + TIDLExecutionProvider.

Split into preprocess / infer_raw / decode so the pipeline (app.py) can run the
C7x inference of one frame while the A72 does NMS for another. NMS is pure numpy
(TIDL doesn't accelerate it and it was deliberately left out of the graph).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # bbai64-deploy/
import config as C            # noqa: E402
import preprocess as P        # noqa: E402


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> List[int]:
    """Greedy NMS on xyxy boxes. Returns kept indices."""
    if boxes.shape[0] == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = (xx2 - xx1).clip(0)
        h = (yy2 - yy1).clip(0)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thres]
    return keep


class YoloRuntime:
    def __init__(self, onnx_path: Path = None, artifacts_dir: Path = None,
                 names: List[str] = None, per_class_conf: Dict[str, float] = None,
                 conf_default: float = None, iou_thres: float = None) -> None:
        import onnxruntime as ort

        onnx_path = onnx_path or C.YOLO.ONNX
        artifacts_dir = artifacts_dir or C.YOLO.TIDL_DIR
        # At inference the TIDL EP only needs the artifacts folder; tidl_tools is
        # PC-only. CPUExecutionProvider catches any A72-fallback subgraphs.
        ep_opts = {"artifacts_folder": str(artifacts_dir)}
        self.sess = ort.InferenceSession(
            str(onnx_path),
            providers=["TIDLExecutionProvider", "CPUExecutionProvider"],
            provider_options=[ep_opts, {}],
        )
        self.iname = self.sess.get_inputs()[0].name

        # Labels + thresholds come from config (which read them from data.yaml);
        # all overridable here so runtime/config.yaml can tune without code edits.
        self.names = names if names is not None else C.YOLO.NAMES
        self.iou = C.YOLO.IOU_THRES if iou_thres is None else iou_thres
        self.conf_default = (C.YOLO.CONF_DEFAULT if conf_default is None
                             else conf_default)
        # Per-class keep-threshold array, indexed by class_id (built once).
        self.conf_by_id = C.YOLO.conf_by_id(
            self.names, per_class_conf, conf_default)
        # Cheapest possible pre-filter before the per-class test.
        self.min_conf = float(self.conf_by_id.min()) if self.conf_by_id.size else 0.0

        # Consistency check: the model's class count must match data.yaml names,
        # or labels/thresholds silently go wrong after a model swap. Output is
        # [1, 4+nc, N]; warn loudly if a static shape disagrees with len(names).
        try:
            oshape = self.sess.get_outputs()[0].shape
            ch = oshape[1] if len(oshape) == 3 else None
            nc_model = (ch - 4) if isinstance(ch, int) else None
        except Exception:  # noqa: BLE001
            nc_model = None
        if nc_model is not None and self.names and nc_model != len(self.names):
            print(f"[yolo] WARNING: model has {nc_model} classes but data.yaml "
                  f"lists {len(self.names)} names - labels/thresholds will be "
                  f"wrong. Repoint config.YOLO.DATA_YAML to this model's yaml.")

    # ── stage A ──
    def preprocess(self, bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        return P.preprocess_yolo(bgr)

    # ── stage B (C7x) ──
    def infer_raw(self, inp: np.ndarray) -> np.ndarray:
        return self.sess.run(None, {self.iname: inp})[0]      # [1, 84, 8400]

    # ── stage C (A72) ──
    def decode(self, raw: np.ndarray, meta: Dict[str, float]) -> List[Dict]:
        preds = raw[0].T                                       # [8400, 84]
        boxes_xywh = preds[:, :4]
        cls_scores = preds[:, 4:]
        class_ids = cls_scores.argmax(1)
        confs = cls_scores.max(1)

        # Cheap global pre-filter, then the exact per-class keep-threshold.
        m = confs >= self.min_conf
        if not np.any(m):
            return []
        boxes_xywh, class_ids, confs = boxes_xywh[m], class_ids[m], confs[m]
        if self.conf_by_id.size:                  # skip only if names unavailable
            # clip guards against a model/data.yaml class-count mismatch (warned
            # at load) so a stray class id can't IndexError mid-stream.
            safe = np.clip(class_ids, 0, self.conf_by_id.size - 1)
            m = confs >= self.conf_by_id[safe]
            if not np.any(m):
                return []
            boxes_xywh, class_ids, confs = boxes_xywh[m], class_ids[m], confs[m]

        # xywh (letterbox 640 space) → xyxy, then back to original image coords.
        cx, cy, w, h = boxes_xywh.T
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        r, left, top = meta["r"], meta["left"], meta["top"]
        sw, sh = meta["src_w"], meta["src_h"]
        x1 = ((x1 - left) / r).clip(0, sw)
        y1 = ((y1 - top) / r).clip(0, sh)
        x2 = ((x2 - left) / r).clip(0, sw)
        y2 = ((y2 - top) / r).clip(0, sh)
        boxes = np.stack([x1, y1, x2, y2], 1)

        # class-aware NMS via per-class coordinate offset.
        offset = class_ids.astype(np.float32)[:, None] * (max(sw, sh) + 1.0)
        keep = _nms(boxes + offset, confs, self.iou)

        out: List[Dict] = []
        for i in keep:
            cid = int(class_ids[i])
            out.append({
                "class_id": cid,
                "class_name": self.names[cid] if cid < len(self.names) else str(cid),
                "confidence": round(float(confs[i]), 4),
                "bbox_xyxy": [round(float(boxes[i, 0]), 1), round(float(boxes[i, 1]), 1),
                              round(float(boxes[i, 2]), 1), round(float(boxes[i, 3]), 1)],
            })
        return out
