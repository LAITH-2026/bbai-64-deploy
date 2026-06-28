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
                 conf_default: float = None, iou_thres: float = None,
                 providers: List[str] = None) -> None:
        import onnxruntime as ort

        # Prefer the head-truncated export (what the C7x can verify) when present;
        # fall back to the full single-output model otherwise.
        if onnx_path is None:
            onnx_path = (C.YOLO.TRUNC_ONNX if C.YOLO.TRUNC_ONNX.exists()
                         else C.YOLO.ONNX)
        artifacts_dir = artifacts_dir or C.YOLO.TIDL_DIR
        # At inference the TIDL EP only needs the artifacts folder; tidl_tools is
        # PC-only. CPUExecutionProvider catches any A72-fallback subgraphs. A pure
        # ["CPUExecutionProvider"] override lets the decode be validated off-board.
        if providers is None:
            providers = ["TIDLExecutionProvider", "CPUExecutionProvider"]
        popts = [{"artifacts_folder": str(artifacts_dir)} if p == "TIDLExecutionProvider"
                 else {} for p in providers]
        self.sess = ort.InferenceSession(
            str(onnx_path), providers=providers, provider_options=popts,
        )
        self.iname = self.sess.get_inputs()[0].name

        # Head-truncated model? The C7x net was cut at the 6 raw detection-conv
        # outputs (3 scales x {box[1,4,H,W] ltrb, cls[1,nc,H,W] logits}) because the
        # board's TIDL firmware can't verify the DFL/Reshape head. So the anchor
        # decode + sigmoid that the full graph used to bake into output0 now run
        # here on the A72. Detected by >1 model output.
        self.truncated = len(self.sess.get_outputs()) > 1
        self.imgsz = int(C.YOLO.IMGSZ)
        self._anchor_cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}

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
            if self.truncated:
                # cls heads are the non-4-channel outputs ([1, nc, H, W]).
                chans = [o.shape[1] for o in self.sess.get_outputs()
                         if len(o.shape) == 4 and isinstance(o.shape[1], int)]
                cls_ch = [c for c in chans if c != 4]
                nc_model = cls_ch[0] if cls_ch else None
            else:
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
    def infer_raw(self, inp: np.ndarray):
        out = self.sess.run(None, {self.iname: inp})
        # Truncated model → the 6 raw conv heads (list); full model → [1, 4+nc, N].
        return out if self.truncated else out[0]

    def _anchors(self, h: int, w: int) -> Tuple[np.ndarray, np.ndarray]:
        """Cached anchor-point grid (x+0.5, y+0.5) in grid units, row-major h*w+w
        to match the conv output's [C, H*W] flatten."""
        key = (h, w)
        a = self._anchor_cache.get(key)
        if a is None:
            ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
            a = ((xs.reshape(-1) + 0.5).astype(np.float32),
                 (ys.reshape(-1) + 0.5).astype(np.float32))
            self._anchor_cache[key] = a
        return a

    def _decode_truncated(self, outs: List[np.ndarray]
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """6 raw heads → (boxes_xyxy in letterbox-640 space, class_ids, confs).
        Per scale: box[1,4,H,W]=ltrb dist, cls[1,nc,H,W]=logits. anchor decode
        (dist2bbox) + sigmoid(cls), stride = imgsz / H. (This yolo26n is reg_max=1,
        i.e. cv2 outputs 4 channels directly — no DFL expectation needed.)"""
        by_hw: Dict[Tuple[int, int], Dict[str, np.ndarray]] = {}
        for o in outs:
            _, c, h, w = o.shape
            by_hw.setdefault((h, w), {})["box" if c == 4 else "cls"] = o
        boxes_all, cls_all = [], []
        for (h, w), pair in sorted(by_hw.items(), key=lambda kv: -kv[0][0]):
            box, cls = pair["box"], pair["cls"]
            stride = self.imgsz / h
            l, t, r, b = box.reshape(4, h * w)            # ltrb distances (grid units)
            ax, ay = self._anchors(h, w)
            x1 = (ax - l) * stride
            y1 = (ay - t) * stride
            x2 = (ax + r) * stride
            y2 = (ay + b) * stride
            boxes_all.append(np.stack([x1, y1, x2, y2], 1))   # [h*w, 4]
            logits = cls.reshape(cls.shape[1], h * w).T        # [h*w, nc]
            cls_all.append(1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50))))
        boxes = np.concatenate(boxes_all, 0)
        scores = np.concatenate(cls_all, 0)
        return boxes, scores.argmax(1), scores.max(1)

    # ── stage C (A72) ──
    def decode(self, raw, meta: Dict[str, float]) -> List[Dict]:
        if self.truncated:
            boxes_xyxy, class_ids, confs = self._decode_truncated(raw)
        else:
            preds = raw[0].T                                   # [8400, 4+nc]
            cx, cy, w, h = preds[:, :4].T                      # xywh, letterbox-640
            boxes_xyxy = np.stack([cx - w / 2, cy - h / 2,
                                   cx + w / 2, cy + h / 2], 1)
            cls_scores = preds[:, 4:]
            class_ids = cls_scores.argmax(1)
            confs = cls_scores.max(1)
        return self._finish(boxes_xyxy, class_ids, confs, meta)

    def _finish(self, boxes_xyxy: np.ndarray, class_ids: np.ndarray,
                confs: np.ndarray, meta: Dict[str, float]) -> List[Dict]:
        """Shared tail: conf filter → undo letterbox → class-aware NMS → dicts.
        `boxes_xyxy` is in the letterbox-640 space (both decode paths converge here)."""
        # Cheap global pre-filter, then the exact per-class keep-threshold.
        m = confs >= self.min_conf
        if not np.any(m):
            return []
        boxes_xyxy, class_ids, confs = boxes_xyxy[m], class_ids[m], confs[m]
        if self.conf_by_id.size:                  # skip only if names unavailable
            # clip guards against a model/data.yaml class-count mismatch (warned
            # at load) so a stray class id can't IndexError mid-stream.
            safe = np.clip(class_ids, 0, self.conf_by_id.size - 1)
            m = confs >= self.conf_by_id[safe]
            if not np.any(m):
                return []
            boxes_xyxy, class_ids, confs = boxes_xyxy[m], class_ids[m], confs[m]

        # letterbox-640 xyxy → original image coords.
        r, left, top = meta["r"], meta["left"], meta["top"]
        sw, sh = meta["src_w"], meta["src_h"]
        x1 = ((boxes_xyxy[:, 0] - left) / r).clip(0, sw)
        y1 = ((boxes_xyxy[:, 1] - top) / r).clip(0, sh)
        x2 = ((boxes_xyxy[:, 2] - left) / r).clip(0, sw)
        y2 = ((boxes_xyxy[:, 3] - top) / r).clip(0, sh)
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
