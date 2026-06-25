"""
Depth-Anything-V2 metric-depth runtime (board) — onnxruntime + TIDLExecutionProvider.

Split into preprocess / infer_raw / decode for the same pipeline-overlap reason as
yolo_runtime / ufld_runtime: the C7x runs depth inference for one frame while the
A72 samples per-object distances for another.

The network emits a metric depth map (metres) at its own working resolution; decode
bicubic-resizes it back to the native frame so every pixel maps 1:1 onto a YOLO box.
`sample_box_depth` (ported verbatim from Integrate-Features/integrate.py) then takes
the median of each box's central region for a robust per-object distance.

Torch-free: the board has no transformers/torch. Only the PC export step needs them.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # bbai64-deploy/
import config as C            # noqa: E402
import preprocess as P        # noqa: E402


def sample_box_depth(depth_map: np.ndarray, box_xyxy, shrink: float = None) -> Optional[float]:
    """Representative metric depth (metres) for one detection box.

    Samples the central sub-region of the box (default inner 50%) and takes the
    median — robust to background pixels leaking in at the box edges and to depth
    noise, far better than a single centre pixel. Returns None if the box is
    degenerate or yields no valid samples. (Verbatim from integrate.py.)
    """
    shrink = C.DEPTH.SAMPLE_SHRINK if shrink is None else shrink
    h, w = depth_map.shape[:2]
    x1, y1, x2, y2 = box_xyxy
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return None
    mx, my = bw * (1.0 - shrink) / 2.0, bh * (1.0 - shrink) / 2.0
    cx1 = max(0, int(x1 + mx)); cy1 = max(0, int(y1 + my))
    cx2 = min(w, int(x2 - mx)); cy2 = min(h, int(y2 - my))
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    region = depth_map[cy1:cy2, cx1:cx2]
    valid = region[np.isfinite(region) & (region > 0)]
    if valid.size == 0:
        return None
    return float(np.median(valid))


class DepthRuntime:
    def __init__(self, onnx_path: Path = None, artifacts_dir: Path = None) -> None:
        import onnxruntime as ort

        onnx_path = onnx_path or C.DEPTH.ONNX
        artifacts_dir = artifacts_dir or C.DEPTH.TIDL_DIR
        # CPUExecutionProvider is the catch-all for ViT subgraphs TIDL won't take
        # (transformer offload is version-dependent) — they run on the A72.
        ep_opts = {"artifacts_folder": str(artifacts_dir)}
        self.sess = ort.InferenceSession(
            str(onnx_path),
            providers=["TIDLExecutionProvider", "CPUExecutionProvider"],
            provider_options=[ep_opts, {}],
        )
        self.iname = self.sess.get_inputs()[0].name

    # ── stage A ──
    def preprocess(self, bgr: np.ndarray) -> np.ndarray:
        return P.preprocess_depth(bgr)

    # ── stage B (C7x, time-shared after YOLO + UFLD) ──
    def infer_raw(self, inp: np.ndarray) -> np.ndarray:
        return self.sess.run(None, {self.iname: inp})[0]      # [1,1,H',W'] metres

    # ── stage C (A72) ──
    def decode(self, raw: np.ndarray, frame_hw) -> np.ndarray:
        """Resize the metric depth map to (frame_h, frame_w) so it aligns 1:1
        with native-resolution boxes. Returns a float32 HxW array in metres."""
        depth = np.asarray(raw, dtype=np.float32)
        depth = np.squeeze(depth)                  # (1,1,H',W') / (1,H',W') → (H',W')
        fh, fw = frame_hw
        if depth.shape != (fh, fw):
            depth = cv2.resize(depth, (fw, fh), interpolation=cv2.INTER_CUBIC)
        return depth

    def attach_depth(self, detections: List[Dict], depth_map: np.ndarray) -> None:
        """Add a `depth_m` field (metres, or None) to each detection in place."""
        for det in detections:
            d = sample_box_depth(depth_map, det["bbox_xyxy"])
            det["depth_m"] = round(d, 2) if d is not None else None
