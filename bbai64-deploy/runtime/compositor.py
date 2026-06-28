"""
Overlay compositor — ports the box-mask occlusion logic from
Integrate-Features/integrate.py to the board (numpy/cv2, no torch).

Lanes are drawn first, then the original pixels inside each detection box are
restored over them, so a lane appears to pass BEHIND a vehicle. Box outlines +
labels go on top. Lane points arrive in CULane vis space and are scaled to the
frame's native resolution so they line up with the (native-space) boxes.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # bbai64-deploy/
import config as C  # noqa: E402

LANE_COLOR = (0, 255, 0)
BOX_COLOR = (0, 0, 255)


def _overlay_masks(frame: np.ndarray, masks: Tuple[np.ndarray, np.ndarray]) -> None:
    """Alpha-blend TwinLite drivable-area + lane-line binary masks onto the frame.
    Masks are produced at the model's working resolution; they are nearest-resized
    to the frame so they line up with the (native-space) detection boxes."""
    da, ll = masks
    fh, fw = frame.shape[:2]
    if da.shape != (fh, fw):
        da = cv2.resize(da, (fw, fh), interpolation=cv2.INTER_NEAREST)
        ll = cv2.resize(ll, (fw, fh), interpolation=cv2.INTER_NEAREST)
    a = C.TWINLITE.OVERLAY_ALPHA
    for mask, color in ((da, C.TWINLITE.DA_COLOR), (ll, C.TWINLITE.LL_COLOR)):
        sel = mask.astype(bool)
        if sel.any():
            frame[sel] = (frame[sel] * (1.0 - a) + np.array(color) * a).astype(frame.dtype)


def _draw_lanes_scaled(frame: np.ndarray, lanes: List[Dict], vis_w: int, vis_h: int) -> None:
    h, w = frame.shape[:2]
    sx, sy = w / float(vis_w), h / float(vis_h)
    for lane in lanes:
        prev = None
        for p in lane.get("points", []):
            if len(p) != 2:
                continue
            x, y = int(p[0] * sx), int(p[1] * sy)
            cv2.circle(frame, (x, y), 4, LANE_COLOR, -1)
            if prev is not None:
                cv2.line(frame, prev, (x, y), LANE_COLOR, 2)
            prev = (x, y)


def composite(frame: np.ndarray, detections: List[Dict], lanes: List[Dict],
              vis_w: int, vis_h: int,
              masks: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> np.ndarray:
    canvas = frame
    clean = frame.copy()

    # Lane visualisation: TwinLite supplies (da, ll) segmentation masks to overlay;
    # UFLD supplies polyline lane points. Exactly one is active per run.
    if masks is not None:
        _overlay_masks(canvas, masks)
    else:
        _draw_lanes_scaled(canvas, lanes, vis_w, vis_h)

    fh, fw = canvas.shape[:2]
    boxes = []
    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det["bbox_xyxy"])
        boxes.append((x1, y1, x2, y2))
        x1c, y1c, x2c, y2c = max(0, x1), max(0, y1), min(fw, x2), min(fh, y2)
        if x2c > x1c and y2c > y1c:                     # restore pixels behind lanes
            canvas[y1c:y2c, x1c:x2c] = clean[y1c:y2c, x1c:x2c]

    for (x1, y1, x2, y2), det in zip(boxes, detections):
        cv2.rectangle(canvas, (x1, y1), (x2, y2), BOX_COLOR, 2)
        label = f"{det['class_name']} {det['confidence']:.2f}"
        if det.get("depth_m") is not None:          # append metric distance
            label += f"  {det['depth_m']:.1f}m"
        cv2.putText(canvas, label, (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, BOX_COLOR, 1, cv2.LINE_AA)
    return canvas
