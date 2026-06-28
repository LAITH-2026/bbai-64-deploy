"""
Per-object metric distance (board) — CLOSED-FORM MONOCULAR GEOMETRY, no network.

Replaces the Depth-Anything-V2 ViT, which TIDL on this SDK could not offload: its
DPT dense head + attention / LayerNorm / GELU ops fell back to the A72 and ran at
seconds-per-frame as the third model time-sharing the one C7x. The runtime only
ever needed ONE scalar distance per YOLO box (the `depth_m` ADAS field; see
compositor / adas_events), so the dense depth map was never necessary — we derive
that scalar from the known CARLA camera geometry instead. Result: zero C7x load
(the engine now runs only yolo + ufld) and native metres with no scale recovery.

Two estimators, routed by detection class (config.DEPTH):
  • Ground-plane / IPM   — the box BOTTOM is the object's road-contact point
    (vehicles, bikes, pedestrians). Back-project the bottom-centre pixel onto the
    flat ground plane using camera height H + pitch δ + intrinsics:
        y_n = (v_bottom − cy) / fy
        Z   = H · (cosδ − y_n·sinδ) / (y_n·cosδ + sinδ)
    (δ = 0 ⇒ the textbook  Z = H·fy / (v_bottom − cy)).
  • Known-size pinhole   — OFF-ground objects (traffic lights, speed signs) and
    boxes with a clipped bottom: Z = fy · H_real[class] / h_px.

Intrinsics come from the live frame size + FOV (fx = fy = W/(2·tan(FOV/2)),
cx = W/2, cy = H/2), so they track the incoming MQTT resolution automatically.

Torch-free and onnxruntime-free: there is no model session to load — `attach_depth`
runs on the A72 in microseconds. Keeps the DepthRuntime / attach_depth interface so
app.py's stage C is a drop-in (the C7x preprocess / infer / decode steps are gone).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # bbai64-deploy/
import config as C            # noqa: E402


class DepthRuntime:
    """Closed-form monocular per-object distance estimator.

    Construction is cheap (no artifacts, no session); the optional positional/
    keyword args are accepted and ignored so existing call sites (`DepthRuntime()`)
    keep working.
    """

    def __init__(self, *_args, **_kwargs) -> None:
        d = C.DEPTH
        self.fov_rad = math.radians(d.FOV_DEG)
        self.h_cam = float(d.CAM_HEIGHT_M)
        self.pitch_rad = math.radians(d.PITCH_DEG)
        self._cosd = math.cos(self.pitch_rad)
        self._sind = math.sin(self.pitch_rad)
        self.class_dims = dict(d.CLASS_DIMS)
        self.unknown_h = float(d.UNKNOWN_HEIGHT_M)
        self.ground = set(d.GROUND_CONTACT)
        self.max_m = float(d.MAX_M)
        self.max_range = float(d.MAX_RANGE_M)
        self.edge_margin = float(d.EDGE_MARGIN_PX)
        self._intr: Optional[Tuple[int, int, float, float, float, float]] = None

    # ── intrinsics (cached per frame resolution) ──────────────────
    def _intrinsics(self, fw: int, fh: int):
        """(fx, fy, cx, cy) for the live frame size. CARLA's FOV is horizontal and
        pixels are square, so fy == fx. Recomputed only when the resolution changes
        (so the boxes' pixel coords and the intrinsics always agree)."""
        if self._intr is None or self._intr[0] != fw or self._intr[1] != fh:
            fx = fw / (2.0 * math.tan(self.fov_rad / 2.0))
            self._intr = (fw, fh, fx, fx, fw / 2.0, fh / 2.0)
        _, _, fx, fy, cx, cy = self._intr
        return fx, fy, cx, cy

    # ── ground-plane / IPM distance from the box-bottom contact point ─
    def _ground_distance(self, v_bottom: float, fy: float, cy: float) -> Optional[float]:
        y_n = (v_bottom - cy) / fy
        denom = y_n * self._cosd + self._sind        # = 0 at the horizon
        if denom <= 1e-6:                            # at / above the horizon: no hit
            return None
        z = self.h_cam * (self._cosd - y_n * self._sind) / denom
        return z if z > 0.0 else None

    # ── known-size pinhole distance from the box pixel height ─────
    def _size_distance(self, h_px: float, real_h: float, fy: float) -> Optional[float]:
        if h_px <= 1.0:
            return None
        return fy * real_h / h_px

    def estimate(self, box_xyxy, class_name: Optional[str],
                 fw: int, fh: int) -> Tuple[Optional[float], Optional[str]]:
        """Per-box distance in metres and the method used ('ipm' | 'size'), or
        (None, None) when no reliable estimate is available."""
        fx, fy, cx, cy = self._intrinsics(fw, fh)
        x1, y1, x2, y2 = box_xyxy
        h_px = y2 - y1
        if h_px <= 0.0 or (x2 - x1) <= 0.0:
            return None, None
        v_bottom = y2
        real_h = self.class_dims.get(class_name, self.unknown_h)

        # Truncation flags: a box touching a frame edge has an unreliable pixel
        # height (pinhole) and, if bottom-clipped, no visible contact point (IPM).
        clipped_bottom = v_bottom >= fh - self.edge_margin
        clipped_height = clipped_bottom or (y1 <= self.edge_margin)

        d_ground = None
        if class_name in self.ground and not clipped_bottom:
            d_ground = self._ground_distance(v_bottom, fy, cy)

        d_size = None
        if real_h is not None and not clipped_height:
            d_size = self._size_distance(h_px, real_h, fy)

        # Routing: ground objects trust the contact-point IPM, falling back to the
        # size estimate; off-ground / clipped objects use the size estimate.
        if d_ground is not None:
            dist, method = d_ground, "ipm"
        elif d_size is not None:
            dist, method = d_size, "size"
        else:
            return None, None

        if dist <= 0.0 or dist > self.max_range:     # too far to trust monocularly
            return None, method
        return float(min(dist, self.max_m)), method

    def attach_depth(self, detections: List[Dict], frame_hw: Tuple[int, int]) -> None:
        """Add a `depth_m` field (metres, or None) to each detection in place.
        `frame_hw` is (frame_h, frame_w) of the native frame the boxes live in."""
        fh, fw = frame_hw
        for det in detections:
            d, _ = self.estimate(det["bbox_xyxy"], det.get("class_name"), fw, fh)
            det["depth_m"] = round(d, 2) if d is not None else None
