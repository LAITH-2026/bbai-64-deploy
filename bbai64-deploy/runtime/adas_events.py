"""
ADAS event publisher — turns detections + lanes into alert payloads and publishes
them over MQTT for the Qt infotainment system.

Every detection is published by its class_name (taken straight from the model's
data.yaml, via config.YOLO.NAMES) plus its class_id; the Qt side maps a name to
its own UI category. Nothing is hard-coded to a fixed taxonomy here, so swapping
the model changes the published names automatically.

Publishing is best-effort: if no broker is reachable the app still runs and
renders locally (publish becomes a no-op with a one-time warning).
"""
from __future__ import annotations

import json
import time
from typing import Dict, List


def build_payload(frame_idx: int, detections: List[Dict], lanes: List[Dict],
                  frame_w: int = None, frame_h: int = None,
                  vis_w: int = None, vis_h: int = None) -> Dict:
    """Build the per-frame message (objects + lane locations).

    Object bboxes are already in frame-pixel coords. Lane points arrive in CULane
    visualization space (vis_w × vis_h); when the frame size is given they are
    scaled to frame-pixel coords so objects and lanes share one coordinate system.
    The MQTT message and the offline JSON file are this same structure.
    """
    objects = [
        {
            "class_id": d["class_id"],
            "class_name": d["class_name"],   # from data.yaml — Qt maps by this
            "confidence": d["confidence"],
            "bbox_xyxy": d["bbox_xyxy"],
            # Per-object metric distance (metres) from geometric monocular ranging
            # (IPM ground-plane + known-size pinhole), or None when depth is
            # disabled / the box is too far / truncated to estimate reliably. Qt
            # uses this for distance-based ADAS warnings.
            "depth_m": d.get("depth_m"),
        }
        for d in detections
    ]
    if frame_w and frame_h and vis_w and vis_h:
        sx, sy = frame_w / float(vis_w), frame_h / float(vis_h)
    else:
        sx = sy = 1.0
    lane_objs = [
        {
            "index": lane.get("index"),
            "points": [[int(round(x * sx)), int(round(y * sy))]
                       for x, y in lane.get("points", [])],
        }
        for lane in lanes
    ]
    return {
        "frame": frame_idx,
        "timestamp": round(time.time(), 3),
        "lanes_detected": len(lanes),
        "lanes": lane_objs,
        "objects": objects,
    }


class AdasPublisher:
    def __init__(self, broker: str, port: int, topic: str, qos: int = 0) -> None:
        self.topic = topic
        self.qos = qos
        self._ok = False
        self._warned = False
        try:
            from mqtt_compat import make_client
            self.client = make_client()
            self.client.reconnect_delay_set(min_delay=1, max_delay=30)
            self.client.connect_async(broker, port, keepalive=30)
            self.client.loop_start()
            self._ok = True
            print(f"[adas] publishing alerts -> {topic} @ {broker}:{port}")
        except Exception as e:  # noqa: BLE001
            print(f"[adas] broker unavailable ({e}); ADAS publishing disabled.")

    def publish(self, payload: Dict) -> None:
        if not self._ok:
            return
        try:
            self.client.publish(self.topic, json.dumps(payload), qos=self.qos)
        except Exception as e:  # noqa: BLE001
            if not self._warned:
                print(f"[adas] publish failed ({e}); further errors suppressed.")
                self._warned = True

    def close(self) -> None:
        if self._ok:
            self.client.loop_stop()
            self.client.disconnect()
