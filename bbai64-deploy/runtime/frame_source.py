"""
Frame sources for the runtime pipeline.

  * MqttFrameSource  — subscribes to CARLA's camera topic, decodes JPEG payloads.
                       Keeps only the LATEST frame (drops stale ones) so the
                       pipeline always works on fresh data — correct behaviour for
                       a real-time ADAS feed.
  * VideoFileSource  — offline bring-up / benchmarking from an MP4, same interface.

Both expose: read(timeout) -> Optional[bgr]  and  close().
The abstraction is the seam where a GStreamer/CSI camera source can drop in later.
"""
from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from typing import Optional

import cv2
import numpy as np


class FrameSource(ABC):
    @abstractmethod
    def read(self, timeout: float = 1.0) -> Optional[np.ndarray]: ...

    @abstractmethod
    def close(self) -> None: ...


class ImageFileSource(FrameSource):
    """Single still image: yields the frame once, then EOF (None). For the
    pipeline-sanity / single-frame mode."""

    def __init__(self, path: str) -> None:
        self.frame = cv2.imread(path)
        if self.frame is None:
            raise RuntimeError(f"cannot read image: {path}")
        self._served = False

    def read(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        if self._served:
            return None
        self._served = True
        return self.frame

    def close(self) -> None:
        pass


class VideoFileSource(FrameSource):
    def __init__(self, path: str, loop: bool = False) -> None:
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open video: {path}")
        self.loop = loop
        # Source frame rate, so an annotated output plays back at the right speed.
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and fps > 0 else 30.0

    def read(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        ok, frame = self.cap.read()
        if not ok:
            if self.loop:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self.cap.read()
            if not ok:
                return None
        return frame

    def close(self) -> None:
        self.cap.release()


class MqttFrameSource(FrameSource):
    def __init__(self, broker: str, port: int, topic: str, qos: int = 0) -> None:
        from mqtt_compat import make_client

        self._lock = threading.Lock()
        self._latest: Optional[np.ndarray] = None
        self._new = threading.Event()

        self.client = make_client()
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self._topic = topic
        self._qos = qos
        # Auto-reconnect: the board may start before the external CARLA server /
        # broker is up; connect_async + loop_start retries in the background and
        # _on_connect re-subscribes on every (re)connect. read() just returns None
        # until frames flow, which Stage A treats as idle.
        self.client.reconnect_delay_set(min_delay=1, max_delay=30)
        self.client.connect_async(broker, port, keepalive=30)
        self.client.loop_start()
        print(f"[mqtt] connecting to {broker}:{port}, topic '{topic}' (auto-reconnect)")

    def _on_connect(self, client, userdata, flags, rc) -> None:
        client.subscribe(self._topic, qos=self._qos)
        print(f"[mqtt] connected (rc={rc}); subscribed to {self._topic}")

    def _on_message(self, client, userdata, msg) -> None:
        arr = np.frombuffer(msg.payload, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return
        with self._lock:
            self._latest = img          # keep only newest; drop any unprocessed
        self._new.set()

    def read(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        if not self._new.wait(timeout):
            return None
        with self._lock:
            frame = self._latest
            self._new.clear()
        return frame

    def close(self) -> None:
        self.client.loop_stop()
        self.client.disconnect()
