"""
TwinLiteNet lane + drivable-area runtime (board) — onnxruntime + TIDLExecutionProvider.

Drop-in replacement for UfldRuntime in app.py's lane stage. UFLDv2's 196 MB FC head
resets the board; TwinLiteNet is a 0.4 M-param conv-only seg net that offloads
127/127 nodes to the C7x and runs at ~70 FPS with no reset.

Same preprocess / infer_raw / decode split as ufld_runtime / yolo_runtime for the
pipeline overlap. Unlike UFLD (which emits lane-curve point lists), TwinLite emits
two raw-logit segmentation heads:
    da [1,2,H,W]  drivable area
    ll [1,2,H,W]  lane lines
decode = numpy argmax over the 2-channel axis on the A72 → two binary masks, which
the compositor alpha-blends onto the frame (it carries no "lane points", so decode
returns an empty lane list and stashes the masks on `self.last_masks` for the
compositor to pick up).

ACCURACY: on this 0x20250429 firmware TIDL mis-quantises the decoder's ConvTranspose
(logit overflow), so the masks are saturated/inaccurate pending a decoder retrain
(Resize+Conv) — see TWINLITE_RETRAIN_HANDOFF.md. Latency/FPS are valid; mask
geometry is not yet deployment-accurate.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # bbai64-deploy/
import config as C            # noqa: E402
import preprocess as P        # noqa: E402


class TwinliteRuntime:
    # Compositor reads these to scale; TwinLite masks are produced at native frame
    # size already, so the "vis" space is the frame itself (scale factor 1).
    VIS_W = C.TWINLITE.TWIN_W
    VIS_H = C.TWINLITE.TWIN_H

    def __init__(self, onnx_path: Path = None, artifacts_dir: Path = None) -> None:
        import onnxruntime as ort

        onnx_path = onnx_path or C.TWINLITE.ONNX
        artifacts_dir = artifacts_dir or C.TWINLITE.TIDL_DIR
        ep_opts = {"artifacts_folder": str(artifacts_dir)}
        self.sess = ort.InferenceSession(
            str(onnx_path),
            providers=["TIDLExecutionProvider", "CPUExecutionProvider"],
            provider_options=[ep_opts, {}],
        )
        self.iname = self.sess.get_inputs()[0].name
        self.onames = [o.name for o in self.sess.get_outputs()]
        # Most recent decoded (da_mask, ll_mask) at frame resolution — the
        # compositor reads this to alpha-blend the segmentation overlay.
        self.last_masks: Optional[Tuple[np.ndarray, np.ndarray]] = None

    # ── stage A ──
    def preprocess(self, bgr: np.ndarray) -> np.ndarray:
        return P.preprocess_twinlite(bgr)

    # ── stage B (C7x) ──
    def infer_raw(self, inp: np.ndarray) -> List[np.ndarray]:
        return self.sess.run(None, {self.iname: inp})

    # ── stage C (A72) ──
    def _argmax_mask(self, logits: np.ndarray) -> np.ndarray:
        """[1,2,H,W] raw logits → [H,W] uint8 {0,1} (channel-1 beats channel-0)."""
        a = np.asarray(logits)
        if a.ndim == 4:
            a = a[0]
        return (a[1] > a[0]).astype(np.uint8)

    def decode(self, raw: List[np.ndarray]) -> List[Dict]:
        """Argmax the da/ll heads into binary masks; stash them for the compositor.
        Returns an empty lane-point list (TwinLite has no polyline lanes — the masks
        carry the lane/drivable info)."""
        by_name = dict(zip(self.onames, raw))
        da_logits = by_name.get(C.TWINLITE.DA_OUTPUT, raw[0])
        ll_logits = by_name.get(C.TWINLITE.LL_OUTPUT, raw[-1])
        da = self._argmax_mask(da_logits)
        ll = self._argmax_mask(ll_logits)
        self.last_masks = (da, ll)
        return []
