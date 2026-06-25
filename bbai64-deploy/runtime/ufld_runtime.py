"""
UFLDv2 lane-detection runtime (board) — onnxruntime + TIDLExecutionProvider.

Split into preprocess / infer_raw / decode (numpy pred2coords) for the same
pipeline-overlap reason as yolo_runtime. Lane coords come back in CULane
visualization space (VIS_W × VIS_H); the compositor scales them to the frame.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # bbai64-deploy/
import config as C            # noqa: E402
import preprocess as P        # noqa: E402
from pred2coords_np import pred2coords_np  # noqa: E402


class UfldRuntime:
    def __init__(self, onnx_path: Path = None, artifacts_dir: Path = None) -> None:
        import onnxruntime as ort

        onnx_path = onnx_path or C.UFLD.ONNX
        artifacts_dir = artifacts_dir or C.UFLD.TIDL_DIR
        ep_opts = {"artifacts_folder": str(artifacts_dir)}
        self.sess = ort.InferenceSession(
            str(onnx_path),
            providers=["TIDLExecutionProvider", "CPUExecutionProvider"],
            provider_options=[ep_opts, {}],
        )
        self.iname = self.sess.get_inputs()[0].name
        self.onames = [o.name for o in self.sess.get_outputs()]

    # ── stage A ──
    def preprocess(self, bgr: np.ndarray) -> np.ndarray:
        return P.preprocess_ufld(bgr)

    # ── stage B (C7x) ──
    def infer_raw(self, inp: np.ndarray) -> List[np.ndarray]:
        return self.sess.run(None, {self.iname: inp})

    # ── stage C (A72) ──
    def decode(self, raw: List[np.ndarray]) -> List[Dict]:
        # Map by output name when preserved; else fall back to export order
        # (loc_row, loc_col, exist_row, exist_col).
        if raw is None or len(raw) < 4:
            raise RuntimeError(
                f"UFLD expected 4 output tensors, got {0 if raw is None else len(raw)} "
                f"(names={self.onames}). The TIDL artifacts likely don't match the "
                f"exported ONNX — recompile from this repo's ufld ONNX.")
        by_name = dict(zip(self.onames, raw))
        try:
            loc_row = by_name["loc_row"]; loc_col = by_name["loc_col"]
            exist_row = by_name["exist_row"]; exist_col = by_name["exist_col"]
        except KeyError:
            loc_row, loc_col, exist_row, exist_col = raw[0], raw[1], raw[2], raw[3]

        coords = pred2coords_np(
            loc_row, loc_col, exist_row, exist_col,
            C.UFLD.ROW_ANCHOR, C.UFLD.COL_ANCHOR,
            local_width=C.UFLD.LOCAL_WIDTH,
            original_image_width=C.UFLD.VIS_W,
            original_image_height=C.UFLD.VIS_H,
        )
        return [{"index": i, "points": [[int(x), int(y)] for x, y in lane]}
                for i, lane in enumerate(coords)]
