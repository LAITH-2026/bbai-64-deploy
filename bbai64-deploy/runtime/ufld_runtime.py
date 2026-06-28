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
    # Lane coords are decoded in CULane visualisation space; the compositor /
    # payload scale them to the frame. Exposed so app.py is lane-source-agnostic.
    VIS_W = C.UFLD.VIS_W
    VIS_H = C.UFLD.VIS_H

    def __init__(self, onnx_path: Path = None, artifacts_dir: Path = None) -> None:
        import onnxruntime as ort

        # Prefer the head-truncated export (single `linear_1` output) when present;
        # the board's 0x20250429 TIDL firmware can't verify UFLD's 4× Slice+Reshape
        # head, so the C7x runs backbone+FC and the head split runs in numpy here.
        if onnx_path is None:
            onnx_path = (C.UFLD.TRUNC_ONNX if C.UFLD.TRUNC_ONNX.exists()
                         else C.UFLD.ONNX)
        artifacts_dir = artifacts_dir or C.UFLD.TIDL_DIR
        ep_opts = {"artifacts_folder": str(artifacts_dir)}
        self.sess = ort.InferenceSession(
            str(onnx_path),
            providers=["TIDLExecutionProvider", "CPUExecutionProvider"],
            provider_options=[ep_opts, {}],
        )
        self.iname = self.sess.get_inputs()[0].name
        self.onames = [o.name for o in self.sess.get_outputs()]
        # Truncated model = single flat output `linear_1` [1, 91224]; numpy splits
        # it into (loc_row, loc_col, exist_row, exist_col). Bounds/shapes mirror the
        # original onnx head exactly (Slice + Reshape on the FC output).
        self.truncated = len(self.onames) == 1
        self._ufld_split = [
            ("loc_row",   0,     57600, (1, 200, 72, 4)),
            ("loc_col",   57600, 90000, (1, 100, 81, 4)),
            ("exist_row", 90000, 90576, (1,   2, 72, 4)),
            ("exist_col", 90576, 91224, (1,   2, 81, 4)),
        ]

    # ── stage A ──
    def preprocess(self, bgr: np.ndarray) -> np.ndarray:
        return P.preprocess_ufld(bgr)

    # ── stage B (C7x) ──
    def infer_raw(self, inp: np.ndarray) -> List[np.ndarray]:
        return self.sess.run(None, {self.iname: inp})

    # ── stage C (A72) ──
    def decode(self, raw: List[np.ndarray]) -> List[Dict]:
        # Head-truncated model: split the flat FC output into the 4 head tensors
        # (the Slice+Reshape the board's TIDL couldn't verify), then decode as usual.
        if self.truncated:
            flat = raw[0].reshape(-1)
            by_name = {name: flat[s:e].reshape(shape)
                       for name, s, e, shape in self._ufld_split}
            coords = pred2coords_np(
                by_name["loc_row"], by_name["loc_col"],
                by_name["exist_row"], by_name["exist_col"],
                C.UFLD.ROW_ANCHOR, C.UFLD.COL_ANCHOR,
                local_width=C.UFLD.LOCAL_WIDTH,
                original_image_width=C.UFLD.VIS_W,
                original_image_height=C.UFLD.VIS_H,
            )
            return [{"index": i, "points": [[int(x), int(y)] for x, y in lane]}
                    for i, lane in enumerate(coords)]
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
