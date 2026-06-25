"""
Torch-free port of ufld_inference.pred2coords (runs on the A72 at runtime).

Input are the 4 numpy arrays produced by the TIDL ONNX session:
    loc_row   [1, num_grid_row, num_cls_row, num_lane]
    loc_col   [1, num_grid_col, num_cls_col, num_lane]
    exist_row [1, 2,            num_cls_row, num_lane]
    exist_col [1, 2,            num_cls_col, num_lane]

Returns a list of lanes, each a list of (x, y) integer points in the CULane
visualization space (original_image_width × original_image_height). Logic mirrors
the original torch implementation exactly, including the row/col validity gates
and lane index ordering ([1, 2] for rows, [0, 3] for cols).
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


def _softmax_1d(v: np.ndarray) -> np.ndarray:
    v = v - np.max(v)
    e = np.exp(v)
    return e / np.sum(e)


def pred2coords_np(
    loc_row: np.ndarray,
    loc_col: np.ndarray,
    exist_row: np.ndarray,
    exist_col: np.ndarray,
    row_anchor: np.ndarray,
    col_anchor: np.ndarray,
    local_width: int = 1,
    original_image_width: int = 1640,
    original_image_height: int = 590,
) -> List[List[Tuple[int, int]]]:
    _, num_grid_row, num_cls_row, _ = loc_row.shape
    _, num_grid_col, num_cls_col, _ = loc_col.shape

    max_indices_row = loc_row.argmax(axis=1)     # [1, num_cls_row, num_lane]
    valid_row = exist_row.argmax(axis=1)         # [1, num_cls_row, num_lane] (0/1)
    max_indices_col = loc_col.argmax(axis=1)
    valid_col = exist_col.argmax(axis=1)

    coords: List[List[Tuple[int, int]]] = []
    row_lane_idx = [1, 2]
    col_lane_idx = [0, 3]

    for i in row_lane_idx:
        tmp: List[Tuple[int, int]] = []
        if valid_row[0, :, i].sum() > num_cls_row / 2:
            for k in range(valid_row.shape[1]):
                if valid_row[0, k, i]:
                    idx = int(max_indices_row[0, k, i])
                    lo = max(0, idx - local_width)
                    hi = min(num_grid_row - 1, idx + local_width) + 1
                    all_ind = np.arange(lo, hi)
                    w = _softmax_1d(loc_row[0, all_ind, k, i])
                    out = float((w * all_ind).sum()) + 0.5
                    out = out / (num_grid_row - 1) * original_image_width
                    tmp.append((int(out), int(row_anchor[k] * original_image_height)))
            coords.append(tmp)

    for i in col_lane_idx:
        tmp = []
        if valid_col[0, :, i].sum() > num_cls_col / 4:
            for k in range(valid_col.shape[1]):
                if valid_col[0, k, i]:
                    idx = int(max_indices_col[0, k, i])
                    lo = max(0, idx - local_width)
                    hi = min(num_grid_col - 1, idx + local_width) + 1
                    all_ind = np.arange(lo, hi)
                    w = _softmax_1d(loc_col[0, all_ind, k, i])
                    out = float((w * all_ind).sum()) + 0.5
                    out = out / (num_grid_col - 1) * original_image_height
                    tmp.append((int(col_anchor[k] * original_image_width), int(out)))
            coords.append(tmp)

    return coords
