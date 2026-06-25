#!/usr/bin/env python3
"""
Stage 2a (PC) — extract evenly-spaced calibration frames from a CARLA / dashcam
clip into bbai64-deploy/calib/. These frames drive INT8 PTQ: the quantizer
watches activation ranges on them, so they must be REPRESENTATIVE of what the
board will see (same scenes/lighting as the CARLA stream).

    python compile/prepare_calib.py --video carla_clip.mp4 --n 25
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # bbai64-deploy/
import config as C  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="clip to sample calibration frames from")
    ap.add_argument("--n", type=int, default=25, help="number of frames (default 25)")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"[calib] cannot open {args.video}")
    C.CALIB_DIR.mkdir(parents=True, exist_ok=True)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // args.n) if total > 0 else 1
    saved = idx = 0
    while saved < args.n:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            cv2.imwrite(str(C.CALIB_DIR / f"calib_{saved:03d}.jpg"), frame)
            saved += 1
        idx += 1
    cap.release()

    print(f"[calib] wrote {saved} frames → {C.CALIB_DIR}")
    if saved < args.n:
        print(f"[calib] note: clip yielded only {saved}/{args.n} frames.")


if __name__ == "__main__":
    main()
