#!/usr/bin/env python3
"""
Stage 1a (PC) — export YOLO .pt → fixed-shape ONNX for TIDL.

Why fixed shape + no NMS:
  * TIDL compiles a static graph; dynamic axes hurt or break offload.
  * ultralytics' built-in NMS export emits ops TIDL won't accelerate, so we keep
    the raw head output ([1, 84, 8400]) and run NMS in numpy on the A72 at
    runtime (runtime/yolo_runtime.py). This keeps 100 % of the *network* on C7x.

Run on an x86_64 Linux / WSL2 box that has `ultralytics` installed:
    python export/export_yolo_onnx.py
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # bbai64-deploy/
import config as C  # noqa: E402


def main() -> None:
    try:
        from ultralytics import YOLO  # lazy import so config stays torch-free
    except Exception as e:  # noqa: BLE001
        sys.exit(f"[export] ultralytics not available ({e}); "
                 f"run on the PC export venv: pip install ultralytics")

    if not C.YOLO.PT.exists():
        sys.exit(f"[export] weights not found: {C.YOLO.PT}\n"
                 f"         drop your trained best.pt there, or repoint "
                 f"config.YOLO.PT / MODEL_DIR.")
    print(f"[export] loading YOLO weights: {C.YOLO.PT}")
    model = YOLO(str(C.YOLO.PT))

    print(f"[export] exporting ONNX  imgsz={C.YOLO.IMGSZ}  opset={C.ONNX_OPSET}  "
          f"(static batch=1, nms=False)")
    out_path = model.export(
        format="onnx",
        imgsz=C.YOLO.IMGSZ,
        opset=C.ONNX_OPSET,
        dynamic=False,     # static shapes — required for clean TIDL offload
        simplify=True,     # fold constants / clean the graph (onnxslim)
        nms=False,         # raw head; NMS done in numpy at runtime
        batch=1,
    )

    C.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    shutil.copy(out_path, C.YOLO.ONNX)
    print(f"[export] wrote → {C.YOLO.ONNX}")
    print("[export] next: python compile/compile_yolo_tidl.py")


if __name__ == "__main__":
    main()
