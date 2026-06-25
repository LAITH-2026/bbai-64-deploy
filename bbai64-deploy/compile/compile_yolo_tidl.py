#!/usr/bin/env python3
"""
Stage 2b (PC) — compile YOLO ONNX → TIDL artifacts (INT8 PTQ).
    export TIDL_TOOLS_PATH=/opt/edgeai-tidl-tools/tidl_tools
    python compile/compile_yolo_tidl.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # bbai64-deploy/
import config as C            # noqa: E402
import preprocess as P        # noqa: E402
from tidl_common import compile_model  # noqa: E402

if __name__ == "__main__":
    compile_model(C.YOLO.ONNX, C.YOLO.TIDL_DIR, P.preprocess_yolo)
