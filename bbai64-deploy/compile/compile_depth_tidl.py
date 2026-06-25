#!/usr/bin/env python3
"""
Stage 2d (PC) — compile Depth-Anything-V2 ONNX → TIDL artifacts (INT8 PTQ).
    export TIDL_TOOLS_PATH=/opt/edgeai-tidl-tools/tidl_tools
    python compile/compile_depth_tidl.py

⚠️ This is a ViT/DINOv2 transformer — the riskiest model to offload:
  * Read the layer table the compile prints. Many ViT ops may land on the A72
    depending on your TIDL version; that is OK (runtime CPU EP handles them) but
    costs latency. This model time-shares the one C7x with YOLO + UFLD.
  * If INT8 depth is inaccurate (ViTs are quantization-sensitive), recompile at
    16-bit for THIS model only:  BBAI64_DEPTH_BITS=16 python compile/compile_depth_tidl.py
    (config.DEPTH.TENSOR_BITS picks it up; YOLO/UFLD stay INT8).
  * If specific transformer ops hurt accuracy, keep them on the A72 via the
    deny_list arg below (e.g. "LayerNormalization", "Softmax").
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # bbai64-deploy/
import config as C            # noqa: E402
import preprocess as P        # noqa: E402
from tidl_common import compile_model  # noqa: E402

if __name__ == "__main__":
    compile_model(
        C.DEPTH.ONNX,
        C.DEPTH.TIDL_DIR,
        P.preprocess_depth,
        tensor_bits=C.DEPTH.TENSOR_BITS,   # per-model (INT8 default; 16 if needed)
        # deny_list="LayerNormalization,Softmax",  # uncomment to pin ViT ops to A72
    )
