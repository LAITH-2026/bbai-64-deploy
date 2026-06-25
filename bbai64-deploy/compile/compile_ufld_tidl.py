#!/usr/bin/env python3
"""
Stage 2c (PC) — compile UFLDv2 ONNX → TIDL artifacts (INT8 PTQ).
    export TIDL_TOOLS_PATH=/opt/edgeai-tidl-tools/tidl_tools
    python compile/compile_ufld_tidl.py

If lane accuracy drops after INT8 (thin lane features are quantization-sensitive):
  1. Quick check: BBAI64_TENSOR_BITS=16 python compile/compile_ufld_tidl.py
  2. Keep LayerNorm on the A72:  pass deny_list="LayerNormalization" below.
  3. Last resort: QAT with TI's edgeai-modeloptimization.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # bbai64-deploy/
import config as C            # noqa: E402
import preprocess as P        # noqa: E402
from tidl_common import compile_model  # noqa: E402

if __name__ == "__main__":
    compile_model(
        C.UFLD.ONNX,
        C.UFLD.TIDL_DIR,
        P.preprocess_ufld,
        # deny_list="LayerNormalization",   # uncomment if INT8 accuracy needs it
    )
