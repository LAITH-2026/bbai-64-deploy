#!/usr/bin/env python3
"""
Stage 2c (PC) — compile TwinLiteNet ONNX → TIDL artifacts (INT8 PTQ).
    export TIDL_TOOLS_PATH=/opt/edgeai-tidl-tools/tidl_tools
    python compile/compile_twinlite_tidl.py

Compiles the ATTENTION-FREE export `artifacts/twinlite_noattn.onnx` (the Dual-
Attention PAM/CAM ReduceMax/Sub/Expand ops hang the perfsim; removing them is
near-lossless). It is conv-only and offloads 127/127 nodes to the C7x — no
deny_list needed. Input is a bare /255 resize to TWIN_W×TWIN_H (preprocess_twinlite),
matching the BDD100K TwinLite recipe (NO ImageNet normalisation).

⚠️ KNOWN ISSUE (0x20250429 firmware): TIDL mis-quantises the decoder's
ConvTranspose (logits overflow) → saturated/inaccurate masks. Latency/FPS are
valid; mask geometry is not yet deployment-accurate. The fix is a decoder retrain
(Resize+Conv instead of ConvTranspose) — see TWINLITE_RETRAIN_HANDOFF.md.
INT16 for the decoder ONLY is worth a try if you must improve masks pre-retrain:
  BBAI64_TENSOR_BITS=16 python compile/compile_twinlite_tidl.py   (whole net at 16)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # bbai64-deploy/
import config as C            # noqa: E402
import preprocess as P        # noqa: E402
from tidl_common import compile_model  # noqa: E402

if __name__ == "__main__":
    compile_model(
        C.TWINLITE.ONNX,
        C.TWINLITE.TIDL_DIR,
        P.preprocess_twinlite,
        # conv-only attention-free graph — no deny_list required.
    )
