#!/usr/bin/env python3
"""
Stage 1b (PC) — export TwinLiteNet → fixed-shape, ATTENTION-FREE ONNX for TIDL.

⚠️ HANDOFF / RECIPE (not yet turnkey). The committed board artifact
`artifacts/twinlite_noattn.onnx` was produced from the upstream TwinLiteNet repo
with the Dual-Attention (PAM/CAM) block removed — those ReduceMax/Sub/Expand ops
hang the TIDL perfsim, and dropping them is near-lossless. This script records the
exact export so the artifact can be reproduced, but it needs the TwinLiteNet source
+ checkpoint, which are NOT vendored here. Point it at them via:

    export BBAI64_TWINLITE_SRC=/path/to/TwinLiteNet        # repo with model.py
    export BBAI64_TWINLITE_WEIGHTS=/path/to/best.pth        # trained checkpoint
    python export/export_twinlite_onnx.py

Export contract the board runtime depends on (config.TWINLITE / twinlite_runtime):
  * input  name "images", static shape 1×3×TWIN_H×TWIN_W (360×640), NCHW
  * outputs named "da" and "ll", each [1,2,H,W] raw logits (drivable / lane)
  * preprocessing is a bare /255 resize (NO ImageNet norm) — see preprocess_twinlite
  * the attention block MUST be removed (or the perfsim hangs at compile)

ACCURACY NOTE: the decoder uses ConvTranspose, which the current board firmware
mis-quantises (saturated masks). The deployment fix is a decoder retrain with
Resize+Conv — see TWINLITE_RETRAIN_HANDOFF.md. Re-run this export after that retrain.

Run on x86_64 Linux / WSL2 with torch installed. NEVER on the board.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # bbai64-deploy/
import config as C  # noqa: E402


def main() -> None:
    src = os.environ.get("BBAI64_TWINLITE_SRC")
    weights = os.environ.get("BBAI64_TWINLITE_WEIGHTS")
    if not src or not Path(src).exists():
        sys.exit(
            "[export-twinlite] TwinLiteNet source not found.\n"
            "  Set BBAI64_TWINLITE_SRC=/path/to/TwinLiteNet (the repo with the\n"
            "  model definition) and BBAI64_TWINLITE_WEIGHTS=/path/to/best.pth,\n"
            "  then re-run. The committed artifacts/twinlite_noattn.onnx was made\n"
            "  this way with the Dual-Attention block removed (see module docstring).")
    if not weights or not Path(weights).exists():
        sys.exit("[export-twinlite] set BBAI64_TWINLITE_WEIGHTS to the trained .pth")

    try:
        import torch  # lazy import keeps config torch-free
    except Exception as e:  # noqa: BLE001
        sys.exit(f"[export-twinlite] torch not available ({e}); run on the PC venv.")

    sys.path.insert(0, str(src))
    try:
        # Upstream module/class names vary by fork; adjust this import to match the
        # repo at BBAI64_TWINLITE_SRC. The model must already have its PAM/CAM
        # attention removed (the "noattn" variant) before export.
        from model import TwinLiteNet  # type: ignore  # noqa: E402
    except Exception as e:  # noqa: BLE001
        sys.exit(f"[export-twinlite] cannot import TwinLiteNet from {src} ({e}).\n"
                 f"  Edit this import to match your fork's model module/class, and\n"
                 f"  ensure the Dual-Attention block is removed (the noattn variant).")

    model = TwinLiteNet().eval()
    sd = torch.load(weights, map_location="cpu")
    model.load_state_dict(sd.get("state_dict", sd), strict=False)

    dummy = torch.zeros(1, 3, C.TWINLITE.TWIN_H, C.TWINLITE.TWIN_W)
    C.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    print(f"[export-twinlite] exporting ONNX in=1x3x{C.TWINLITE.TWIN_H}x{C.TWINLITE.TWIN_W} "
          f"opset={C.ONNX_OPSET} → {C.TWINLITE.ONNX}")
    with torch.no_grad():
        torch.onnx.export(
            model, dummy, str(C.TWINLITE.ONNX),
            input_names=["images"],
            output_names=[C.TWINLITE.DA_OUTPUT, C.TWINLITE.LL_OUTPUT],  # "da","ll"
            opset_version=C.ONNX_OPSET,
            do_constant_folding=True,
            dynamic_axes=None,           # fully static — required for clean offload
        )
    print(f"[export-twinlite] wrote → {C.TWINLITE.ONNX}")
    print("[export-twinlite] next: python compile/compile_twinlite_tidl.py")


if __name__ == "__main__":
    main()
