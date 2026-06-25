"""
Shared TIDL compilation helper (PC, x86_64 Linux / WSL2).

Compilation = run the ONNX model through onnxruntime's TIDLCompilationProvider on
a handful of calibration frames. This (a) decides which layers offload to C7x vs
fall back to A72, (b) quantizes weights/activations to INT8 using the calibration
statistics, and (c) writes the artifacts the board loads. Output → artifacts_dir.

Requires:  pip install onnxruntime-tidl  (TI fork)  and  TIDL_TOOLS_PATH set.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, List, Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # bbai64-deploy/
import config as C  # noqa: E402


def list_calib_images() -> List[Path]:
    imgs = sorted(C.CALIB_DIR.glob("*.jpg")) + sorted(C.CALIB_DIR.glob("*.png"))
    if not imgs:
        sys.exit(f"[compile] no calibration frames in {C.CALIB_DIR} — "
                 f"run: python compile/prepare_calib.py --video <clip.mp4>")
    return imgs


def compile_model(
    onnx_path: Path,
    artifacts_dir: Path,
    preprocess_fn: Callable[[np.ndarray], object],
    deny_list: Optional[str] = None,
    tensor_bits: Optional[int] = None,
) -> None:
    try:
        import onnxruntime as ort
    except Exception as e:  # noqa: BLE001
        sys.exit(f"[compile] onnxruntime not importable ({e})")

    if "TIDLCompilationProvider" not in ort.get_available_providers():
        sys.exit("[compile] TIDLCompilationProvider not available — install TI's "
                 "fork (onnxruntime-tidl from edgeai-tidl-tools) and source its env. "
                 f"providers={ort.get_available_providers()}")
    if not C.TIDL_TOOLS_PATH:
        sys.exit("[compile] TIDL_TOOLS_PATH not set — "
                 "export TIDL_TOOLS_PATH=/opt/edgeai-tidl-tools/tidl_tools")
    if not Path(onnx_path).exists():
        sys.exit(f"[compile] missing ONNX: {onnx_path} — run the export step first.")

    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    calib = list_calib_images()

    bits = C.TENSOR_BITS if tensor_bits is None else tensor_bits   # per-model override
    options = {
        "tidl_tools_path": C.TIDL_TOOLS_PATH,
        "artifacts_folder": str(artifacts_dir),
        "tensor_bits": bits,                      # 8 = INT8, 16 = INT16
        "accuracy_level": 1,                      # 1 = advanced calibration
        "advanced_options:calibration_frames": len(calib),
        "advanced_options:calibration_iterations": 3,
        "debug_level": 1,
    }
    if deny_list:
        # Comma-separated op types to KEEP on the A72 (e.g. "LayerNormalization")
        # if they hurt accuracy or aren't supported by your TIDL version.
        options["deny_list"] = deny_list

    sess = ort.InferenceSession(
        str(onnx_path),
        providers=["TIDLCompilationProvider", "CPUExecutionProvider"],
        provider_options=[options, {}],
    )
    iname = sess.get_inputs()[0].name
    print(f"[compile] {Path(onnx_path).name}: calibrating on {len(calib)} frames "
          f"(INT{bits}, input '{iname}')")

    for i, p in enumerate(calib, 1):
        bgr = cv2.imread(str(p))
        if bgr is None:
            print(f"  [skip] unreadable {p.name}")
            continue
        inp = preprocess_fn(bgr)
        if isinstance(inp, tuple):       # preprocess_yolo returns (tensor, meta)
            inp = inp[0]
        sess.run(None, {iname: np.ascontiguousarray(inp, dtype=np.float32)})
        print(f"  calib {i}/{len(calib)}  {p.name}")

    print(f"[compile] artifacts → {artifacts_dir}")
    print("[compile] check the layer table above: anything NOT on C7x ran on A72.")
