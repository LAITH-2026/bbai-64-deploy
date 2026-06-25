#!/usr/bin/env python3
"""
Stage 1c (PC) — export Depth-Anything-V2 (metric) → fixed-shape ONNX for TIDL.

The model (Hugging Face AutoModelForDepthEstimation) is a DINOv2 ViT encoder + a
DPT depth head. We export a thin wrapper that returns ONLY the metric depth
tensor (the HF forward returns a ModelOutput dataclass; ONNX outputs must be
plain tensors). The per-object distance sampling (`sample_box_depth`) is NOT
exported — it runs on the A72 in numpy at runtime (runtime/depth_runtime.py).

⚠️ This is a transformer. The export graph is large and TIDL's offload of ViT ops
is version-dependent — expect the compile step to place some layers on the A72.
That is acceptable (the runtime CPU EP handles them); it just costs latency. If
INT8 accuracy is poor, recompile with BBAI64_DEPTH_BITS=16.

Run on x86_64 Linux / WSL2 with torch + transformers installed. NEVER on the
board.
    pip install torch transformers
    python export/export_depth_onnx.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # bbai64-deploy/
import config as C  # noqa: E402


def main() -> None:
    try:
        import torch  # lazy import keeps config torch-free
        from transformers import AutoModelForDepthEstimation
    except Exception as e:  # noqa: BLE001
        sys.exit(f"[export-depth] torch/transformers not available ({e}); "
                 f"run on the PC export venv: pip install torch transformers")

    print(f"[export-depth] loading HF checkpoint '{C.DEPTH.HF_MODEL_ID}' "
          f"(first run downloads it) ...")
    model = AutoModelForDepthEstimation.from_pretrained(C.DEPTH.HF_MODEL_ID).cpu().eval()

    class Wrap(torch.nn.Module):
        """Return just the metric-depth tensor (ONNX-friendly)."""

        def __init__(self, m: torch.nn.Module) -> None:
            super().__init__()
            self.m = m

        def forward(self, pixel_values):  # noqa: D401
            out = self.m(pixel_values=pixel_values)
            d = out.predicted_depth                    # (1, H', W') metres
            if d.dim() == 3:
                d = d.unsqueeze(1)                     # → (1, 1, H', W')
            return d

    wrapped = Wrap(model).eval()
    dummy = torch.zeros(*C.DEPTH.IN_SHAPE)             # 1×3×518×518

    C.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    print(f"[export-depth] exporting ONNX  in={C.DEPTH.IN_SHAPE}  "
          f"opset={C.DEPTH.ONNX_OPSET}  (static batch=1)")
    with torch.no_grad():
        torch.onnx.export(
            wrapped,
            dummy,
            str(C.DEPTH.ONNX),
            input_names=[C.DEPTH.IN_NAME],
            output_names=[C.DEPTH.OUT_NAME],
            opset_version=C.DEPTH.ONNX_OPSET,
            do_constant_folding=True,
            dynamic_axes=None,           # fully static — required for clean offload
        )
    print(f"[export-depth] wrote → {C.DEPTH.ONNX}")
    print("[export-depth] per-object depth sampling stays on A72 — not in graph.")
    print("[export-depth] next: python compile/compile_depth_tidl.py")


if __name__ == "__main__":
    main()
