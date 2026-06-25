#!/usr/bin/env python3
"""
Stage 1b (PC) — export UFLDv2 (CULane/ResNet-18) → fixed-shape ONNX for TIDL.

The network (ufld_inference.ParsingNet) is a plain CNN: ResNet-18 → Conv(512→8) →
flatten → LayerNorm → Linear → ReLU → Linear → reshape into 4 tensors. No
GridSample / custom ops, so it offloads cleanly. The messy decode (`pred2coords`)
is NOT exported — it runs on the A72 in numpy (runtime/pred2coords_np.py).

We export a thin wrapper that returns the 4 head tensors as an ordered TUPLE
(loc_row, loc_col, exist_row, exist_col) — ONNX outputs must be tensors, not the
dict the PyTorch forward returns.

Run on x86_64 Linux / WSL2 with the lane venv (torch + torchvision). NEVER on the
board.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # bbai64-deploy/
import config as C  # noqa: E402

# UFLD repo must be importable: ufld_inference → utils.config, model.backbone.
sys.path.insert(0, str(C.UFLD_ROOT))


def main() -> None:
    try:
        import torch  # lazy import keeps config torch-free
    except Exception as e:  # noqa: BLE001
        sys.exit(f"[export-ufld] torch not available ({e}); run on the lane venv.")
    try:
        from ufld_inference import ParsingNet, build_cfg  # noqa: E402
    except Exception as e:  # noqa: BLE001
        sys.exit(f"[export-ufld] cannot import UFLD repo ({e}); check UFLD_ROOT in "
                 f"config.py points at Ultra-Fast-Lane-Detection-v2.")

    if not C.UFLD.WEIGHTS.exists():
        sys.exit(f"[export-ufld] missing checkpoint: {C.UFLD.WEIGHTS}")
    if not Path(C.UFLD.CONFIG).exists():
        sys.exit(f"[export-ufld] missing config: {C.UFLD.CONFIG}")

    cfg = build_cfg(str(C.UFLD.CONFIG))
    # NOTE: ParsingNet builds resnet(pretrained=True) → may download ImageNet
    # weights once on a fresh machine; they are immediately overwritten by the
    # checkpoint load below, so the download is harmless (needs internet once).
    net = ParsingNet(
        backbone=cfg.backbone,
        num_grid_row=cfg.num_cell_row,
        num_cls_row=cfg.num_row,
        num_grid_col=cfg.num_cell_col,
        num_cls_col=cfg.num_col,
        num_lane_on_row=cfg.num_lanes,
        num_lane_on_col=cfg.num_lanes,
        input_height=cfg.train_height,
        input_width=cfg.train_width,
        fc_norm=getattr(cfg, "fc_norm", False),
    ).cpu()

    state = torch.load(str(C.UFLD.WEIGHTS), map_location="cpu")["model"]
    compatible = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    net.load_state_dict(compatible, strict=False)
    net.eval()

    class Wrap(torch.nn.Module):
        """Return the 4 head tensors as a tuple (ONNX-friendly)."""

        def __init__(self, m: torch.nn.Module) -> None:
            super().__init__()
            self.m = m

        def forward(self, x):  # noqa: D401
            d = self.m(x)
            return d["loc_row"], d["loc_col"], d["exist_row"], d["exist_col"]

    wrapped = Wrap(net).eval()
    dummy = torch.zeros(*C.UFLD.IN_SHAPE)   # 1×3×320×1600

    C.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    print(f"[export-ufld] exporting ONNX  in={C.UFLD.IN_SHAPE}  opset={C.ONNX_OPSET}")
    with torch.no_grad():
        torch.onnx.export(
            wrapped,
            dummy,
            str(C.UFLD.ONNX),
            input_names=[C.UFLD.IN_NAME],
            output_names=list(C.UFLD.OUT_NAMES),
            opset_version=C.ONNX_OPSET,
            do_constant_folding=True,
            dynamic_axes=None,           # fully static
        )
    print(f"[export-ufld] wrote → {C.UFLD.ONNX}")
    print("[export-ufld] decode (pred2coords) stays on A72 — not in the graph.")
    print("[export-ufld] next: python compile/compile_ufld_tidl.py")


if __name__ == "__main__":
    main()
