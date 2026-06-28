#!/usr/bin/env python3
"""
Stage 1.5 (PC) — head-truncate the YOLO / UFLD ONNX for the board's TIDL.

The board's 0x20250429 TIDL firmware cannot VERIFY two specific head subgraphs, so
they're cut from the graph and re-implemented in numpy on the A72 at runtime:
  * yolo26n's DFL/Reshape detection head → cut to the 6 raw conv outputs
    (3 scales × {box [1,4,H,W] ltrb, cls [1,nc,H,W] logits}); anchor-decode +
    sigmoid run in yolo_runtime._decode_truncated.
  * UFLDv2's 4× Slice+Reshape head       → cut to the single flat `linear_1`
    [1,91224]; ufld_runtime splits it into the 4 head tensors.
The board runtime auto-prefers the `*_trunc.onnx` when it exists (YOLO/UFLD config
TRUNC_ONNX). This is a pure ONNX graph surgery (onnx.utils.extract_model) on the
full export — NO retrain, NO weight change; the trunc model reuses the same
external `.onnx.data` as the full UFLD export.

Run order (PC, after the full exports):
    python export/export_yolo_onnx.py
    python export/export_ufld_onnx.py
    python export/truncate_onnx.py            # writes *_trunc.onnx beside them
    python compile/compile_yolo_tidl.py       # compiles the trunc (auto-preferred)
    python compile/compile_ufld_tidl.py

UFLD's cut tensor is known (`linear_1`). YOLO's 6 conv outputs are auto-discovered
(Conv outputs with 4 or nc channels at the 3 stride scales) and printed for you to
sanity-check; override with --yolo-outputs name1,name2,... if your export differs
(open the onnx in netron and pick the conv outputs feeding the DFL/Reshape head).

Needs onnx on the PC:  pip install onnx
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # bbai64-deploy/
import config as C  # noqa: E402


def _onnx():
    try:
        import onnx
        from onnx import shape_inference, utils
        return onnx, shape_inference, utils
    except Exception as e:  # noqa: BLE001
        sys.exit(f"[truncate] onnx not available ({e}); pip install onnx (PC only).")


def _input_name(model) -> str:
    inits = {i.name for i in model.graph.initializer}
    ins = [i.name for i in model.graph.input if i.name not in inits]
    return ins[0] if ins else model.graph.input[0].name


def truncate_ufld() -> None:
    onnx, shape_inference, utils = _onnx()
    src, dst = C.UFLD.ONNX, C.UFLD.TRUNC_ONNX
    if not src.exists():
        print(f"[truncate] UFLD full ONNX missing ({src}); run export_ufld_onnx.py — skipping")
        return
    model = onnx.load(str(src), load_external_data=False)
    iname = _input_name(model)
    names = {n for node in model.graph.node for n in node.output}
    if "linear_1" not in names:
        sys.exit("[truncate] UFLD: tensor 'linear_1' not found — open the onnx in "
                 "netron and pass the FC output name (the tensor just before the "
                 "4× Slice/Reshape head).")
    # extract_model keeps the external-weights reference (same .onnx.data).
    utils.extract_model(str(src), str(dst), [iname], ["linear_1"])
    print(f"[truncate] UFLD → {dst.name}  (input '{iname}' → output 'linear_1' [1,91224])")


def _discover_yolo_outputs(model, shape_inference) -> list:
    """Conv node outputs whose channel count is 4 (box) or nc (cls), at the 3
    detection scales (H = imgsz/8, /16, /32). Returns up to 6 tensor names."""
    nc = C.YOLO.NC or 8
    imgsz = C.YOLO.IMGSZ
    scales = {imgsz // s for s in (8, 16, 32)}
    inferred = shape_inference.infer_shapes(model)
    shp = {vi.name: [d.dim_value for d in vi.type.tensor_type.shape.dim]
           for vi in list(inferred.graph.value_info) + list(inferred.graph.output)}
    out = []
    for node in model.graph.node:
        if node.op_type != "Conv":
            continue
        o = node.output[0]
        s = shp.get(o)
        if s and len(s) == 4 and s[1] in (4, nc) and s[2] in scales:
            out.append(o)
    return out


def truncate_yolo(explicit: list | None) -> None:
    onnx, shape_inference, utils = _onnx()
    src, dst = C.YOLO.ONNX, C.YOLO.TRUNC_ONNX
    if not src.exists():
        print(f"[truncate] YOLO full ONNX missing ({src}); run export_yolo_onnx.py — skipping")
        return
    model = onnx.load(str(src))
    iname = _input_name(model)
    outs = explicit or _discover_yolo_outputs(model, shape_inference)
    if len(outs) != 6:
        sys.exit(f"[truncate] YOLO: expected 6 conv head outputs, found {len(outs)}: "
                 f"{outs}\n  Pass them explicitly: --yolo-outputs n1,n2,...,n6 "
                 f"(the 3×{{box[1,4,H,W], cls[1,{C.YOLO.NC},H,W]}} conv outputs).")
    utils.extract_model(str(src), str(dst), [iname], outs)
    print(f"[truncate] YOLO → {dst.name}  (input '{iname}' → 6 conv outputs: {outs})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Head-truncate YOLO/UFLD ONNX for board TIDL.")
    ap.add_argument("--only", choices=["yolo", "ufld"], help="truncate just one model")
    ap.add_argument("--yolo-outputs", help="comma-separated 6 conv output names (override)")
    args = ap.parse_args()
    yolo_outs = args.yolo_outputs.split(",") if args.yolo_outputs else None
    if args.only != "ufld":
        truncate_yolo(yolo_outs)
    if args.only != "yolo":
        truncate_ufld()
    print("[truncate] done. Next: compile/compile_yolo_tidl.py + compile/compile_ufld_tidl.py")


if __name__ == "__main__":
    main()
