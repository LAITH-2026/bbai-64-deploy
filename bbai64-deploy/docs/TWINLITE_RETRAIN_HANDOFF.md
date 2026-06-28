# TwinLiteNet decoder retrain handoff — fixing the saturated masks

**Audience:** whoever owns the TwinLiteNet model (PC/training side).
**Status:** TwinLiteNet **runs** on the board (C7x, 127/127 nodes, ~70 FPS, no
reset) but the **masks are not deployment-accurate yet**. FPS/latency KPIs are
valid; geometry is not. This doc is the fix.

---

## 1. Symptom

On-board, the drivable-area (`da`) and lane-line (`ll`) masks come out
**saturated** — nearly all-positive or all-negative after the per-pixel argmax —
instead of tracking the road/lane. The float ONNX (PC, onnxruntime CPU) produces
correct masks, so this is a **quantisation** problem introduced at TIDL compile,
not a model/preprocess bug.

Confirm with `board_twin.py` on the board: `drivable-area px` / `lane px` are
non-zero (segmentation "works") but visually the overlay floods or vanishes.

## 2. Root cause

The decoder upsamples with **`ConvTranspose2d`**. On the board's `0x20250429`
firmware TIDL mis-quantises ConvTranspose (the logits overflow the INT8 range), so
the 2-channel head logits saturate and argmax collapses. The encoder (ESPNet-C,
plain convs) quantises fine — only the transposed-conv decoder is affected.

## 3. The fix — replace ConvTranspose with Resize + Conv, then retrain

TIDL quantises **Resize (bilinear/nearest) + Conv2d** cleanly. Swap every decoder
`ConvTranspose2d(C_in, C_out, k, stride=2)` for:

```python
# was: nn.ConvTranspose2d(C_in, C_out, kernel_size=2, stride=2)
nn.Sequential(
    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # or "nearest"
    nn.Conv2d(C_in, C_out, kernel_size=3, padding=1),
)
```

Keep channel counts and the two output heads (`da`, `ll`) unchanged. Then:

1. **Fine-tune**, don't train from scratch — only the decoder changed, so warm-start
   from the current weights and fine-tune on the TwinLite BDD100K recipe (lane +
   drivable-area) for a handful of epochs until val IoU recovers. (A full retrain
   also works if warm-start is awkward.)
2. Keep the **attention block removed** (the "noattn" variant) — the Dual-Attention
   PAM/CAM ReduceMax/Sub/Expand ops hang the TIDL perfsim and were dropped near-
   losslessly. Do not reintroduce them.

## 4. Re-export → re-compile → verify (unchanged contract)

The board runtime and `config.TWINLITE` must keep working with **no code change**,
so preserve this export contract exactly:

| | value |
|---|---|
| input name | `images` |
| input shape | `1×3×360×640` (static, NCHW) |
| preprocess | resize 640×360, RGB, `/255` (NO ImageNet norm) |
| outputs | `da`, `ll` — each `[1,2,H,W]` raw logits |
| ops | conv-only; no attention; **no ConvTranspose** |

Then:

```bash
# PC
export BBAI64_TWINLITE_SRC=/path/to/TwinLiteNet  BBAI64_TWINLITE_WEIGHTS=/path/to/retrained.pth
python export/export_twinlite_onnx.py            # → artifacts/twinlite_noattn.onnx
python compile/compile_twinlite_tidl.py          # → artifacts/twinlite_tidl/
#   copy artifacts to the board
# Board
python board_twin.py        # masks should now be geometrically correct, not saturated
```

## 5. Acceptance check

- Float ONNX vs INT8-TIDL mask **IoU** ≥ ~0.9 on a held-out clip (the current
  saturated state scores near 0 or near 1-everywhere).
- On-board overlay (`runtime/app.py --source image --image test.jpg`) shows the
  green drivable area on the road surface and red lane lines on the markings.

## 6. Stop-gap if a retrain isn't possible yet

- Try a 16-bit compile (`BBAI64_TENSOR_BITS=16 python compile/compile_twinlite_tidl.py`):
  more headroom for the ConvTranspose logits — **may** reduce saturation, but does
  not reliably fix it. Measure IoU before trusting it.
- Until fixed, treat the lane overlay as **non-authoritative** (the FPS/latency
  numbers and the rest of the pipeline are valid); the YOLO + depth ADAS path does
  not depend on the lane masks.
