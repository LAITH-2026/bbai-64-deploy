# bbai64-deploy — YOLO + UFLDv2 + Depth-Anything-V2 on BeagleBone AI-64 (TDA4VM)

Deploys the three graduation-project ML features — **object detection** (YOLO,
the fine-tuned 8-class CARLA model in `Fined-Tuned-Model/`), **lane detection**
(UFLDv2), and **monocular metric depth** (Depth-Anything-V2 → per-object distance
in metres) — onto the BeagleBone AI-64's TDA4VM, running on the **C7x+MMA
deep-learning accelerator** via TIDL, fed by **MQTT frames from CARLA**, rendering
to a display and publishing **ADAS alerts** for the Qt infotainment.

On-device successor to `Integrate-Features/integrate.py` (which ran all three
models sequentially on a Windows CUDA GPU). Depth is **ON by default** and adds a
`depth_m` field to every detection; disable it with `--no-depth` for a leaner
YOLO+UFLD pipeline.

> **Packaging:** native-on-board first; Docker is deferred. The `runtime/` code is
> container-agnostic, so wrapping it in a container later is a drop-in with no code
> changes (see *Deferred: Docker* below).

---

## Hardware reality (read this first)

The TDA4VM has **one** C7x+MMA (8 TOPS), **not two**. The three models therefore
**time-share a single accelerator** — inference is serial, so combined C7x latency
= `yolo + ufld + depth`; there is no `max(...)` latency halving. The win over the
Windows/CUDA baseline comes from:

1. **INT8 quantization on the MMA** — the dominant speedup.
2. **Pipeline overlap** — A72 cores run preprocessing / NMS / `pred2coords` /
   depth box-sampling / compositing for one frame while the C7x runs inference for
   the next. The single-GPU `integrate.py` pays all of that serially; here it is
   hidden.

So "running all three concurrently" on this chip = a **3-stage software pipeline**
(capture → infer → decode/composite), implemented in `runtime/app.py`.

> **Depth is the heavy one.** Depth-Anything-V2 is a ViT/DINOv2 transformer; it is
> the largest model and the riskiest to offload (TIDL transformer support is
> version-dependent). Run with `--no-depth` if the live ADAS FPS budget can't
> absorb a third serial model on the one C7x.

---

## Three stages, two machines

```
┌──────────────────── x86_64 Linux / WSL2 (your PC) ─────────────────────┐
│ 1. EXPORT    PyTorch/HF → ONNX (fixed shape, batch 1)                    │
│    export/export_yolo_onnx.py  → artifacts/yolo26n_carla8.onnx           │
│    export/export_ufld_onnx.py  → artifacts/ufld_culane_res18.onnx        │
│    export/export_depth_onnx.py → artifacts/depth_anything_v2_metric_s.onnx│
│                                                                          │
│ 2. COMPILE   ONNX → TIDL artifacts (INT8 PTQ + calibration)              │
│    compile/prepare_calib.py     → calib/*.jpg (frames from a CARLA clip) │
│    compile/compile_yolo_tidl.py → artifacts/yolo_tidl/                   │
│    compile/compile_ufld_tidl.py → artifacts/ufld_tidl/                   │
│    compile/compile_depth_tidl.py→ artifacts/depth_tidl/                  │
└──────────────────────────────────────────────────────────────────────── ┘
                              │  copy ./artifacts to the board
                              ▼
┌─────────────────────── BeagleBone AI-64 (native) ──────────────────────┐
│ 3. RUNTIME   onnxruntime + TIDLExecutionProvider, 3-stage pipeline       │
│    MQTT(CARLA) → preprocess → [C7x: YOLO, UFLD, then Depth] → numpy       │
│    decode (NMS · pred2coords · per-box depth) → composite overlay →       │
│    display  +  ADAS alerts (+depth_m) → MQTT (Qt infotainment)           │
│    runtime/app.py                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## ⚠️ Version pinning (the #1 cause of silent failure)

The `edgeai-tidl-tools` version used in **stage 2** MUST match the TIDL runtime
baked into the **board's Yocto image** (stage 3). Artifacts are version-locked to
the firmware. Pick one SDK version (e.g. `09.02`) and use it everywhere:

```bash
export TIDL_TOOLS_PATH=/opt/edgeai-tidl-tools/tidl_tools   # stage 2 (PC)
# board firmware: Yocto image built from processor-sdk-analytics 09.02
```

`config.py` is the single source of truth for shapes / paths / quantization, so
the two stages cannot drift.

---

## What runs where (what does NOT go on the board)

| Component                   | PC (export/compile) | Board (runtime) |
|-----------------------------|:-------------------:|:---------------:|
| torch / ultralytics         | ✅ YOLO export      | ❌ **never**    |
| torch + transformers        | ✅ depth export     | ❌ **never**    |
| edgeai-tidl-tools           | ✅                  | ❌              |
| onnxruntime (TIDL EP)       | ✅ compile EP       | ✅ exec EP      |
| numpy / opencv              | ✅                  | ✅              |
| paho-mqtt                   | —                   | ✅              |
| artifacts/                  | produced            | consumed        |

The board never imports PyTorch or transformers. `pred2coords`, YOLO NMS, and the
depth box-sampling are reimplemented in **pure numpy** (`runtime/pred2coords_np.py`,
`runtime/yolo_runtime.py`, `runtime/depth_runtime.py`).

---

## Swapping in a new trained model (no code edits)

The model identity is **data-driven**, not hard-coded:

- **YOLO** — drop the new `best.pt` + its `data.yaml` into `Fined-Tuned-Model/`
  (or repoint `YOLO.MODEL_DIR` in `config.py`). Class **names** are read from
  that `data.yaml` once at startup (`config.load_class_names`) and used by id at
  runtime, so labels and ADAS names follow automatically. Re-run export →
  compile → copy artifacts (the `data.yaml` ships with them). Per-class
  confidence lives in `config.py` (`PER_CLASS_CONF` / `CONF_DEFAULT`) and is
  overridable at runtime via `runtime/config.yaml` `thresholds:`.
- **UFLD** — repoint `UFLD.WEIGHTS` / `UFLD.CONFIG` in `config.py`. ⚠️ the head
  geometry constants in `config.py` (`NUM_ROW`, anchors, `CROP_RATIO`, …) are a
  hand-copy of `configs/culane_res18.py`; if you retrain with a different
  backbone/anchors, update them to match (still a single file to edit).
- **Depth** — set `BBAI64_DEPTH_MODEL` (or edit `DEPTH.HF_MODEL_ID`) to any HF
  `*-Metric-*-hf` Depth-Anything-V2 checkpoint, then re-export → compile. A
  different ViT size may change `INPUT_SIZE` (keep it a multiple of 14). The
  metric **Outdoor** head returns absolute metres; an **Indoor** or **relative**
  checkpoint would change the units/scale of every `depth_m`.

## Model facts (from the existing codebase)

| | YOLO (fine-tuned yolo26n, 8-class CARLA) | UFLDv2 (CULane res18) | Depth-Anything-V2 (Metric-Outdoor-Small) |
|---|---|---|---|
| ONNX input | `1×3×640×640`¹ | `1×3×320×1600` | `1×3×518×518` |
| Graph | conv backbone + detect head | ResNet-18 + Conv(512→8) + LayerNorm + MLP | **DINOv2 ViT-S/14 encoder + DPT depth head** |
| Output | raw head `[1,84,8400]` | 4 head tensors | metric depth map (metres) |
| On C7x (TIDL) | whole backbone+head | whole net | as much as TIDL takes (ViT ops may fall to A72) |
| On A72 (numpy) | NMS / box decode | `pred2coords` argmax+softmax decode | resize-to-frame + per-box median sampling |
| Preprocess | letterbox, RGB, /255 | resize→(533,1600), crop bottom 320, RGB, ImageNet norm | square-resize 518, RGB, ImageNet norm |
| Op(s) to watch | — | `LayerNorm` (fc_norm) | **transformer ops** (Attention/MatMul/Softmax/LayerNorm) — version-dependent offload; ViTs are quantization-sensitive (try INT16 via `BBAI64_DEPTH_BITS=16`) |

¹ YOLO input is a **user choice** (`BBAI64_YOLO_IMGSZ`, default 640). The fine-tuned
model was trained @1280 — set `1280` (or `960`) before export+compile for best
small-object/speed-sign accuracy, at ~4×/~2× the C7x cost. Keep the same value at
runtime (config reads it once so the two stages cannot disagree).

Normalization (mean/scale) is done in **numpy `preprocess.py` on the A72**, applied
identically at INT8 calibration and at runtime. It is **not** folded into the TIDL
compile config — adding mean/scale there would double-normalize and wreck accuracy.

---

## Run order

```bash
# ── PC (x86_64 Linux / WSL2) ──
python export/export_yolo_onnx.py
python export/export_ufld_onnx.py
python export/export_depth_onnx.py                      # needs torch + transformers
python compile/prepare_calib.py --video /path/to/carla_clip.mp4 --n 25
python compile/compile_yolo_tidl.py
python compile/compile_ufld_tidl.py
python compile/compile_depth_tidl.py                   # ViT — watch the layer table
#   → copy ./artifacts to the board (e.g. scp -r artifacts debian@bbai64:~/bbai64-deploy/)

# Optional: bump YOLO resolution before export+compile (keep set at runtime too)
#   export BBAI64_YOLO_IMGSZ=1280
# Optional: compile depth at INT16 if INT8 depth is inaccurate
#   export BBAI64_DEPTH_BITS=16   # then re-run compile/compile_depth_tidl.py

# ── Board (BeagleBone AI-64) ──
python3 runtime/app.py            # or ./run_native.sh   (depth ON by default)
python3 runtime/app.py --no-depth # YOLO + UFLD only (leaner real-time path)
```

---

## Input modes (what each produces)

The same pipeline serves three input types, selected by `--source`:

| Mode | Command | Outputs |
|------|---------|---------|
| **image** — single-frame pipeline check | `app.py --source image --image frame.png` | `frame_annotated.png` (boxes + lanes drawn) **+** `frame_result.json` (objects + lane points) |
| **video** — offline clip | `app.py --source video --video clip.mp4` | `clip_annotated.mp4` **+** `clip_result.json` (per-frame objects + lanes for the whole stream) **+** runtime KPIs |
| **mqtt** — live CARLA stream (default) | `app.py` | per-frame JSON **published to Qt over MQTT** (`adas/alerts`) continuously until stopped **+** runtime KPIs |

- Output paths default next to the input (`<stem>_annotated.*`, `<stem>_result.json`); override with `--out` / `--json`.
- `--display imshow` adds an optional live preview window in any mode; saved artifacts do not depend on it.
- The **JSON schema is identical** in all three modes (it is exactly the MQTT
  message): `{frame, timestamp, lanes_detected, lanes:[{index, points}], objects:[{class_id, class_name, confidence, bbox_xyxy, depth_m}]}`. Object boxes and lane points are both in **frame-pixel coordinates**; `depth_m` is the per-object metric distance in **metres** (`null` when depth is disabled or a box yields no valid depth samples).

---

## KPIs / analytics

The runtime documents performance two ways: a **per-frame** log
(`runtime/runtime_kpis.csv`, flushed every frame) and an **end-of-run** summary
(`runtime/runtime_analytics.txt`, min/max/avg/p95/p99). Metrics cover throughput
(avg + per-frame FPS, inference-bound ceiling), C7x/MMA utilization (duty cycle),
the full latency breakdown (A72 preprocess · YOLO · UFLD · **Depth** · A72
decode+composite), and memory (RSS avg/peak). With depth ON the combined-C7x line
is `yolo+ufld+depth`; the per-frame CSV gains a `depth_ms` column.

**What every KPI means, what it clarifies, and its honest limits is documented in
[docs/06-onboard-kpis.md](../docs/06-onboard-kpis.md).**

## Deferred: Docker

Containerization is intentionally postponed. When added, it is pure packaging:

- base image = **TI's edgeai runtime image** (so in-container TIDL libs match host
  firmware — same version-pinning rule);
- `docker run` passes through the C7x device nodes (`/dev/dri`, `/dev/dma_heap`,
  remoteproc) — no zero-copy path to break, since frames arrive as MQTT JPEGs;
- the `runtime/` code is unchanged.

Performance note: containers run the accelerator at **native speed**; the only real
costs are RAM/eMMC (keep the image lean) and device passthrough config.
