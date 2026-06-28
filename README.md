# bbai-64-deploy

On-board deployment of a three-feature perception stack — **object detection**
(fine-tuned YOLO), **lane detection** (UFLDv2), and **monocular metric depth**
(per-object distance in metres, by closed-form camera geometry) — for the
**BeagleBone AI-64 (TI TDA4VM)**. The two CNNs run on the **C7x+MMA** accelerator
via **TIDL**; depth is **A72 geometry, not a model** (Depth-Anything-V2 was a ViT
TIDL couldn't offload — see `bbai64-deploy/README.md`). Fed by **MQTT frames from
CARLA**.

This repository is a **self-contained slice** of the larger "Connected Intelligent
Vehicle Platform" graduation project: the deploy code plus exactly the
dependencies it needs to run export → compile → runtime.

> **Start here:** the full deployment guide lives in
> [`bbai64-deploy/README.md`](bbai64-deploy/README.md). This top-level file only
> explains the repo layout and the one asset you must fetch separately.

## Layout

```
bbai-64-deploy/
├── bbai64-deploy/                  ← the deployment project (all the code + docs)
│   ├── config.py                   single source of truth (paths/shapes/thresholds)
│   ├── export/  compile/  runtime/ the 3 stages (export → compile → board runtime)
│   ├── preprocess.py               shared numpy preprocessing (calibration == runtime)
│   └── README.md  docs/            full guide + Yocto image requirements
├── Fined-Tuned-Model/              ← the deployable object-detection model
│   ├── best.pt                     fine-tuned yolo26n, 8-class CARLA (committed, 15 MB)
│   └── data.yaml                   class names (read at runtime — single source)
└── Ultra-Fast-Lane-Detection-v2/   ← lane-model SOURCE needed only by the export step
    ├── ufld_inference.py  model/  utils/  configs/
    └── (culane_res18.pth is NOT here — see below)
```

`config.py` finds `Fined-Tuned-Model/` and `Ultra-Fast-Lane-Detection-v2/` as
**siblings of `bbai64-deploy/`**, which is exactly this layout — so nothing needs
repointing after cloning.

## What is NOT in the repo (and why)

| Asset | Size | How to get it |
|-------|-----:|---------------|
| `Ultra-Fast-Lane-Detection-v2/culane_res18.pth` | ~825 MB | The standard UFLDv2 **CULane / ResNet-18** checkpoint. Download from the [Ultra-Fast-Lane-Detection-v2 release](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2) and drop it in that folder. **Only the PC lane-export step needs it; the board does not.** |
| `bbai64-deploy/artifacts/` | tens of MB | Produced by the compile stage (`compile/compile_*_tidl.py`) and **version-locked to the board firmware** — rebuild per SDK version, never commit. |
| `bbai64-deploy/calib/` | — | Calibration frames extracted from a CARLA clip (`compile/prepare_calib.py`). |

> Depth needs no asset at all — it is closed-form geometry from the CARLA camera
> parameters (`config.py` `class DEPTH`), not a downloaded checkpoint.

> The board runtime needs **none** of the `.pth`/`.pt` weights or PyTorch — it
> consumes the compiled TIDL artifacts plus `Fined-Tuned-Model/data.yaml`
> (class names). The weights and lane source here exist only to re-run the PC
> **export** stage.

## Quick start

See [`bbai64-deploy/README.md`](bbai64-deploy/README.md) for the full run order.
In brief, on an x86_64 Linux / WSL2 PC:

```bash
# (fetch culane_res18.pth into Ultra-Fast-Lane-Detection-v2/ first)
python bbai64-deploy/export/export_yolo_onnx.py
python bbai64-deploy/export/export_ufld_onnx.py
python bbai64-deploy/compile/prepare_calib.py --video <carla_clip.mp4> --n 25
python bbai64-deploy/compile/compile_yolo_tidl.py
python bbai64-deploy/compile/compile_ufld_tidl.py
#   → copy bbai64-deploy/artifacts/ to the board, then:  python3 runtime/app.py
#   (depth = A72 geometry; no export/compile, just set config.py DEPTH for CARLA)
```
