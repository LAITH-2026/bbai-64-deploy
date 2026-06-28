# Yocto Image Requirements — BeagleBone AI-64 (TDA4VM) ADAS runtime

**Audience:** the Embedded Linux engineer building the custom image.
**Goal:** produce an image that can run `bbai64-deploy/runtime/` natively — two
ML models (YOLO + UFLDv2) on the **C7x+MMA via TIDL**, fed by MQTT, with no
PyTorch on the device.

This is **native-first** (no Docker yet). Containerization is a later, additive
step and does not change anything below.

---

## 0. The one constraint that overrides everything: SDK version pinning

The TIDL **artifacts** we hand you are compiled on the PC with a specific
`edgeai-tidl-tools` version. The **TIDL runtime + C7x firmware** in your image
MUST be the matching version, or the models load as garbage / fail to init.

**Action — agree on a single SDK version and tell us back what you pick**, e.g.
`Processor SDK Linux for Edge AI / J721E, 09.02`. We will compile the artifacts
with the exact matching `edgeai-tidl-tools` tag. Everything below assumes that
one version is used consistently.

---

## 1. Start from TI's Edge AI image (don't assemble by hand)

The full inference stack already exists as a branded image target. Use TI's
`oe-layersetup` with the **processor-sdk-analytics** config for the agreed
version, then:

```sh
# conf/local.conf
ARAGO_BRAND = "edgeai"

# build
bitbake -k tisdk-edgeai-image
```

`tisdk-edgeai-image` already bundles: the **C7x/MMA TIDL runtime + DSP firmware**,
**onnxruntime with the TIDL execution provider**, OpenVX/TIOVX, GStreamer,
OpenCV, Python 3, numpy. Layers pulled in: `meta-ti-bsp`, `meta-arago`,
`meta-edgeai`.

> If you have a strong reason to base on `tisdk-default-image` instead, you must
> add the edgeai runtime recipes yourself (`onnxruntime`, TIDL runtime, firmware).
> Strongly prefer the edgeai image — it's the supported path.

---

## 2. Packages to ADD on top (our app's extra deps)

Append to your image recipe (or `local.conf`):

```bitbake
IMAGE_INSTALL:append = " \
    python3-core \
    python3-numpy \
    python3-opencv \
    python3-pyyaml \
    python3-paho-mqtt \
    python3-pillow \
"
```

Notes per package:

| Package | Why | If the recipe is missing |
|---|---|---|
| `python3-numpy` | all pre/post-processing, NMS, lane decode | should be in edgeai image already |
| `python3-opencv` | frame decode, resize, overlay drawing | verify it's the **Python** binding, not just libopencv |
| `python3-pyyaml` | reads `runtime/config.yaml` | optional; app falls back to defaults |
| `python3-paho-mqtt` | MQTT frame source + ADAS publish | from `meta-openembedded/meta-python`; if absent, `run_native.sh` pip-installs it to `--user` at first launch |
| `python3-pillow` | only if a transform path needs it | low priority |

**onnxruntime + TIDL EP must already be present** (from the edgeai image). Verify
in §5 — this is the single most important check.

---

## 3. Do NOT include these

```
torch / pytorch / torchvision      ← never on the board (~2 GB, unused)
ultralytics                        ← never; YOLO ships as a compiled artifact
transformers / huggingface_hub     ← never; not needed at all (no depth model —
                                     depth is closed-form geometry on the A72)
edgeai-tidl-tools / onnxruntime-tidl (compile build) ← PC-only
```

The board runs **compiled artifacts** (YOLO + UFLD) through the runtime
onnxruntime, not PyTorch/transformers, plus pure-numpy/math decode. Keeping the
heavy stacks out matters: the board has **4 GB RAM** shared with the inference
buffers. Depth adds essentially nothing — it is per-box arithmetic, not a model
with working buffers — so memory is dominated by the two CNNs' TIDL buffers.

---

## 4. Hardware / firmware enablement (must be on at boot)

These are default in `tisdk-edgeai-image` — confirm they survive any image
trimming:

- **C7x DSP + MMA firmware** loaded via **remoteproc** at boot (this *is* the
  deep-learning accelerator — without it everything falls back to the A72 and is
  unusably slow).
- **dma-heap / CMA** carveout for TIDL's contiguous buffers (kernel default for
  J721E; don't strip it).
- **MSMC / shared memory** between A72 and C7x (TI K3 default).

If you customize the kernel/devicetree, keep the J721E remoteproc + dma-heap +
TIDL reserved-memory nodes intact.

---

## 5. First-boot validation checklist (hand this back to us if any fail)

Run on the booted board:

```sh
# (a) C7x firmware is up
ls /dev/rpmsg* ; cat /sys/class/remoteproc/remoteproc*/state    # expect "running"

# (b) onnxruntime present WITH the TIDL execution provider
python3 -c "import onnxruntime as o; print(o.__version__); print(o.get_available_providers())"
#   → the list MUST contain 'TIDLExecutionProvider'

# (c) the app's Python deps import
python3 -c "import numpy, cv2, yaml, paho.mqtt.client; print('deps ok')"

# (d) RAM headroom
free -m
```

If (b) does **not** list `TIDLExecutionProvider`, the image is missing the TIDL
runtime — that blocks the entire project. Tell us before proceeding.

---

## 6. Display path (decide with us)

Our runtime can render the annotated frame three ways (`--display`):

- `none` — headless; only publishes ADAS alerts over MQTT (no GUI deps needed).
- `file` — writes an MP4 (bring-up/benchmarking; no GUI deps needed).
- `imshow` — `cv2.imshow` window, which needs **OpenCV built with a GUI/highgui
  backend** (Wayland/Qt). Headless edgeai images often omit this.

For the **dedicated ADAS display** in production, the cleaner route is to render
through the existing **Qt infotainment** app or a `kmssink`, not `cv2.imshow`. So
unless you specifically want `imshow` for debugging, **GUI-enabled OpenCV is not
required** — plan the display via Qt/KMS. Let's confirm which you'll wire up.

---

## 7. Filesystem / storage

- The image only needs the **runtime deps** above. The **TIDL artifacts** and our
  Python code are **not baked into the image** — we copy them to the rootfs after
  flashing (`scp -r bbai64-deploy <user>@board:~/`). Leave a few hundred MB free
  on the data partition; artifacts are small (tens of MB).
- 16 GB eMMC is plenty for the edgeai image + our app.

---

## 8. What we deliver to you vs. what you deliver to us

| We (ML side) deliver | You (BSP side) deliver |
|---|---|
| `bbai64-deploy/` source | The flashed image with §1–4 satisfied |
| TIDL artifacts compiled at the agreed SDK version | The **exact SDK version** you built (so we match it) |
| This document | §5 checklist results (esp. `TIDLExecutionProvider` present) |

---

## TL;DR for the engineer

1. Pick an Edge AI SDK version, **tell us which**.
2. `oe-layersetup` (processor-sdk-analytics, that version) → `ARAGO_BRAND="edgeai"`
   → `bitbake tisdk-edgeai-image`.
3. Add `python3-numpy python3-opencv python3-pyyaml python3-paho-mqtt` to
   `IMAGE_INSTALL`.
4. Keep C7x firmware/remoteproc/dma-heap enabled. Keep torch/ultralytics OUT.
5. Boot, run the §5 checks, confirm `TIDLExecutionProvider` is listed.
