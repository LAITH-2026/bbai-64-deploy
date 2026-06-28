#!/usr/bin/env python3
"""
bbai64 runtime — concurrent YOLO + UFLDv2 on the TDA4VM, as a 3-stage software
pipeline (the faithful single-C7x realization of "one combined app, parallel"):

    Stage A  (thread)  capture frame  + preprocess BOTH models      [A72 + I/O]
    Stage B  (thread)  YOLO infer, then UFLD infer                  [C7x + MMA]
    Stage C  (thread)  NMS + pred2coords + composite + publish      [A72]

Because the three stages run on different threads, the A72 decode/compositing of
frame N overlaps the C7x inference of frame N+1 and the capture of N+2. The single
GPU `integrate.py` paid all of that serially; here throughput ≈ 1 / (C7x stage).

Source: MQTT JPEG frames from CARLA (default) or a local video file (bring-up).
Outputs: annotated display + ADAS alerts on MQTT + on-exit analytics summary.

Three input modes:
    python3 runtime/app.py --source image --image frame.png   # single-frame check
    python3 runtime/app.py --source video --video clip.mp4     # offline clip
    python3 runtime/app.py                                     # live CARLA stream (MQTT)

  image/video → write an annotated artifact + a JSON of all detections+lanes.
  mqtt        → publish per-frame JSON to Qt over MQTT + document runtime KPIs.
"""
from __future__ import annotations

import argparse
import csv
import json
import queue
import signal
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # bbai64-deploy/
import config as C                       # noqa: E402
import preflight                          # noqa: E402
from adas_events import AdasPublisher, build_payload  # noqa: E402
from compositor import composite         # noqa: E402
from depth_runtime import DepthRuntime   # noqa: E402
from frame_source import ImageFileSource, MqttFrameSource, VideoFileSource  # noqa: E402
from ufld_runtime import UfldRuntime     # noqa: E402
from yolo_runtime import YoloRuntime     # noqa: E402

SENTINEL = None


# ─────────────────────────────────────────────────────────────
def _yaml_overrides() -> dict:
    p = Path(__file__).resolve().parent / "config.yaml"
    if not p.exists():
        return {}
    try:
        import yaml
        return yaml.safe_load(p.read_text()) or {}
    except Exception as e:  # noqa: BLE001
        print(f"[app] config.yaml ignored ({e})")
        return {}


def _pct(vals: list, p: float) -> float:
    """Linear-interpolated p-th percentile of vals (sorted copy)."""
    if not vals:
        return float("nan")
    s = sorted(vals)
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _rss_mb():
    """Resident-set size of this process in MB, or None if unavailable.

    /proc/self/status is the accurate *current* RSS on the Linux board and needs
    no dependency. On non-Linux dev machines this returns None (KPI shows N/A),
    which is honest rather than reporting a wrong number.
    """
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0          # kB → MB
    except OSError:
        pass
    return None


def _open_writer(canvas, out_path, fps):
    """Open a VideoWriter, falling back mp4v -> MJPG/.avi if the board's OpenCV
    lacks the mp4 codec (common without ffmpeg). Returns (writer|None, path|None)
    so a missing codec degrades to a clear warning, never a silent empty file."""
    h, w = canvas.shape[:2]
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if vw.isOpened():
        return vw, str(out_path)
    alt = str(Path(out_path).with_suffix(".avi"))
    vw = cv2.VideoWriter(alt, cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    if vw.isOpened():
        print(f"[app] mp4v unavailable; writing {alt} (MJPG) instead")
        return vw, alt
    print("[app] WARNING: no working VideoWriter codec; annotated video disabled")
    return None, None


class Stat:
    """Online min/max/mean + a bounded reservoir for percentiles.

    O(1) memory regardless of stream length, so the indefinite MQTT mode cannot
    leak. Percentiles are exact while the run fits the reservoir (covers image /
    video and short streams) and a uniform random estimate beyond it. The full
    per-frame series still lands in the CSV; this is only for the exit summary.
    """

    __slots__ = ("n", "_sum", "min", "max", "last", "_res", "_cap", "_rng")

    def __init__(self, cap: int = 20000) -> None:
        import random
        self.n = 0
        self._sum = 0.0
        self.min = float("inf")
        self.max = float("-inf")
        self.last = float("nan")
        self._res: list = []
        self._cap = cap
        self._rng = random.Random(0)

    def add(self, x: float) -> None:
        self.n += 1
        self._sum += x
        self.last = x
        if x < self.min:
            self.min = x
        if x > self.max:
            self.max = x
        if len(self._res) < self._cap:           # reservoir sampling (Vitter R)
            self._res.append(x)
        else:
            j = self._rng.randint(0, self.n - 1)
            if j < self._cap:
                self._res[j] = x

    @property
    def avg(self) -> float:
        return self._sum / self.n if self.n else float("nan")

    def block(self, label: str) -> str:
        if not self.n:
            return f"  {label}: N/A\n"
        return (f"  {label}:\n"
                f"    min : {self.min:.3f}\n"
                f"    max : {self.max:.3f}\n"
                f"    avg : {self.avg:.3f}\n"
                f"    p95 : {_pct(self._res, 95):.3f}\n"
                f"    p99 : {_pct(self._res, 99):.3f}\n")


# ─────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["mqtt", "video", "image"], default="mqtt",
                    help="mqtt = live CARLA stream; video = file; image = single frame")
    ap.add_argument("--video", help="input path for --source video")
    ap.add_argument("--image", help="input path for --source image")
    ap.add_argument("--loop", action="store_true", help="loop the video file")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = unbounded")
    _disp_default = C.DISPLAY_SINK if C.DISPLAY_SINK in ("imshow", "none") else "none"
    ap.add_argument("--display", choices=["imshow", "none"], default=_disp_default,
                    help="optional live preview window; saved artifacts are automatic")
    ap.add_argument("--no-depth", dest="depth", action="store_false",
                    help="skip per-object distance (geometric monocular ranging; "
                         "A72-only, no C7x cost). Depth is ON by default.")
    ap.set_defaults(depth=C.DEPTH.ENABLED)
    ap.add_argument("--out", help="annotated output path (image/video). "
                    "Default: alongside the input as <stem>_annotated.<ext>")
    ap.add_argument("--json", dest="json_out", help="detections+lanes JSON path "
                    "(image/video). Default: alongside input as <stem>_result.json")
    args = ap.parse_args()
    ov = _yaml_overrides()

    # ── resolve input + default output paths (offline modes) ───
    in_path = None
    if args.source == "video":
        if not args.video:
            sys.exit("[app] --source video requires --video <path>")
        in_path = args.video
    elif args.source == "image":
        if not args.image:
            sys.exit("[app] --source image requires --image <path>")
        in_path = args.image
    if in_path is not None:
        stem = str(Path(in_path).with_suffix(""))
        ext = ".mp4" if args.source == "video" else ".png"
        out_path = Path(args.out) if args.out else Path(f"{stem}_annotated{ext}")
        json_path = Path(args.json_out) if args.json_out else Path(f"{stem}_result.json")
    else:
        out_path = json_path = None

    # config.yaml `depth: false` also disables it; --no-depth wins regardless.
    depth_enabled = bool(args.depth and ov.get("depth", True))

    # ── preflight: fail early with one clear message ───────────
    hard, warn = preflight.check(args.source, depth=depth_enabled)
    for w in warn:
        print(f"[preflight] WARN: {w}")
    if hard:
        print("[preflight] cannot start — fix these and retry "
              "(run `python3 runtime/preflight.py` for details):")
        for h in hard:
            print(f"  - {h}")
        sys.exit(2)

    # ── models (load TIDL artifacts) ───────────────────────────
    print("[app] loading TIDL sessions ...")
    th = ov.get("thresholds", {})        # optional runtime override (config.yaml)
    try:
        yolo = YoloRuntime(
            per_class_conf=th.get("per_class"),
            conf_default=th.get("conf_default"),
            iou_thres=th.get("iou"),
        )
        ufld = UfldRuntime()
        # Geometric distance estimator — no TIDL artifacts, A72-only (see
        # depth_runtime.py); construction is cheap and cannot fail on the board.
        depth = DepthRuntime() if depth_enabled else None
    except Exception as e:  # noqa: BLE001
        sys.exit(f"[app] failed to load TIDL sessions: {e}\n"
                 f"      check that ./artifacts matches the board's TIDL/SDK "
                 f"version and was copied intact.")
    print(f"[app] depth: {'ON (geometric monocular ranging)' if depth else 'OFF'}")

    print("[app] warming up ...")
    try:
        dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
        yi, ym = yolo.preprocess(dummy); yolo.infer_raw(yi)
        ufld.infer_raw(ufld.preprocess(dummy))
        # depth needs no warm-up: it is closed-form arithmetic, not a C7x model.
    except Exception as e:  # noqa: BLE001
        sys.exit(f"[app] warm-up inference failed: {e}\n"
                 f"      likely an artifact/firmware version mismatch or a bad "
                 f"input shape.")

    # ── frame source ───────────────────────────────────────────
    if args.source == "video":
        src = VideoFileSource(in_path, loop=args.loop)
    elif args.source == "image":
        src = ImageFileSource(in_path)
    else:
        m = ov.get("mqtt", {})
        src = MqttFrameSource(m.get("broker", C.MQTT.BROKER),
                              int(m.get("port", C.MQTT.PORT)),
                              m.get("topic_frames", C.MQTT.TOPIC_FRAMES),
                              C.MQTT.QOS)

    # Live stream publishes JSON to Qt over MQTT; offline modes accumulate the
    # same per-frame payloads into a list and write them to a JSON file at exit.
    adas = (AdasPublisher(C.MQTT.BROKER, C.MQTT.PORT, C.MQTT.TOPIC_ADAS, C.MQTT.QOS)
            if args.source == "mqtt" else None)
    records: list = []
    out_fps = float(getattr(src, "fps", 30.0))   # annotated-video playback rate

    qB: queue.Queue = queue.Queue(maxsize=2)
    qC: queue.Queue = queue.Queue(maxsize=2)
    stop = threading.Event()

    # Clean shutdown for a headless service: SIGTERM (kill / systemctl stop) sets
    # `stop`, so stages drain, the JSON/KPI summary is still written, and MQTT is
    # closed. SIGINT (Ctrl-C) keeps its KeyboardInterrupt path below.
    def _request_stop(signum, _frame):  # noqa: ANN001
        print(f"\n[app] signal {signum} -> stopping ...")
        stop.set()
    try:
        signal.signal(signal.SIGTERM, _request_stop)
    except (ValueError, OSError, AttributeError):
        pass            # not in main thread / platform without SIGTERM
    # Per-frame KPI accumulators (bounded memory; summarized at exit).
    pre_st, yolo_st, ufld_st = Stat(), Stat(), Stat()   # A72 pre, YOLO C7x, UFLD C7x (ms)
    depth_st = Stat()                                    # Depth C7x (ms; 0 when off)
    c7x_st, post_st = Stat(), Stat()                     # combined C7x, A72 decode+comp (ms)
    fps_st, ceil_st, occ_st, rss_st = Stat(), Stat(), Stat(), Stat()
    processed = [0]
    last_done = {"t": None}                   # perf_counter of previous completed frame
    rss = {"mb": None, "peak": 0.0, "next": 0.0}   # throttled memory sampler state
    writer = {"w": None, "path": None}

    # Per-frame KPI log (one row per frame, flushed as we go so a crash/kill still
    # leaves a complete record up to the last frame).
    kpi_path = Path(__file__).resolve().parent / "runtime_kpis.csv"
    kpi_file = open(kpi_path, "w", newline="", encoding="utf-8")
    kpi_csv = csv.writer(kpi_file)
    kpi_csv.writerow([
        "frame", "t_wall_s", "pre_ms", "yolo_ms", "ufld_ms", "depth_ms", "c7x_ms",
        "post_ms", "frame_interval_ms", "fps_inst", "fps_infer_ceiling",
        "c7x_occupancy_pct", "rss_mb", "n_dets", "n_lanes",
    ])

    # ── Stage A: capture + preprocess ──────────────────────────
    def stage_a() -> None:
        idx = 0
        try:
            while not stop.is_set():
                if args.max_frames and idx >= args.max_frames:
                    break
                frame = src.read(timeout=1.0)
                if frame is None:
                    if args.source in ("video", "image"):
                        break                # EOF / single frame served
                    continue                 # MQTT idle — keep waiting
                tp = time.perf_counter()
                yi, ym = yolo.preprocess(frame)
                ui = ufld.preprocess(frame)
                pre_ms = (time.perf_counter() - tp) * 1e3      # A72 preprocess latency
                qB.put((idx, frame, yi, ym, ui, pre_ms))
                idx += 1
        except Exception as e:  # noqa: BLE001 — surface, don't hang
            print(f"[app] stage A (capture/preprocess) error: {e}")
            stop.set()
        finally:
            qB.put(SENTINEL)

    # ── Stage B: inference on the C7x (YOLO then UFLD) ──────────
    def stage_b() -> None:
        try:
            while True:
                item = qB.get()
                if item is SENTINEL:
                    return
                idx, frame, yi, ym, ui, pre_ms = item
                t0 = time.perf_counter()
                y_raw = yolo.infer_raw(yi)
                t1 = time.perf_counter()
                u_raw = ufld.infer_raw(ui)
                t2 = time.perf_counter()
                # Depth is now closed-form geometry on the A72 (stage C), not a C7x
                # model — the engine runs only yolo + ufld here.
                # yolo, ufld, combined-C7x (serial on the single engine)
                timings = ((t1 - t0) * 1e3, (t2 - t1) * 1e3, (t2 - t0) * 1e3)
                qC.put((idx, frame, y_raw, ym, u_raw, pre_ms, timings))
        except Exception as e:  # noqa: BLE001
            print(f"[app] stage B (inference) error: {e}")
            stop.set()
        finally:
            qC.put(SENTINEL)     # always release stage C — never hang

    # ── Stage C: decode + composite + publish + display ────────
    def stage_c() -> None:
        try:
            while True:
                item = qC.get()
                if item is SENTINEL:
                    return
                idx, frame, y_raw, ym, u_raw, pre_ms, timings = item
                yolo_ms, ufld_ms, c7x_ms = timings
                tpost = time.perf_counter()
                dets = yolo.decode(y_raw, ym)
                lanes = ufld.decode(u_raw)
                td = time.perf_counter()
                if depth is not None:               # per-object metric distance (A72)
                    depth.attach_depth(dets, frame.shape[:2])
                depth_ms = (time.perf_counter() - td) * 1e3     # A72 geometric ranging
                canvas = composite(frame, dets, lanes, C.UFLD.VIS_W, C.UFLD.VIS_H)
                post_ms = (time.perf_counter() - tpost) * 1e3   # A72 decode + depth + composite

                # JSON payload (objects + lane points, both in frame-pixel coords).
                fh, fw = frame.shape[:2]
                payload = build_payload(idx, dets, lanes, fw, fh, C.UFLD.VIS_W, C.UFLD.VIS_H)
                if adas is not None:                # live stream → MQTT to Qt
                    adas.publish(payload)
                else:                               # offline → accumulate for JSON file
                    records.append(payload)

                # Optional live preview (any mode).
                if args.display == "imshow":
                    cv2.imshow("bbai64 ADAS", canvas)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        stop.set()
                # Saved annotated artifact (driven by source, not display).
                if args.source == "video":
                    if writer["w"] is None and writer["path"] is None:
                        writer["w"], writer["path"] = _open_writer(canvas, out_path, out_fps)
                        writer["path"] = writer["path"] or ""   # don't retry every frame
                    if writer["w"] is not None:
                        writer["w"].write(canvas)
                elif args.source == "image":
                    if not cv2.imwrite(str(out_path), canvas):
                        print(f"[app] WARNING: could not write image {out_path}")

                # ── per-frame KPIs ──────────────────────────────────
                # FPS is the true pipelined throughput: reciprocal of the wall-
                # clock gap between successive frame COMPLETIONS (captures any
                # stall, not just inference). C7x occupancy = how much of that gap
                # the engine was busy — a duty-cycle proxy for accelerator use
                # (100 % ⇒ C7x-bound; <100 % ⇒ source/CPU-limited, MMA headroom).
                # True MMA compute-array % needs TIDL perfsim, not timing.
                now = time.perf_counter()
                prev = last_done["t"]
                last_done["t"] = now
                pre_st.add(pre_ms)
                yolo_st.add(yolo_ms)
                ufld_st.add(ufld_ms)
                if depth is not None:
                    depth_st.add(depth_ms)
                c7x_st.add(c7x_ms)
                post_st.add(post_ms)

                # Inference-bound ceiling: FPS if the C7x were the only limit.
                ceil = 1000.0 / c7x_ms if c7x_ms > 0 else 0.0
                ceil_st.add(ceil)

                # Memory: throttled to ~1 Hz; carry the last sample into each row.
                if now >= rss["next"]:
                    mb = _rss_mb()
                    if mb is not None:
                        rss["mb"] = mb
                        rss["peak"] = max(rss["peak"], mb)
                        rss_st.add(mb)
                    rss["next"] = now + 1.0
                rss_s = f"{rss['mb']:.1f}" if rss["mb"] is not None else ""

                if prev is None:                   # first frame: no interval yet
                    fps_s = int_s = occ_s = ""
                else:
                    interval_ms = (now - prev) * 1e3
                    fps_inst = 1000.0 / interval_ms if interval_ms > 0 else 0.0
                    occ = min(100.0, c7x_ms / interval_ms * 100.0) if interval_ms > 0 else 0.0
                    fps_st.add(fps_inst)
                    occ_st.add(occ)
                    fps_s, int_s, occ_s = f"{fps_inst:.2f}", f"{interval_ms:.3f}", f"{occ:.1f}"
                kpi_csv.writerow([
                    idx, f"{now - wall0:.3f}", f"{pre_ms:.3f}", f"{yolo_ms:.3f}",
                    f"{ufld_ms:.3f}", f"{depth_ms:.3f}", f"{c7x_ms:.3f}",
                    f"{post_ms:.3f}", int_s, fps_s, f"{ceil:.2f}", occ_s, rss_s,
                    len(dets), len(lanes),
                ])
                kpi_file.flush()
                processed[0] += 1
        except Exception as e:  # noqa: BLE001
            print(f"[app] stage C (decode/composite) error: {e}")
            stop.set()

    threads = [threading.Thread(target=f, name=n, daemon=True)
               for f, n in ((stage_a, "A"), (stage_b, "B"), (stage_c, "C"))]
    wall0 = time.perf_counter()
    for t in threads:
        t.start()
    try:
        # progress while running
        while threads[2].is_alive():
            threads[2].join(timeout=2.0)
            n = processed[0]
            if n and c7x_st.n:
                el = time.perf_counter() - wall0
                fps_now = fps_st.last if fps_st.n else n / el
                occ_now = occ_st.last if occ_st.n else 0.0
                rss_now = f"{rss['mb']:.0f}MB" if rss["mb"] is not None else "n/a"
                depth_now = f"depth={depth_st.last:5.1f}ms  " if depth_st.n else ""
                print(f"  frames={n:>6}  fps={fps_now:5.1f}  avg_fps={n/el:5.1f}  "
                      f"det={yolo_st.last:5.1f}ms  lane={ufld_st.last:5.1f}ms  "
                      f"{depth_now}c7x={c7x_st.last:5.1f}ms  "
                      f"c7x_util={occ_now:4.0f}%  rss={rss_now}")
    except KeyboardInterrupt:
        print("\n[app] stopping ...")
        stop.set()
        for t in threads:
            t.join(timeout=3.0)

    wall = time.perf_counter() - wall0
    if writer["w"] is not None:
        writer["w"].release()
    if args.display == "imshow":
        cv2.destroyAllWindows()
    kpi_file.close()
    src.close()
    if adas is not None:
        adas.close()

    # ── offline JSON output (image / video) ────────────────────
    if json_path is not None:
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump({
                "source": args.source,
                "input": str(in_path),
                "depth_enabled": depth is not None,
                "depth_method": C.DEPTH.METHOD if depth is not None else None,
                "depth_unit": "metres" if depth is not None else None,
                "frame_count": len(records),
                "frames": records,
            }, jf, indent=2)
        print(f"[app] {len(records)} frame record(s) -> {json_path}")
        ann = writer["path"] if (args.source == "video" and writer["path"]) else out_path
        print(f"[app] annotated output -> {ann}")

    # ── analytics summary ──────────────────────────────────────
    n = processed[0]
    fps = n / wall if wall > 0 else 0.0           # avg FPS across all frames
    avg_occ = occ_st.avg if occ_st.n else 0.0
    rss_line = (f"{rss_st.avg:.1f} avg / {rss['peak']:.1f} peak"
                if rss_st.n else "N/A (non-Linux dev host)")
    out = Path(__file__).resolve().parent / "runtime_analytics.txt"
    with open(out, "w", encoding="utf-8") as f:
        f.write("=" * 56 + "\n  BBAI-64 RUNTIME ANALYTICS (YOLO + UFLDv2, TIDL)\n")
        f.write("=" * 56 + "\n\n")
        f.write(f"  Source         : {args.source}\n")
        f.write(f"  Frames         : {n}\n")
        f.write(f"  Wall time      : {wall:.2f} s\n")
        f.write(f"  Avg FPS        : {fps:.2f}  (frames / wall time; pipelined)\n")
        f.write(f"  Avg C7x util   : {avg_occ:.1f} %  (duty cycle; see note below)\n")
        f.write(f"  Memory (RSS MB): {rss_line}\n")
        f.write(f"  Tensor bits    : INT{C.TENSOR_BITS}\n")
        f.write(f"  Per-frame KPIs : {kpi_path.name}\n\n")
        f.write("── Instantaneous FPS (per-frame throughput) ────────────\n")
        f.write(fps_st.block("fps"))
        f.write("\n── Inference-bound FPS ceiling (1000 / c7x_ms) ─────────\n")
        f.write(ceil_st.block("fps ceiling"))
        f.write("\n── C7x occupancy (%, duty cycle = busy / frame interval) ─\n")
        f.write(occ_st.block("c7x util"))
        f.write("\n── A72 preprocess latency (ms, stage A) ────────────────\n")
        f.write(pre_st.block("preprocess"))
        f.write("\n── YOLO C7x latency (ms) ───────────────────────────────\n")
        f.write(yolo_st.block("yolo infer"))
        f.write("\n── UFLD C7x latency (ms) ───────────────────────────────\n")
        f.write(ufld_st.block("ufld infer"))
        f.write("\n── Depth A72 latency (ms, geometric monocular ranging) ─\n")
        f.write(depth_st.block("depth est") if depth_st.n else "  depth: OFF\n")
        f.write("\n── Combined C7x latency (ms, all models, one engine) ───\n")
        f.write(c7x_st.block("yolo+ufld"))
        f.write("\n── A72 decode+composite latency (ms, stage C) ──────────\n")
        f.write(post_st.block("decode+comp"))
        f.write("\n  NOTE: 'C7x util' is a TIME duty cycle (fraction of each frame's\n")
        f.write("  wall interval the engine was computing), not silicon MMA-array\n")
        f.write("  occupancy. 100% ⇒ C7x is the bottleneck; <100% ⇒ source/CPU-\n")
        f.write("  limited with MMA headroom. For true MMA compute %, use TIDL\n")
        f.write("  perfsim / graph stats at compile time.\n")
        f.write("\n" + "=" * 56 + "\n")
    print(f"\n[app] {n} frames, avg {fps:.2f} FPS, avg C7x util {avg_occ:.1f}% "
          f"-> summary: {out.name}, per-frame: {kpi_path.name}")


if __name__ == "__main__":
    main()
