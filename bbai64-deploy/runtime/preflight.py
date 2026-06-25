#!/usr/bin/env python3
"""
Preflight environment check for the bbai64 runtime.

Run it standalone on the board BEFORE app.py to get a clear PASS/FAIL list of
what the runtime needs, instead of a mid-run traceback:

    python3 runtime/preflight.py            # check for the default (mqtt) source
    python3 runtime/preflight.py --source video

app.py also calls check() at startup and aborts with one actionable message if a
hard requirement is missing.

Distinguishes HARD failures (cannot run) from WARNINGs (will run but degraded,
e.g. TIDL EP missing -> falls back to slow CPU).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # bbai64-deploy/
import config as C  # noqa: E402


def _dir_has_files(p: Path) -> bool:
    return p.is_dir() and any(p.iterdir())


def check(source: str = "mqtt", depth: bool = None) -> tuple[list[str], list[str]]:
    """Return (hard_failures, warnings). Empty hard list == safe to run.

    `depth` toggles the Depth-Anything-V2 artifact check; None = config default.
    """
    if depth is None:
        depth = C.DEPTH.ENABLED
    hard: list[str] = []
    warn: list[str] = []

    # ── core python deps ───────────────────────────────────────
    for mod in ("cv2", "numpy"):
        try:
            __import__(mod)
        except Exception as e:  # noqa: BLE001
            hard.append(f"missing python module '{mod}' ({e})")

    # ── onnxruntime + TIDL execution provider ──────────────────
    try:
        import onnxruntime as ort
        provs = ort.get_available_providers()
        if "TIDLExecutionProvider" not in provs:
            warn.append("TIDLExecutionProvider NOT available -> inference will "
                        "fall back to CPU (very slow). Is this the board's "
                        f"onnxruntime-tidl? providers={provs}")
    except Exception as e:  # noqa: BLE001
        hard.append(f"onnxruntime not importable ({e})")

    # ── model artifacts (produced by export+compile on the PC) ─
    needed = [
        ("YOLO", C.YOLO.ONNX, C.YOLO.TIDL_DIR),
        ("UFLD", C.UFLD.ONNX, C.UFLD.TIDL_DIR),
    ]
    if depth:
        needed.append(("DEPTH", C.DEPTH.ONNX, C.DEPTH.TIDL_DIR))
    for name, onnx, tdir in needed:
        if not Path(onnx).exists():
            hard.append(f"{name}: ONNX missing: {onnx} (run export, copy artifacts)")
        if not _dir_has_files(Path(tdir)):
            hard.append(f"{name}: TIDL artifacts missing/empty: {tdir} "
                        f"(run compile on PC, copy ./artifacts to the board)")

    # ── class names ────────────────────────────────────────────
    if not C.YOLO.NAMES:
        warn.append(f"no class names loaded from {C.YOLO.DATA_YAML} -> detections "
                    "will be labelled by numeric id")

    # ── source-specific deps ───────────────────────────────────
    if source == "mqtt":
        try:
            import paho.mqtt.client  # noqa: F401
        except Exception as e:  # noqa: BLE001
            hard.append(f"--source mqtt needs paho-mqtt ({e}); pip3 install paho-mqtt")
    try:
        import yaml  # noqa: F401
    except Exception:  # noqa: BLE001
        warn.append("PyYAML not installed -> runtime/config.yaml overrides ignored "
                    "(config.py defaults still apply)")

    return hard, warn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["mqtt", "video", "image"], default="mqtt")
    ap.add_argument("--no-depth", dest="depth", action="store_false",
                    help="skip the Depth-Anything-V2 artifact check")
    ap.set_defaults(depth=C.DEPTH.ENABLED)
    args = ap.parse_args()

    hard, warn = check(args.source, depth=args.depth)
    print("=" * 56)
    print("  BBAI64 PREFLIGHT")
    print("=" * 56)
    for w in warn:
        print(f"  [WARN] {w}")
    for h in hard:
        print(f"  [FAIL] {h}")
    if hard:
        print(f"\n  RESULT: NOT READY ({len(hard)} blocker(s)).")
        return 1
    print(f"\n  RESULT: READY{' (with warnings)' if warn else ''}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
