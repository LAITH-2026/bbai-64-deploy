#!/usr/bin/env bash
# Native launch on the BeagleBone AI-64 (no Docker — that step is deferred).
# Assumes the Yocto edgeai image (onnxruntime + TIDL runtime present) and that
# ./artifacts was copied over from the PC compile stage.
set -euo pipefail
cd "$(dirname "$0")"

# Lightweight deps the app needs beyond the edgeai image's onnxruntime/opencv.
python3 - <<'PY' 2>/dev/null || pip3 install --user paho-mqtt pyyaml
import paho.mqtt.client, yaml  # noqa
PY

if [ ! -d artifacts/yolo_tidl ] || [ ! -d artifacts/ufld_tidl ]; then
  echo "ERROR: artifacts/ missing. Copy the compiled TIDL artifacts from the PC:"
  echo "  scp -r artifacts <user>@<board>:$(pwd)/"
  exit 1
fi

exec python3 runtime/app.py "$@"
