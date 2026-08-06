#!/usr/bin/env bash
# Cross-platform friendly launcher (Linux / macOS / Git Bash / WSL).
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  if [[ -x "venv/bin/python" ]]; then
    PYTHON="venv/bin/python"
  elif [[ -x "venv/Scripts/python.exe" ]]; then
    PYTHON="venv/Scripts/python.exe"
  else
    PYTHON="python3"
  fi
fi

export DETECTOR_HOST="${DETECTOR_HOST:-0.0.0.0}"
export DETECTOR_PORT="${DETECTOR_PORT:-7860}"
export DETECTOR_DEVICE="${DETECTOR_DEVICE:-auto}"

echo "Starting object-detector with $PYTHON on ${DETECTOR_HOST}:${DETECTOR_PORT} (device=${DETECTOR_DEVICE})"
exec "$PYTHON" app.py
