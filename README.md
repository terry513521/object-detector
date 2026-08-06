# Object Detector

Cross-platform viewer for miner+ONNX and Ultralytics `.pt` models (Linux, macOS, Windows).

## Requirements

- Python **3.10–3.12**
- Optional: NVIDIA CUDA + drivers for GPU mode
- Optional: [pm2](https://pm2.keymetrics.io/) for process management

## Quick start

```bash
# Linux / macOS
python3 -m venv venv
source venv/bin/activate

# Windows (cmd)
python -m venv venv
venv\Scripts\activate
```

Install deps (pick one):

```bash
# CUDA hosts (default)
pip install -r requirements-gpu.txt

# CPU-only or macOS / Apple Silicon
pip install -r requirements-cpu.txt
```

Run:

```bash
# Linux / macOS / Git Bash
./start.sh

# Windows
start.bat

# or
python app.py
```

Open `http://127.0.0.1:7860`.

## Environment

| Variable | Default | Meaning |
|---|---|---|
| `DETECTOR_HOST` | `0.0.0.0` | Bind address |
| `DETECTOR_PORT` | `7860` | Port |
| `DETECTOR_DEVICE` | `auto` | `auto` \| `cpu` \| `cuda` |
| `DETECTOR_ROOT` | auto | Root used for model IDs / sibling `layer` & `full` trees |
| `DETECTOR_SCAN_ROOTS` | empty | Extra model dirs (`:` on Unix, `;` on Windows) |

On macOS, `auto` may use the CoreML ONNX Runtime EP when available.

## pm2

```bash
pm2 start ecosystem.config.cjs
pm2 save
```

`ecosystem.config.cjs` resolves `venv/bin/python` or `venv\Scripts\python.exe` from `__dirname`.

## Layout

- Upload models via the UI → stored under `uploads/`
- Optional sibling workspace: `layer/models`, `full/exports`, etc. (scanned when present under `DETECTOR_ROOT`)
- Result cache: `cache/results/`
