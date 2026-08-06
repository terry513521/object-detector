#!/usr/bin/env python3
"""Object Detector viewer: miner+ONNX or Ultralytics .pt, drop an image, see boxes.

Cross-platform (Linux / macOS / Windows). Paths are project-relative; optional
workspace scanning for sibling model trees. GPU (CUDA) and CPU modes both
supported; CoreML EP is used on macOS when available.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import importlib.util
import json
import os
import platform
import re
import shutil
import sys
import threading
import time
import traceback
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
UPLOADS = HERE / "uploads"
UPLOADS.mkdir(parents=True, exist_ok=True)
RESULTS = HERE / "cache" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def _default_id_root() -> Path:
    """Project root for model IDs. Use parent when sibling model trees exist."""
    env = os.environ.get("DETECTOR_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    parent = HERE.parent
    if any((parent / name).is_dir() for name in ("layer", "full")):
        return parent.resolve()
    return HERE.resolve()


ID_ROOT = _default_id_root()
# Back-compat alias used throughout older code / docs.
CAR_WASH = ID_ROOT

# Default inference device: auto | cpu | cuda (gpu/0 accepted as cuda aliases).
DEFAULT_DEVICE_MODE = os.environ.get("DETECTOR_DEVICE", "auto")


def _build_scan_layout() -> tuple[list[Path], list[Path], list[tuple[Path, tuple[str, ...]]]]:
    """Discover model scan roots relative to ID_ROOT / env (cross-platform)."""
    roots: list[Path] = [UPLOADS]
    extra = os.environ.get("DETECTOR_SCAN_ROOTS", "")
    for part in extra.split(os.pathsep):
        part = part.strip()
        if part:
            roots.append(Path(part).expanduser().resolve())

    # Optional sibling / workspace layout (car-wash style), only if present.
    for rel in (
        ("layer", "models"),
        ("layer", "_cand"),
        ("layer", "exports"),
        ("full", "exports"),
    ):
        p = ID_ROOT.joinpath(*rel)
        if p.is_dir():
            roots.append(p)

    pt_files: list[Path] = []
    best = ID_ROOT / "full" / "best.pt"
    if best.is_file():
        pt_files.append(best)

    pt_globs: list[tuple[Path, tuple[str, ...]]] = []
    for base, names in (
        (ID_ROOT / "full" / "exports", ("best.pt", "best_tw.pt", "last.pt", "weights.pt", "model.pt")),
        (ID_ROOT / "layer" / "exports", ("best.pt", "best_tw.pt", "last.pt", "weights.pt", "model.pt")),
        (ID_ROOT / "full" / "runs", ("best.pt", "best_tw.pt")),
    ):
        if base.is_dir():
            pt_globs.append((base, names))

    # Deduplicate roots while preserving order
    seen: set[Path] = set()
    uniq: list[Path] = []
    for r in roots:
        try:
            key = r.resolve()
        except OSError:
            key = r
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)
    return uniq, pt_files, pt_globs


SCAN_ROOTS, PT_EXTRA_FILES, PT_DIR_GLOBS = _build_scan_layout()

CLASS_COLORS = {
    # Matched to Ultralytics-style reference annotation colors
    "broom": (220, 70, 25),
    "drainage gate": (150, 205, 30),
    "drainage_gate": (150, 205, 30),
    "nozzle": (30, 200, 100),
    "track": (30, 110, 215),
}
# Scalable fallback palette (RGB) — used by class index when name is unknown.
CLASS_PALETTE = [
    (220, 70, 25),
    (150, 205, 30),
    (30, 200, 100),
    (30, 110, 215),
    (155, 89, 182),
    (46, 204, 113),
    (241, 196, 15),
    (230, 126, 34),
    (26, 188, 156),
    (52, 73, 94),
    (149, 165, 166),
    (192, 57, 43),
]
DEFAULT_COLOR = (180, 180, 180)
FALLBACK_NAMES = ["broom", "drainage gate", "nozzle", "track"]
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
PT_NAME_PREFER = ("best_tw.pt", "best.pt", "last.pt", "weights.pt", "model.pt")


def color_for_class(cls_id: int, raw_name: str) -> tuple[int, int, int]:
    """Prefer known class colors; otherwise pick a stable palette color by index."""
    key = (raw_name or "").strip()
    named = CLASS_COLORS.get(key) or CLASS_COLORS.get(key.replace("_", " "))
    if named:
        return named
    if CLASS_PALETTE:
        return CLASS_PALETTE[int(cls_id) % len(CLASS_PALETTE)]
    return DEFAULT_COLOR

app = FastAPI(title="Object Detector")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

_cache_lock = threading.Lock()
_model_cache: dict[str, dict[str, Any]] = {}
_pt_meta_cache: dict[str, tuple[float, int, list[str], list[float]]] = {}
_miner_meta_cache: dict[str, tuple[float, list[str], list[float]]] = {}
_runtime_ready = False


def cuda_available() -> bool:
    """True only when a usable CUDA device is present (Linux/Windows/macOS-safe)."""
    try:
        import torch

        if torch.cuda.is_available():
            return True
    except Exception:
        pass
    # Linux device node (works even if torch is CPU-only build)
    if sys.platform.startswith("linux") and Path("/dev/nvidia0").exists():
        return True
    return False


def normalize_device_mode(raw: str | None = None) -> str:
    """Return canonical mode: auto | cpu | cuda."""
    v = (raw if raw is not None else DEFAULT_DEVICE_MODE) or "auto"
    v = str(v).strip().lower()
    if v in ("gpu", "cuda", "0"):
        return "cuda"
    if v in ("cpu",):
        return "cpu"
    return "auto"


def resolve_torch_device(mode: str | None = None) -> str:
    """Ultralytics device string: 'cpu' or '0' (GPU). Keeps GPU path when requested/available."""
    choice = normalize_device_mode(mode)
    if choice == "cpu":
        return "cpu"
    if choice == "cuda":
        if not cuda_available():
            raise HTTPException(400, "GPU (CUDA) requested but CUDA is not available on this host")
        return "0"
    return "0" if cuda_available() else "cpu"


def device_label(torch_device: str) -> str:
    return "cuda" if torch_device not in ("cpu", "", None) else "cpu"


def get_device(mode: str | None = None) -> str:
    """Backward-compatible helper — resolves auto/cpu/cuda to torch device."""
    return resolve_torch_device(mode)


def ensure_runtime_ready() -> None:
    """One-time CUDA / ORT warm-up so the first user request is not a 30–60s hit."""
    global _runtime_ready
    if _runtime_ready:
        return
    with _cache_lock:
        if _runtime_ready:
            return
        mode = normalize_device_mode()
        try:
            import torch

            if mode != "cpu" and torch.cuda.is_available():
                torch.zeros(1, device="cuda").item()
                torch.cuda.synchronize()
        except Exception as e:
            print(f"torch cuda warm-up skipped: {e}")
        try:
            import onnxruntime as ort

            so = ort.SessionOptions()
            so.log_severity_level = 3
            print("ORT providers:", ort.get_available_providers())
            print(
                f"platform={platform.system()} DETECTOR_DEVICE={mode} "
                f"cuda_available={cuda_available()} id_root={ID_ROOT}"
            )
        except Exception as e:
            print(f"ORT warm-up skipped: {e}")
        _runtime_ready = True


def _ort_providers(mode: str | None = None) -> list[str]:
    """Pick ORT EPs for the current OS. GPU mode prefers CUDA; auto may use CoreML on macOS."""
    try:
        import onnxruntime as ort

        available = set(ort.get_available_providers())
    except Exception:
        return ["CPUExecutionProvider"]

    choice = normalize_device_mode(mode)
    if choice == "cpu":
        return ["CPUExecutionProvider"] if "CPUExecutionProvider" in available else list(available) or [
            "CPUExecutionProvider"
        ]
    if choice == "cuda":
        if not cuda_available() or "CUDAExecutionProvider" not in available:
            raise HTTPException(
                400,
                "GPU (CUDA) requested but CUDA is not available on this host",
            )
        order = ["CUDAExecutionProvider"]
        if "CPUExecutionProvider" in available:
            order.append("CPUExecutionProvider")
        return order
    # auto: CUDA when real GPU present; CoreML on macOS; always keep CPU fallback
    order = []
    if cuda_available() and "CUDAExecutionProvider" in available:
        order.append("CUDAExecutionProvider")
    if sys.platform == "darwin" and "CoreMLExecutionProvider" in available:
        order.append("CoreMLExecutionProvider")
    if "CPUExecutionProvider" in available:
        order.append("CPUExecutionProvider")
    return order or ["CPUExecutionProvider"]


def _cache_key(model_id: str, torch_device: str) -> str:
    return f"{model_id}:::{device_label(torch_device)}"


@dataclass
class DetBox:
    x1: int
    y1: int
    x2: int
    y2: int
    cls_id: int
    conf: float


class ModelInfo(BaseModel):
    id: str
    name: str
    group: str
    path: str
    weights: str
    weights_mb: float
    kind: str  # "miner" | "pt"
    uploaded: bool = False
    default_imgsz: int | None = None
    default_conf: float = 0.25
    class_names: list[str] = []
    default_confs: list[float] = []


# Sensible car-wash defaults when class set matches.
_CW_DEFAULT_CONFS = {
    "broom": 0.35,
    "drainage gate": 0.70,
    "drainage_gate": 0.70,
    "nozzle": 0.40,
    "track": 0.70,
}


def _find_onnx(folder: Path) -> Path | None:
    preferred = folder / "weights.onnx"
    if preferred.is_file():
        return preferred
    onnxs = sorted(folder.glob("*.onnx"))
    return onnxs[0] if onnxs else None


def _find_pt(folder: Path) -> Path | None:
    pts = [p for p in folder.glob("*.pt") if p.is_file()]
    if not pts:
        return None
    rank = {n: i for i, n in enumerate(PT_NAME_PREFER)}
    pts.sort(key=lambda p: (rank.get(p.name.lower(), 99), p.name.lower()))
    return pts[0]


def _is_miner_model(folder: Path) -> bool:
    return (folder / "miner.py").is_file() and _find_onnx(folder) is not None


def _is_pt_folder(folder: Path) -> bool:
    return _find_pt(folder) is not None and not _is_miner_model(folder)


def _is_under(path: Path, root: Path) -> bool:
    """True if path is root or a descendant (Windows-safe; no string prefix tricks)."""
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        return False


def _model_id_for(path: Path) -> str:
    resolved = path.resolve()
    for root in (ID_ROOT, HERE):
        try:
            return str(resolved.relative_to(root.resolve())).replace("\\", "/")
        except ValueError:
            continue
    raise HTTPException(400, f"Path outside project roots: {path}")


def _group_for(path: Path) -> str:
    resolved = path.resolve()
    if UPLOADS.resolve() == resolved or UPLOADS.resolve() in resolved.parents:
        return "uploads"
    try:
        rel = resolved.relative_to(ID_ROOT.resolve())
    except ValueError:
        rel = resolved.relative_to(HERE.resolve())
    if len(rel.parts) >= 2:
        return f"{rel.parts[0]}/{rel.parts[1]}"
    return rel.parts[0]


def _imgsz_from_pt(path: Path) -> int:
    try:
        import torch

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        args = ckpt.get("train_args") or {}
        imgsz = args.get("imgsz")
        if imgsz is None:
            return 640
        if isinstance(imgsz, (list, tuple)):
            return int(imgsz[0])
        return int(imgsz)
    except Exception:
        return 640


def _names_from_pt(path: Path) -> list[str]:
    try:
        import torch

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = ckpt.get("model")
        names = getattr(model, "names", None) if model is not None else None
        if isinstance(names, dict):
            return [str(names[i]) for i in sorted(names)]
        if isinstance(names, (list, tuple)):
            return [str(n) for n in names]
    except Exception:
        pass
    return list(FALLBACK_NAMES)


def _default_confs_for(names: list[str]) -> list[float]:
    out: list[float] = []
    for n in names:
        key = n.lower().strip()
        out.append(float(_CW_DEFAULT_CONFS.get(key, _CW_DEFAULT_CONFS.get(key.replace("_", " "), 0.25))))
    return out


def _pt_meta(path: Path) -> tuple[int, list[str], list[float]]:
    key = str(path.resolve())
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    cached = _pt_meta_cache.get(key)
    if cached and cached[0] == mtime:
        return cached[1], cached[2], cached[3]
    # Single torch.load for both imgsz + names
    imgsz = 640
    names = list(FALLBACK_NAMES)
    try:
        import torch

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        args = ckpt.get("train_args") or {}
        raw = args.get("imgsz")
        if isinstance(raw, (list, tuple)):
            imgsz = int(raw[0])
        elif raw is not None:
            imgsz = int(raw)
        model = ckpt.get("model")
        n = getattr(model, "names", None) if model is not None else None
        if isinstance(n, dict):
            names = [str(n[i]) for i in sorted(n)]
        elif isinstance(n, (list, tuple)):
            names = [str(x) for x in n]
    except Exception:
        pass
    confs = _default_confs_for(names)
    _pt_meta_cache[key] = (mtime, imgsz, names, confs)
    return imgsz, names, confs


def _extract_py_list(src: str, *patterns: str) -> list[Any] | None:
    """Find the first assignable Python list literal matching any pattern."""
    for pat in patterns:
        m = re.search(pat, src, flags=re.MULTILINE)
        if not m:
            continue
        try:
            parsed = ast.literal_eval(m.group(1))
        except Exception:
            continue
        if isinstance(parsed, (list, tuple)) and parsed:
            return list(parsed)
    return None


def _miner_meta(folder: Path) -> tuple[list[str], list[float]]:
    """Read class_names + per-class conf thresholds from miner.py (no ONNX load).

    Supports both instance attrs (`self.class_names = [...]`) and class attrs
    (`class_names = [...]`), including multiline np.array([...]) assignments.
    """
    path = folder / "miner.py"
    key = str(path.resolve())
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    cached = _miner_meta_cache.get(key)
    if cached and cached[0] == mtime:
        return cached[1], cached[2]

    names: list[str] = []
    confs: list[float] = []
    try:
        src = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        _miner_meta_cache[key] = (mtime, names, confs)
        return names, confs

    # Prefer self.* (instance) then bare class-level attributes.
    raw_names = _extract_py_list(
        src,
        r"self\.class_names\s*=\s*(\[[^\]]*\])",
        r"(?m)^\s*class_names\s*=\s*(\[[^\]]*\])",
    )
    if raw_names:
        names = [str(x) for x in raw_names]

    raw_confs = _extract_py_list(
        src,
        r"self\._conf_thres_array\s*=\s*np\.array\s*\(\s*(\[[^\]]*\])",
        r"(?m)^\s*_conf_thres_array\s*=\s*np\.array\s*\(\s*(\[[^\]]*\])",
    )
    if raw_confs:
        try:
            confs = [float(x) for x in raw_confs]
        except Exception:
            confs = []

    if names and not confs:
        confs = _default_confs_for(names)
    if not names and confs:
        names = [f"class {i}" for i in range(len(confs))]

    if len(confs) < len(names):
        confs = confs + [0.25] * (len(names) - len(confs))
    elif len(confs) > len(names):
        confs = confs[: len(names)]

    _miner_meta_cache[key] = (mtime, names, confs)
    return names, confs


def _append_miner(models: list[ModelInfo], seen: set[Path], folder: Path) -> None:
    resolved = folder.resolve()
    if resolved in seen or not _is_miner_model(folder):
        return
    seen.add(resolved)
    onnx = _find_onnx(folder)
    assert onnx is not None
    group = _group_for(folder)
    names, confs = _miner_meta(folder)
    models.append(
        ModelInfo(
            id=_model_id_for(folder),
            name=folder.name,
            group=group,
            path=str(folder),
            weights=onnx.name,
            weights_mb=round(onnx.stat().st_size / (1024 * 1024), 2),
            kind="miner",
            uploaded=(group == "uploads"),
            default_conf=float(min(confs) if confs else 0.25),
            class_names=[display_name(n) for n in names],
            default_confs=confs,
        )
    )


def _append_pt(models: list[ModelInfo], seen: set[Path], pt: Path, name: str | None = None) -> None:
    resolved = pt.resolve()
    if resolved in seen or not pt.is_file():
        return
    if not (_is_under(resolved, ID_ROOT) or _is_under(resolved, HERE)):
        return
    seen.add(resolved)
    group = _group_for(pt)
    imgsz, names, confs = _pt_meta(pt)
    models.append(
        ModelInfo(
            id=_model_id_for(pt),
            name=name or pt.stem,
            group=group,
            path=str(pt),
            weights=pt.name,
            weights_mb=round(pt.stat().st_size / (1024 * 1024), 2),
            kind="pt",
            uploaded=(group == "uploads"),
            default_imgsz=imgsz,
            default_conf=float(min(confs) if confs else 0.25),
            class_names=[display_name(n) for n in names],
            default_confs=confs,
        )
    )


def discover_models() -> list[ModelInfo]:
    seen: set[Path] = set()
    models: list[ModelInfo] = []

    for root in SCAN_ROOTS:
        if not root.is_dir():
            continue
        for folder in sorted(p for p in root.iterdir() if p.is_dir()):
            if folder.name.startswith("."):
                continue
            if _is_miner_model(folder):
                _append_miner(models, seen, folder)
            elif _is_pt_folder(folder):
                pt = _find_pt(folder)
                if pt:
                    _append_pt(models, seen, pt, name=folder.name)

    for pt in PT_EXTRA_FILES:
        _append_pt(models, seen, pt)

    for base, names in PT_DIR_GLOBS:
        if not base.is_dir():
            continue
        for pt in base.rglob("*.pt"):
            if pt.name.lower() not in {n.lower() for n in names}:
                continue
            # Skip noisy intermediate paths under runs/*/weights/epoch*
            if pt.name.lower().startswith("epoch"):
                continue
            label = pt.parent.name if pt.parent.name != "weights" else pt.parent.parent.name
            _append_pt(models, seen, pt, name=f"{label}/{pt.stem}")

    order = {
        "uploads": 0,
        "layer/models": 1,
        "layer/exports": 2,
        "full/exports": 3,
        "full/runs": 4,
        "layer/_cand": 5,
    }
    models.sort(key=lambda m: (0 if m.kind == "pt" and m.uploaded else 1,
                               order.get(m.group, 9),
                               0 if m.kind == "miner" else 1,
                               m.name.lower()))
    # Prefer uploads first overall
    models.sort(key=lambda m: (0 if m.group == "uploads" else 1,
                               order.get(m.group, 9),
                               0 if m.kind == "miner" else 1,
                               m.name.lower()))
    return models


def _resolve_model(model_id: str) -> tuple[str, Path]:
    mid = model_id.replace("\\", "/").lstrip("/")
    candidates = [(ID_ROOT / mid).resolve(), (HERE / mid).resolve()]
    path = None
    for cand in candidates:
        if _is_under(cand, ID_ROOT) or _is_under(cand, HERE):
            if cand.exists():
                path = cand
                break
    if path is None:
        raise HTTPException(404, f"Unknown model: {model_id}")
    if path.is_file() and path.suffix.lower() == ".pt":
        return "pt", path
    if path.is_dir() and _is_miner_model(path):
        return "miner", path
    if path.is_dir() and _find_pt(path):
        return "pt", _find_pt(path)  # type: ignore[return-value]
    raise HTTPException(404, f"Unknown model: {model_id}")


def load_model(model_id: str, device_mode: str | None = None) -> dict[str, Any]:
    ensure_runtime_ready()
    mode = normalize_device_mode(device_mode)
    torch_device = resolve_torch_device(mode)
    key = _cache_key(model_id, torch_device)
    with _cache_lock:
        if key in _model_cache:
            return _model_cache[key]

    kind, path = _resolve_model(model_id)
    if kind == "miner":
        folder = path
        mod_name = "viewer_miner_" + re.sub(r"[^A-Za-z0-9_]", "_", model_id)
        spec = importlib.util.spec_from_file_location(mod_name, folder / "miner.py")
        if spec is None or spec.loader is None:
            raise HTTPException(500, "Failed to load miner.py")
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
            # Skip multi-iter miner warm-up; we warm CUDA once at process start
            # and do a single real-frame warm predict below.
            if hasattr(mod.Miner, "_warmup"):
                mod.Miner._warmup = lambda self, iters=1: None  # type: ignore[method-assign]
            obj = mod.Miner(folder)
            # Align ORT session with requested device mode (GPU kept when selected/available).
            try:
                import onnxruntime as ort

                providers = _ort_providers(mode)
                want_cuda = "CUDAExecutionProvider" in providers
                have_cuda = "CUDAExecutionProvider" in obj.session.get_providers()
                need_rebuild = (want_cuda and not have_cuda) or (not want_cuda and have_cuda) or (
                    obj.session.get_providers()[: len(providers)] != providers
                )
                if need_rebuild:
                    onnx_path = folder / "weights.onnx"
                    if not onnx_path.is_file():
                        onnx_path = next(folder.glob("*.onnx"))
                    so = ort.SessionOptions()
                    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    obj.session = ort.InferenceSession(
                        str(onnx_path), sess_options=so, providers=providers,
                    )
                    print("Rebuilt ORT session with", obj.session.get_providers(), f"(mode={mode})")
            except HTTPException:
                raise
            except Exception as e:
                print(f"ORT session rebuild skipped: {e}")
            # One warm inference
            try:
                dummy = np.zeros((640, 640, 3), dtype=np.uint8)
                obj.predict_batch([dummy], 0, 0)
            except Exception as e:
                print(f"miner warm predict skipped: {e}")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, f"Miner init failed: {e}") from e
        used = (
            "cuda"
            if "CUDAExecutionProvider" in getattr(obj.session, "get_providers", lambda: [])()
            else "cpu"
        )
        entry = {
            "kind": "miner",
            "obj": obj,
            "path": folder,
            "device": used,
            "device_mode": mode,
            "cache_key": key,
        }
    else:
        try:
            from ultralytics import YOLO

            obj = YOLO(str(path))
            # Move / warm on target device once
            imgsz = 640
            try:
                imgsz = _pt_meta(path)[0]
            except Exception:
                pass
            dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
            obj.predict(dummy, imgsz=imgsz, device=torch_device, verbose=False)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, f"YOLO .pt load failed: {e}") from e
        entry = {
            "kind": "pt",
            "obj": obj,
            "path": path,
            "device": torch_device,
            "device_mode": mode,
            "cache_key": key,
        }

    with _cache_lock:
        _model_cache[key] = entry
    return entry


def class_names_for(entry: dict[str, Any]) -> list[str]:
    if entry["kind"] == "miner":
        names = getattr(entry["obj"], "class_names", None)
        if names is None:
            return list(FALLBACK_NAMES)
        return list(names)
    names = getattr(entry["obj"], "names", None) or {}
    if isinstance(names, dict):
        return [str(names[i]) for i in sorted(names)]
    return list(names) if names else list(FALLBACK_NAMES)


def display_name(name: str) -> str:
    return name.replace("_", " ")


def predict_miner(
    entry: dict[str, Any],
    image: np.ndarray,
    conf: float | None = None,
    confs: list[float] | None = None,
) -> tuple[list[DetBox], list[float]]:
    """Run miner inference; temporarily apply UI per-class thresholds when provided."""
    obj = entry["obj"]
    names = class_names_for(entry)
    n_cls = max(1, len(names))

    native = getattr(obj, "_conf_thres_array", None)
    if confs is None or len(confs) == 0:
        if native is not None and len(native):
            thr = [float(x) for x in list(native)]
        elif conf is not None:
            thr = [float(np.clip(conf, 0.01, 0.99))] * n_cls
        else:
            thr = _default_confs_for(names)
    else:
        thr = [float(np.clip(c, 0.01, 0.99)) for c in confs]

    if len(thr) < n_cls:
        thr = thr + [thr[-1] if thr else 0.25] * (n_cls - len(thr))
    elif len(thr) > n_cls:
        thr = thr[:n_cls]

    prev = None
    if hasattr(obj, "_conf_thres_array"):
        prev = np.array(obj._conf_thres_array, dtype=np.float32, copy=True)
        obj._conf_thres_array = np.array(thr, dtype=np.float32)
    try:
        results = obj.predict_batch([image], 0, 0)
        boxes = results[0].boxes if results else []
        out = [
            DetBox(
                x1=int(b.x1), y1=int(b.y1), x2=int(b.x2), y2=int(b.y2),
                cls_id=int(b.cls_id), conf=float(b.conf),
            )
            for b in boxes
        ]
    finally:
        if prev is not None:
            obj._conf_thres_array = prev
    return out, thr


def predict_pt(
    entry: dict[str, Any],
    image: np.ndarray,
    conf: float | None,
    imgsz: int,
    confs: list[float] | None = None,
) -> tuple[list[DetBox], list[float]]:
    names = class_names_for(entry)
    n_cls = max(1, len(names))
    if confs is None or len(confs) == 0:
        thr = [float(np.clip(conf if conf is not None else 0.25, 0.01, 0.99))] * n_cls
    else:
        thr = [float(np.clip(c, 0.01, 0.99)) for c in confs]
        if len(thr) < n_cls:
            thr = thr + [thr[-1]] * (n_cls - len(thr))
        elif len(thr) > n_cls:
            thr = thr[:n_cls]

    floor = float(min(thr)) if thr else 0.25
    imgsz = int(np.clip(imgsz, 32, 2048))
    device = entry.get("device") or resolve_torch_device(entry.get("device_mode"))
    results = entry["obj"].predict(
        source=image,
        conf=floor,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )
    out: list[DetBox] = []
    if not results:
        return out, thr
    r0 = results[0]
    if r0.boxes is None or len(r0.boxes) == 0:
        return out, thr
    xyxy = r0.boxes.xyxy.cpu().numpy()
    scores = r0.boxes.conf.cpu().numpy()
    clss = r0.boxes.cls.cpu().numpy().astype(int)
    h, w = image.shape[:2]
    for (x1, y1, x2, y2), sc, cid in zip(xyxy, scores, clss):
        cid_i = int(cid)
        t = thr[cid_i] if 0 <= cid_i < len(thr) else floor
        if float(sc) < t:
            continue
        out.append(
            DetBox(
                x1=max(0, min(w, int(x1))),
                y1=max(0, min(h, int(y1))),
                x2=max(0, min(w, int(x2))),
                y2=max(0, min(h, int(y2))),
                cls_id=cid_i,
                conf=float(sc),
            )
        )
    return out, thr


def parse_confs_form(confs_raw: str | None, conf: float, n_hint: int = 4) -> list[float] | None:
    """Parse confs JSON array from form; None means use single conf."""
    if confs_raw:
        try:
            data = json.loads(confs_raw)
            if isinstance(data, list) and data:
                return [float(x) for x in data]
        except Exception:
            pass
    return None


def _model_cache_dir(model_id: str) -> Path:
    digest = hashlib.sha1(model_id.encode("utf-8")).hexdigest()[:20]
    path = (RESULTS / digest).resolve()
    if not _is_under(path, RESULTS):
        raise HTTPException(400, "Invalid model cache path")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _image_hash(raw: bytes) -> str:
    return hashlib.sha1(raw).hexdigest()


def _settings_key(
    device_mode: str | None,
    confs: list[float] | None,
    imgsz: int | None,
) -> str:
    payload = {
        "device": normalize_device_mode(device_mode),
        "confs": [round(float(c), 4) for c in (confs or [])],
        "imgsz": int(imgsz) if imgsz is not None else None,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _result_id(image_hash: str, settings_key: str) -> str:
    return f"{image_hash[:24]}_{settings_key}"


def _result_dir(model_id: str, result_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f_]{8,80}", result_id):
        raise HTTPException(400, "Invalid result id")
    root = _model_cache_dir(model_id)
    path = (root / result_id).resolve()
    if not _is_under(path, root):
        raise HTTPException(400, "Invalid result path")
    return path


def _write_jpeg(path: Path, image_bgr: np.ndarray, quality: int = 90) -> None:
    ok, buf = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    path.write_bytes(buf.tobytes())


def _write_thumb(annotated_bgr: np.ndarray, path: Path, max_w: int = 480) -> None:
    h, w = annotated_bgr.shape[:2]
    if w > max_w:
        scale = max_w / float(w)
        annotated_bgr = cv2.resize(
            annotated_bgr,
            (max_w, max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    _write_jpeg(path, annotated_bgr, quality=70)


def _result_urls(model_id: str, result_id: str) -> dict[str, str]:
    from urllib.parse import quote

    q = f"model_id={quote(model_id, safe='')}"
    rid = quote(result_id, safe="")
    return {
        "image_url": f"/api/results/{rid}/annotated?{q}",
        "original_url": f"/api/results/{rid}/original?{q}",
        "preview_url": f"/api/results/{rid}/preview?{q}",
    }


def _cached_file_response(path: Path) -> FileResponse:
    if not path.is_file():
        raise HTTPException(404, "Image not found")
    return FileResponse(
        path,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "public, max-age=31536000, immutable",
        },
    )


def save_result(
    *,
    model_id: str,
    filename: str,
    image_hash: str,
    settings_key: str,
    image_bgr: np.ndarray,
    annotated_bgr: np.ndarray,
    payload: dict[str, Any],
) -> dict[str, Any]:
    rid = _result_id(image_hash, settings_key)
    dest = _result_dir(model_id, rid)
    dest.mkdir(parents=True, exist_ok=True)
    _write_jpeg(dest / "original.jpg", image_bgr, quality=82)
    _write_jpeg(dest / "annotated.jpg", annotated_bgr, quality=85)
    _write_thumb(annotated_bgr, dest / "thumb.jpg")
    meta = {
        "id": rid,
        "model_id": model_id,
        "filename": filename or "image.jpg",
        "image_hash": image_hash,
        "settings_key": settings_key,
        "created_at": time.time(),
        "kind": payload.get("kind"),
        "width": payload.get("width"),
        "height": payload.get("height"),
        "inference_ms": payload.get("inference_ms"),
        "load_ms": payload.get("load_ms"),
        "num_detections": payload.get("num_detections"),
        "counts": payload.get("counts") or {},
        "class_names": payload.get("class_names") or [],
        "detections": payload.get("detections") or [],
        "conf": payload.get("conf"),
        "confs": payload.get("confs"),
        "imgsz": payload.get("imgsz"),
        "device": payload.get("device"),
        "device_mode": payload.get("device_mode"),
        "cached": True,
    }
    (dest / "meta.json").write_text(json.dumps(meta, separators=(",", ":")), encoding="utf-8")
    return meta


def load_result_meta(model_id: str, result_id: str) -> dict[str, Any]:
    dest = _result_dir(model_id, result_id)
    meta_path = dest / "meta.json"
    if not meta_path.is_file():
        raise HTTPException(404, "Result not found")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(500, f"Corrupt result meta: {e}") from e
    meta["id"] = result_id
    meta["model_id"] = model_id
    return meta


def result_payload_fast(model_id: str, result_id: str) -> dict[str, Any]:
    """Meta + image URLs only (no base64) for fast cached loads."""
    dest = _result_dir(model_id, result_id)
    if not (dest / "meta.json").is_file():
        raise HTTPException(404, "Result not found")
    if not (dest / "annotated.jpg").is_file() or not (dest / "original.jpg").is_file():
        raise HTTPException(404, "Result images missing")
    # Backfill thumb for older cache entries
    thumb = dest / "thumb.jpg"
    if not thumb.is_file():
        try:
            ann = cv2.imread(str(dest / "annotated.jpg"))
            if ann is not None:
                _write_thumb(ann, thumb)
        except Exception:
            pass
    meta = dict(load_result_meta(model_id, result_id))
    meta.update(_result_urls(model_id, result_id))
    meta["result_id"] = result_id
    meta["from_cache"] = True
    return meta


def find_cached_result(
    model_id: str,
    image_hash: str,
    settings_key: str,
) -> dict[str, Any] | None:
    rid = _result_id(image_hash, settings_key)
    dest = _result_dir(model_id, rid)
    if not (dest / "meta.json").is_file():
        return None
    try:
        return result_payload_fast(model_id, rid)
    except HTTPException:
        return None


def list_results_for_model(model_id: str) -> list[dict[str, Any]]:
    root = _model_cache_dir(model_id)
    items: list[dict[str, Any]] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        meta_path = child / "meta.json"
        if not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        rid = child.name
        # Lazy thumb for old entries
        if not (child / "thumb.jpg").is_file() and (child / "annotated.jpg").is_file():
            try:
                ann = cv2.imread(str(child / "annotated.jpg"))
                if ann is not None:
                    _write_thumb(ann, child / "thumb.jpg")
            except Exception:
                pass
        items.append(
            {
                "id": rid,
                "model_id": model_id,
                "filename": meta.get("filename") or "image.jpg",
                "created_at": meta.get("created_at") or 0,
                "num_detections": meta.get("num_detections") or 0,
                "counts": meta.get("counts") or {},
                "inference_ms": meta.get("inference_ms"),
                "device": meta.get("device"),
                "width": meta.get("width"),
                "height": meta.get("height"),
                "preview_url": f"/api/results/{rid}/preview?model_id={model_id}",
            }
        )
    items.sort(key=lambda x: float(x.get("created_at") or 0), reverse=True)
    return items


@app.post("/api/predict")
async def api_predict(
    model_id: str = Form(...),
    file: UploadFile = File(...),
    conf: float = Form(0.25),
    imgsz: int = Form(640),
    confs: str = Form(""),
    device: str = Form(""),
    force: str = Form("false"),
):
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Empty file")
    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "Could not decode image")

    force_run = str(force).strip().lower() in ("1", "true", "yes", "on")
    device_mode = device.strip() or None
    conf_list = parse_confs_form(confs, conf, 4)
    # imgsz only matters for .pt; include requested value in cache key always
    settings_key = _settings_key(device_mode, conf_list, imgsz)
    image_hash = _image_hash(raw)
    filename = Path(file.filename or "image.jpg").name

    if not force_run:
        cached = find_cached_result(model_id, image_hash, settings_key)
        if cached is not None:
            cached["from_cache"] = True
            cached["load_ms"] = 0
            cached["result_id"] = cached.get("id")
            return cached

    t_load0 = time.perf_counter()
    try:
        entry = load_model(model_id, device_mode=device_mode)
        names = class_names_for(entry)
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Model load failed: {e}") from e
    load_ms = (time.perf_counter() - t_load0) * 1000

    t0 = time.perf_counter()
    try:
        conf_list = parse_confs_form(confs, conf, len(names)) or conf_list
        if entry["kind"] == "miner":
            boxes, used_confs = predict_miner(
                entry, image, conf=conf, confs=conf_list,
            )
            used_conf = float(min(used_confs)) if used_confs else float(conf)
            used_imgsz = None
        else:
            boxes, used_confs = predict_pt(
                entry, image, conf=conf, imgsz=imgsz, confs=conf_list,
            )
            used_conf = float(min(used_confs)) if used_confs else float(conf)
            used_imgsz = int(np.clip(imgsz, 32, 2048))
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Inference failed: {e}") from e
    ms = (time.perf_counter() - t0) * 1000

    annotated = draw_boxes(image, boxes, names)
    detections = [
        {
            "cls_id": int(b.cls_id),
            "class": display_name(
                names[int(b.cls_id)] if 0 <= int(b.cls_id) < len(names) else f"cls{b.cls_id}"
            ),
            "conf": round(float(b.conf), 4),
            "bbox": [int(b.x1), int(b.y1), int(b.x2), int(b.y2)],
        }
        for b in boxes
    ]
    counts: dict[str, int] = {}
    for d in detections:
        counts[d["class"]] = counts.get(d["class"], 0) + 1

    # Recompute settings key with resolved confs/imgsz so cache matches what was used
    settings_key = _settings_key(
        entry.get("device_mode") or device_mode,
        used_confs,
        used_imgsz if entry["kind"] == "pt" else imgsz,
    )

    payload = {
        "model_id": model_id,
        "kind": entry["kind"],
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "inference_ms": round(ms, 1),
        "load_ms": round(load_ms, 1),
        "num_detections": len(detections),
        "counts": counts,
        "class_names": [display_name(n) for n in names],
        "detections": detections,
        "conf": used_conf,
        "confs": used_confs,
        "imgsz": used_imgsz,
        "device": device_label(str(entry.get("device") or "cpu")),
        "device_mode": entry.get("device_mode") or normalize_device_mode(device_mode),
        "from_cache": False,
    }
    try:
        meta = save_result(
            model_id=model_id,
            filename=filename,
            image_hash=image_hash,
            settings_key=settings_key,
            image_bgr=image,
            annotated_bgr=annotated,
            payload=payload,
        )
        rid = meta["id"]
        payload["result_id"] = rid
        payload.update(_result_urls(model_id, rid))
    except Exception as e:
        print(f"result cache save skipped: {e}")
        # Fallback when disk cache fails: inline images once
        payload["image_b64"] = encode_jpeg_b64(annotated)
        payload["original_b64"] = encode_jpeg_b64(image, quality=82)
    return payload


@app.get("/api/results")
def api_list_results(model_id: str):
    return {"model_id": model_id, "results": list_results_for_model(model_id)}


@app.get("/api/results/{result_id}")
def api_get_result(result_id: str, model_id: str):
    return result_payload_fast(model_id, result_id)


@app.get("/api/results/{result_id}/annotated")
def api_result_annotated(result_id: str, model_id: str):
    return _cached_file_response(_result_dir(model_id, result_id) / "annotated.jpg")


@app.get("/api/results/{result_id}/original")
def api_result_original(result_id: str, model_id: str):
    return _cached_file_response(_result_dir(model_id, result_id) / "original.jpg")


@app.get("/api/results/{result_id}/preview")
def api_result_preview(result_id: str, model_id: str):
    dest = _result_dir(model_id, result_id)
    thumb = dest / "thumb.jpg"
    if thumb.is_file():
        return _cached_file_response(thumb)
    return _cached_file_response(dest / "annotated.jpg")


@app.delete("/api/results/{result_id}")
def api_delete_result(result_id: str, model_id: str):
    dest = _result_dir(model_id, result_id)
    if not dest.exists():
        raise HTTPException(404, "Result not found")
    shutil.rmtree(dest)
    return {"ok": True, "deleted": result_id}


def draw_boxes(
    image_bgr: np.ndarray,
    boxes: list[DetBox],
    names: list[str],
) -> np.ndarray:
    """Draw detections with visible box edges even when labels crowd together.

    Labels are small outline text (no filled background). Box outlines are
    redrawn last so rectangles stay visible under dense detections.
    """
    out = image_bgr.copy()
    h, w = out.shape[:2]
    thickness = max(2, int(round(min(h, w) / 480)))
    font_scale = max(0.58, min(0.85, min(h, w) / 850))
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_thick = 2
    pad = 2

    # Lower confidence first so stronger detections / labels end up on top.
    ordered = sorted(boxes, key=lambda b: float(b.conf))

    def _meta(b: DetBox) -> tuple[int, str, tuple[int, int, int], int, int, int, int]:
        cls_id = int(b.cls_id)
        raw = names[cls_id] if 0 <= cls_id < len(names) else f"cls{cls_id}"
        name = display_name(raw)
        color = color_for_class(cls_id, raw)
        bgr = (int(color[2]), int(color[1]), int(color[0]))
        x1, y1, x2, y2 = int(b.x1), int(b.y1), int(b.x2), int(b.y2)
        x1, y1 = max(0, min(w - 1, x1)), max(0, min(h - 1, y1))
        x2, y2 = max(0, min(w - 1, x2)), max(0, min(h - 1, y2))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        return cls_id, name, bgr, x1, y1, x2, y2

    # Pass 1 — rectangles only
    for b in ordered:
        _, _, bgr, x1, y1, x2, y2 = _meta(b)
        cv2.rectangle(out, (x1, y1), (x2, y2), bgr, thickness)

    def _overlap_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0
        return (ix2 - ix1) * (iy2 - iy1)

    occupied: list[tuple[int, int, int, int]] = []

    # Pass 2 — small text labels, no filled background
    for b in ordered:
        _, name, bgr, x1, y1, x2, y2 = _meta(b)
        label = f"{name} {float(b.conf):.2f}"
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, text_thick)
        lw = tw + pad * 2
        lh = th + baseline + pad * 2

        candidates = [
            (x1, y1 - lh),
            (x2 - lw, y1 - lh),
            (x1, y2),
            (x2 - lw, y2),
            (x1, y1),
            (x2 - lw, y1),
            (x1, y2 - lh),
            (x2 - lw, y2 - lh),
            ((x1 + x2 - lw) // 2, y1 - lh),
            ((x1 + x2 - lw) // 2, y2),
        ]

        best: tuple[int, int, int, int] | None = None
        best_score = None
        for lx, ly in candidates:
            lx = int(max(0, min(w - lw, lx)))
            ly = int(max(0, min(h - lh, ly)))
            rect = (lx, ly, lx + lw, ly + lh)
            score = sum(_overlap_area(rect, o) for o in occupied)
            if lx >= x1 and ly >= y1 and lx + lw <= x2 and ly + lh <= y2:
                score += 50
            if best_score is None or score < best_score:
                best_score = score
                best = rect
                if score == 0:
                    break

        assert best is not None
        lx1, ly1, lx2, ly2 = best
        occupied.append(best)

        tx = lx1 + pad
        ty = ly1 + pad + th
        cv2.putText(out, label, (tx, ty), font, font_scale, bgr, text_thick, cv2.LINE_AA)

    # Pass 3 — redraw box outlines so edges stay visible under stacked labels
    for b in ordered:
        _, _, bgr, x1, y1, x2, y2 = _meta(b)
        cv2.rectangle(out, (x1, y1), (x2, y2), bgr, thickness)

    return out


def encode_jpeg_b64(image_bgr: np.ndarray, quality: int = 90) -> str:
    ok, buf = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _sanitize_name(name: str) -> str:
    name = (name or "").strip().replace(" ", "_")
    if not SAFE_NAME.match(name):
        raise HTTPException(
            400,
            "Invalid name. Use 1–64 chars: letters, numbers, . _ - (must start alphanumeric).",
        )
    return name


def _safe_extract_zip(zf: zipfile.ZipFile, dest: Path) -> None:
    dest = dest.resolve()
    for info in zf.infolist():
        if info.is_dir():
            continue
        name = info.filename.replace("\\", "/")
        if name.startswith("/") or ".." in Path(name).parts:
            raise HTTPException(400, f"Unsafe path in zip: {info.filename}")
        if name.startswith("__MACOSX/") or name.endswith(".DS_Store"):
            continue
        target = (dest / name).resolve()
        if not _is_under(target, dest):
            raise HTTPException(400, f"Zip slip blocked: {info.filename}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(info) as src, open(target, "wb") as out:
            shutil.copyfileobj(src, out)


def _flatten_if_single_subdir(folder: Path) -> None:
    entries = [p for p in folder.iterdir() if p.name not in {".", ".."}]
    complete = _is_miner_model(folder) or _find_pt(folder) is not None
    if len(entries) == 1 and entries[0].is_dir() and not complete:
        sub = entries[0]
        for child in list(sub.iterdir()):
            dest = folder / child.name
            if dest.exists():
                if dest.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()
            shutil.move(str(child), str(dest))
        sub.rmdir()


def _ensure_weights_onnx(folder: Path) -> None:
    if (folder / "weights.onnx").is_file():
        return
    onnx = _find_onnx(folder)
    if onnx is None:
        return
    shutil.copy2(onnx, folder / "weights.onnx")


@app.on_event("startup")
def _startup() -> None:
    # Kick CUDA/ORT in background so first click is fast
    threading.Thread(target=ensure_runtime_ready, daemon=True).start()


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/models")
def api_models():
    return {
        "models": [m.model_dump() for m in discover_models()],
        "device": {
            "default": normalize_device_mode(),
            "cuda_available": cuda_available(),
            "options": ["auto", "cpu", "cuda"],
            "platform": platform.system(),
            "machine": platform.machine(),
        },
    }


@app.get("/api/device")
def api_device():
    return {
        "default": normalize_device_mode(),
        "cuda_available": cuda_available(),
        "resolved": device_label(resolve_torch_device()),
        "options": ["auto", "cpu", "cuda"],
        "platform": platform.system(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "id_root": str(ID_ROOT),
    }


@app.post("/api/models/upload")
async def api_upload_model(
    name: str = Form(...),
    files: list[UploadFile] = File(...),
    overwrite: bool = Form(True),
):
    """Upload miner+ONNX, a .pt file, or a zip containing either."""
    model_name = _sanitize_name(name)
    dest = UPLOADS / model_name
    if dest.exists() and not overwrite:
        raise HTTPException(409, f"Model '{model_name}' already exists.")

    if not files:
        raise HTTPException(400, "No files uploaded")

    tmp = UPLOADS / f".tmp_{model_name}_{int(time.time() * 1000)}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    try:
        if len(files) == 1 and (files[0].filename or "").lower().endswith(".zip"):
            raw = await files[0].read()
            zip_path = tmp / "pack.zip"
            zip_path.write_bytes(raw)
            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    _safe_extract_zip(zf, tmp)
            except zipfile.BadZipFile as e:
                raise HTTPException(400, f"Invalid zip: {e}") from e
            zip_path.unlink(missing_ok=True)
            _flatten_if_single_subdir(tmp)
        else:
            for uf in files:
                fname = Path(uf.filename or "").name
                if not fname or fname in {".", ".."}:
                    continue
                lower = fname.lower()
                allowed = (
                    lower == "miner.py"
                    or lower.endswith((".onnx", ".pt", ".py", ".yml", ".yaml", ".json", ".txt", ".md"))
                )
                if not allowed:
                    raise HTTPException(400, f"Unsupported file type: {fname}")
                data = await uf.read()
                # Normalize single .pt to weights.pt for consistency
                if lower.endswith(".pt") and len(files) == 1:
                    fname = "weights.pt"
                (tmp / fname).write_bytes(data)

        has_miner = _is_miner_model(tmp)
        has_pt = _find_pt(tmp) is not None
        if has_miner:
            _ensure_weights_onnx(tmp)
        elif has_pt:
            pass
        else:
            raise HTTPException(
                400,
                "Upload must be miner.py + .onnx, or a .pt weights file (or a zip of either).",
            )

        if dest.exists():
            mid_candidates = []
            mid_candidates.append(_model_id_for(dest))
            pt = _find_pt(dest)
            if pt:
                mid_candidates.append(_model_id_for(pt))
            with _cache_lock:
                for mid in mid_candidates:
                    prefix = f"{mid}:::"
                    for key in [k for k in _model_cache if k == mid or k.startswith(prefix)]:
                        _model_cache.pop(key, None)
            shutil.rmtree(dest)
        shutil.move(str(tmp), str(dest))
        tmp = None
    finally:
        if tmp is not None and tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)

    info = next((m for m in discover_models() if m.name == model_name and m.uploaded), None)
    if info is None:
        # .pt id may use file path; match by folder name
        info = next(
            (m for m in discover_models() if m.uploaded and model_name in m.id.split("/")),
            None,
        )
    if info is None:
        raise HTTPException(500, "Saved but model not discoverable")
    return {"ok": True, "model": info.model_dump()}


@app.delete("/api/models/{model_name}")
def api_delete_uploaded(model_name: str):
    model_name = _sanitize_name(model_name)
    dest = (UPLOADS / model_name).resolve()
    if not _is_under(dest, UPLOADS):
        raise HTTPException(400, "Invalid name")
    if not dest.is_dir():
        raise HTTPException(404, f"Uploaded model not found: {model_name}")
    mids = [_model_id_for(dest)]
    pt = _find_pt(dest)
    if pt:
        mids.append(_model_id_for(pt))
    with _cache_lock:
        for mid in mids:
            prefix = f"{mid}:::"
            for key in [k for k in _model_cache if k == mid or k.startswith(prefix)]:
                _model_cache.pop(key, None)
    shutil.rmtree(dest)
    return {"ok": True, "deleted": model_name}


@app.get("/api/models/{model_name}/download")
def api_download_uploaded(model_name: str):
    """Download an uploaded model folder as a zip."""
    import io

    model_name = _sanitize_name(model_name)
    dest = (UPLOADS / model_name).resolve()
    if not _is_under(dest, UPLOADS):
        raise HTTPException(400, "Invalid name")
    if not dest.is_dir():
        raise HTTPException(404, f"Uploaded model not found: {model_name}")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(dest.rglob("*")):
            if not path.is_file():
                continue
            if path.name.startswith("."):
                continue
            zf.write(path, path.relative_to(dest).as_posix())
    data = buf.getvalue()
    return Response(
        content=data,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{model_name}.zip"',
            "Content-Length": str(len(data)),
        },
    )


@app.post("/api/unload")
def api_unload(model_id: str = Form(...)):
    with _cache_lock:
        prefix = f"{model_id}:::"
        removed = [k for k in list(_model_cache) if k == model_id or k.startswith(prefix)]
        for key in removed:
            _model_cache.pop(key, None)
    return {"ok": True, "removed": removed, "remaining": list(_model_cache)}


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("DETECTOR_HOST", "0.0.0.0")
    port = int(os.environ.get("DETECTOR_PORT", "7860"))
    uvicorn.run("app:app", host=host, port=port, reload=False)
