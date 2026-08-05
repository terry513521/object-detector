#!/usr/bin/env python3
"""Object Detector viewer: pick a miner+ONNX pair, drop an image, see boxes."""

from __future__ import annotations

import base64
import importlib.util
import re
import shutil
import threading
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

HERE = Path(__file__).resolve().parent
CAR_WASH = HERE.parent
STATIC = HERE / "static"
UPLOADS = HERE / "uploads"
UPLOADS.mkdir(parents=True, exist_ok=True)

# Roots to scan for (miner.py + *.onnx) pairs.
SCAN_ROOTS = [
    UPLOADS,
    CAR_WASH / "layer" / "models",
    CAR_WASH / "layer" / "_cand",
    CAR_WASH / "layer" / "exports",
    CAR_WASH / "full" / "exports",
]

CLASS_COLORS = {
    "broom": (46, 196, 182),          # teal
    "drainage gate": (255, 159, 28),  # amber
    "nozzle": (231, 76, 60),          # red
    "track": (52, 152, 219),          # blue
}
DEFAULT_COLOR = (180, 180, 180)
FALLBACK_NAMES = ["broom", "drainage gate", "nozzle", "track"]
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

app = FastAPI(title="Object Detector")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

_cache_lock = threading.Lock()
_miner_cache: dict[str, Any] = {}


class ModelInfo(BaseModel):
    id: str
    name: str
    group: str
    path: str
    onnx: str
    onnx_mb: float
    uploaded: bool = False


def _find_onnx(folder: Path) -> Path | None:
    preferred = folder / "weights.onnx"
    if preferred.is_file():
        return preferred
    onnxs = sorted(folder.glob("*.onnx"))
    return onnxs[0] if onnxs else None


def _is_complete_model(folder: Path) -> bool:
    return (folder / "miner.py").is_file() and _find_onnx(folder) is not None


def _model_id_for(folder: Path) -> str:
    """Stable id relative to CAR_WASH (works for uploads and builtin)."""
    return str(folder.resolve().relative_to(CAR_WASH.resolve())).replace("\\", "/")


def _group_for(folder: Path) -> str:
    if UPLOADS.resolve() in folder.resolve().parents or folder.resolve() == UPLOADS.resolve():
        return "uploads"
    rel = folder.resolve().relative_to(CAR_WASH.resolve())
    if len(rel.parts) >= 2:
        return f"{rel.parts[0]}/{rel.parts[1]}"
    return rel.parts[0]


def discover_models() -> list[ModelInfo]:
    seen: set[Path] = set()
    models: list[ModelInfo] = []
    for root in SCAN_ROOTS:
        if not root.is_dir():
            continue
        for folder in sorted(p for p in root.iterdir() if p.is_dir()):
            resolved = folder.resolve()
            if resolved in seen:
                continue
            if not _is_complete_model(folder):
                continue
            seen.add(resolved)
            onnx = _find_onnx(folder)
            assert onnx is not None
            group = _group_for(folder)
            models.append(
                ModelInfo(
                    id=_model_id_for(folder),
                    name=folder.name,
                    group=group,
                    path=str(folder),
                    onnx=onnx.name,
                    onnx_mb=round(onnx.stat().st_size / (1024 * 1024), 2),
                    uploaded=(group == "uploads"),
                )
            )
    order = {
        "uploads": 0,
        "layer/models": 1,
        "layer/exports": 2,
        "full/exports": 3,
        "layer/_cand": 4,
    }
    models.sort(key=lambda m: (order.get(m.group, 9), m.name.lower()))
    return models


def _resolve_model_folder(model_id: str) -> Path:
    folder = (CAR_WASH / model_id).resolve()
    root = CAR_WASH.resolve()
    if not str(folder).startswith(str(root)):
        raise HTTPException(400, "Invalid model path")
    if not folder.is_dir():
        raise HTTPException(404, f"Unknown model: {model_id}")
    if not _is_complete_model(folder):
        raise HTTPException(404, f"Model incomplete: {model_id}")
    return folder


def load_miner(model_id: str):
    with _cache_lock:
        if model_id in _miner_cache:
            return _miner_cache[model_id]

    folder = _resolve_model_folder(model_id)

    mod_name = "viewer_miner_" + re.sub(r"[^A-Za-z0-9_]", "_", model_id)
    spec = importlib.util.spec_from_file_location(mod_name, folder / "miner.py")
    if spec is None or spec.loader is None:
        raise HTTPException(500, "Failed to load miner.py")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        miner = mod.Miner(folder)
    except Exception as e:
        raise HTTPException(500, f"Miner init failed: {e}") from e

    with _cache_lock:
        _miner_cache[model_id] = miner
    return miner


def class_names_for(miner) -> list[str]:
    names = getattr(miner, "class_names", None)
    if names is None:
        return list(FALLBACK_NAMES)
    return list(names)


def draw_boxes(
    image_bgr: np.ndarray,
    boxes: list,
    names: list[str],
) -> np.ndarray:
    out = image_bgr.copy()
    h, w = out.shape[:2]
    thickness = max(2, int(round(min(h, w) / 400)))
    font_scale = max(0.45, min(h, w) / 900)

    for b in boxes:
        cls_id = int(b.cls_id)
        name = names[cls_id] if 0 <= cls_id < len(names) else f"cls{cls_id}"
        color = CLASS_COLORS.get(name, DEFAULT_COLOR)
        bgr = (color[2], color[1], color[0])
        x1, y1, x2, y2 = int(b.x1), int(b.y1), int(b.x2), int(b.y2)
        cv2.rectangle(out, (x1, y1), (x2, y2), bgr, thickness)
        label = f"{name} {float(b.conf):.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        ty = max(0, y1 - th - baseline - 4)
        cv2.rectangle(out, (x1, ty), (x1 + tw + 6, y1), bgr, -1)
        cv2.putText(
            out,
            label,
            (x1 + 3, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (20, 20, 20),
            max(1, thickness - 1),
            cv2.LINE_AA,
        )
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
        # Skip junk / absolute / traversal
        name = info.filename.replace("\\", "/")
        if name.startswith("/") or ".." in Path(name).parts:
            raise HTTPException(400, f"Unsafe path in zip: {info.filename}")
        if name.startswith("__MACOSX/") or name.endswith(".DS_Store"):
            continue
        target = (dest / name).resolve()
        if not str(target).startswith(str(dest)):
            raise HTTPException(400, f"Zip slip blocked: {info.filename}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(info) as src, open(target, "wb") as out:
            shutil.copyfileobj(src, out)


def _flatten_if_single_subdir(folder: Path) -> None:
    """If zip unpacked to one top-level dir, hoist its contents up."""
    entries = [p for p in folder.iterdir() if p.name not in {".", ".."}]
    if len(entries) == 1 and entries[0].is_dir() and not _is_complete_model(folder):
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
    """If miner expects weights.onnx but zip used another name, symlink/copy."""
    if (folder / "weights.onnx").is_file():
        return
    onnx = _find_onnx(folder)
    if onnx is None:
        return
    # Prefer copy so Windows-ish envs still work; same filesystem hardlink-ish via copy
    shutil.copy2(onnx, folder / "weights.onnx")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/models")
def api_models():
    return {"models": [m.model_dump() for m in discover_models()]}


@app.post("/api/models/upload")
async def api_upload_model(
    name: str = Form(...),
    files: list[UploadFile] = File(...),
    overwrite: bool = Form(False),
):
    """Upload a model pack: either a .zip, or miner.py + *.onnx (multi-file)."""
    model_name = _sanitize_name(name)
    dest = UPLOADS / model_name
    if dest.exists() and not overwrite:
        raise HTTPException(409, f"Model '{model_name}' already exists. Enable overwrite to replace.")

    if not files:
        raise HTTPException(400, "No files uploaded")

    tmp = UPLOADS / f".tmp_{model_name}_{int(time.time() * 1000)}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    try:
        # Single zip?
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
                # Only allow expected extensions
                lower = fname.lower()
                if not (
                    lower == "miner.py"
                    or lower.endswith(".onnx")
                    or lower in {"chute_config.yml", "chute_config.yaml", "readme.md", "config.json"}
                ):
                    # Still allow other small sidecar files commonly shipped with miners
                    if lower.endswith((".py", ".yml", ".yaml", ".json", ".txt", ".md")):
                        pass
                    else:
                        raise HTTPException(400, f"Unsupported file type: {fname}")
                data = await uf.read()
                (tmp / fname).write_bytes(data)

        if not (tmp / "miner.py").is_file():
            raise HTTPException(400, "Upload must include miner.py")
        if _find_onnx(tmp) is None:
            raise HTTPException(400, "Upload must include an .onnx file")
        _ensure_weights_onnx(tmp)

        if dest.exists():
            # Drop cached miner for this id
            mid = _model_id_for(dest) if dest.is_dir() else None
            with _cache_lock:
                if mid:
                    _miner_cache.pop(mid, None)
            shutil.rmtree(dest)
        shutil.move(str(tmp), str(dest))
        tmp = None  # moved
    finally:
        if tmp is not None and tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)

    info = next((m for m in discover_models() if m.name == model_name and m.uploaded), None)
    if info is None:
        raise HTTPException(500, "Saved but model not discoverable")
    return {"ok": True, "model": info.model_dump()}


@app.delete("/api/models/{model_name}")
def api_delete_uploaded(model_name: str):
    model_name = _sanitize_name(model_name)
    dest = (UPLOADS / model_name).resolve()
    if not str(dest).startswith(str(UPLOADS.resolve())):
        raise HTTPException(400, "Invalid name")
    if not dest.is_dir():
        raise HTTPException(404, f"Uploaded model not found: {model_name}")
    mid = _model_id_for(dest)
    with _cache_lock:
        _miner_cache.pop(mid, None)
    shutil.rmtree(dest)
    return {"ok": True, "deleted": model_name}


@app.post("/api/predict")
async def api_predict(
    model_id: str = Form(...),
    file: UploadFile = File(...),
):
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Empty file")
    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "Could not decode image")

    t0 = time.perf_counter()
    try:
        miner = load_miner(model_id)
        names = class_names_for(miner)
        results = miner.predict_batch([image], 0, 0)
        boxes = results[0].boxes if results else []
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
            "class": names[int(b.cls_id)] if 0 <= int(b.cls_id) < len(names) else f"cls{b.cls_id}",
            "conf": round(float(b.conf), 4),
            "bbox": [int(b.x1), int(b.y1), int(b.x2), int(b.y2)],
        }
        for b in boxes
    ]
    counts: dict[str, int] = {}
    for d in detections:
        counts[d["class"]] = counts.get(d["class"], 0) + 1

    return {
        "model_id": model_id,
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "inference_ms": round(ms, 1),
        "num_detections": len(detections),
        "counts": counts,
        "class_names": names,
        "detections": detections,
        "image_b64": encode_jpeg_b64(annotated),
        "original_b64": encode_jpeg_b64(image, quality=85),
    }


@app.post("/api/unload")
def api_unload(model_id: str = Form(...)):
    with _cache_lock:
        _miner_cache.pop(model_id, None)
    return {"ok": True, "remaining": list(_miner_cache)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)
