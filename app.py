#!/usr/bin/env python3
"""Object Detector viewer: miner+ONNX or Ultralytics .pt, drop an image, see boxes."""

from __future__ import annotations

import base64
import importlib.util
import re
import shutil
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

# Extra single .pt files / dirs to surface (avoid every epoch checkpoint).
PT_EXTRA_FILES = [
    CAR_WASH / "full" / "best.pt",
]
PT_DIR_GLOBS = [
    (CAR_WASH / "full" / "exports", ("best.pt", "best_tw.pt", "last.pt", "weights.pt", "model.pt")),
    (CAR_WASH / "layer" / "exports", ("best.pt", "best_tw.pt", "last.pt", "weights.pt", "model.pt")),
    (CAR_WASH / "full" / "runs", ("best.pt", "best_tw.pt")),
]

CLASS_COLORS = {
    "broom": (46, 196, 182),
    "drainage gate": (255, 159, 28),
    "drainage_gate": (255, 159, 28),
    "nozzle": (231, 76, 60),
    "track": (52, 152, 219),
}
DEFAULT_COLOR = (180, 180, 180)
FALLBACK_NAMES = ["broom", "drainage gate", "nozzle", "track"]
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
PT_NAME_PREFER = ("best_tw.pt", "best.pt", "last.pt", "weights.pt", "model.pt")

app = FastAPI(title="Object Detector")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

_cache_lock = threading.Lock()
_model_cache: dict[str, dict[str, Any]] = {}


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


@dataclass
class DetBox:
    x1: int
    y1: int
    x2: int
    y2: int
    cls_id: int
    conf: float


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


def _model_id_for(path: Path) -> str:
    return str(path.resolve().relative_to(CAR_WASH.resolve())).replace("\\", "/")


def _group_for(path: Path) -> str:
    resolved = path.resolve()
    if UPLOADS.resolve() == resolved or UPLOADS.resolve() in resolved.parents:
        return "uploads"
    rel = resolved.relative_to(CAR_WASH.resolve())
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


def _append_miner(models: list[ModelInfo], seen: set[Path], folder: Path) -> None:
    resolved = folder.resolve()
    if resolved in seen or not _is_miner_model(folder):
        return
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
            weights=onnx.name,
            weights_mb=round(onnx.stat().st_size / (1024 * 1024), 2),
            kind="miner",
            uploaded=(group == "uploads"),
        )
    )


def _append_pt(models: list[ModelInfo], seen: set[Path], pt: Path, name: str | None = None) -> None:
    resolved = pt.resolve()
    if resolved in seen or not pt.is_file():
        return
    root = CAR_WASH.resolve()
    if not str(resolved).startswith(str(root)):
        return
    seen.add(resolved)
    group = _group_for(pt)
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
            default_imgsz=_imgsz_from_pt(pt),
            default_conf=0.25,
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
    path = (CAR_WASH / model_id).resolve()
    root = CAR_WASH.resolve()
    if not str(path).startswith(str(root)):
        raise HTTPException(400, "Invalid model path")
    if path.is_file() and path.suffix.lower() == ".pt":
        return "pt", path
    if path.is_dir() and _is_miner_model(path):
        return "miner", path
    if path.is_dir() and _find_pt(path):
        return "pt", _find_pt(path)  # type: ignore[return-value]
    raise HTTPException(404, f"Unknown model: {model_id}")


def load_model(model_id: str) -> dict[str, Any]:
    with _cache_lock:
        if model_id in _model_cache:
            return _model_cache[model_id]

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
            obj = mod.Miner(folder)
        except Exception as e:
            raise HTTPException(500, f"Miner init failed: {e}") from e
        entry = {"kind": "miner", "obj": obj, "path": folder}
    else:
        try:
            from ultralytics import YOLO

            obj = YOLO(str(path))
        except Exception as e:
            raise HTTPException(500, f"YOLO .pt load failed: {e}") from e
        entry = {"kind": "pt", "obj": obj, "path": path}

    with _cache_lock:
        _model_cache[model_id] = entry
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


def predict_miner(entry: dict[str, Any], image: np.ndarray) -> list[DetBox]:
    results = entry["obj"].predict_batch([image], 0, 0)
    boxes = results[0].boxes if results else []
    return [
        DetBox(
            x1=int(b.x1), y1=int(b.y1), x2=int(b.x2), y2=int(b.y2),
            cls_id=int(b.cls_id), conf=float(b.conf),
        )
        for b in boxes
    ]


def predict_pt(
    entry: dict[str, Any],
    image: np.ndarray,
    conf: float,
    imgsz: int,
) -> list[DetBox]:
    conf = float(np.clip(conf, 0.01, 0.99))
    imgsz = int(np.clip(imgsz, 32, 2048))
    # Ultralytics accepts BGR numpy
    results = entry["obj"].predict(
        source=image,
        conf=conf,
        imgsz=imgsz,
        verbose=False,
    )
    out: list[DetBox] = []
    if not results:
        return out
    r0 = results[0]
    if r0.boxes is None or len(r0.boxes) == 0:
        return out
    xyxy = r0.boxes.xyxy.cpu().numpy()
    scores = r0.boxes.conf.cpu().numpy()
    clss = r0.boxes.cls.cpu().numpy().astype(int)
    h, w = image.shape[:2]
    for (x1, y1, x2, y2), sc, cid in zip(xyxy, scores, clss):
        out.append(
            DetBox(
                x1=max(0, min(w, int(x1))),
                y1=max(0, min(h, int(y1))),
                x2=max(0, min(w, int(x2))),
                y2=max(0, min(h, int(y2))),
                cls_id=int(cid),
                conf=float(sc),
            )
        )
    return out


def draw_boxes(
    image_bgr: np.ndarray,
    boxes: list[DetBox],
    names: list[str],
) -> np.ndarray:
    out = image_bgr.copy()
    h, w = out.shape[:2]
    thickness = max(2, int(round(min(h, w) / 400)))
    font_scale = max(0.45, min(h, w) / 900)

    for b in boxes:
        cls_id = int(b.cls_id)
        raw = names[cls_id] if 0 <= cls_id < len(names) else f"cls{cls_id}"
        name = display_name(raw)
        color = CLASS_COLORS.get(raw) or CLASS_COLORS.get(name) or DEFAULT_COLOR
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
    """Upload miner+ONNX, a .pt file, or a zip containing either."""
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
                    _model_cache.pop(mid, None)
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
    if not str(dest).startswith(str(UPLOADS.resolve())):
        raise HTTPException(400, "Invalid name")
    if not dest.is_dir():
        raise HTTPException(404, f"Uploaded model not found: {model_name}")
    mids = [_model_id_for(dest)]
    pt = _find_pt(dest)
    if pt:
        mids.append(_model_id_for(pt))
    with _cache_lock:
        for mid in mids:
            _model_cache.pop(mid, None)
    shutil.rmtree(dest)
    return {"ok": True, "deleted": model_name}


@app.post("/api/predict")
async def api_predict(
    model_id: str = Form(...),
    file: UploadFile = File(...),
    conf: float = Form(0.25),
    imgsz: int = Form(640),
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
        entry = load_model(model_id)
        names = class_names_for(entry)
        if entry["kind"] == "miner":
            boxes = predict_miner(entry, image)
            used_conf = None
            used_imgsz = None
        else:
            boxes = predict_pt(entry, image, conf=conf, imgsz=imgsz)
            used_conf = float(np.clip(conf, 0.01, 0.99))
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

    return {
        "model_id": model_id,
        "kind": entry["kind"],
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "inference_ms": round(ms, 1),
        "num_detections": len(detections),
        "counts": counts,
        "class_names": [display_name(n) for n in names],
        "detections": detections,
        "conf": used_conf,
        "imgsz": used_imgsz,
        "image_b64": encode_jpeg_b64(annotated),
        "original_b64": encode_jpeg_b64(image, quality=85),
    }


@app.post("/api/unload")
def api_unload(model_id: str = Form(...)):
    with _cache_lock:
        _model_cache.pop(model_id, None)
    return {"ok": True, "remaining": list(_model_cache)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)
