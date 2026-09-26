"""Standalone HTTP boundary for a versioned Vehicle ReID inference bundle."""

from __future__ import annotations

import io
import logging
import os
from threading import Lock

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from app.schemas import BBox, EmbeddingResponse, SearchResponse

from .runtime import GalleryNotReady, MLRuntime


logger = logging.getLogger(__name__)
app = FastAPI(title="LCT Vehicle ReID ML", version="1.0.0")
MAX_IMAGE_BYTES = 20 * 1024 * 1024


class GalleryBuildRequest(BaseModel):
    job_id: str


_runtime_lock = Lock()
_runtime_instance: MLRuntime | None = None
_runtime_key: tuple[str, str, str] | None = None


def _load_runtime(artifact_dir: str, gallery_path: str, device: str) -> MLRuntime:
    # functools.lru_cache permits duplicate concurrent cache misses. A cold
    # /ready burst would construct several 1.2 GB models and OOM the container.
    global _runtime_instance, _runtime_key
    key = (artifact_dir, gallery_path, device)
    with _runtime_lock:
        if _runtime_instance is not None and _runtime_key != key:
            raise RuntimeError("ML artifact configuration changed; restart the service")
        if _runtime_instance is None:
            loaded = MLRuntime(artifact_dir, gallery_path or None, device)
            _runtime_instance, _runtime_key = loaded, key
        return _runtime_instance


def _runtime(require_gallery: bool = False) -> MLRuntime:
    artifact_dir = os.getenv("LCT_ML_ARTIFACT_DIR", "")
    if not artifact_dir:
        raise HTTPException(status_code=503, detail="ML artifact directory is not configured")
    try:
        runtime = _load_runtime(
            artifact_dir, os.getenv("LCT_ML_GALLERY_PATH", ""),
            os.getenv("LCT_ML_DEVICE", "auto"),
        )
    except Exception as exc:
        logger.exception("ML artifacts could not be loaded")
        raise HTTPException(status_code=503, detail="ML artifacts are not ready") from exc
    if require_gallery and not runtime.search_ready:
        raise HTTPException(status_code=503, detail="ML gallery is not configured")
    return runtime


def _bbox(x: int, y: int, w: int, h: int) -> tuple[int, int, int, int]:
    try:
        value = BBox(x=x, y=y, w=w, h=h)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return value.x, value.y, value.w, value.h


def _decode_image(data: bytes) -> Image.Image:
    if not data:
        raise HTTPException(status_code=422, detail="Image file is empty")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image file is too large")
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format not in {"JPEG", "PNG"}:
                raise HTTPException(status_code=422, detail="Only JPEG and PNG are supported")
            if image.width * image.height > 50_000_000:
                raise HTTPException(status_code=413, detail="Image dimensions are too large")
            image.load()
            return image.copy()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="Invalid image file") from exc


def _image_bytes(upload: UploadFile) -> bytes:
    if not upload.filename:
        raise HTTPException(status_code=422, detail="Image file is required")
    data = upload.file.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image file is too large")
    return data


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/ready")
def ready():
    try:
        runtime = _runtime(require_gallery=True)
    except HTTPException:
        return JSONResponse(status_code=503, content={"status": "not_ready"})
    return {
        "status": "ready",
        "model_version": runtime.model_version,
        "preprocessing_version": runtime.preprocessing_version,
        "embedding_dimension": runtime.encoder.dimension,
        "gallery_size": len(runtime.gallery_ids),
        "ranking": "exact_cosine",
    }


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
def embedding(
    image: UploadFile = File(...),
    x: int = Form(...), y: int = Form(...),
    w: int = Form(...), h: int = Form(...),
):
    bbox = _bbox(x, y, w, h)
    source = _decode_image(_image_bytes(image))
    runtime = _runtime()
    try:
        vector = runtime.embed(source, bbox)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "model_version": runtime.model_version,
        "preprocessing_version": runtime.preprocessing_version,
        "dimension": runtime.encoder.dimension,
        "embedding": vector.tolist(),
    }


@app.post("/v1/search", response_model=SearchResponse)
def search(
    image: UploadFile = File(...),
    x: int = Form(...), y: int = Form(...),
    w: int = Form(...), h: int = Form(...),
    topk: int = Form(10),
    gallery_id: str | None = Form(None),
):
    if not 1 <= topk <= 100:
        raise HTTPException(status_code=422, detail="topk must be between 1 and 100")
    bbox = _bbox(x, y, w, h)
    source = _decode_image(_image_bytes(image))
    runtime = _runtime(require_gallery=gallery_id is None)
    try:
        vector = runtime.embed(source, bbox)
        if gallery_id is None:
            return runtime.search(vector, topk=topk)
        return runtime.search(vector, topk=topk, gallery_id=gallery_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Gallery not found") from exc
    except GalleryNotReady as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/internal/galleries/{gallery_id}/build")
def build_gallery(gallery_id: str, payload: GalleryBuildRequest):
    runtime = _runtime()
    try:
        return runtime.build_gallery(gallery_id, payload.job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Gallery not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
