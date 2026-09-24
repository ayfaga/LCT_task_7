from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from .database import Base, engine, get_db
from .ml_gateway import MAX_IMAGE_BYTES, call_ml, get_ml_ready
from .models import IdentificationRequest
from .schemas import (
    IdentificationRequestCreate,
    IdentificationRequestRead,
    IdentificationRequestUpdate,
    EmbeddingResponse,
    SearchResponse,
)

Base.metadata.create_all(bind=engine)
app = FastAPI(title="LCT Case API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def disable_cache(request: Request, call_next):
    response = await call_next(request)
    if request.url.path in {"/", "/solo", "/many"} or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "frontend")
UPLOAD_DIR = Path(os.getenv("LCT_UPLOAD_DIR", os.path.join(STATIC_DIR, "uploads"))).resolve()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

async def _upload_bytes(image: UploadFile) -> bytes:
    if not image.filename:
        raise HTTPException(status_code=422, detail="Image file is required")
    data = await image.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image file is too large")
    if not data:
        raise HTTPException(status_code=422, detail="Image file is empty")
    return data


@app.get("/")
def read_index() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/solo")
def read_solo() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "solo.html"))


@app.get("/many")
def read_many() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "many.html"))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        ml_state = await get_ml_ready()
    except Exception:
        return JSONResponse(status_code=503, content={"status": "not_ready"})
    return {"status": "ready", **{key: ml_state[key] for key in (
        "model_version", "preprocessing_version", "embedding_dimension", "gallery_size", "ranking")}}


@app.get("/api/health")
def read_health():
    return {"status": "ok"}


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embedding(
    image: UploadFile = File(...),
    x: int = Form(...),
    y: int = Form(...),
    w: int = Form(...),
    h: int = Form(...),
):
    return await call_ml(
        "/v1/embeddings", await _upload_bytes(image), image.filename,
        image.content_type, {"x": x, "y": y, "w": w, "h": h},
    )


@app.post("/v1/search", response_model=SearchResponse)
@app.post("/api/identify", response_model=SearchResponse)
@app.post("/api/infer", response_model=SearchResponse)
async def search_matches(
    image: UploadFile = File(...),
    x: int = Form(...),
    y: int = Form(...),
    w: int = Form(...),
    h: int = Form(...),
    topk: int = Form(default=10),
):
    if not 1 <= topk <= 100:
        raise HTTPException(status_code=422, detail="topk must be between 1 and 100")
    return await call_ml(
        "/v1/search", await _upload_bytes(image), image.filename,
        image.content_type, {"x": x, "y": y, "w": w, "h": h, "topk": topk},
    )


@app.post("/api/requests", response_model=IdentificationRequestRead, status_code=status.HTTP_201_CREATED)
def create_request(payload: IdentificationRequestCreate, db: Annotated[Session, Depends(get_db)]):
    entry = IdentificationRequest(
        title=payload.title,
        description=payload.description,
        status="pending",
        image_name=payload.image_name,
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry


@app.get("/api/requests", response_model=list[IdentificationRequestRead])
def get_requests(db: Annotated[Session, Depends(get_db)]):
    result = db.execute(select(IdentificationRequest).order_by(IdentificationRequest.created_at.desc())).scalars().all()
    return result


@app.get("/api/requests/{request_id}", response_model=IdentificationRequestRead)
def get_request(request_id: int, db: Annotated[Session, Depends(get_db)]):
    item = db.get(IdentificationRequest, request_id)
    if not item:
        raise HTTPException(status_code=404, detail="Request not found")
    return item


@app.patch("/api/requests/{request_id}", response_model=IdentificationRequestRead)
def update_request(request_id: int, payload: IdentificationRequestUpdate, db: Annotated[Session, Depends(get_db)]):
    item = db.get(IdentificationRequest, request_id)
    if not item:
        raise HTTPException(status_code=404, detail="Request not found")

    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(item, field, value)

    db.commit()
    db.refresh(item)
    return item


@app.delete("/api/requests/{request_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_request(request_id: int, db: Annotated[Session, Depends(get_db)]):
    item = db.get(IdentificationRequest, request_id)
    if not item:
        raise HTTPException(status_code=404, detail="Request not found")
    db.delete(item)
    db.commit()


@app.post("/api/upload", response_model=IdentificationRequestRead)
def upload_image(
    title: str,
    description: str | None = None,
    mode: str = "solo",
    file: UploadFile = File(...),
    db: Annotated[Session, Depends(get_db)] = None,
):
    if not file.filename:
        raise HTTPException(status_code=422, detail="Filename is required")
    extension = Path(file.filename).suffix.lower()
    if extension not in {".jpg", ".jpeg", ".png"}:
        raise HTTPException(status_code=422, detail="Only JPEG and PNG are supported")
    payload = file.file.read(MAX_IMAGE_BYTES + 1)
    if len(payload) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image file is too large")
    try:
        with Image.open(io.BytesIO(payload)) as source:
            if source.format not in {"JPEG", "PNG"} or source.width * source.height > 50_000_000:
                raise ValueError("Unsupported image")
            source.verify()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="Invalid image file") from exc
    safe_name = f"{uuid4().hex}{extension}"
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    (UPLOAD_DIR / safe_name).write_bytes(payload)

    item = IdentificationRequest(
        title=title or ("Solo identification" if mode == "solo" else "Multi identification"),
        description=description,
        status="pending",
        image_name=safe_name,
    )
    db.add(item)
    db.commit()
    db.refresh(item)
    return item


@app.get("/api/images/{filename}")
def get_image(filename: str):
    if Path(filename).name != filename or filename in {".", ".."}:
        raise HTTPException(status_code=404, detail="Image not found")
    uploaded = UPLOAD_DIR / filename
    bundled = Path(STATIC_DIR) / "images" / filename
    image_path = uploaded if uploaded.is_file() else bundled
    if not image_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(image_path)
