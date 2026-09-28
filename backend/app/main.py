from __future__ import annotations

import io
import logging
import os
import csv
import json
import tarfile
import zipfile
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from .database import Base, engine, get_db
from .gallery_store import (
    COMMON_GALLERY_ID, create_gallery, ensure_common_gallery, fail_import,
    gallery_dir, list_galleries, public_gallery, read_common_gallery,
    read_gallery, retry_import, stage_import,
)
from .ml_gateway import MAX_IMAGE_BYTES, build_ml_gallery, call_ml, get_ml_ready
from .models import IdentificationRequest, ReplenishmentFile, ReplenishmentItem
from .schemas import (
    IdentificationRequestCreate,
    IdentificationRequestRead,
    IdentificationRequestUpdate,
    EmbeddingResponse,
    ReplenishmentRead,
    SearchResponse,
)

Base.metadata.create_all(bind=engine)
app = FastAPI(title="LCT Case API", version="1.0.0")
logger = logging.getLogger(__name__)

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
    if request.url.path in {"/", "/solo", "/many", "/gallery", "/replenishment"} or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "frontend")
UPLOAD_DIR = Path(os.getenv("LCT_UPLOAD_DIR", os.path.join(STATIC_DIR, "uploads"))).resolve()
REPLENISHMENT_DIR = (UPLOAD_DIR / "replenishment").resolve()
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


@app.get("/gallery")
def read_gallery_page() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "gallery.html"))


@app.get("/replenishment")
def read_replenishment() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "replenishment.html"))


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
        "model_version", "preprocessing_version", "embedding_dimension", "gallery_size", "ranking")},
            "gallery_ready": ml_state.get("gallery_ready", ml_state["gallery_size"] > 0)}


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
    gallery_id: str | None = Form(default=None),
):
    if not 1 <= topk <= 100:
        raise HTTPException(status_code=422, detail="topk must be between 1 and 100")
    if gallery_id:
        info = public_gallery(read_gallery(gallery_id))
        if not info["search_ready"]:
            raise HTTPException(status_code=409, detail="Add at least ten gallery images before searching")
    form = {"x": x, "y": y, "w": w, "h": h, "topk": topk}
    if gallery_id:
        form["gallery_id"] = gallery_id
    return await call_ml(
        "/v1/search", await _upload_bytes(image), image.filename,
        image.content_type, form,
    )


async def _finish_gallery_import(gallery_id: str, job_id: str) -> None:
    try:
        await build_ml_gallery(gallery_id, job_id)
    except Exception as exc:
        logger.exception("Gallery import failed: %s", gallery_id)
        fail_import(gallery_id, job_id, f"Embedding failed: {type(exc).__name__}")


@app.post("/api/galleries", status_code=201)
def new_gallery(name: str = Form(...)):
    return create_gallery(name)


@app.get("/api/galleries")
def galleries():
    return list_galleries()


@app.get("/api/common-gallery")
def common_gallery():
    return read_common_gallery()


@app.post("/api/common-gallery/images", status_code=202)
async def import_common_gallery_images(
    background_tasks: BackgroundTasks,
    images: list[UploadFile] = File(default=[]),
    archive: UploadFile | None = File(None),
    manifest: UploadFile | None = File(None),
):
    gallery_id = ensure_common_gallery()
    result = await stage_import(gallery_id, images, archive, manifest)
    background_tasks.add_task(_finish_gallery_import, gallery_id, result["job_id"])
    return result


@app.post("/api/common-gallery/retry", status_code=202)
def retry_common_gallery(background_tasks: BackgroundTasks):
    result = retry_import(COMMON_GALLERY_ID)
    background_tasks.add_task(_finish_gallery_import, COMMON_GALLERY_ID, result["job_id"])
    return result


@app.get("/api/galleries/{gallery_id}")
def gallery(gallery_id: str):
    return public_gallery(read_gallery(gallery_id))


@app.post("/api/galleries/{gallery_id}/images", status_code=202)
async def import_gallery_images(
    gallery_id: str,
    background_tasks: BackgroundTasks,
    images: list[UploadFile] = File(default=[]),
    archive: UploadFile | None = File(None),
    manifest: UploadFile | None = File(None),
):
    result = await stage_import(gallery_id, images, archive, manifest)
    background_tasks.add_task(_finish_gallery_import, gallery_id, result["job_id"])
    return result


@app.post("/api/galleries/{gallery_id}/retry", status_code=202)
def retry_gallery_images(gallery_id: str, background_tasks: BackgroundTasks):
    result = retry_import(gallery_id)
    background_tasks.add_task(_finish_gallery_import, gallery_id, result["job_id"])
    return result


@app.get("/api/galleries/{gallery_id}/images/{image_key}")
def gallery_image(gallery_id: str, image_key: str):
    meta = read_gallery(gallery_id)
    if not any(row["image_key"] == image_key for row in meta["images"]):
        raise HTTPException(status_code=404, detail="Gallery image not found")
    path = gallery_dir(gallery_id) / "images" / image_key
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Gallery image not found")
    return FileResponse(path)


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


def _save_replenishment_pair(db: Session, label: str, image_bytes: bytes, image_filename: str, text_bytes: bytes, text_filename: str) -> ReplenishmentRead:
    REPLENISHMENT_DIR.mkdir(parents=True, exist_ok=True)
    image_ext = Path(image_filename).suffix.lower() or ".png"
    text_ext = Path(text_filename).suffix.lower() or ".txt"
    image_name = f"{uuid4().hex}{image_ext}"
    text_name = f"{uuid4().hex}{text_ext}"
    (REPLENISHMENT_DIR / image_name).write_bytes(image_bytes)
    (REPLENISHMENT_DIR / text_name).write_bytes(text_bytes)

    item = ReplenishmentItem(
        label=label,
        image_name=image_name,
        text_name=text_name,
        image_path=str(REPLENISHMENT_DIR / image_name),
        text_path=str(REPLENISHMENT_DIR / text_name),
    )
    db.add(item)
    db.commit()
    db.refresh(item)

    result = ReplenishmentRead.model_validate(item)
    result.image_url = f"/api/replenishment/images/{item.image_name}"
    result.text_url = f"/api/replenishment/texts/{item.text_name}"
    return result


def _file_payload(item: ReplenishmentFile) -> dict:
    return {
        "id": item.id,
        "name": item.name,
        "kind": item.kind,
        "url": f"/api/replenishment/files/{item.id}/content",
        "created_at": item.created_at.isoformat(),
    }


BBOX_COLUMNS = ("image_id", "x", "y", "w", "h", "vehicle_id", "camera_id")


def _validate_bbox_table(payload: bytes, filename: str) -> None:
    """Validate the operator table before saving it to the replenishment DB."""
    try:
        decoded = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=422, detail="The table must be UTF-8 encoded") from exc

    suffix = Path(filename).suffix.lower()
    try:
        if suffix == ".json":
            value = json.loads(decoded)
            rows = value.get("rows", value) if isinstance(value, dict) else value
            columns = set(rows[0]) if isinstance(rows, list) and rows and isinstance(rows[0], dict) else set()
        else:
            sample = decoded[:4096]
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
            reader = csv.DictReader(io.StringIO(decoded), dialect=dialect)
            columns = set(reader.fieldnames or [])
            for row in reader:
                for numeric in ("x", "y", "w", "h"):
                    if row.get(numeric, "").strip() == "":
                        raise ValueError(f"empty {numeric}")
                    float(row[numeric])
                break
    except (ValueError, TypeError, csv.Error, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422, detail="Cannot read the coordinate table") from exc

    missing = [column for column in BBOX_COLUMNS if column not in columns]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"The table must contain columns: {', '.join(BBOX_COLUMNS)}",
        )


def _migrate_legacy_replenishment_files(db: Session) -> None:
    """Expose paired MVP uploads as independently manageable files once."""
    known = {
        (entry.source_item_id, entry.kind)
        for entry in db.execute(select(ReplenishmentFile).where(ReplenishmentFile.source_item_id.is_not(None))).scalars()
    }
    legacy = db.execute(select(ReplenishmentItem)).scalars().all()
    changed = False
    for item in legacy:
        for kind, name, path in (
            ("image", item.image_name, item.image_path),
            ("text", item.text_name, item.text_path),
        ):
            if (item.id, kind) not in known and Path(path).is_file():
                db.add(ReplenishmentFile(name=name, kind=kind, path=path, source_item_id=item.id))
                changed = True
    if changed:
        db.commit()


@app.get("/api/replenishment/files")
def list_replenishment_files(db: Annotated[Session, Depends(get_db)]):
    _migrate_legacy_replenishment_files(db)
    files = db.execute(
        select(ReplenishmentFile).order_by(ReplenishmentFile.created_at.desc(), ReplenishmentFile.id.desc())
    ).scalars().all()
    return [_file_payload(item) for item in files if Path(item.path).is_file()]


@app.post("/api/replenishment/files", status_code=status.HTTP_201_CREATED)
async def create_replenishment_file(
    kind: str = Form(...),
    file: UploadFile = File(...),
    db: Annotated[Session, Depends(get_db)] = None,
):
    if kind not in {"image", "text"}:
        raise HTTPException(status_code=422, detail="kind must be image or text")
    if not file.filename:
        raise HTTPException(status_code=422, detail="Filename is required")
    max_size = MAX_IMAGE_BYTES if kind == "image" else 1024 * 1024
    payload = await file.read(max_size + 1)
    if not payload:
        raise HTTPException(status_code=422, detail="File is empty")
    if len(payload) > max_size:
        raise HTTPException(status_code=413, detail="File is too large")

    suffix = Path(file.filename).suffix.lower()
    if kind == "image":
        if suffix not in {".png", ".jpg", ".jpeg"}:
            raise HTTPException(status_code=422, detail="Only PNG and JPEG images are supported")
        try:
            with Image.open(io.BytesIO(payload)) as source:
                if source.format not in {"JPEG", "PNG"} or source.width * source.height > 50_000_000:
                    raise ValueError("Unsupported image")
                source.verify()
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise HTTPException(status_code=422, detail="Invalid image file") from exc
    elif suffix not in {".txt", ".csv", ".json"}:
        raise HTTPException(status_code=422, detail="Upload a CSV, TXT or JSON coordinate table")
    else:
        _validate_bbox_table(payload, file.filename)

    REPLENISHMENT_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = f"{uuid4().hex}{suffix or ('.png' if kind == 'image' else '.txt')}"
    path = REPLENISHMENT_DIR / safe_name
    path.write_bytes(payload)
    entry = ReplenishmentFile(name=Path(file.filename).name, kind=kind, path=str(path))
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return _file_payload(entry)


@app.get("/api/replenishment/files/{file_id}/content")
def get_replenishment_file(file_id: int, db: Annotated[Session, Depends(get_db)]):
    item = db.get(ReplenishmentFile, file_id)
    if not item or not Path(item.path).is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(item.path, filename=item.name)


@app.patch("/api/replenishment/files/{file_id}")
def rename_replenishment_file(
    file_id: int,
    name: str = Form(...),
    db: Annotated[Session, Depends(get_db)] = None,
):
    item = db.get(ReplenishmentFile, file_id)
    clean_name = Path(name).name.strip()
    if not item:
        raise HTTPException(status_code=404, detail="File not found")
    if not clean_name:
        raise HTTPException(status_code=422, detail="Name is required")
    item.name = clean_name
    db.commit()
    db.refresh(item)
    return _file_payload(item)


@app.delete("/api/replenishment/files/{file_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_replenishment_file(file_id: int, db: Annotated[Session, Depends(get_db)]):
    item = db.get(ReplenishmentFile, file_id)
    if not item:
        raise HTTPException(status_code=404, detail="File not found")
    path = Path(item.path)
    db.delete(item)
    db.commit()
    if path.is_file():
        path.unlink()


async def _read_archive_entries(archive: UploadFile) -> dict[str, bytes]:
    data = await archive.read(150 * 1024 * 1024 + 1)
    if len(data) > 150 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Archive is too large")

    entries: dict[str, bytes] = {}
    suffix = Path(archive.filename or "").suffix.lower()
    if suffix == ".zip":
        with zipfile.ZipFile(io.BytesIO(data)) as zipped:
            for info in zipped.infolist():
                if info.is_dir() or info.filename.startswith("__MACOSX/"):
                    continue
                filename = info.filename.replace("\\", "/")
                if filename.startswith("/") or any(part in {".", "..", ""} for part in filename.split("/")):
                    continue
                entries[filename] = zipped.read(info)
    elif suffix in {".tar", ".tgz", ".gz", ".bz2", ".xz"}:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tfile:
            for member in tfile.getmembers():
                if not member.isfile():
                    continue
                filename = member.name.replace("\\", "/")
                if filename.startswith("/") or any(part in {".", "..", ""} for part in filename.split("/")):
                    continue
                entries[filename] = tfile.extractfile(member).read()
    else:
        raise HTTPException(status_code=422, detail="Archive format is not supported")

    if not entries:
        raise HTTPException(status_code=422, detail="Archive does not contain readable files")
    return entries


@app.post("/api/replenishment/archive", status_code=status.HTTP_201_CREATED)
async def create_replenishment_archive(
    archive: UploadFile = File(...),
    db: Annotated[Session, Depends(get_db)] = None,
):
    entries = await _read_archive_entries(archive)
    allowed = {".png": "image", ".jpg": "image", ".jpeg": "image", ".txt": "text", ".json": "text", ".csv": "text"}
    REPLENISHMENT_DIR.mkdir(parents=True, exist_ok=True)
    uploaded: list[ReplenishmentFile] = []
    for archive_name, payload in entries.items():
        kind = allowed.get(Path(archive_name).suffix.lower())
        if not kind or not payload:
            continue
        suffix = Path(archive_name).suffix.lower()
        stored_name = f"{uuid4().hex}{suffix}"
        path = REPLENISHMENT_DIR / stored_name
        path.write_bytes(payload)
        entry = ReplenishmentFile(name=Path(archive_name).name, kind=kind, path=str(path))
        db.add(entry)
        uploaded.append(entry)
    if not uploaded:
        raise HTTPException(status_code=422, detail="Archive does not contain supported images or text files")
    db.commit()
    for entry in uploaded:
        db.refresh(entry)
    return {"files": [_file_payload(entry) for entry in uploaded], "count": len(uploaded)}


@app.get("/api/replenishment", response_model=list[ReplenishmentRead])
def list_replenishment_items(db: Annotated[Session, Depends(get_db)]):
    items = db.execute(
        select(ReplenishmentItem).order_by(ReplenishmentItem.created_at.desc())
    ).scalars().all()
    payload = []
    for item in items:
        result = ReplenishmentRead.model_validate(item)
        result.image_url = f"/api/replenishment/images/{item.image_name}"
        result.text_url = f"/api/replenishment/texts/{item.text_name}"
        payload.append(result)
    return payload


@app.post("/api/replenishment", response_model=ReplenishmentRead, status_code=status.HTTP_201_CREATED)
async def create_replenishment_item(
    label: str = Form(...),
    image: UploadFile = File(...),
    text: UploadFile = File(...),
    db: Annotated[Session, Depends(get_db)] = None,
):
    clean_label = label.strip()
    if not clean_label:
        raise HTTPException(status_code=422, detail="Label is required")
    if not image.filename or not text.filename:
        raise HTTPException(status_code=422, detail="Image and text files are required")

    image_bytes = await image.read(MAX_IMAGE_BYTES + 1)
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image file is too large")
    if not image_bytes:
        raise HTTPException(status_code=422, detail="Image file is empty")

    text_bytes = await text.read(1024 * 1024 + 1)
    if len(text_bytes) > 1024 * 1024:
        raise HTTPException(status_code=413, detail="Text file is too large")

    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            if source.format not in {"JPEG", "PNG"} or source.width * source.height > 50_000_000:
                raise ValueError("Unsupported image")
            source.verify()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="Invalid image file") from exc

    return _save_replenishment_pair(
        db,
        clean_label,
        image_bytes,
        image.filename,
        text_bytes,
        text.filename,
    )


@app.get("/api/replenishment/images/{filename}")
def get_replenishment_image(filename: str):
    if Path(filename).name != filename or filename in {".", ".."}:
        raise HTTPException(status_code=404, detail="Image not found")
    path = REPLENISHMENT_DIR / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path)


@app.get("/api/replenishment/texts/{filename}")
def get_replenishment_text(filename: str):
    if Path(filename).name != filename or filename in {".", ".."}:
        raise HTTPException(status_code=404, detail="Text not found")
    path = REPLENISHMENT_DIR / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Text not found")
    return FileResponse(path)
