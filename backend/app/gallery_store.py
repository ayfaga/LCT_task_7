"""Append-only gallery uploads for local manual ReID checks.

The backend stores original images and metadata. The ML service writes versioned
embedding archives into the same volume; uploaded images are never removed.
"""

from __future__ import annotations

import csv
import errno
import io
import json
import os
import re
import shutil
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from threading import Lock
from uuid import uuid4
from zipfile import BadZipFile, ZipFile

from fastapi import HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool


MAX_PIXELS = 50_000_000
COPY_CHUNK_BYTES = 1024 * 1024
DISK_RESERVE_BYTES = 256 * 1024 * 1024
PREVIEW_IMAGES = 100
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
GALLERY_ID_RE = re.compile(r"[0-9a-f]{32}\Z")
COMMON_GALLERY_ID = "00000000000000000000000000000001"
COMMON_GALLERY_NAME = "Общая галерея поиска"
_write_lock = Lock()


def root() -> Path:
    return Path(os.getenv("LCT_GALLERY_STATE_DIR", "/tmp/lct_gallery_state"))


def gallery_dir(gallery_id: str) -> Path:
    if not GALLERY_ID_RE.fullmatch(gallery_id):
        raise HTTPException(status_code=404, detail="Gallery not found")
    return root() / gallery_id


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def read_gallery(gallery_id: str) -> dict:
    path = gallery_dir(gallery_id) / "meta.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Gallery not found")
    return json.loads(path.read_text())


def public_gallery(meta: dict) -> dict:
    gallery_id = meta["gallery_id"]
    image_count = len(meta["images"])
    images = [
        {"gallery_id": row["gallery_id"], "filename": row["filename"],
         "image_url": f"/api/galleries/{gallery_id}/images/{row['image_key']}"}
        for row in meta["images"][-PREVIEW_IMAGES:]
    ]
    return {
        "gallery_id": gallery_id, "name": meta["name"], "state": meta["state"],
        "image_count": image_count, "images_truncated": image_count > PREVIEW_IMAGES,
        "minimum_for_search": 10,
        "search_ready": bool(meta.get("gallery_file") and image_count >= 10),
        "processed": meta.get("processed", 0),
        "pending_count": len(meta.get("pending", [])), "error": meta.get("error"),
        "images": images,
    }


def create_gallery(name: str) -> dict:
    name = name.strip()
    if not 1 <= len(name) <= 100:
        raise HTTPException(status_code=422, detail="Gallery name must have 1–100 characters")
    gallery_id = uuid4().hex
    folder = gallery_dir(gallery_id)
    (folder / "images").mkdir(parents=True, exist_ok=False)
    meta = _initial_meta(gallery_id, name)
    _write_json(folder / "meta.json", meta)
    return public_gallery(meta)


def _initial_meta(gallery_id: str, name: str) -> dict:
    return {"gallery_id": gallery_id, "name": name, "state": "collecting",
            "images": [], "pending": [], "gallery_file": None, "generation": 0,
            "processed": 0, "job_id": None, "error": None}


def read_common_gallery() -> dict:
    """Read the sole UI gallery; an empty gallery does not write to disk on GET."""
    path = gallery_dir(COMMON_GALLERY_ID) / "meta.json"
    return public_gallery(read_gallery(COMMON_GALLERY_ID) if path.is_file()
                          else _initial_meta(COMMON_GALLERY_ID, COMMON_GALLERY_NAME))


def ensure_common_gallery() -> str:
    """Create the stable UI gallery on first upload without touching other galleries."""
    folder = gallery_dir(COMMON_GALLERY_ID)
    with _write_lock:
        if not (folder / "meta.json").is_file():
            (folder / "images").mkdir(parents=True, exist_ok=True)
            _write_json(folder / "meta.json", _initial_meta(COMMON_GALLERY_ID, COMMON_GALLERY_NAME))
    return COMMON_GALLERY_ID


def list_galleries() -> list[dict]:
    base = root()
    if not base.is_dir():
        return []
    items = []
    for path in sorted(base.glob("*/meta.json")):
        if GALLERY_ID_RE.fullmatch(path.parent.name):
            try:
                items.append(public_gallery(json.loads(path.read_text())))
            except (OSError, ValueError, KeyError):
                continue
    return items


def _clean_filename(name: str) -> str:
    if not name or "\\" in name or name.startswith("/"):
        raise HTTPException(status_code=422, detail="Invalid archive filename")
    path = PurePosixPath(name)
    if any(part in {"", ".", ".."} for part in path.parts):
        raise HTTPException(status_code=422, detail="Unsafe archive path")
    return str(path)


def _valid_image(name: str, path: Path) -> tuple[int, int]:
    if Path(name).suffix.lower() not in IMAGE_EXTENSIONS:
        raise HTTPException(status_code=422, detail=f"Unsupported image: {name}")
    try:
        with Image.open(path) as image:
            if image.format not in {"JPEG", "PNG"} or image.width * image.height > MAX_PIXELS:
                raise ValueError("Unsupported image or dimensions")
            image.load()
            return image.size
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid image: {name}") from exc


def _parse_manifest(source) -> dict[str, dict]:
    if source is None:
        return {}
    try:
        with io.TextIOWrapper(source, encoding="utf-8-sig", newline="") as text:
            rows = csv.DictReader(text)
            if not rows.fieldnames or "filename" not in rows.fieldnames:
                raise ValueError("Manifest needs a filename column")
            result = {}
            for row in rows:
                filename = _clean_filename((row.get("filename") or "").strip())
                if filename in result:
                    raise ValueError("Duplicate manifest filename")
                result[filename] = row
            return result
    except (UnicodeDecodeError, ValueError, csv.Error) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid manifest: {exc}") from exc


def _copy_stream(source, destination: Path) -> None:
    """Copy a single member without accumulating the archive or images in RAM."""
    try:
        with destination.open("wb") as output:
            while chunk := source.read(COPY_CHUNK_BYTES):
                if shutil.disk_usage(destination.parent).free < len(chunk) + DISK_RESERVE_BYTES:
                    raise HTTPException(status_code=507, detail="Not enough free disk space for gallery upload")
                output.write(chunk)
    except OSError as exc:
        if exc.errno == errno.ENOSPC:
            raise HTTPException(status_code=507, detail="Not enough free disk space for gallery upload") from exc
        raise


def _uploaded_images(images: list[UploadFile] | None, archive: UploadFile | None,
                     manifest: UploadFile | None, scratch: Path) -> list[tuple[str, Path, dict]]:
    if bool(images) == bool(archive):
        raise HTTPException(status_code=422, detail="Supply images or one ZIP archive")
    raw: list[tuple[str, Path]] = []
    internal_manifest = None
    if manifest:
        manifest.file.seek(0)
    external_manifest = _parse_manifest(manifest.file) if manifest else None
    if archive:
        if Path(archive.filename or "").suffix.lower() != ".zip":
            raise HTTPException(status_code=422, detail="Archive must be ZIP")
        archive.file.seek(0)
        try:
            with ZipFile(archive.file) as zipped:
                selected = []
                for entry in zipped.infolist():
                    if entry.is_dir() or entry.filename.startswith("__MACOSX/"):
                        continue
                    filename = _clean_filename(entry.filename)
                    if Path(filename).name.startswith("._") or Path(filename).name == ".DS_Store":
                        continue
                    if entry.flag_bits & 1 or (entry.external_attr >> 16) & 0o170000 == 0o120000:
                        raise HTTPException(status_code=422, detail="Encrypted files and links are unsupported")
                    if filename == "manifest.csv":
                        if internal_manifest is not None:
                            raise HTTPException(status_code=422, detail="Invalid ZIP manifest")
                        with zipped.open(entry) as source:
                            internal_manifest = _parse_manifest(source)
                        continue
                    if Path(filename).suffix.lower() not in IMAGE_EXTENSIONS:
                        raise HTTPException(status_code=422, detail=f"Unsupported ZIP entry: {filename}")
                    selected.append((filename, entry))
                if external_manifest is not None and internal_manifest is not None:
                    raise HTTPException(status_code=422, detail="Supply one manifest only")
                manifest_rows = external_manifest if external_manifest is not None else (internal_manifest or {})
                _validate_manifest_names(manifest_rows, [name for name, _ in selected],
                                         external_manifest is not None or internal_manifest is not None)
                for filename, entry in selected:
                    path = scratch / uuid4().hex
                    with zipped.open(entry) as source:
                        _copy_stream(source, path)
                    raw.append((filename, path))
        except (BadZipFile, EOFError, RuntimeError) as exc:
            raise HTTPException(status_code=422, detail="Invalid ZIP archive") from exc
    else:
        manifest_rows = external_manifest or {}
        _validate_manifest_names(manifest_rows, [_clean_filename(image.filename or "") for image in images],
                                 external_manifest is not None)
        for image in images:
            filename = _clean_filename(image.filename or "")
            image.file.seek(0)
            path = scratch / uuid4().hex
            _copy_stream(image.file, path)
            raw.append((filename, path))
    if not raw:
        raise HTTPException(status_code=422, detail="Supply JPEG/PNG images")
    result = []
    seen_filenames = set()
    seen_ids = set()
    for filename, path in raw:
        if filename in seen_filenames:
            raise HTTPException(status_code=422, detail=f"Duplicate filename: {filename}")
        seen_filenames.add(filename)
        width, height = _valid_image(filename, path)
        row = manifest_rows.get(filename, {})
        label = (row.get("gallery_id") or PurePosixPath(filename).stem).strip()
        if not label or len(label) > 128 or label in seen_ids:
            raise HTTPException(status_code=422, detail=f"Invalid or duplicate gallery ID: {label}")
        seen_ids.add(label)
        fields = [row.get(key, "") for key in ("x", "y", "w", "h")]
        if any(value not in ("", None) for value in fields):
            try:
                x, y, w, h = map(int, fields)
            except (TypeError, ValueError) as exc:
                raise HTTPException(status_code=422, detail=f"Invalid BBox for {filename}") from exc
            if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > width or y + h > height:
                raise HTTPException(status_code=422, detail=f"BBox outside {filename}")
            bbox = [x, y, w, h]
        else:
            bbox = [0, 0, width, height]
        result.append((filename, path, {"gallery_id": label, "bbox": bbox}))
    return result


def _validate_manifest_names(rows: dict[str, dict], filenames: list[str], required: bool) -> None:
    if len(set(filenames)) != len(filenames):
        raise HTTPException(status_code=422, detail="Duplicate image filename in upload")
    if required and set(rows) != set(filenames):
        missing = sorted(set(filenames) - set(rows))[:3]
        extra = sorted(set(rows) - set(filenames))[:3]
        raise HTTPException(status_code=422, detail=f"Manifest filenames must match uploaded images; missing={missing}, extra={extra}. Include ZIP paths such as images/car.jpg")


async def stage_import(gallery_id: str, images: list[UploadFile] | None,
                       archive: UploadFile | None, manifest: UploadFile | None) -> dict:
    return await run_in_threadpool(_stage_import_sync, gallery_id, images, archive, manifest)


def _stage_import_sync(gallery_id: str, images: list[UploadFile] | None,
                       archive: UploadFile | None, manifest: UploadFile | None) -> dict:
    folder = gallery_dir(gallery_id)
    with TemporaryDirectory(prefix=".upload-", dir=folder) as temporary:
        incoming = _uploaded_images(images, archive, manifest, Path(temporary))
        return _commit_import(gallery_id, incoming)


def _commit_import(gallery_id: str, incoming: list[tuple[str, Path, dict]]) -> dict:
    with _write_lock:
        meta = read_gallery(gallery_id)
        if meta["state"] == "building":
            raise HTTPException(status_code=409, detail="Gallery import is already running")
        existing_ids = {row["gallery_id"] for row in meta["images"]}
        if existing_ids & {record["gallery_id"] for _, _, record in incoming}:
            raise HTTPException(status_code=409, detail="Gallery ID already exists")
        folder = gallery_dir(gallery_id)
        pending = []
        for filename, path, record in incoming:
            image_key = uuid4().hex + Path(filename).suffix.lower()
            os.replace(path, folder / "images" / image_key)
            pending.append({**record, "filename": filename, "image_key": image_key})
        meta.update(state="building", pending=pending, processed=0,
                    job_id=uuid4().hex, error=None)
        _write_json(folder / "meta.json", meta)
        return {"gallery_id": gallery_id, "job_id": meta["job_id"],
                "state": "building", "pending_count": len(pending)}


def retry_import(gallery_id: str) -> dict:
    with _write_lock:
        meta = read_gallery(gallery_id)
        if meta["state"] != "failed" or not meta["pending"]:
            raise HTTPException(status_code=409, detail="No failed import to retry")
        meta.update(state="building", processed=0, job_id=uuid4().hex, error=None)
        _write_json(gallery_dir(gallery_id) / "meta.json", meta)
        return {"gallery_id": gallery_id, "job_id": meta["job_id"],
                "state": "building", "pending_count": len(meta["pending"])}


def fail_import(gallery_id: str, job_id: str, reason: str) -> None:
    with _write_lock:
        meta = read_gallery(gallery_id)
        if meta.get("job_id") != job_id:
            return
        meta.update(state="failed", error=reason[:300])
        _write_json(gallery_dir(gallery_id) / "meta.json", meta)
