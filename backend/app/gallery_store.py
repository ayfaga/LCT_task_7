"""Append-only gallery uploads for local manual ReID checks.

The backend stores original images and metadata. The ML service writes versioned
embedding archives into the same volume; uploaded images are never removed.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
from pathlib import Path, PurePosixPath
from threading import Lock
from uuid import uuid4
from zipfile import BadZipFile, ZipFile

from fastapi import HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError


MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_ARCHIVE_BYTES = 100 * 1024 * 1024
MAX_TOTAL_BYTES = 200 * 1024 * 1024
MAX_IMAGES_PER_IMPORT = 200
MAX_GALLERY_IMAGES = 1000
MAX_PIXELS = 50_000_000
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
GALLERY_ID_RE = re.compile(r"[0-9a-f]{32}\Z")
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
    images = [
        {"gallery_id": row["gallery_id"], "filename": row["filename"],
         "image_url": f"/api/galleries/{gallery_id}/images/{row['image_key']}"}
        for row in meta["images"]
    ]
    return {
        "gallery_id": gallery_id, "name": meta["name"], "state": meta["state"],
        "image_count": len(images), "minimum_for_search": 10,
        "search_ready": bool(meta.get("gallery_file") and len(images) >= 10),
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
    meta = {"gallery_id": gallery_id, "name": name, "state": "collecting",
            "images": [], "pending": [], "gallery_file": None, "generation": 0,
            "processed": 0, "job_id": None, "error": None}
    _write_json(folder / "meta.json", meta)
    return public_gallery(meta)


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


def _valid_image(name: str, data: bytes) -> tuple[int, int]:
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail=f"Image {name} exceeds 20 MiB")
    if Path(name).suffix.lower() not in IMAGE_EXTENSIONS:
        raise HTTPException(status_code=422, detail=f"Unsupported image: {name}")
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format not in {"JPEG", "PNG"} or image.width * image.height > MAX_PIXELS:
                raise ValueError("Unsupported image or dimensions")
            image.load()
            return image.size
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid image: {name}") from exc


def _parse_manifest(data: bytes | None) -> dict[str, dict]:
    if data is None:
        return {}
    if len(data) > 1024 * 1024:
        raise HTTPException(status_code=413, detail="Manifest exceeds 1 MiB")
    try:
        rows = csv.DictReader(io.StringIO(data.decode("utf-8-sig")))
        if not rows.fieldnames or "filename" not in rows.fieldnames:
            raise ValueError("Manifest needs a filename column")
        result = {}
        for row in rows:
            filename = _clean_filename((row.get("filename") or "").strip())
            if filename in result:
                raise ValueError("Duplicate manifest filename")
            result[filename] = row
        return result
    except (UnicodeDecodeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid manifest: {exc}") from exc


async def _read_upload(upload: UploadFile, limit: int) -> bytes:
    data = await upload.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(status_code=413, detail=f"Upload exceeds {limit // (1024 * 1024)} MiB")
    return data


async def _uploaded_images(images: list[UploadFile] | None, archive: UploadFile | None,
                           manifest: UploadFile | None) -> list[tuple[str, bytes, dict]]:
    if bool(images) == bool(archive):
        raise HTTPException(status_code=422, detail="Supply images or one ZIP archive")
    raw: list[tuple[str, bytes]] = []
    internal_manifest = None
    if archive:
        if Path(archive.filename or "").suffix.lower() != ".zip":
            raise HTTPException(status_code=422, detail="Archive must be ZIP")
        compressed = await _read_upload(archive, MAX_ARCHIVE_BYTES)
        try:
            with ZipFile(io.BytesIO(compressed)) as zipped:
                entries = zipped.infolist()
                if len(entries) > MAX_IMAGES_PER_IMPORT + 20:
                    raise HTTPException(status_code=413, detail="Too many ZIP entries")
                total = 0
                for entry in entries:
                    if entry.is_dir() or entry.filename.startswith("__MACOSX/"):
                        continue
                    filename = _clean_filename(entry.filename)
                    if Path(filename).name.startswith("._") or Path(filename).name == ".DS_Store":
                        continue
                    if entry.flag_bits & 1 or (entry.external_attr >> 16) & 0o170000 == 0o120000:
                        raise HTTPException(status_code=422, detail="Encrypted files and links are unsupported")
                    if filename == "manifest.csv":
                        if internal_manifest is not None or entry.file_size > 1024 * 1024:
                            raise HTTPException(status_code=422, detail="Invalid ZIP manifest")
                        internal_manifest = zipped.read(entry)
                        continue
                    if Path(filename).suffix.lower() not in IMAGE_EXTENSIONS:
                        raise HTTPException(status_code=422, detail=f"Unsupported ZIP entry: {filename}")
                    total += entry.file_size
                    if entry.file_size > MAX_IMAGE_BYTES or total > MAX_TOTAL_BYTES:
                        raise HTTPException(status_code=413, detail="ZIP images exceed size limit")
                    raw.append((filename, zipped.read(entry)))
        except BadZipFile as exc:
            raise HTTPException(status_code=422, detail="Invalid ZIP archive") from exc
    else:
        if len(images) > MAX_IMAGES_PER_IMPORT:
            raise HTTPException(status_code=413, detail="Too many images")
        total = 0
        for image in images:
            filename = _clean_filename(image.filename or "")
            data = await _read_upload(image, MAX_IMAGE_BYTES)
            total += len(data)
            if total > MAX_TOTAL_BYTES:
                raise HTTPException(status_code=413, detail="Images exceed 200 MiB")
            raw.append((filename, data))
    if not raw or len(raw) > MAX_IMAGES_PER_IMPORT:
        raise HTTPException(status_code=422, detail="Supply 1–200 JPEG/PNG images")
    if manifest and internal_manifest is not None:
        raise HTTPException(status_code=422, detail="Supply one manifest only")
    manifest_rows = _parse_manifest(await _read_upload(manifest, 1024 * 1024) if manifest else internal_manifest)
    if manifest_rows and set(manifest_rows) != {filename for filename, _ in raw}:
        raise HTTPException(status_code=422, detail="Manifest filenames must match uploaded images")
    result = []
    seen_filenames = set()
    seen_ids = set()
    for filename, data in raw:
        if filename in seen_filenames:
            raise HTTPException(status_code=422, detail=f"Duplicate filename: {filename}")
        seen_filenames.add(filename)
        width, height = _valid_image(filename, data)
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
        result.append((filename, data, {"gallery_id": label, "bbox": bbox}))
    return result


async def stage_import(gallery_id: str, images: list[UploadFile] | None,
                       archive: UploadFile | None, manifest: UploadFile | None) -> dict:
    incoming = await _uploaded_images(images, archive, manifest)
    with _write_lock:
        meta = read_gallery(gallery_id)
        if meta["state"] == "building":
            raise HTTPException(status_code=409, detail="Gallery import is already running")
        if len(meta["images"]) + len(incoming) > MAX_GALLERY_IMAGES:
            raise HTTPException(status_code=413, detail="Gallery exceeds 1000 images")
        existing_ids = {row["gallery_id"] for row in meta["images"]}
        if existing_ids & {record["gallery_id"] for _, _, record in incoming}:
            raise HTTPException(status_code=409, detail="Gallery ID already exists")
        folder = gallery_dir(gallery_id)
        pending = []
        for filename, data, record in incoming:
            image_key = uuid4().hex + Path(filename).suffix.lower()
            (folder / "images" / image_key).write_bytes(data)
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
