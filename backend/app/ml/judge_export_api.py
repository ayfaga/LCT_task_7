"""Local, versioned organizer export jobs sharing the live ML encoder.

The request body is a raw ZIP stream, never a multipart UploadFile. This avoids
Starlette's temporary copy of a multi-gigabyte upload. Only one export may run
per ML worker; no incomplete files are offered for download.
"""

from __future__ import annotations

import errno
import asyncio
import json
import logging
import os
import re
import shutil
from pathlib import Path
from threading import Lock, Thread
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from ml.import_organizer_archive import (ComponentInput, IMAGE_ID, MAX_CSV_BYTES,
                                         MAX_JPEG_BYTES, build, inspect_archive)


logger = logging.getLogger(__name__)
JOB_ID = re.compile(r"[0-9a-f]{32}\Z")
DOWNLOADS = {"submission.csv", "embeddings.npy", "candidates.csv", "manifest.json"}
DISK_RESERVE_BYTES = 512 * 1024 * 1024
COPY_CHUNK_BYTES = 1024 * 1024
STATUS_EVERY_IMAGES = 16


def create_router(runtime_loader):
    router = APIRouter(prefix="/internal/judge-exports", tags=["jury export"])
    active_lock = Lock()
    part_upload_lock = asyncio.Lock()
    active = {"job_id": None}

    def enabled() -> None:
        if os.getenv("LCT_JUDGE_EXPORT_ENABLED") != "1":
            raise HTTPException(status_code=403, detail="Judge export is disabled")

    def root() -> Path:
        return Path(os.getenv("LCT_JUDGE_EXPORT_DIR", "/tmp/lct_judge_exports"))

    def folder(job_id: str) -> Path:
        if not JOB_ID.fullmatch(job_id):
            raise HTTPException(status_code=404, detail="Export job not found")
        return root() / job_id

    def write_status(path: Path, payload: dict) -> None:
        temporary = path / f".status.{uuid4().hex}.tmp"
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        os.replace(temporary, path / "status.json")

    def finish_active(job_id: str) -> None:
        with active_lock:
            if active["job_id"] == job_id:
                active["job_id"] = None

    def run_job(job_id: str, counts: dict, source: Path | ComponentInput,
                allow_partial: bool = False) -> None:
        path = folder(job_id)
        status = {"job_id": job_id, "state": "running", "counts": counts,
                  "processed": {"gallery": 0, "query": 0}}
        write_status(path, status)
        try:
            runtime = runtime_loader()
            artifact_dir = Path(os.environ["LCT_ML_ARTIFACT_DIR"])
            model = json.loads((artifact_dir / "model_manifest.json").read_text())
            if model["model_version"] != runtime.model_version or model["model_sha256"] != runtime.model_sha256:
                raise ValueError("Live encoder and export bundle have different versions")

            def progress(split: str, done: int, total: int) -> None:
                status["processed"][split] = done
                if done == total or done % STATUS_EVERY_IMAGES == 0:
                    write_status(path, status)

            result = build(source, artifact_dir, path / "output",
                           batch_size=int(os.getenv("LCT_JUDGE_EXPORT_BATCH_SIZE", "4")),
                           encoder=runtime.encoder, inference_lock=runtime._inference_lock,
                           progress=progress, allow_partial=allow_partial)
            allowed_scope = ("full organizer test archive", "PARTIAL TEST ONLY; not a full submission")
            if result["status"] != "completed" or result["scope"] not in allowed_scope:
                raise RuntimeError("Export did not produce a verified submission or labeled partial test")
            status.update(state="completed", validation=result["submission_validation"],
                          input_sha256=result["input_sha256"], scope=result["scope"])
            write_status(path, status)
        except Exception as exc:
            logger.exception("Judge export %s failed", job_id)
            status.update(state="failed", error=f"{type(exc).__name__}: {exc}"[:500])
            write_status(path, status)
        finally:
            finish_active(job_id)

    @router.post("", status_code=202)
    async def upload(request: Request) -> dict:
        enabled()
        if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/zip":
            raise HTTPException(status_code=415, detail="Send the organizer ZIP as application/zip")
        job_id = uuid4().hex
        with active_lock:
            if active["job_id"] is not None:
                raise HTTPException(status_code=409, detail="Another judge export is active")
            active["job_id"] = job_id
        path = folder(job_id)
        try:
            path.mkdir(parents=True, exist_ok=False)
            state = {"job_id": job_id, "state": "uploading", "received_bytes": 0}
            write_status(path, state)
            total = 0
            with (path / "archive.zip").open("wb") as output:
                async for chunk in request.stream():
                    for start in range(0, len(chunk), COPY_CHUNK_BYTES):
                        part = chunk[start:start + COPY_CHUNK_BYTES]
                        if not part:
                            continue
                        if shutil.disk_usage(path).free < len(part) + DISK_RESERVE_BYTES:
                            raise HTTPException(status_code=507, detail="Insufficient disk space for export")
                        output.write(part)
                        total += len(part)
            if total == 0:
                raise HTTPException(status_code=422, detail="ZIP is empty")
            _, counts = inspect_archive(path / "archive.zip")
            if counts["gallery"] < 10 or counts["query"] < 1:
                raise HTTPException(status_code=422, detail="Need at least 10 gallery and one query image")
            write_status(path, {"job_id": job_id, "state": "queued", "received_bytes": total,
                                "counts": counts, "processed": {"gallery": 0, "query": 0}})
            Thread(target=run_job, args=(job_id, counts, path / "archive.zip"), daemon=True,
                   name=f"judge-export-{job_id[:8]}").start()
            return {"job_id": job_id, "state": "queued", "counts": counts}
        except HTTPException as exc:
            if path.is_dir():
                write_status(path, {"job_id": job_id, "state": "failed", "error": exc.detail})
            finish_active(job_id)
            raise
        except Exception as exc:
            if path.is_dir():
                write_status(path, {"job_id": job_id, "state": "failed", "error": str(exc)[:500]})
            finish_active(job_id)
            if isinstance(exc, OSError) and exc.errno == errno.ENOSPC:
                raise HTTPException(status_code=507, detail="Insufficient disk space for export") from exc
            raise HTTPException(status_code=422, detail=f"Invalid organizer archive: {exc}") from exc

    def collecting(job_id: str) -> Path:
        path = folder(job_id)
        status_path = path / "status.json"
        if not status_path.is_file():
            raise HTTPException(status_code=404, detail="Export job not found")
        state = json.loads(status_path.read_text()).get("state")
        with active_lock:
            current = active["job_id"]
        if current != job_id or state != "collecting":
            raise HTTPException(status_code=409, detail="Component job is not collecting files")
        return path

    async def stream_part(request: Request, target: Path, max_bytes: int | None = None) -> int:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise HTTPException(status_code=409, detail=f"Duplicate component: {target.name}")
        temporary = target.parent / f".upload.{uuid4().hex}.tmp"
        total = 0
        try:
            with temporary.open("wb") as output:
                async for chunk in request.stream():
                    for start in range(0, len(chunk), COPY_CHUNK_BYTES):
                        part = chunk[start:start + COPY_CHUNK_BYTES]
                        if not part:
                            continue
                        total += len(part)
                        if max_bytes is not None and total > max_bytes:
                            raise HTTPException(status_code=413, detail=f"Component exceeds {max_bytes} bytes")
                        if shutil.disk_usage(target.parent).free < len(part) + DISK_RESERVE_BYTES:
                            raise HTTPException(status_code=507, detail="Insufficient disk space for export")
                        output.write(part)
            if total == 0:
                raise HTTPException(status_code=422, detail="Empty component")
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return total

    @router.post("/components", status_code=201)
    def create_components() -> dict:
        enabled()
        job_id = uuid4().hex
        with active_lock:
            if active["job_id"] is not None:
                raise HTTPException(status_code=409, detail="Another judge export is active")
            active["job_id"] = job_id
        path = folder(job_id)
        try:
            path.mkdir(parents=True, exist_ok=False)
            write_status(path, {"job_id": job_id, "state": "collecting",
                                "input_kind": "components", "uploaded": {"gallery": 0, "query": 0}})
        except Exception:
            finish_active(job_id)
            raise
        return {"job_id": job_id, "state": "collecting"}

    @router.put("/components/{job_id}/{split}/csv")
    async def upload_csv(job_id: str, split: str, request: Request) -> dict:
        enabled()
        if split not in {"gallery", "query"}:
            raise HTTPException(status_code=404, detail="Unknown split")
        async with part_upload_lock:
            path = collecting(job_id)
            target = path / "parts" / f"{split}.csv"
            size = await stream_part(request, target, MAX_CSV_BYTES)
        return {"split": split, "kind": "csv", "bytes": size}

    @router.post("/components/{job_id}/{split}/zip", status_code=201)
    async def upload_zip(job_id: str, split: str, request: Request) -> dict:
        enabled()
        if split not in {"gallery", "query"}:
            raise HTTPException(status_code=404, detail="Unknown split")
        if request.headers.get("content-type", "").split(";", 1)[0].lower() != "application/zip":
            raise HTTPException(status_code=415, detail="Send ZIP as application/zip")
        async with part_upload_lock:
            path = collecting(job_id)
            source_id = uuid4().hex
            size = await stream_part(request, path / "parts" / split / "zips" / f"{source_id}.zip")
        return {"split": split, "kind": "zip", "source_id": source_id, "bytes": size}

    @router.put("/components/{job_id}/{split}/images/{image_id}")
    async def upload_image(job_id: str, split: str, image_id: str, request: Request) -> dict:
        enabled()
        if split not in {"gallery", "query"} or not IMAGE_ID.fullmatch(image_id):
            raise HTTPException(status_code=404, detail="Unknown split or image_id")
        async with part_upload_lock:
            path = collecting(job_id)
            size = await stream_part(request, path / "parts" / split / "images" /
                                     f"{image_id}.jpg", MAX_JPEG_BYTES)
        return {"split": split, "kind": "image", "image_id": image_id, "bytes": size}

    @router.post("/components/{job_id}/start", status_code=202)
    async def start_components(job_id: str, allow_partial: bool = False) -> dict:
        enabled()
        async with part_upload_lock:
            path = collecting(job_id)
            parts = path / "parts"
            def sources(split: str) -> tuple[Path, ...]:
                folder_path = parts / split / "images"
                zips = sorted((parts / split / "zips").glob("*.zip"))
                return tuple(([folder_path] if folder_path.is_dir() else []) + zips)
            spec = ComponentInput(parts / "gallery.csv", sources("gallery"),
                                  parts / "query.csv", sources("query"))
            if not spec.gallery_images or not spec.query_images:
                raise HTTPException(status_code=422, detail="Provide images for gallery and query")
            try:
                _, counts = inspect_archive(spec, allow_partial=allow_partial)
                if counts["gallery"] < 10 or counts["query"] < 1:
                    raise ValueError("Need at least 10 gallery and one query image")
            except Exception as exc:
                write_status(path, {"job_id": job_id, "state": "failed", "error": str(exc)[:500]})
                finish_active(job_id)
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            write_status(path, {"job_id": job_id, "state": "queued", "input_kind": "components",
                                "counts": counts, "processed": {"gallery": 0, "query": 0}})
            Thread(target=run_job, args=(job_id, counts, spec, allow_partial), daemon=True,
                   name=f"judge-export-{job_id[:8]}").start()
            return {"job_id": job_id, "state": "queued", "counts": counts}

    @router.post("/components/{job_id}/abort")
    async def abort_components(job_id: str) -> dict:
        enabled()
        async with part_upload_lock:
            path = collecting(job_id)
            write_status(path, {"job_id": job_id, "state": "interrupted",
                                "error": "Upload aborted; retained for diagnostics"})
            finish_active(job_id)
            return {"job_id": job_id, "state": "interrupted"}

    @router.get("/{job_id}")
    def get_job(job_id: str) -> dict:
        enabled()
        path = folder(job_id) / "status.json"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Export job not found")
        state = json.loads(path.read_text())
        if state["state"] in {"uploading", "collecting", "queued", "running"}:
            with active_lock:
                if active["job_id"] != job_id:
                    state.update(state="interrupted", error="Worker stopped; start a new export")
        return state

    @router.get("/{job_id}/files/{name}")
    def get_file(job_id: str, name: str) -> FileResponse:
        enabled()
        if name not in DOWNLOADS:
            raise HTTPException(status_code=404, detail="Export file not found")
        path = folder(job_id)
        status_file = path / "status.json"
        if not status_file.is_file() or json.loads(status_file.read_text()).get("state") != "completed":
            raise HTTPException(status_code=409, detail="Export is not complete")
        target = path / "output" / "submission" / name
        if not target.is_file():
            raise HTTPException(status_code=404, detail="Export file not found")
        return FileResponse(target, filename=name)

    return router
