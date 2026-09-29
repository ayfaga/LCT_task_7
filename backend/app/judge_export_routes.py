"""Same-origin browser gateway for the isolated ML export worker."""

from __future__ import annotations

import os
import re
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from .ml_gateway import ml_url


router = APIRouter(prefix="/api/judge-export", tags=["jury export"])
JOB_ID = re.compile(r"[0-9a-f]{32}\Z")
DOWNLOADS = {"submission.csv", "embeddings.npy", "candidates.csv", "manifest.json"}


def _enabled() -> None:
    if os.getenv("LCT_JUDGE_EXPORT_ENABLED") != "1":
        raise HTTPException(status_code=403, detail="Judge export is disabled; see docs/JUDGE_EXPORT.md")


def _same_origin(request: Request) -> None:
    origin = request.headers.get("origin")
    if origin and urlparse(origin).netloc != request.headers.get("host"):
        raise HTTPException(status_code=403, detail="Cross-origin export upload is forbidden")
    if request.headers.get("sec-fetch-site") == "cross-site":
        raise HTTPException(status_code=403, detail="Cross-site export upload is forbidden")


def _job_id(value: str) -> None:
    if not JOB_ID.fullmatch(value):
        raise HTTPException(status_code=404, detail="Export job not found")


def _upstream_error(response: httpx.Response) -> HTTPException:
    try:
        detail = response.json().get("detail", "Export service failed")
    except (ValueError, AttributeError):
        detail = "Export service failed"
    return HTTPException(status_code=response.status_code, detail=detail)


@router.get("/config")
def config() -> dict:
    return {"enabled": os.getenv("LCT_JUDGE_EXPORT_ENABLED") == "1",
            "input": "one ZIP with test_gallery.csv, test_query.csv and images/<image_id>.jpg",
            "downloads": sorted(DOWNLOADS)}


@router.post("/jobs", status_code=202)
async def create_job(request: Request) -> dict:
    _enabled()
    _same_origin(request)
    if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/zip":
        raise HTTPException(status_code=415, detail="Send one ZIP as application/zip")
    try:
        timeout = httpx.Timeout(connect=5.0, read=3600.0, write=3600.0, pool=5.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(f"{ml_url()}/internal/judge-exports",
                                         content=request.stream(),
                                         headers={"content-type": "application/zip"})
    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail="ML export service is unavailable") from exc
    if response.status_code != 202:
        raise _upstream_error(response)
    return response.json()


@router.get("/jobs/{job_id}")
async def job_status(job_id: str) -> dict:
    _enabled()
    _job_id(job_id)
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{ml_url()}/internal/judge-exports/{job_id}")
    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail="ML export service is unavailable") from exc
    if response.status_code != 200:
        raise _upstream_error(response)
    return response.json()


@router.get("/jobs/{job_id}/files/{name}")
async def download(job_id: str, name: str) -> StreamingResponse:
    _enabled()
    _job_id(job_id)
    if name not in DOWNLOADS:
        raise HTTPException(status_code=404, detail="Export file not found")
    client = httpx.AsyncClient(timeout=httpx.Timeout(connect=5.0, read=600.0, write=5.0, pool=5.0))
    try:
        request = client.build_request("GET", f"{ml_url()}/internal/judge-exports/{job_id}/files/{name}")
        response = await client.send(request, stream=True)
    except httpx.RequestError as exc:
        await client.aclose()
        raise HTTPException(status_code=503, detail="ML export service is unavailable") from exc
    if response.status_code != 200:
        await response.aread()
        error = _upstream_error(response)
        await response.aclose()
        await client.aclose()
        raise error

    async def chunks():
        try:
            async for part in response.aiter_bytes(chunk_size=1024 * 1024):
                yield part
        finally:
            await response.aclose()
            await client.aclose()

    content_type = "application/octet-stream" if name.endswith(".npy") else (
        "application/json" if name.endswith(".json") else "text/csv; charset=utf-8")
    return StreamingResponse(chunks(), media_type=content_type,
                             headers={"Content-Disposition": f'attachment; filename="{name}"',
                                      "Cache-Control": "no-store"})
