"""Small backend client for the isolated ML HTTP service."""

from __future__ import annotations

import os

import httpx
from fastapi import HTTPException


MAX_IMAGE_BYTES = 20 * 1024 * 1024


def ml_url() -> str:
    return os.getenv("LCT_ML_SERVICE_URL", "http://127.0.0.1:8001").rstrip("/")


async def get_ml_ready() -> dict:
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(5.0, connect=2.0)) as client:
            response = await client.get(f"{ml_url()}/ready")
    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail="ML service is unavailable") from exc
    if response.status_code != 200:
        raise HTTPException(status_code=503, detail="ML service is not ready")
    return response.json()


async def call_ml(path: str, data: bytes, filename: str, content_type: str | None,
                  form: dict[str, int]) -> dict:
    if not filename or not data:
        raise HTTPException(status_code=422, detail="Image file is required")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image file is too large")
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(90.0, connect=3.0)) as client:
            response = await client.post(
                f"{ml_url()}{path}",
                data=form,
                files={"image": (filename, data, content_type or "application/octet-stream")},
            )
    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail="ML service is unavailable") from exc
    if response.status_code >= 500:
        raise HTTPException(status_code=503, detail="ML service failed")
    if response.status_code not in {200, 201}:
        try:
            detail = response.json().get("detail", "ML request failed")
        except ValueError:
            detail = "ML request failed"
        raise HTTPException(status_code=response.status_code, detail=detail)
    try:
        return response.json()
    except ValueError as exc:
        raise HTTPException(status_code=502, detail="Invalid ML service response") from exc
