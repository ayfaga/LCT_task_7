from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class IdentificationRequestCreate(BaseModel):
    title: str
    description: str | None = None
    image_name: str | None = None
    mode: str = "solo"


class IdentificationRequestUpdate(BaseModel):
    title: str | None = None
    description: str | None = None
    status: str | None = None
    result: str | None = None
    confidence: float | None = None
    image_name: str | None = None


class IdentificationRequestRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    description: str | None = None
    status: str
    result: str | None = None
    confidence: float | None = None
    image_name: str | None = None
    created_at: datetime
    updated_at: datetime


class IdentificationRequestProcess(BaseModel):
    result: str
    confidence: float
    image_name: str | None = None
