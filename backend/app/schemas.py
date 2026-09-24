from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class BBox(BaseModel):
    x: int = Field(..., ge=0)
    y: int = Field(..., ge=0)
    w: int = Field(..., gt=0)
    h: int = Field(..., gt=0)

    @model_validator(mode="after")
    def validate_bbox(self):
        if self.w <= 0 or self.h <= 0:
            raise ValueError("BBox width and height must be positive")
        return self


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


class EmbeddingCandidate(BaseModel):
    gallery_id: str
    similarity: float


class EmbeddingResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_version: str
    preprocessing_version: str
    dimension: int
    embedding: list[float]


class SearchResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_version: str
    ranked: list[EmbeddingCandidate]
    accepted: list[EmbeddingCandidate]
    candidates: list[EmbeddingCandidate]
    threshold: float
    confidence_semantics: str
