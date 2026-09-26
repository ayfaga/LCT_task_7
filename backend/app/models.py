from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Double, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from .database import Base


class IdentificationRequest(Base):
    __tablename__ = "identification_requests"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(50), default="pending", nullable=False)
    result: Mapped[str | None] = mapped_column(Text, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Double, default=0.0, nullable=True)
    image_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


class ReplenishmentItem(Base):
    __tablename__ = "replenishment_items"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    label: Mapped[str] = mapped_column(String(200), nullable=False)
    image_name: Mapped[str] = mapped_column(String(255), nullable=False)
    text_name: Mapped[str] = mapped_column(String(255), nullable=False)
    image_path: Mapped[str] = mapped_column(String(512), nullable=False)
    text_path: Mapped[str] = mapped_column(String(512), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


class ReplenishmentFile(Base):
    """An independently uploaded image or text asset for replenishment."""

    __tablename__ = "replenishment_files"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    path: Mapped[str] = mapped_column(String(512), nullable=False)
    # Legacy paired items are represented here too, without copying their files.
    source_item_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
