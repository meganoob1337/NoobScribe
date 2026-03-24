"""SQLAlchemy ORM models for recordings and transcription results."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy import JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.db import Base

if TYPE_CHECKING:
    pass


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Recording(Base):
    __tablename__ = "recordings"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(512), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(1024), nullable=False)
    file_path: Mapped[str] = mapped_column(String(2048), nullable=False)
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    file_size_bytes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    mime_type: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    hide_in_recordings: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="0"
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    transcriptions: Mapped[List["TranscriptionResult"]] = relationship(
        "TranscriptionResult",
        back_populates="recording",
        cascade="all, delete-orphan",
    )


class TranscriptionResult(Base):
    __tablename__ = "transcription_results"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    recording_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("recordings.id", ondelete="CASCADE"), nullable=False, index=True
    )
    full_text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    segments_json: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    language: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    model_id: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    diarization_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    speakers_json: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    word_timestamps: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    recording: Mapped["Recording"] = relationship("Recording", back_populates="transcriptions")
