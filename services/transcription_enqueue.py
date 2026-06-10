"""Shared helpers for enqueueing transcription work from API handlers."""
from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from config import get_config
from services.job_queue import check_redis_connection
from services.transcription_jobs import (
    JOB_TYPE_OPENAI,
    JOB_TYPE_RECORDING,
    JOB_TYPE_SPEAKER_EXTRACT,
    create_and_enqueue_job,
)


def require_queue_available() -> None:
    cfg = get_config()
    if not cfg.queue_enabled:
        raise HTTPException(
            status_code=503,
            detail="Transcription queue is disabled (QUEUE_TRANSCRIPTION=0).",
        )
    if not check_redis_connection():
        raise HTTPException(
            status_code=503,
            detail="Redis is unavailable. Start redis and a transcription worker.",
        )


async def enqueue_recording_transcription(
    session: AsyncSession,
    recording_id: str,
    params: Dict[str, Any],
    *,
    job_type: str = JOB_TYPE_RECORDING,
) -> str:
    require_queue_available()
    return await create_and_enqueue_job(
        session,
        job_type=job_type,
        recording_id=recording_id,
        params=params,
    )


async def enqueue_openai_transcription(
    session: AsyncSession,
    params: Dict[str, Any],
) -> str:
    require_queue_available()
    return await create_and_enqueue_job(
        session,
        job_type=JOB_TYPE_OPENAI,
        recording_id=None,
        params=params,
    )


def build_transcribe_params(
    *,
    language: Optional[str] = None,
    response_format: str = "verbose_json",
    timestamps: bool = True,
    word_timestamps: bool = False,
    diarize: bool = True,
    include_diarization_in_text: Optional[bool] = None,
    temperature: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "language": language,
        "response_format": response_format,
        "timestamps": timestamps,
        "word_timestamps": word_timestamps,
        "diarize": diarize,
        "include_diarization_in_text": include_diarization_in_text,
        "temperature": temperature,
    }
    if extra:
        params.update(extra)
    return params
