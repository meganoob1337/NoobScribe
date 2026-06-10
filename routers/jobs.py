"""Transcription job status endpoints."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from database.db import get_session
from database.models import Recording as RecordingORM
from database.models import TranscriptionJob as TranscriptionJobORM
from models import TranscriptionJobListResponse, TranscriptionJobReconcileResponse, TranscriptionJobResponse
from services.job_queue import cancel_rq_job
from services.transcription_jobs import (
    ACTIVE_JOB_STATUSES,
    JOB_STATUS_CANCELLED,
    job_to_dict,
    job_visible_in_ui,
    reconcile_stale_queued_jobs,
)

router = APIRouter(prefix="/transcription-jobs", tags=["transcription-jobs"])


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _to_response(job: TranscriptionJobORM, *, recording_name: Optional[str] = None) -> TranscriptionJobResponse:
    d = job_to_dict(job, recording_name=recording_name)
    return TranscriptionJobResponse(**d)


async def _recording_names_for_jobs(
    session: AsyncSession, jobs: List[TranscriptionJobORM]
) -> Dict[str, str]:
    recording_ids = {j.recording_id for j in jobs if j.recording_id}
    if not recording_ids:
        return {}
    q = select(RecordingORM).where(RecordingORM.id.in_(recording_ids))
    res = await session.execute(q)
    return {r.id: r.name for r in res.scalars().all()}


@router.get("", response_model=TranscriptionJobListResponse)
async def list_transcription_jobs(
    limit: int = 50,
    offset: int = 0,
    session: AsyncSession = Depends(get_session),
):
    """List visible jobs: active plus historic while RQ metadata still exists."""
    q = (
        select(TranscriptionJobORM)
        .order_by(TranscriptionJobORM.created_at.desc())
        .limit(max(1, min(limit, 200)))
        .offset(max(0, offset))
    )
    res = await session.execute(q)
    rows = list(res.scalars().all())
    visible = [j for j in rows if job_visible_in_ui(j)]
    names = await _recording_names_for_jobs(session, visible)
    return TranscriptionJobListResponse(
        data=[
            _to_response(j, recording_name=names.get(j.recording_id) if j.recording_id else None)
            for j in visible
        ]
    )


@router.post("/reconcile", response_model=TranscriptionJobReconcileResponse)
async def reconcile_transcription_jobs(
    stale_seconds: int = 60,
    session: AsyncSession = Depends(get_session),
):
    """Re-enqueue queued jobs stuck after lost or finished RQ work (idempotent)."""
    result = await reconcile_stale_queued_jobs(session, stale_seconds=max(30, stale_seconds))
    return TranscriptionJobReconcileResponse(**result)


@router.get("/{job_id}", response_model=TranscriptionJobResponse)
async def get_transcription_job(job_id: str, session: AsyncSession = Depends(get_session)):
    job = await session.get(TranscriptionJobORM, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    recording_name = None
    if job.recording_id:
        rec = await session.get(RecordingORM, job.recording_id)
        if rec:
            recording_name = rec.name
    return _to_response(job, recording_name=recording_name)


@router.post("/{job_id}/cancel", response_model=TranscriptionJobResponse)
async def cancel_transcription_job(job_id: str, session: AsyncSession = Depends(get_session)):
    job = await session.get(TranscriptionJobORM, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status not in ACTIVE_JOB_STATUSES:
        raise HTTPException(
            status_code=409,
            detail=f"Job cannot be cancelled (status={job.status})",
        )
    job.status = JOB_STATUS_CANCELLED
    job.completed_at = _utcnow()
    job.error_message = "Cancelled by user"
    job.progress_stage = "cancelled"
    if job.rq_job_id:
        cancel_rq_job(job.rq_job_id)
    await session.flush()
    recording_name = None
    if job.recording_id:
        rec = await session.get(RecordingORM, job.recording_id)
        if rec:
            recording_name = rec.name
    return _to_response(job, recording_name=recording_name)
