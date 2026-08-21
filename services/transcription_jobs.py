"""Create and serialize transcription job records."""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from database.models import TranscriptionJob as TranscriptionJobORM
from services.job_queue import (
    enqueue_transcription_job,
    get_rq_job_status,
    rq_job_exists,
    rq_job_needs_reenqueue,
)

logger = logging.getLogger(__name__)

JOB_TYPE_RECORDING = "recording_transcribe"
JOB_TYPE_OPENAI = "openai_transcribe"
JOB_TYPE_SPEAKER_EXTRACT = "speaker_extract"

JOB_STATUS_QUEUED = "queued"
JOB_STATUS_RUNNING = "running"
JOB_STATUS_COMPLETED = "completed"
JOB_STATUS_FAILED = "failed"
JOB_STATUS_CANCELLED = "cancelled"

ACTIVE_JOB_STATUSES = (JOB_STATUS_QUEUED, JOB_STATUS_RUNNING)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


async def create_and_enqueue_job(
    session: AsyncSession,
    *,
    job_type: str,
    recording_id: Optional[str],
    params: Dict[str, Any],
) -> str:
    job_id = str(uuid.uuid4())
    job = TranscriptionJobORM(
        id=job_id,
        job_type=job_type,
        status=JOB_STATUS_QUEUED,
        recording_id=recording_id,
        params_json=params,
    )
    session.add(job)
    await session.flush()
    # Commit before Redis enqueue so the worker can see the row immediately.
    await session.commit()

    try:
        job.rq_job_id = enqueue_transcription_job(job_id)
    except Exception as e:
        job.status = JOB_STATUS_FAILED
        job.error_message = f"Failed to enqueue transcription job: {e}"
        job.completed_at = _utcnow()
        await session.flush()
        await session.commit()
        raise

    await session.flush()
    await session.commit()
    return job_id


async def reconcile_stale_queued_jobs(
    session: AsyncSession,
    *,
    stale_seconds: int = 60,
) -> Dict[str, int]:
    """Re-enqueue queued jobs whose RQ work was lost or already finished."""
    cutoff = _utcnow() - timedelta(seconds=stale_seconds)
    q = select(TranscriptionJobORM).where(
        TranscriptionJobORM.status == JOB_STATUS_QUEUED,
        TranscriptionJobORM.created_at < cutoff,
    )
    rows = list((await session.execute(q)).scalars().all())

    re_enqueued = 0
    skipped = 0
    failed = 0

    for job in rows:
        rq_status = get_rq_job_status(job.rq_job_id) if job.rq_job_id else None
        if not rq_job_needs_reenqueue(rq_status):
            skipped += 1
            continue
        try:
            job.rq_job_id = enqueue_transcription_job(job.id)
            re_enqueued += 1
            logger.info("Re-enqueued stale queued job %s (rq_status=%s)", job.id, rq_status)
        except Exception as e:
            job.status = JOB_STATUS_FAILED
            job.error_message = f"Reconcile failed to re-enqueue: {e}"
            job.completed_at = _utcnow()
            failed += 1
            logger.error("Failed to re-enqueue stale job %s: %s", job.id, e)

    if re_enqueued or failed:
        await session.flush()

    return {
        "examined": len(rows),
        "re_enqueued": re_enqueued,
        "skipped": skipped,
        "failed": failed,
    }


def job_visible_in_ui(job: TranscriptionJobORM) -> bool:
    """Active jobs always visible; finished jobs only while RQ metadata remains."""
    if job.status in ACTIVE_JOB_STATUSES:
        return True
    if not job.rq_job_id:
        return False
    return rq_job_exists(job.rq_job_id)


def job_to_dict(job: TranscriptionJobORM, *, recording_name: Optional[str] = None) -> dict:
    return {
        "job_id": job.id,
        "job_type": job.job_type,
        "status": job.status,
        "recording_id": job.recording_id,
        "recording_name": recording_name,
        "transcription_id": job.transcription_id,
        "params": job.params_json,
        "progress_stage": job.progress_stage,
        "progress_current": job.progress_current,
        "progress_total": job.progress_total,
        "error_message": job.error_message,
        "result": job.result_json,
        "cancellable": job.status in ACTIVE_JOB_STATUSES,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
    }


def update_job_progress_sync(
    session: Session,
    job_id: str,
    *,
    stage: str,
    current: int,
    total: int,
) -> bool:
    """Write progress. Returns False if the job is already cancelled."""
    job = session.get(TranscriptionJobORM, job_id)
    if not job:
        return True
    if job.status == JOB_STATUS_CANCELLED:
        return False
    job.progress_stage = stage
    job.progress_current = current
    job.progress_total = total
    session.flush()
    return True


def list_jobs_for_recording_sync(session: Session, recording_id: str) -> List[TranscriptionJobORM]:
    q = (
        select(TranscriptionJobORM)
        .where(TranscriptionJobORM.recording_id == recording_id)
        .order_by(TranscriptionJobORM.created_at.desc())
    )
    return list(session.execute(q).scalars().all())
