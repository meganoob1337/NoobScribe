"""RQ task handlers for transcription jobs."""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from sqlalchemy import update

from audio import convert_audio_to_wav
from config import get_config
from database.models import Recording as RecordingORM
from database.models import TranscriptionJob as TranscriptionJobORM
from database.sync_db import get_sync_session, init_sync_db
from services.stored_recording_transcribe import (
    persist_pipeline_transcription_sync,
    run_transcription_for_recording_path,
    transcription_response_json_body,
)
from services.transcription_jobs import (
    JOB_STATUS_CANCELLED,
    JOB_STATUS_COMPLETED,
    JOB_STATUS_FAILED,
    JOB_STATUS_QUEUED,
    JOB_STATUS_RUNNING,
    JOB_TYPE_OPENAI,
    JOB_TYPE_RECORDING,
    JOB_TYPE_SPEAKER_EXTRACT,
    processing_seconds_from_datetimes,
    update_job_progress_sync,
)
from services.transcription_pipeline import run_transcription_pipeline
from services.worker_bootstrap import get_worker_asr_model, get_worker_speaker_db
from transcription import format_srt, format_vtt

logger = logging.getLogger(__name__)


class JobCancelled(Exception):
    """Raised when a job was cancelled cooperatively."""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _check_job_cancelled(job_id: str) -> None:
    with get_sync_session() as session:
        job = session.get(TranscriptionJobORM, job_id)
        if job and job.status == JOB_STATUS_CANCELLED:
            raise JobCancelled(f"Job {job_id} was cancelled")


def _make_progress_callback(job_id: str):
    def on_progress(stage: str, current: int, total: int) -> None:
        with get_sync_session() as session:
            if not update_job_progress_sync(
                session,
                job_id,
                stage=stage,
                current=current,
                total=total,
            ):
                raise JobCancelled(f"Job {job_id} was cancelled")

    return on_progress


def _params(job: TranscriptionJobORM) -> Dict[str, Any]:
    return job.params_json or {}


def _pipeline_result_to_job_payload(
    pr,
    *,
    transcription_id: Optional[str],
    recording_id: Optional[str],
    response_format: str,
) -> Dict[str, Any]:
    if response_format == "text":
        return {"text": pr.response.text, "transcription_id": transcription_id, "recording_id": recording_id}
    if response_format == "srt":
        return {
            "text": format_srt(pr.all_segments),
            "format": "srt",
            "transcription_id": transcription_id,
            "recording_id": recording_id,
        }
    if response_format == "vtt":
        return {
            "text": format_vtt(pr.all_segments),
            "format": "vtt",
            "transcription_id": transcription_id,
            "recording_id": recording_id,
        }
    body = transcription_response_json_body(pr, transcription_id or "")
    if recording_id:
        body["recording_id"] = recording_id
    return body


def _run_recording_job(job_id: str, recording_id: str) -> None:
    cfg = get_config()
    with get_sync_session() as session:
        job = session.get(TranscriptionJobORM, job_id)
        if not job:
            raise RuntimeError(f"Job {job_id} not found")
        if job.status == JOB_STATUS_CANCELLED:
            raise JobCancelled(f"Job {job_id} was cancelled")
        rec = session.get(RecordingORM, recording_id)
        if not rec:
            raise FileNotFoundError(f"Recording {recording_id} not found")
        if not rec.file_path or not os.path.exists(rec.file_path):
            raise FileNotFoundError(f"Audio file not found for recording {recording_id}")
        file_path = rec.file_path
        params = _params(job)

    asr_model = get_worker_asr_model()
    speaker_db = get_worker_speaker_db()
    on_progress = _make_progress_callback(job_id)

    pr = run_transcription_for_recording_path(
        file_path,
        asr_model,
        cfg,
        speaker_db,
        language=params.get("language"),
        word_timestamps=bool(params.get("word_timestamps", False)),
        diarize=bool(params.get("diarize", True)),
        include_diarization_in_text=params.get("include_diarization_in_text"),
        response_format=params.get("response_format", "verbose_json"),
        timestamps=bool(params.get("timestamps", True)),
        on_progress=on_progress,
    )

    with get_sync_session() as session:
        job_row = session.get(TranscriptionJobORM, job_id)
        if not job_row or job_row.status == JOB_STATUS_CANCELLED:
            raise JobCancelled(f"Job {job_id} was cancelled")
        rec_row = session.get(RecordingORM, recording_id)
        if not rec_row:
            raise RuntimeError(f"Recording {recording_id} not found")
        completed_at = _utcnow()
        tid = persist_pipeline_transcription_sync(
            session,
            rec_row,
            pr,
            diarize=bool(params.get("diarize", True)),
            word_timestamps=bool(params.get("word_timestamps", False)),
            processing_seconds=processing_seconds_from_datetimes(
                job_row.started_at, completed_at
            ),
        )
        payload = _pipeline_result_to_job_payload(
            pr,
            transcription_id=tid,
            recording_id=recording_id,
            response_format=params.get("response_format", "verbose_json"),
        )
        result = session.execute(
            update(TranscriptionJobORM)
            .where(
                TranscriptionJobORM.id == job_id,
                TranscriptionJobORM.status == JOB_STATUS_RUNNING,
            )
            .values(
                transcription_id=tid,
                result_json=payload,
                status=JOB_STATUS_COMPLETED,
                completed_at=completed_at,
                progress_stage="completed",
            )
        )
        if result.rowcount == 0:
            raise JobCancelled(f"Job {job_id} was cancelled")


def _run_openai_job(job_id: str) -> None:
    cfg = get_config()
    with get_sync_session() as session:
        job = session.get(TranscriptionJobORM, job_id)
        if not job:
            raise RuntimeError(f"Job {job_id} not found")
        if job.status == JOB_STATUS_CANCELLED:
            raise JobCancelled(f"Job {job_id} was cancelled")
        params = _params(job)
    temp_file = params.get("temp_file")
    if not temp_file or not os.path.exists(temp_file):
        raise FileNotFoundError(f"Upload file not found: {temp_file}")

    wav_file: Optional[str] = None
    pr = None
    try:
        _check_job_cancelled(job_id)
        wav_file = convert_audio_to_wav(temp_file)
        _check_job_cancelled(job_id)
        asr_model = get_worker_asr_model()
        speaker_db = get_worker_speaker_db()
        on_progress = _make_progress_callback(job_id)
        pr = run_transcription_pipeline(
            wav_file,
            asr_model,
            cfg,
            speaker_db,
            language=params.get("language"),
            word_timestamps=bool(params.get("word_timestamps", False)),
            diarize=bool(params.get("diarize", True)),
            include_diarization_in_text=params.get("include_diarization_in_text"),
            response_format=params.get("response_format", "json"),
            timestamps=bool(params.get("timestamps", False)),
            on_progress=on_progress,
        )
        with get_sync_session() as session:
            job_row = session.get(TranscriptionJobORM, job_id)
            if not job_row or job_row.status == JOB_STATUS_CANCELLED:
                raise JobCancelled(f"Job {job_id} was cancelled")
            payload = _pipeline_result_to_job_payload(
                pr,
                transcription_id=None,
                recording_id=None,
                response_format=params.get("response_format", "json"),
            )
            result = session.execute(
                update(TranscriptionJobORM)
                .where(
                    TranscriptionJobORM.id == job_id,
                    TranscriptionJobORM.status == JOB_STATUS_RUNNING,
                )
                .values(
                    result_json=payload,
                    status=JOB_STATUS_COMPLETED,
                    completed_at=_utcnow(),
                    progress_stage="completed",
                )
            )
            if result.rowcount == 0:
                raise JobCancelled(f"Job {job_id} was cancelled")
    finally:
        if pr is not None:
            for chunk_path in pr.paths_to_cleanup:
                try:
                    if os.path.exists(chunk_path):
                        os.unlink(chunk_path)
                except OSError:
                    pass
        try:
            if temp_file and os.path.exists(temp_file):
                os.unlink(temp_file)
        except OSError:
            pass
        if wav_file and os.path.exists(wav_file) and wav_file != temp_file:
            try:
                os.unlink(wav_file)
            except OSError:
                pass


def _acquire_job_for_processing(job_id: str) -> Optional[tuple[str, Optional[str]]]:
    """Load job and mark running; retry briefly if the API commit is still in flight."""
    for attempt in range(5):
        with get_sync_session() as session:
            job = session.get(TranscriptionJobORM, job_id)
            if not job:
                if attempt < 4:
                    time.sleep(0.2 * (attempt + 1))
                    continue
                logger.error("Job %s not found after retries", job_id)
                raise RuntimeError(f"Job {job_id} not found in database after retries")
            if job.status == JOB_STATUS_CANCELLED:
                logger.info("Job %s was cancelled; skipping", job_id)
                return None
            if job.status not in (JOB_STATUS_RUNNING, JOB_STATUS_QUEUED):
                logger.info("Job %s already in status %s; skipping", job_id, job.status)
                return None
            values: Dict[str, Any] = {"status": JOB_STATUS_RUNNING}
            if job.started_at is None:
                values["started_at"] = _utcnow()
            result = session.execute(
                update(TranscriptionJobORM)
                .where(
                    TranscriptionJobORM.id == job_id,
                    TranscriptionJobORM.status.in_((JOB_STATUS_QUEUED, JOB_STATUS_RUNNING)),
                )
                .values(**values)
            )
            if result.rowcount == 0:
                logger.info("Job %s was cancelled before acquire; skipping", job_id)
                return None
            session.flush()
            return job.job_type, job.recording_id
    raise RuntimeError(f"Job {job_id} not found in database after retries")


def process_transcription_job(job_id: str) -> None:
    """RQ entrypoint: run a queued transcription job."""
    init_sync_db()

    acquired = _acquire_job_for_processing(job_id)
    if acquired is None:
        return
    job_type, recording_id = acquired

    try:
        _check_job_cancelled(job_id)
        if job_type in (JOB_TYPE_RECORDING, JOB_TYPE_SPEAKER_EXTRACT):
            if not recording_id:
                raise ValueError("recording_id is required for stored transcription jobs")
            _run_recording_job(job_id, recording_id)
        elif job_type == JOB_TYPE_OPENAI:
            _run_openai_job(job_id)
        else:
            raise ValueError(f"Unknown job type: {job_type}")

        with get_sync_session() as session:
            session.execute(
                update(TranscriptionJobORM)
                .where(
                    TranscriptionJobORM.id == job_id,
                    TranscriptionJobORM.status == JOB_STATUS_RUNNING,
                )
                .values(
                    status=JOB_STATUS_COMPLETED,
                    completed_at=_utcnow(),
                    progress_stage="completed",
                )
            )
    except JobCancelled:
        logger.info("Job %s cancelled during processing", job_id)
        with get_sync_session() as session:
            job_row = session.get(TranscriptionJobORM, job_id)
            if job_row:
                job_row.status = JOB_STATUS_CANCELLED
                if not job_row.completed_at:
                    job_row.completed_at = _utcnow()
                job_row.progress_stage = "cancelled"
    except Exception as e:
        logger.error("Job %s failed: %s", job_id, e)
        with get_sync_session() as session:
            session.execute(
                update(TranscriptionJobORM)
                .where(
                    TranscriptionJobORM.id == job_id,
                    TranscriptionJobORM.status == JOB_STATUS_RUNNING,
                )
                .values(
                    status=JOB_STATUS_FAILED,
                    error_message=str(e),
                    completed_at=_utcnow(),
                )
            )
        raise
