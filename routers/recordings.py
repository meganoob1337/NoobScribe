"""Recording CRUD, audio download, and stored transcription runs."""
from __future__ import annotations

import logging
import os
import stat
import uuid
from pathlib import Path
from typing import List, Optional

import anyio
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from starlette.types import Receive, Scope, Send
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from audio import convert_audio_to_wav, get_wav_duration_seconds
from config import get_config
from database.db import get_session
from database.models import Recording as RecordingORM
from database.models import TranscriptionResult as TranscriptionResultORM
from models import (
    RecordingDetailResponse,
    RecordingListResponse,
    RecordingResponse,
    RecordingUpdate,
    TranscriptionJobCreateResponse,
    TranscriptionJobListResponse,
    TranscriptionJobResponse,
    TranscriptionResultListResponse,
    TranscriptionResultResponse,
)
from database.models import TranscriptionJob as TranscriptionJobORM
from services.transcription_enqueue import build_transcribe_params, enqueue_recording_transcription
from services.transcription_jobs import job_to_dict, processing_seconds_from_datetimes


class FullRecordingFileResponse(FileResponse):
    """
    Like FileResponse but ignores Range so the response is always 200 with the full file.
    Starlette's default FileResponse answers Range with 206; that breaks duration/seek UI for
    some formats in the browser's <audio> player. Snippet WAVs keep standard FileResponse.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.headers["accept-ranges"] = "none"

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        send_header_only = scope["method"].upper() == "HEAD"
        send_pathsend = "http.response.pathsend" in scope.get("extensions", {})

        if self.stat_result is None:
            try:
                stat_result = await anyio.to_thread.run_sync(os.stat, self.path)
                self.set_stat_headers(stat_result)
            except FileNotFoundError:
                raise RuntimeError(f"File at path {self.path} does not exist.") from None
            else:
                if not stat.S_ISREG(stat_result.st_mode):
                    raise RuntimeError(f"File at path {self.path} is not a file.")

        await self._handle_simple(send, send_header_only, send_pathsend)
        if self.background is not None:
            await self.background()


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/recordings", tags=["recordings"])


def _orm_recording_to_response(
    rec: RecordingORM,
    transcription_count: int,
) -> RecordingResponse:
    p = Path(rec.file_path)
    return RecordingResponse(
        id=rec.id,
        name=rec.name,
        original_filename=rec.original_filename,
        stored_filename=p.name,
        duration_seconds=rec.duration_seconds,
        file_size_bytes=rec.file_size_bytes,
        mime_type=rec.mime_type,
        hide_in_recordings=bool(getattr(rec, "hide_in_recordings", False)),
        created_at=rec.created_at.isoformat() if rec.created_at else "",
        updated_at=rec.updated_at.isoformat() if rec.updated_at else "",
        transcription_count=transcription_count,
    )


def _orm_transcription_to_response(
    row: TranscriptionResultORM,
    *,
    job: Optional[TranscriptionJobORM] = None,
) -> TranscriptionResultResponse:
    processing = row.processing_seconds
    if processing is None and job is not None:
        processing = processing_seconds_from_datetimes(job.started_at, job.completed_at)
    return TranscriptionResultResponse(
        id=row.id,
        recording_id=row.recording_id,
        full_text=row.full_text,
        segments=row.segments_json,
        language=row.language,
        model_id=row.model_id,
        diarization_enabled=row.diarization_enabled,
        speakers=row.speakers_json,
        duration_seconds=row.duration_seconds,
        processing_seconds=processing,
        word_timestamps=row.word_timestamps,
        created_at=row.created_at.isoformat() if row.created_at else "",
    )


async def _jobs_by_transcription_id(
    session: AsyncSession,
    transcription_ids: List[str],
) -> dict:
    if not transcription_ids:
        return {}
    q = (
        select(TranscriptionJobORM)
        .where(TranscriptionJobORM.transcription_id.in_(transcription_ids))
        .order_by(TranscriptionJobORM.completed_at.desc())
    )
    rows = list((await session.execute(q)).scalars().all())
    out: dict = {}
    for job in rows:
        tid = job.transcription_id
        if tid and tid not in out:
            out[tid] = job
    return out


def _recording_detail(
    rec: RecordingORM,
    transcription_count: int,
    transcriptions: List[TranscriptionResultORM],
    jobs_by_tid: Optional[dict] = None,
) -> RecordingDetailResponse:
    p = Path(rec.file_path)
    return RecordingDetailResponse(
        id=rec.id,
        name=rec.name,
        original_filename=rec.original_filename,
        stored_filename=p.name,
        duration_seconds=rec.duration_seconds,
        file_size_bytes=rec.file_size_bytes,
        mime_type=rec.mime_type,
        hide_in_recordings=bool(getattr(rec, "hide_in_recordings", False)),
        created_at=rec.created_at.isoformat() if rec.created_at else "",
        updated_at=rec.updated_at.isoformat() if rec.updated_at else "",
        transcription_count=transcription_count,
        transcriptions=[
            _orm_transcription_to_response(
                t, job=(jobs_by_tid or {}).get(t.id)
            )
            for t in transcriptions
        ],
    )


@router.get("", response_model=RecordingListResponse)
async def list_recordings(
    limit: int = 50,
    offset: int = 0,
    session: AsyncSession = Depends(get_session),
):
    limit = max(1, min(limit, 200))
    offset = max(0, offset)

    q = (
        select(RecordingORM)
        .where(RecordingORM.hide_in_recordings.is_(False))
        .order_by(RecordingORM.created_at.desc())
        .limit(limit + 1)
        .offset(offset)
    )
    res = await session.execute(q)
    rows = list(res.scalars().all())
    has_more = len(rows) > limit
    rows = rows[:limit]

    data: List[RecordingResponse] = []
    for rec in rows:
        cnt = await session.scalar(
            select(func.count())
            .select_from(TranscriptionResultORM)
            .where(TranscriptionResultORM.recording_id == rec.id)
        )
        data.append(_orm_recording_to_response(rec, int(cnt or 0)))

    return RecordingListResponse(data=data, has_more=has_more)


@router.post("", response_model=RecordingResponse)
async def create_recording(
    file: UploadFile = File(...),
    name: Optional[str] = Form(None),
    hide_in_recordings: bool = Form(False),
    session: AsyncSession = Depends(get_session),
):
    cfg = get_config()
    rid = str(uuid.uuid4())
    suffix = Path(file.filename or "audio").suffix or ".bin"
    dest = Path(cfg.recordings_path) / f"{rid}{suffix}"
    dest.parent.mkdir(parents=True, exist_ok=True)

    content = await file.read()
    dest.write_bytes(content)

    duration = None
    mime = file.content_type
    try:
        if suffix.lower() == ".wav":
            duration = get_wav_duration_seconds(str(dest))
        else:
            wav = convert_audio_to_wav(str(dest))
            duration = get_wav_duration_seconds(wav)
            if os.path.exists(wav):
                os.unlink(wav)
    except Exception as e:
        logger.debug("Could not compute duration on upload: %s", e)

    display_name = (name or "").strip() or (file.filename or rid)
    rec = RecordingORM(
        id=rid,
        name=display_name,
        original_filename=file.filename or "upload",
        file_path=str(dest.resolve()),
        duration_seconds=duration,
        file_size_bytes=len(content),
        mime_type=mime,
        hide_in_recordings=bool(hide_in_recordings),
    )
    session.add(rec)
    await session.flush()

    return _orm_recording_to_response(rec, 0)


@router.get("/{recording_id}", response_model=RecordingDetailResponse)
async def get_recording(recording_id: str, session: AsyncSession = Depends(get_session)):
    rec = await session.get(RecordingORM, recording_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Recording not found")
    q = (
        select(TranscriptionResultORM)
        .where(TranscriptionResultORM.recording_id == recording_id)
        .order_by(TranscriptionResultORM.created_at.desc())
    )
    res = await session.execute(q)
    rows = list(res.scalars().all())
    jobs_by_tid = await _jobs_by_transcription_id(session, [t.id for t in rows])
    return _recording_detail(rec, len(rows), rows, jobs_by_tid)


@router.patch("/{recording_id}", response_model=RecordingResponse)
async def patch_recording(
    recording_id: str,
    body: RecordingUpdate,
    session: AsyncSession = Depends(get_session),
):
    rec = await session.get(RecordingORM, recording_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Recording not found")
    rec.name = body.name.strip()
    session.add(rec)
    await session.flush()
    cnt = await session.scalar(
        select(func.count())
        .select_from(TranscriptionResultORM)
        .where(TranscriptionResultORM.recording_id == rec.id)
    )
    return _orm_recording_to_response(rec, int(cnt or 0))


@router.delete("/{recording_id}")
async def delete_recording(recording_id: str, session: AsyncSession = Depends(get_session)):
    rec = await session.get(RecordingORM, recording_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Recording not found")
    fp = rec.file_path
    await session.delete(rec)
    await session.flush()
    try:
        if fp and os.path.exists(fp):
            os.unlink(fp)
    except OSError as e:
        logger.warning("Could not delete recording file %s: %s", fp, e)
    return JSONResponse(status_code=200, content={"message": f"Recording {recording_id} deleted"})


@router.get("/{recording_id}/audio")
async def get_recording_audio(recording_id: str, session: AsyncSession = Depends(get_session)):
    rec = await session.get(RecordingORM, recording_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Recording not found")
    if not rec.file_path or not os.path.exists(rec.file_path):
        raise HTTPException(status_code=404, detail="Audio file not found on disk")
    return FullRecordingFileResponse(
        rec.file_path,
        filename=rec.original_filename or Path(rec.file_path).name,
        media_type=rec.mime_type or "application/octet-stream",
    )


@router.post(
    "/{recording_id}/transcribe",
    response_model=TranscriptionJobCreateResponse,
    status_code=202,
)
async def transcribe_recording(
    recording_id: str,
    language: Optional[str] = Form(None),
    response_format: str = Form("verbose_json"),
    temperature: float = Form(0.0),
    timestamps: bool = Form(True),
    word_timestamps: bool = Form(False),
    diarize: bool = Form(True),
    include_diarization_in_text: Optional[bool] = Form(None),
    session: AsyncSession = Depends(get_session),
):
    _ = temperature  # accepted for API parity; NeMo path does not use it here
    rec = await session.get(RecordingORM, recording_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Recording not found")
    if not rec.file_path or not os.path.exists(rec.file_path):
        raise HTTPException(status_code=404, detail="Audio file not found on disk")

    try:
        params = build_transcribe_params(
            language=language,
            response_format=response_format,
            timestamps=timestamps,
            word_timestamps=word_timestamps,
            diarize=diarize,
            include_diarization_in_text=include_diarization_in_text,
            temperature=temperature,
        )
        job_id = await enqueue_recording_transcription(session, recording_id, params)
        return TranscriptionJobCreateResponse(
            job_id=job_id,
            status="queued",
            recording_id=recording_id,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to queue transcription for recording %s: %s", recording_id, e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{recording_id}/transcription-jobs", response_model=TranscriptionJobListResponse)
async def list_recording_transcription_jobs(
    recording_id: str,
    session: AsyncSession = Depends(get_session),
):
    rec = await session.get(RecordingORM, recording_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Recording not found")
    q = (
        select(TranscriptionJobORM)
        .where(TranscriptionJobORM.recording_id == recording_id)
        .order_by(TranscriptionJobORM.created_at.desc())
    )
    res = await session.execute(q)
    rows = list(res.scalars().all())
    return TranscriptionJobListResponse(
        data=[TranscriptionJobResponse(**job_to_dict(r, recording_name=rec.name)) for r in rows]
    )


@router.get("/{recording_id}/transcriptions", response_model=TranscriptionResultListResponse)
async def list_transcriptions(recording_id: str, session: AsyncSession = Depends(get_session)):
    rec = await session.get(RecordingORM, recording_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Recording not found")
    q = (
        select(TranscriptionResultORM)
        .where(TranscriptionResultORM.recording_id == recording_id)
        .order_by(TranscriptionResultORM.created_at.desc())
    )
    res = await session.execute(q)
    rows = list(res.scalars().all())
    jobs_by_tid = await _jobs_by_transcription_id(session, [r.id for r in rows])
    return TranscriptionResultListResponse(
        data=[_orm_transcription_to_response(r, job=jobs_by_tid.get(r.id)) for r in rows]
    )


@router.get("/{recording_id}/transcriptions/{transcription_id}", response_model=TranscriptionResultResponse)
async def get_transcription(
    recording_id: str,
    transcription_id: str,
    session: AsyncSession = Depends(get_session),
):
    tr = await session.get(TranscriptionResultORM, transcription_id)
    if not tr or tr.recording_id != recording_id:
        raise HTTPException(status_code=404, detail="Transcription not found")
    jobs_by_tid = await _jobs_by_transcription_id(session, [tr.id])
    return _orm_transcription_to_response(tr, job=jobs_by_tid.get(tr.id))
