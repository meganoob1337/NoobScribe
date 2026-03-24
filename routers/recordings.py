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
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
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
    TranscriptionResultListResponse,
    TranscriptionResultResponse,
)
from services.stored_recording_transcribe import (
    persist_pipeline_transcription,
    run_transcription_for_recording_path,
    transcription_response_json_body,
)


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


def _require_asr_model():
    import api as api_module

    if api_module.asr_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet. Please try again in a few moments.")
    return api_module.asr_model


def _speaker_db():
    import api as api_module

    return api_module.speaker_db


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


def _orm_transcription_to_response(row: TranscriptionResultORM) -> TranscriptionResultResponse:
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
        word_timestamps=row.word_timestamps,
        created_at=row.created_at.isoformat() if row.created_at else "",
    )


def _recording_detail(
    rec: RecordingORM,
    transcription_count: int,
    transcriptions: List[TranscriptionResultORM],
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
        transcriptions=[_orm_transcription_to_response(t) for t in transcriptions],
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
    return _recording_detail(rec, len(rows), rows)


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
    session.delete(rec)
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


@router.post("/{recording_id}/transcribe")
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
    cfg = get_config()
    rec = await session.get(RecordingORM, recording_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Recording not found")
    if not rec.file_path or not os.path.exists(rec.file_path):
        raise HTTPException(status_code=404, detail="Audio file not found on disk")

    asr_model = _require_asr_model()
    speaker_db = _speaker_db()

    try:
        pr = run_transcription_for_recording_path(
            rec.file_path,
            asr_model,
            cfg,
            speaker_db,
            language=language,
            word_timestamps=word_timestamps,
            diarize=diarize,
            include_diarization_in_text=include_diarization_in_text,
            response_format=response_format,
            timestamps=timestamps,
        )
    except Exception as e:
        logger.error("Transcription failed for recording %s: %s", recording_id, e)
        raise HTTPException(status_code=500, detail=str(e))

    tid = await persist_pipeline_transcription(
        session,
        rec,
        pr,
        diarize=diarize,
        word_timestamps=word_timestamps,
    )

    def _json_body() -> dict:
        return transcription_response_json_body(pr, tid)

    if response_format == "json":
        return _json_body()
    if response_format == "text":
        return PlainTextResponse(pr.response.text)
    if response_format == "srt":
        from transcription import format_srt

        return PlainTextResponse(format_srt(pr.all_segments))
    if response_format == "vtt":
        from transcription import format_vtt

        return PlainTextResponse(format_vtt(pr.all_segments))
    if response_format == "verbose_json":
        return _json_body()
    raise HTTPException(status_code=400, detail=f"Unsupported response format: {response_format}")


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
    rows = res.scalars().all()
    return TranscriptionResultListResponse(data=[_orm_transcription_to_response(r) for r in rows])


@router.get("/{recording_id}/transcriptions/{transcription_id}", response_model=TranscriptionResultResponse)
async def get_transcription(
    recording_id: str,
    transcription_id: str,
    session: AsyncSession = Depends(get_session),
):
    tr = await session.get(TranscriptionResultORM, transcription_id)
    if not tr or tr.recording_id != recording_id:
        raise HTTPException(status_code=404, detail="Transcription not found")
    return _orm_transcription_to_response(tr)
