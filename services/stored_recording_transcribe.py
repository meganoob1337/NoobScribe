"""Run ASR (+ diarization) on a stored recording file and persist TranscriptionResult."""
from __future__ import annotations

import logging
import os
import uuid
from typing import Any, Callable, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from audio import convert_audio_to_wav
from config import get_config
from database.models import Recording as RecordingORM
from database.models import TranscriptionResult as TranscriptionResultORM
from services.transcription_pipeline import (
    PipelineRunResult,
    run_transcription_pipeline,
    segment_to_dict,
    speaker_info_to_dict,
)

logger = logging.getLogger(__name__)


def run_transcription_for_recording_path(
    recording_file_path: str,
    asr_model: Any,
    cfg: Any,
    speaker_db: Any,
    *,
    language: Optional[str] = None,
    word_timestamps: bool = False,
    diarize: bool = True,
    include_diarization_in_text: Optional[bool] = None,
    response_format: str = "verbose_json",
    timestamps: bool = True,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> PipelineRunResult:
    if on_progress:
        on_progress("preparing", 0, 1)
    wav_file = convert_audio_to_wav(recording_file_path)
    try:
        pr = run_transcription_pipeline(
            wav_file,
            asr_model,
            cfg,
            speaker_db,
            language=language,
            word_timestamps=word_timestamps,
            diarize=diarize,
            include_diarization_in_text=include_diarization_in_text,
            response_format=response_format,
            timestamps=timestamps,
            on_progress=on_progress,
        )
    except Exception:
        if os.path.exists(wav_file) and wav_file != recording_file_path:
            try:
                os.unlink(wav_file)
            except OSError:
                pass
        raise

    for p in pr.paths_to_cleanup:
        try:
            if os.path.exists(p):
                os.unlink(p)
        except OSError:
            pass
    if os.path.exists(wav_file) and wav_file != recording_file_path:
        try:
            os.unlink(wav_file)
        except OSError:
            pass

    return pr


def _build_transcription_result_row(
    rec: RecordingORM,
    pr: PipelineRunResult,
    *,
    diarize: bool,
    word_timestamps: bool,
    processing_seconds: Optional[float] = None,
) -> tuple[str, TranscriptionResultORM]:
    cfg = get_config()
    seg_json = [segment_to_dict(s) for s in pr.all_segments]
    spk_json = [speaker_info_to_dict(s) for s in pr.speakers_list] if pr.speakers_list else None
    dur = None
    if pr.all_segments:
        dur = max(s.end for s in pr.all_segments)
    if rec.duration_seconds is None and dur is not None:
        rec.duration_seconds = dur

    tid = str(uuid.uuid4())
    tr = TranscriptionResultORM(
        id=tid,
        recording_id=rec.id,
        full_text=pr.response.text,
        segments_json=seg_json,
        language=pr.response.language,
        model_id=cfg.get_transcription_model_name(),
        diarization_enabled=diarize,
        speakers_json=spk_json,
        duration_seconds=dur,
        processing_seconds=processing_seconds,
        word_timestamps=word_timestamps,
    )
    return tid, tr


def persist_pipeline_transcription_sync(
    session: Session,
    rec: RecordingORM,
    pr: PipelineRunResult,
    *,
    diarize: bool,
    word_timestamps: bool,
    processing_seconds: Optional[float] = None,
) -> str:
    tid, tr = _build_transcription_result_row(
        rec,
        pr,
        diarize=diarize,
        word_timestamps=word_timestamps,
        processing_seconds=processing_seconds,
    )
    session.add(tr)
    session.add(rec)
    session.flush()
    return tid


async def persist_pipeline_transcription(
    session: AsyncSession,
    rec: RecordingORM,
    pr: PipelineRunResult,
    *,
    diarize: bool,
    word_timestamps: bool,
    processing_seconds: Optional[float] = None,
) -> str:
    tid, tr = _build_transcription_result_row(
        rec,
        pr,
        diarize=diarize,
        word_timestamps=word_timestamps,
        processing_seconds=processing_seconds,
    )
    session.add(tr)
    session.add(rec)
    await session.flush()
    return tid


def transcription_response_json_body(pr: PipelineRunResult, transcription_id: str) -> dict:
    if hasattr(pr.response, "model_dump"):
        d = pr.response.model_dump()
    else:
        d = pr.response.dict()
    d["transcription_id"] = transcription_id
    d["stored"] = True
    return d
