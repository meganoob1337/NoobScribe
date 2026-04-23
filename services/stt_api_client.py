"""
OpenAI-compatible remote STT (Whisper-style audio transcriptions API).

Used when ``USE_API`` is set in configuration; diarization remains local.
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

from audio import get_wav_duration_seconds
from models import WhisperSegment

logger = logging.getLogger(__name__)

_client: Any = None
_client_signature: Optional[Tuple[Optional[str], Optional[str]]] = None


def _get_openai_client(config: Any) -> Any:
    global _client, _client_signature
    sig = (getattr(config, "stt_base_url", None), getattr(config, "stt_api_key", None))
    if _client is None or _client_signature != sig:
        from openai import OpenAI

        _client = OpenAI(
            base_url=config.stt_base_url,
            api_key=config.stt_api_key,
        )
        _client_signature = sig
    return _client


def _response_to_dict(response: Any) -> dict:
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if hasattr(response, "dict"):
        return response.dict()
    if isinstance(response, dict):
        return response
    return {}


def transcribe_api_chunk(
    audio_path: str,
    model_id: str,
    config: Any,
    *,
    language: Optional[str] = None,
) -> Tuple[str, List[WhisperSegment]]:
    """
    Transcribe one WAV chunk via ``POST /v1/audio/transcriptions`` (OpenAI-compatible).

    Returns:
        Tuple of (full text, list of ``WhisperSegment`` for this chunk, times relative to chunk start).
    """
    client = _get_openai_client(config)
    kwargs: dict = {
        "model": model_id,
        "response_format": "verbose_json",
        "timestamp_granularities": ["segment"],
    }
    if language:
        kwargs["language"] = language

    try:
        with open(audio_path, "rb") as audio_file:
            kwargs["file"] = audio_file
            response = client.audio.transcriptions.create(**kwargs)
    except Exception as e:
        logger.error("Remote STT API transcription failed for %s: %s", audio_path, e)
        raise

    data = _response_to_dict(response)
    text = (data.get("text") or getattr(response, "text", None) or "").strip()

    raw_segments = data.get("segments")
    if raw_segments is None and hasattr(response, "segments") and response.segments is not None:
        raw_segments = [
            s.model_dump() if hasattr(s, "model_dump") else (s.dict() if hasattr(s, "dict") else s)
            for s in response.segments
        ]

    segments: List[WhisperSegment] = []
    if raw_segments:
        for i, seg in enumerate(raw_segments):
            if not isinstance(seg, dict):
                seg = seg.model_dump() if hasattr(seg, "model_dump") else (
                    seg.dict() if hasattr(seg, "dict") else {}
                )
            st = float(seg.get("start", 0.0) or 0.0)
            en = float(seg.get("end", st) or st)
            seg_text = (seg.get("text") or "").strip()
            _rid = seg.get("id", i)
            try:
                seg_id = int(_rid) if _rid is not None else i
            except (TypeError, ValueError):
                seg_id = i
            segments.append(
                WhisperSegment(
                    id=seg_id,
                    seek=int(seg.get("seek", 0)),
                    start=st,
                    end=en,
                    text=seg_text,
                    tokens=list(seg.get("tokens") or []),
                    temperature=float(seg.get("temperature", 0.0)),
                    avg_logprob=float(seg.get("avg_logprob", 0.0)),
                    compression_ratio=float(seg.get("compression_ratio", 1.0)),
                    no_speech_prob=float(seg.get("no_speech_prob", 0.1)),
                )
            )
    else:
        duration = get_wav_duration_seconds(audio_path)
        end_t = float(duration) if duration is not None and duration > 0 else max(len(text.split()) / 2.0, 0.1)
        segments.append(
            WhisperSegment(
                id=0,
                start=0.0,
                end=end_t,
                text=text or "",
            )
        )
        if not text and segments:
            text = segments[0].text

    return text, segments
