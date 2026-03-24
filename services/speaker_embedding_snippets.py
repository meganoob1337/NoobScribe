"""Find stored transcription segments whose diarization embeddings match a Chroma enrollment."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database.models import Recording as RecordingORM
from database.models import TranscriptionResult as TranscriptionResultORM
from database.speakers import SpeakerDB

logger = logging.getLogger(__name__)


def _normalize_diarization_speaker_id(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    if s.lower() == "unknown":
        return "unknown"
    if s.startswith("speaker_"):
        s = s[10:]
    return s


def embedding_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in [-1, 1]; returns -1.0 if vectors are invalid or length-mismatched."""
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return -1.0
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return -1.0
    return float(np.dot(a, b) / (na * nb))


def _longest_matching_segment(
    segments: List[Any], want: str
) -> Optional[Dict[str, Any]]:
    """Return the longest segment (by duration) whose speaker label matches ``want``."""
    best: Optional[Dict[str, Any]] = None
    best_dur = -1.0
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        if _normalize_diarization_speaker_id(seg.get("speaker")) != want:
            continue
        try:
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", start))
        except (TypeError, ValueError):
            continue
        dur = end - start
        if dur > best_dur:
            best_dur = dur
            best = seg
    return best


async def collect_snippets_for_enrolled_embedding(
    session: AsyncSession,
    speaker_db: SpeakerDB,
    speaker_id: str,
    embedding_index: int,
    similarity_threshold: float,
) -> Optional[List[Dict[str, Any]]]:
    """
    Returns ``None`` if the enrolled embedding id/index does not exist.

    Otherwise returns one snippet per **recording** (the longest matching segment
    across all transcriptions of that recording). Keys: transcription_id, recording_id,
    recording_name, preview_url, segment_text, start, end.
    """
    target = speaker_db.get_embedding_vector(speaker_id, embedding_index)
    if target is None:
        return None

    result = await session.execute(select(TranscriptionResultORM))
    rows: List[TranscriptionResultORM] = list(result.scalars().all())

    # Accumulate the best (longest) candidate per recording_id.
    # Value: dict with transcription_id, start, end, text, duration.
    best_per_recording: Dict[str, Dict[str, Any]] = {}

    for row in rows:
        speakers_json = row.speakers_json
        # Do not use `if not speakers_json`: JSON may deserialize as numpy arrays,
        # which raise on boolean evaluation.
        if speakers_json is None or not isinstance(speakers_json, list):
            continue

        best_sim = -1.0
        best_label: Optional[str] = None

        for item in speakers_json:
            if not isinstance(item, dict):
                continue
            emb_raw = item.get("embedding")
            if emb_raw is None:
                continue
            try:
                emb = np.asarray(emb_raw, dtype=np.float32)
            except (TypeError, ValueError):
                continue
            sim = embedding_cosine_similarity(target, emb)
            if sim > best_sim:
                best_sim = sim
                lid = item.get("id")
                best_label = str(lid) if lid is not None else None

        if best_sim < similarity_threshold or best_label is None or best_label == "":
            continue

        want = _normalize_diarization_speaker_id(best_label)
        if want is None or want == "" or want == "unknown":
            continue

        segments = row.segments_json
        if segments is None:
            segments = []
        elif not isinstance(segments, list):
            continue

        chosen = _longest_matching_segment(segments, want)
        if chosen is None:
            continue

        try:
            start = float(chosen.get("start", 0.0))
            end = float(chosen.get("end", start))
        except (TypeError, ValueError):
            continue

        if end <= start:
            continue

        duration = end - start
        rid = row.recording_id

        existing = best_per_recording.get(rid)
        if existing is None or duration > existing["duration"]:
            _txt = chosen.get("text")
            best_per_recording[rid] = {
                "transcription_id": row.id,
                "start": start,
                "end": end,
                "text": "" if _txt is None else str(_txt),
                "duration": duration,
            }

    # Resolve recording names and build output list.
    out: List[Dict[str, Any]] = []
    for rid, candidate in best_per_recording.items():
        rec = await session.get(RecordingORM, rid)
        recording_name = rec.name if rec else None
        qs = urlencode({"recording_id": rid, "start": candidate["start"], "end": candidate["end"]})
        out.append(
            {
                "transcription_id": candidate["transcription_id"],
                "recording_id": rid,
                "recording_name": recording_name,
                "preview_url": f"/v1/audio/snippet?{qs}",
                "segment_text": candidate["text"],
                "start": candidate["start"],
                "end": candidate["end"],
            }
        )

    return out
