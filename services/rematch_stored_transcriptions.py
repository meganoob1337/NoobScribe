"""Re-match stored transcription diarization rows against the current Chroma speaker index."""
from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import flag_modified

from database.models import TranscriptionResult
from database.speakers import SpeakerDB

logger = logging.getLogger(__name__)


def _norm_display_name(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


def _meta_changed(before: dict, after: dict) -> bool:
    return _norm_display_name(before.get("display_name")) != _norm_display_name(
        after.get("display_name")
    ) or bool(before.get("matched")) != bool(after.get("matched"))


def rematch_transcription_speakers_json(
    speakers_json: Optional[List[Any]], speaker_db: SpeakerDB
) -> Tuple[Optional[List[Any]], bool]:
    """
    For each speaker dict with an ``embedding``, set ``display_name`` / ``matched`` via
    ``SpeakerDB.find_similar_speaker``. Rows without ``embedding`` are left unchanged.

    Returns:
        (possibly_new_list, changed) — if not changed, returns the original ``speakers_json``.
    """
    if not speakers_json or not isinstance(speakers_json, list):
        return speakers_json, False

    new_list: List[Any] = []
    changed = False

    for item in speakers_json:
        if not isinstance(item, dict):
            new_list.append(item)
            continue

        emb_raw = item.get("embedding")
        if emb_raw is None:
            new_list.append(item)
            continue

        try:
            emb = np.asarray(emb_raw, dtype=np.float32)
            if emb.size == 0:
                logger.warning("Skipping speaker row with empty embedding (id=%r)", item.get("id"))
                new_list.append(item)
                continue
        except (TypeError, ValueError) as e:
            logger.warning(
                "Skipping speaker row with invalid embedding (id=%r): %s",
                item.get("id"),
                e,
            )
            new_list.append(item)
            continue

        match = speaker_db.find_similar_speaker(emb)
        new_item = dict(item)
        if match:
            new_item["display_name"] = match.display_name
            new_item["matched"] = True
        else:
            new_item["display_name"] = None
            new_item["matched"] = False

        if _meta_changed(item, new_item):
            changed = True
        new_list.append(new_item)

    if not changed:
        return speakers_json, False
    return new_list, True


async def rematch_all_stored_transcriptions(
    session: AsyncSession, speaker_db: SpeakerDB
) -> int:
    """
    Re-run ``rematch_transcription_speakers_json`` for every stored transcription row
    that has ``speakers_json``. Persists updates with ``flag_modified``.

    Returns:
        Number of transcription rows whose ``speakers_json`` was updated.
    """
    result = await session.execute(select(TranscriptionResult))
    rows = result.scalars().all()
    updated = 0
    for row in rows:
        new_json, did_change = rematch_transcription_speakers_json(row.speakers_json, speaker_db)
        if did_change:
            row.speakers_json = new_json
            flag_modified(row, "speakers_json")
            updated += 1
    return updated
