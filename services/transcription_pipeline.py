"""
Shared transcription + optional diarization pipeline used by HTTP handlers.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

from models import SpeakerInfo, TranscriptionResponse, WhisperSegment
from audio import split_audio_into_chunks
from transcription import transcribe_audio_chunk
from diarization import Diarizer
from services.language_id import resolve_transcription_language

logger = logging.getLogger(__name__)


@dataclass
class PipelineRunResult:
    """Result of running ASR (+ optional diarization) on a WAV file."""

    response: TranscriptionResponse
    paths_to_cleanup: List[str]
    all_segments: List[WhisperSegment]
    speakers_list: Optional[List[SpeakerInfo]]


def run_transcription_pipeline(
    wav_file: str,
    asr_model: Any,
    config: Any,
    speaker_db: Any,
    *,
    language: Optional[str] = None,
    word_timestamps: bool = False,
    diarize: bool = True,
    include_diarization_in_text: Optional[bool] = None,
    response_format: str = "json",
    timestamps: bool = False,
) -> PipelineRunResult:
    """
    Transcribe ``wav_file`` (16 kHz mono WAV). Optionally diarize on full file.

    Returns a ``TranscriptionResponse`` and a list of extra filesystem paths
    (chunk files) that the caller should delete; the caller owns ``wav_file``.
    """
    effective_language = resolve_transcription_language(language, wav_file, config)

    chunk_duration = config.chunk_duration
    audio_chunks = split_audio_into_chunks(wav_file, chunk_duration=chunk_duration)

    diarizer = None
    if diarize:
        local_path = getattr(config, "diarization_model_path", None)
        hf_token = config.get_hf_token()
        if local_path:
            diarizer = Diarizer(model_path=local_path)
        elif hf_token:
            diarizer = Diarizer(access_token=hf_token)
        else:
            logger.warning("Diarization requested but no local diarization path or HuggingFace token")

    diarization_result = None
    if diarizer and getattr(diarizer, "pipeline", None) is not None:
        logger.info("Performing speaker diarization")
        diarization_result = diarizer.diarize(wav_file, return_embeddings=True)
        logger.info("Diarization found %s speakers", diarization_result.num_speakers)
    elif diarize:
        logger.warning("Diarization requested but pipeline is not available")

    speaker_mapping = None
    if (
        diarizer
        and getattr(diarizer, "pipeline", None) is not None
        and diarization_result
        and diarization_result.embeddings
        and speaker_db
    ):
        logger.info("Matching speakers against stored embeddings")
        speaker_mapping = diarizer.match_speakers(
            diarization_result.embeddings,
            speaker_db,
            threshold=config.speaker_similarity_threshold,
        )

    all_text: List[str] = []
    all_segments: List[WhisperSegment] = []

    for i, chunk_path in enumerate(audio_chunks):
        logger.info("Processing chunk %s/%s", i + 1, len(audio_chunks))
        chunk_text, chunk_segments = transcribe_audio_chunk(
            asr_model,
            chunk_path,
            language=effective_language,
            word_timestamps=word_timestamps,
        )
        if i > 0:
            offset = i * chunk_duration
            for segment in chunk_segments:
                segment.start += offset
                segment.end += offset
        all_text.append(chunk_text)
        all_segments.extend(chunk_segments)

    full_text = " ".join(all_text)

    if diarizer and diarization_result and diarization_result.segments:
        logger.info("Found %s speakers", diarization_result.num_speakers)

        all_segments = diarizer.merge_with_transcription(
            diarization_result,
            all_segments,
            speaker_mapping=speaker_mapping,
        )

        use_diarization_in_text = (
            include_diarization_in_text
            if include_diarization_in_text is not None
            else config.include_diarization_in_text
        )

        if use_diarization_in_text:
            logger.info("Including speaker labels in transcript text")
            previous_speaker_id = None
            for segment in all_segments:
                if hasattr(segment, "speaker") and segment.speaker:
                    speaker_id = segment.speaker
                    if speaker_id == "unknown":
                        continue
                    if speaker_id.startswith("speaker_"):
                        speaker_id = speaker_id.replace("speaker_", "")
                    if speaker_id and speaker_id.startswith("SPEAKER_") and speaker_id != previous_speaker_id:
                        segment.text = f"{speaker_id}: {segment.text}"
                        previous_speaker_id = speaker_id
            full_text = " ".join(segment.text for segment in all_segments)
        else:
            logger.info("Speaker diarization applied to segments but not included in text")
    else:
        logger.warning("Diarization not applied or returned no speakers")

    speakers_list = None
    if diarization_result and diarization_result.embeddings:
        speakers_with_segments = set()
        for segment in all_segments:
            if hasattr(segment, "speaker") and segment.speaker:
                speaker_id = segment.speaker
                if speaker_id.startswith("speaker_"):
                    speaker_id = speaker_id.replace("speaker_", "")
                if speaker_id.startswith("SPEAKER_") and speaker_id != "unknown":
                    speakers_with_segments.add(speaker_id)

        speakers_list = []
        for speaker_id, embedding in diarization_result.embeddings.items():
            if speaker_id not in speakers_with_segments:
                continue
            embedding_list = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
            display_name = None
            matched = False
            if speaker_mapping:
                mapped_name = speaker_mapping.get(speaker_id)
                if mapped_name and mapped_name != speaker_id:
                    display_name = mapped_name
                    matched = True
            speakers_list.append(
                SpeakerInfo(
                    id=speaker_id,
                    display_name=display_name,
                    embedding=embedding_list,
                    matched=matched,
                )
            )

    duration_est = (
        sum(len(segment.text.split()) for segment in all_segments) / 150.0 if all_segments else 0.0
    )

    model_id = getattr(config, "model_id", None)
    include_speakers = (
        response_format in ("json", "verbose_json") and speakers_list is not None
    )

    response = TranscriptionResponse(
        text=full_text,
        segments=all_segments if timestamps or response_format == "verbose_json" else None,
        language=effective_language,
        duration=duration_est,
        model=model_id,
        speakers=speakers_list if include_speakers else None,
    )

    paths_to_cleanup = [p for p in audio_chunks if p != wav_file and os.path.exists(p)]
    return PipelineRunResult(
        response=response,
        paths_to_cleanup=paths_to_cleanup,
        all_segments=all_segments,
        speakers_list=speakers_list,
    )


def segment_to_dict(segment: WhisperSegment) -> dict:
    if hasattr(segment, "model_dump"):
        return segment.model_dump()
    return segment.dict()


def speaker_info_to_dict(s: SpeakerInfo) -> dict:
    if hasattr(s, "model_dump"):
        return s.model_dump()
    return s.dict()
