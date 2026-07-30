"""
Shared transcription + optional diarization pipeline used by HTTP handlers.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

from models import SpeakerInfo, TranscriptionResponse, WhisperSegment
from audio import split_audio_into_chunks
from transcription import load_model, transcribe_audio_chunk, unload_model
from services.stt_api_client import transcribe_api_chunk
from diarization import diarizer_session, merge_diarization_with_transcription
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
    on_progress: Callable[[str, int, int], None] | None = None,
) -> PipelineRunResult:
    """
    Transcribe ``wav_file`` (16 kHz mono WAV). Optionally diarize on full file.

    Returns a ``TranscriptionResponse`` and a list of extra filesystem paths
    (chunk files) that the caller should delete; the caller owns ``wav_file``.

    When ``config.load_models_on_demand`` is set and local ASR is used, the ASR
    model is loaded after language ID / diarization and unloaded before return.
    """
    effective_language = resolve_transcription_language(language, wav_file, config)

    audio_chunks = split_audio_into_chunks(
        wav_file,
        min_duration=float(config.chunk_min_duration),
        max_duration=float(config.chunk_max_duration),
        silence_db=float(config.chunk_silence_db),
        silence_min_duration=float(config.chunk_silence_min_duration),
    )
    total_chunks = len(audio_chunks)

    if on_progress:
        on_progress("preparing", 0, total_chunks)

    diarization_result = None
    speaker_mapping = None
    if diarize:
        local_path = getattr(config, "diarization_model_path", None)
        hf_token = config.get_hf_token()
        if not local_path and not hf_token:
            logger.warning("Diarization requested but no local diarization path or HuggingFace token")
        else:
            if on_progress:
                on_progress("loading_diarization_model", 0, 1)
            logger.info("Loading diarization model")
            with diarizer_session(
                access_token=hf_token if not local_path else None,
                model_path=local_path,
            ) as diarizer:
                if getattr(diarizer, "pipeline", None) is not None:
                    if on_progress:
                        on_progress("diarization", 0, 1)
                    logger.info("Performing speaker diarization")
                    diarization_result = diarizer.diarize(wav_file, return_embeddings=True)
                    logger.info("Diarization found %s speakers", diarization_result.num_speakers)
                    if on_progress:
                        on_progress("diarization", 1, 1)
                    if diarization_result.embeddings and speaker_db:
                        logger.info("Matching speakers against stored embeddings")
                        speaker_mapping = diarizer.match_speakers(
                            diarization_result.embeddings,
                            speaker_db,
                            threshold=config.speaker_similarity_threshold,
                        )
                else:
                    logger.warning("Diarization requested but pipeline is not available")

    active_asr = asr_model
    loaded_on_demand = False
    use_api = getattr(config, "use_api", False)
    on_demand = bool(getattr(config, "load_models_on_demand", False))

    try:
        if not use_api and on_demand:
            if active_asr is None:
                resolved = getattr(config, "resolved_asr_model", None)
                if resolved is None:
                    raise RuntimeError("LOAD_MODELS_ON_DEMAND set but no resolved_asr_model configured")
                if on_progress:
                    on_progress("loading_transcription_model", 0, 1)
                logger.info("Loading ASR model on demand for transcription")
                active_asr = load_model(resolved)
                loaded_on_demand = True

        all_text: List[str] = []
        all_segments: List[WhisperSegment] = []

        for i, chunk in enumerate(audio_chunks):
            if on_progress:
                on_progress("transcribing", i + 1, total_chunks)
            logger.info("Processing chunk %s/%s (start=%.2fs)", i + 1, total_chunks, chunk.start_sec)
            if use_api:
                chunk_text, chunk_segments = transcribe_api_chunk(
                    chunk.path,
                    config.get_transcription_model_name(),
                    config,
                    language=effective_language,
                )
            else:
                chunk_text, chunk_segments = transcribe_audio_chunk(
                    active_asr,
                    chunk.path,
                    language=effective_language,
                    word_timestamps=word_timestamps,
                )
            if chunk.start_sec > 0:
                for segment in chunk_segments:
                    segment.start += chunk.start_sec
                    segment.end += chunk.start_sec
            all_text.append(chunk_text)
            all_segments.extend(chunk_segments)

        full_text = " ".join(all_text)

        if diarization_result and diarization_result.segments:
            logger.info("Found %s speakers", diarization_result.num_speakers)

            all_segments = merge_diarization_with_transcription(
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

            threshold = config.speaker_similarity_threshold
            speakers_list = []
            for speaker_id, embedding in diarization_result.embeddings.items():
                if speaker_id not in speakers_with_segments:
                    continue
                embedding_list = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
                display_name = None
                matched = False
                match_similarity = None
                if speaker_db:
                    best, similarity = speaker_db.query_best_match(embedding)
                    if best and similarity >= threshold:
                        display_name = best.display_name
                        matched = True
                        match_similarity = round(float(similarity), 4)
                speakers_list.append(
                    SpeakerInfo(
                        id=speaker_id,
                        display_name=display_name,
                        embedding=embedding_list,
                        matched=matched,
                        match_similarity=match_similarity,
                    )
                )

        duration_est = (
            sum(len(segment.text.split()) for segment in all_segments) / 150.0 if all_segments else 0.0
        )

        model_id = config.get_transcription_model_name()
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

        paths_to_cleanup = [
            c.path for c in audio_chunks if c.path != wav_file and os.path.exists(c.path)
        ]
        return PipelineRunResult(
            response=response,
            paths_to_cleanup=paths_to_cleanup,
            all_segments=all_segments,
            speakers_list=speakers_list,
        )
    finally:
        if loaded_on_demand:
            unload_model(active_asr)


def segment_to_dict(segment: WhisperSegment) -> dict:
    if hasattr(segment, "model_dump"):
        return segment.model_dump()
    return segment.dict()


def speaker_info_to_dict(s: SpeakerInfo) -> dict:
    if hasattr(s, "model_dump"):
        return s.model_dump()
    return s.dict()

