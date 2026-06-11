# Originally from https://github.com/jfgonsalves/parakeet-diarized (commit 6abadfd)
# Copyright (c) jfgonsalves - MIT License
# Modified by meganoob1337 for the NoobScribe project
#
# Speaker diarization module for NoobScribe (pyannote.audio integration)

from contextlib import contextmanager
from typing import Dict, Iterator, List, Optional, Any
import gc
import os
import logging
import numpy as np
import torch
from pydantic import BaseModel

from config import DEFAULT_DIARIZATION_MODEL, use_cuda

logger = logging.getLogger(__name__)


def _normalize_speaker_id(speaker: Any) -> str:
    """Convert pyannote speaker labels to SPEAKER_XX format."""
    if isinstance(speaker, str) and not speaker.startswith("SPEAKER_"):
        return f"SPEAKER_{speaker}"
    return str(speaker)


def _to_numpy(embedding: Any) -> np.ndarray:
    if isinstance(embedding, torch.Tensor):
        return embedding.cpu().numpy()
    if isinstance(embedding, np.ndarray):
        return embedding
    return np.array(embedding)


class SpeakerSegment(BaseModel):
    """A segment of speech from a specific speaker"""
    start: float
    end: float
    speaker: str


class DiarizationResult(BaseModel):
    """Result of speaker diarization"""
    segments: List[SpeakerSegment]
    num_speakers: int
    embeddings: Optional[Dict[str, Any]] = None  # Mapping of speaker labels to embeddings (stored as lists for JSON serialization)

    class Config:
        arbitrary_types_allowed = True


class Diarizer:
    """Speaker diarization using pyannote.audio"""

    def __init__(self, access_token: Optional[str] = None, model_path: Optional[str] = None):
        self.pipeline = None
        self.access_token = access_token
        self.model_path = model_path
        self.device = "cuda" if use_cuda() else "cpu"
        self._initialize()

    def _initialize(self):
        """Initialize the diarization pipeline"""
        try:
            from pyannote.audio import Pipeline

            if self.model_path:
                logger.info("Loading diarization pipeline from local path: %s", self.model_path)
                self.pipeline = Pipeline.from_pretrained(self.model_path)
            else:
                if not self.access_token:
                    logger.warning(
                        "No access token provided. Using HUGGINGFACE_ACCESS_TOKEN environment variable."
                    )
                    self.access_token = os.environ.get("HUGGINGFACE_ACCESS_TOKEN")

                if not self.access_token:
                    logger.error("No access token available. Diarization will not work.")
                    return

                self.pipeline = Pipeline.from_pretrained(
                    DEFAULT_DIARIZATION_MODEL,
                    token=self.access_token,
                )

            self.pipeline.to(torch.device(self.device))
            logger.info("Diarization pipeline initialized on %s", self.device)

        except ImportError:
            logger.error("Failed to import pyannote.audio. Please install it with 'pip install pyannote.audio'")
        except Exception as e:
            logger.error("Failed to initialize diarization pipeline: %s", e)

    def unload(self) -> None:
        """Release the diarization pipeline and free GPU memory."""
        if self.pipeline is None:
            return
        try:
            if self.device == "cuda":
                self.pipeline.to(torch.device("cpu"))
        except Exception as e:
            logger.debug("Could not move diarization pipeline to CPU: %s", e)
        del self.pipeline
        self.pipeline = None
        gc.collect()
        if use_cuda():
            torch.cuda.empty_cache()
        logger.info("Diarization pipeline unloaded")

    def diarize(self, audio_path: str, num_speakers: Optional[int] = None, return_embeddings: Optional[bool] = False) -> DiarizationResult:
        """
        Perform speaker diarization on an audio file

        Args:
            audio_path: Path to the audio file
            num_speakers: Optional number of speakers (if known)
            return_embeddings: Whether to return speaker embeddings

        Returns:
            DiarizationResult with speaker segments and optional embeddings
        """
        if self.pipeline is None:
            logger.error("Diarization pipeline not initialized")
            return DiarizationResult(segments=[], num_speakers=0, embeddings=None)

        try:
            pipeline_kwargs = {}
            if num_speakers is not None:
                pipeline_kwargs["num_speakers"] = num_speakers

            output = self.pipeline(audio_path, **pipeline_kwargs)

            # Exclusive diarization aligns better with transcription timestamps.
            diarization = output.exclusive_speaker_diarization
            raw_embeddings = output.speaker_embeddings if return_embeddings else None

            segments = []
            speakers = set()
            embeddings_dict = {}

            if return_embeddings and raw_embeddings is not None:
                speaker_labels = list(diarization.labels())
                for i, speaker in enumerate(speaker_labels):
                    speaker_id = _normalize_speaker_id(speaker)
                    if i >= len(raw_embeddings):
                        logger.warning("Could not extract embedding for speaker %s", speaker_id)
                        continue
                    embedding = _to_numpy(raw_embeddings[i])
                    embeddings_dict[speaker_id] = embedding
                    logger.debug("Extracted embedding for %s: shape %s", speaker_id, embedding.shape)

            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_id = _normalize_speaker_id(speaker)
                segments.append(SpeakerSegment(
                    start=turn.start,
                    end=turn.end,
                    speaker=f"speaker_{speaker_id}"
                ))
                speakers.add(speaker_id)

            segments.sort(key=lambda x: x.start)

            return DiarizationResult(
                segments=segments,
                num_speakers=len(speakers),
                embeddings=embeddings_dict if embeddings_dict else None
            )

        except Exception as e:
            logger.error("Diarization failed: %s", e)
            return DiarizationResult(segments=[], num_speakers=0, embeddings=None)

    def match_speakers(self, embeddings: Dict[str, Any], speaker_db, threshold: Optional[float] = None) -> Dict[str, str]:
        """
        Match speaker embeddings against stored speakers in the database

        Args:
            embeddings: Dictionary mapping speaker labels (e.g., "SPEAKER_00") to embeddings
            speaker_db: SpeakerDB instance for querying
            threshold: Optional similarity threshold (uses database default if not provided)

        Returns:
            Dictionary mapping speaker labels to display names (or original label if no match)
        """
        if speaker_db is None:
            logger.warning("Speaker database not available, returning original labels")
            return {label: label for label in embeddings.keys()}

        return speaker_db.match_speakers(embeddings, threshold)

    def merge_with_transcription(
        self,
        diarization: DiarizationResult,
        transcription_segments: list,
        speaker_mapping: Optional[Dict[str, str]] = None,
    ) -> list:
        """Merge diarization results with transcription segments."""
        return merge_diarization_with_transcription(
            diarization, transcription_segments, speaker_mapping=speaker_mapping
        )


def merge_diarization_with_transcription(
    diarization: DiarizationResult,
    transcription_segments: list,
    speaker_mapping: Optional[Dict[str, str]] = None,
) -> list:
    """
    Merge diarization results with transcription segments.

    Does not require a loaded Diarizer pipeline.
    """
    if not diarization.segments:
        return transcription_segments

    for segment in transcription_segments:
        start = segment.start
        end = segment.end

        overlapping = []
        for spk_segment in diarization.segments:
            overlap_start = max(start, spk_segment.start)
            overlap_end = min(end, spk_segment.end)

            if overlap_end > overlap_start:
                duration = overlap_end - overlap_start
                overlapping.append((spk_segment.speaker, duration))

        if overlapping:
            overlapping.sort(key=lambda x: x[1], reverse=True)
            original_speaker = overlapping[0][0]

            if original_speaker.startswith("speaker_"):
                speaker_id = original_speaker.replace("speaker_", "")
            else:
                speaker_id = original_speaker

            setattr(segment, "speaker", speaker_id)
        else:
            setattr(segment, "speaker", "unknown")

    return transcription_segments


@contextmanager
def diarizer_session(
    *,
    access_token: Optional[str] = None,
    model_path: Optional[str] = None,
) -> Iterator[Diarizer]:
    """Load a diarizer for the block and unload it when done."""
    diarizer = Diarizer(access_token=access_token, model_path=model_path)
    try:
        yield diarizer
    finally:
        diarizer.unload()
