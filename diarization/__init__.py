# Originally from https://github.com/jfgonsalves/parakeet-diarized (commit 6abadfd)
# Copyright (c) jfgonsalves - MIT License
# Modified by meganoob1337 for the NoobScribe project
#
# Speaker diarization module for NoobScribe (pyannote.audio integration)

from typing import Dict, List, Optional, Tuple, Union, Any
import os
import logging
import tempfile
import numpy as np
import torch
from pydantic import BaseModel

logger = logging.getLogger(__name__)

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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
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
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=self.access_token,
                )

            # Move to GPU if available
            self.pipeline.to(torch.device(self.device))
            logger.info(f"Diarization pipeline initialized on {self.device}")

        except ImportError:
            logger.error("Failed to import pyannote.audio. Please install it with 'pip install pyannote.audio'")
        except Exception as e:
            logger.error(f"Failed to initialize diarization pipeline: {str(e)}")

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
            # Run the diarization pipeline
            diarization, embeddings = self.pipeline(
                audio_path,
                num_speakers=num_speakers,
                return_embeddings=return_embeddings
            )
            
            # Convert to our format
            segments = []
            speakers = set()
            embeddings_dict = {}
            
            # Process embeddings if available
            if return_embeddings and embeddings is not None:
                # Handle different possible formats of embeddings from pyannote
                # It might be a dict, list, or tensor
                if isinstance(embeddings, dict):
                    # If embeddings is already a dict, use it directly
                    for speaker, embedding in embeddings.items():
                        # Convert speaker label to consistent format
                        if isinstance(speaker, str) and not speaker.startswith("SPEAKER_"):
                            speaker_id = f"SPEAKER_{speaker}"
                        else:
                            speaker_id = speaker
                        
                        # Convert to numpy array
                        if isinstance(embedding, torch.Tensor):
                            embedding = embedding.cpu().numpy()
                        elif not isinstance(embedding, np.ndarray):
                            embedding = np.array(embedding)
                        
                        embeddings_dict[speaker_id] = embedding
                        logger.debug(f"Extracted embedding for {speaker_id}: shape {embedding.shape}")
                else:
                    # If embeddings is a list or array, iterate through speakers
                    speaker_labels = list(diarization.labels())
                    for i, speaker in enumerate(speaker_labels):
                        # Convert speaker label to consistent format
                        if isinstance(speaker, str) and not speaker.startswith("SPEAKER_"):
                            speaker_id = f"SPEAKER_{speaker}"
                        else:
                            speaker_id = speaker
                        
                        # Get embedding for this speaker
                        if isinstance(embeddings, (list, tuple)) and i < len(embeddings):
                            embedding = embeddings[i]
                        elif isinstance(embeddings, np.ndarray) and i < embeddings.shape[0]:
                            embedding = embeddings[i]
                        elif isinstance(embeddings, torch.Tensor) and i < embeddings.shape[0]:
                            embedding = embeddings[i]
                        else:
                            logger.warning(f"Could not extract embedding for speaker {speaker_id}")
                            continue
                        
                        # Convert to numpy array
                        if isinstance(embedding, torch.Tensor):
                            embedding = embedding.cpu().numpy()
                        elif not isinstance(embedding, np.ndarray):
                            embedding = np.array(embedding)
                        
                        embeddings_dict[speaker_id] = embedding
                        logger.debug(f"Extracted embedding for {speaker_id}: shape {embedding.shape}")
            
            # Process the diarization result
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                # Convert speaker label to consistent format
                # This handles different formats from pyannote.audio versions
                if isinstance(speaker, str) and not speaker.startswith("SPEAKER_"):
                    speaker_id = f"SPEAKER_{speaker}"
                else:
                    speaker_id = speaker

                segments.append(SpeakerSegment(
                    start=turn.start,
                    end=turn.end,
                    speaker=f"speaker_{speaker_id}"
                ))
                speakers.add(speaker_id)

            # Sort segments by start time
            segments.sort(key=lambda x: x.start)

            return DiarizationResult(
                segments=segments,
                num_speakers=len(speakers),
                embeddings=embeddings_dict if embeddings_dict else None
            )

        except Exception as e:
            logger.error(f"Diarization failed: {str(e)}")
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

    def merge_with_transcription(self,
                                diarization: DiarizationResult,
                                transcription_segments: list,
                                speaker_mapping: Optional[Dict[str, str]] = None) -> list:
        """
        Merge diarization results with transcription segments

        Args:
            diarization: Speaker diarization result
            transcription_segments: List of transcription segments with start/end times
            speaker_mapping: Optional mapping from speaker labels to display names

        Returns:
            Merged list of segments with speaker information
        """
        # If no diarization results, return original transcription
        if not diarization.segments:
            return transcription_segments

        # For each transcription segment, find the dominant speaker
        for segment in transcription_segments:
            # Get segment time bounds
            start = segment.start
            end = segment.end

            # Find overlapping diarization segments
            overlapping = []
            for spk_segment in diarization.segments:
                # Calculate overlap
                overlap_start = max(start, spk_segment.start)
                overlap_end = min(end, spk_segment.end)

                if overlap_end > overlap_start:
                    # There is an overlap
                    duration = overlap_end - overlap_start
                    overlapping.append((spk_segment.speaker, duration))

            # Assign the speaker with most overlap
            if overlapping:
                # Sort by duration (descending)
                overlapping.sort(key=lambda x: x[1], reverse=True)
                # Get the original speaker label (e.g., "speaker_SPEAKER_00")
                original_speaker = overlapping[0][0]
                
                # Always use speaker ID format (SPEAKER_XX) instead of display names
                # Extract the speaker ID from the label (e.g., "speaker_SPEAKER_00" -> "SPEAKER_00")
                if original_speaker.startswith("speaker_"):
                    speaker_id = original_speaker.replace("speaker_", "")
                else:
                    speaker_id = original_speaker
                
                # Ensure it's in SPEAKER_XX format
                if not speaker_id.startswith("SPEAKER_"):
                    # If it's not in the expected format, try to extract or use as-is
                    speaker_id = speaker_id
                
                setattr(segment, "speaker", speaker_id)
            else:
                # No overlap found, assign unknown
                setattr(segment, "speaker", "unknown")

        return transcription_segments
