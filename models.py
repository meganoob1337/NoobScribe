# Originally from https://github.com/jfgonsalves/parakeet-diarized (commit 6abadfd)
# Copyright (c) jfgonsalves - MIT License
# Modified by meganoob1337 for the NoobScribe project
from __future__ import annotations

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field

class WhisperSegment(BaseModel):
    """Represents a segment in the transcription"""
    id: int
    seek: int = 0
    start: float
    end: float
    text: str
    tokens: List[int] = []
    temperature: float = 0.0
    avg_logprob: float = 0.0
    compression_ratio: float = 1.0
    no_speech_prob: float = 0.1
    speaker: Optional[str] = None  # For speaker diarization

class SpeakerInfo(BaseModel):
    """Information about a speaker with embedding"""
    id: str  # Speaker identifier (e.g., "SPEAKER_00")
    display_name: Optional[str] = None  # Display name if matched to stored speaker
    embedding: List[float]  # The embedding vector
    matched: bool  # Whether speaker was matched to stored identity

class TranscriptionResponse(BaseModel):
    """Represents the response format for transcription"""
    text: str
    segments: Optional[List[WhisperSegment]] = None
    language: Optional[str] = None
    task: str = "transcribe"
    duration: Optional[float] = None
    model: Optional[str] = None
    speakers: Optional[List[SpeakerInfo]] = None  # Speaker embeddings and metadata
    
    class Config:
        schema_extra = {"example": {"text": "Hello world", "segments": []}}
    
    def dict(self, **kwargs):
        """Custom dict method to handle response format"""
        # If we don't need segments, remove them
        result = super().dict(**kwargs)
        if not self.segments:
            result.pop("segments", None)
        # Remove speakers if None
        if not self.speakers:
            result.pop("speakers", None)
        return result

class ModelInfo(BaseModel):
    """Information about a model available in the API"""
    id: str
    object: str = "model"
    created: int
    owned_by: str
    permission: List[Dict[str, Any]] = []
    root: str
    parent: Optional[str] = None

class ModelList(BaseModel):
    """List of available models"""
    object: str = "list"
    data: List[ModelInfo]

class SpeakerCreate(BaseModel):
    """Request model for creating a speaker"""
    display_name: str
    embedding: List[float]

class SpeakerUpdate(BaseModel):
    """Request model for updating a speaker with additional embedding"""
    embedding: List[float]

class SpeakerResponse(BaseModel):
    """Response model for speaker operations"""
    id: str
    display_name: str
    created_at: str
    embedding_count: int

class SpeakerList(BaseModel):
    """List of speakers"""
    object: str = "list"
    data: List[SpeakerResponse]


class SpeakerEmbeddingDetail(BaseModel):
    """One enrolled embedding for a speaker (metadata only; no vector in API)."""

    embedding_index: int
    created_at: str


class SpeakerEmbeddingListResponse(BaseModel):
    """List embeddings for one speaker."""

    speaker_id: str
    display_name: str
    data: List[SpeakerEmbeddingDetail]


class EmbeddingSnippet(BaseModel):
    """One stored transcription whose diarization embedding matches an enrolled vector."""

    transcription_id: str
    recording_id: str
    recording_name: Optional[str] = None
    preview_url: str
    segment_text: str
    start: float
    end: float


class EmbeddingSnippetListResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingSnippet]


# --- Recording management (REST + Web UI) ---


class RecordingUpdate(BaseModel):
    """PATCH body for renaming a recording."""

    name: str


class RecordingResponse(BaseModel):
    """Single recording metadata."""

    id: str
    name: str
    original_filename: str
    stored_filename: str
    duration_seconds: Optional[float] = None
    file_size_bytes: Optional[int] = None
    mime_type: Optional[str] = None
    hide_in_recordings: bool = False
    created_at: str
    updated_at: str
    transcription_count: int = 0


class RecordingListResponse(BaseModel):
    """Paginated list of recordings."""

    object: str = "list"
    data: List[RecordingResponse]
    has_more: bool = False


class RecordingDetailResponse(BaseModel):
    """Recording metadata plus stored transcription history (for GET by id)."""

    id: str
    name: str
    original_filename: str
    stored_filename: str
    duration_seconds: Optional[float] = None
    file_size_bytes: Optional[int] = None
    mime_type: Optional[str] = None
    hide_in_recordings: bool = False
    created_at: str
    updated_at: str
    transcription_count: int = 0
    transcriptions: List["TranscriptionResultResponse"] = []


class TranscriptionResultResponse(BaseModel):
    """Stored transcription + diarization snapshot."""

    id: str
    recording_id: str
    full_text: str
    segments: Optional[List[Dict[str, Any]]] = None
    language: Optional[str] = None
    model_id: Optional[str] = None
    diarization_enabled: bool = False
    speakers: Optional[List[Dict[str, Any]]] = None
    duration_seconds: Optional[float] = None
    word_timestamps: bool = False
    created_at: str


class TranscriptionResultListResponse(BaseModel):
    object: str = "list"
    data: List[TranscriptionResultResponse]