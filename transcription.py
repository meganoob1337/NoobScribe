# Originally from https://github.com/jfgonsalves/parakeet-diarized (commit 6abadfd)
# Copyright (c) jfgonsalves - MIT License
# Modified by meganoob1337 for the NoobScribe project
import os
import logging
import tempfile
from typing import List, Optional, Dict, Any, Union, Tuple

import torch
import numpy as np

from config import use_cuda
from models import WhisperSegment, TranscriptionResponse

logger = logging.getLogger(__name__)

def load_model(model_id: str = "nvidia/canary-1b-v2", model_path: Optional[str] = None):
    """
    Load the ASR model (NVIDIA NeMo, default: Canary 1B v2)

    Args:
        model_id: The HuggingFace / NeMo model ID to load when ``model_path`` is not set
        model_path: Optional path to a local ``.nemo`` checkpoint (fully offline)

    Returns:
        The loaded model
    """
    try:
        # from nemo.collections.asr.models import EncDecMultiTaskModel
        from nemo.collections.asr.models import EncDecCTCModelBPE
        from nemo.collections.asr.models import ASRModel

        if model_path:
            logger.info("Loading ASR model from local path: %s", model_path)
            model = ASRModel.restore_from(restore_path=model_path)
        else:
            logger.info("Loading model %s", model_id)
            model = ASRModel.from_pretrained(model_name=model_id)

        # Move model to GPU if available (unless FORCE_CPU=1)
        if use_cuda():
            model = model.cuda()
            logger.info(f"Model loaded on GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("Running on CPU (will be slow)")

        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def _format_timestamp(seconds: float, always_include_hours: bool = False,
                     decimal_marker: str = '.') -> str:
    """
    Format a timestamp as a string (HH:MM:SS.mmm)

    Args:
        seconds: Time in seconds
        always_include_hours: Always include hours in the output
        decimal_marker: Marker to use for decimal point

    Returns:
        Formatted timestamp string
    """
    hours = int(seconds / 3600)
    seconds = seconds % 3600
    minutes = int(seconds / 60)
    seconds = seconds % 60

    hours_marker = f"{hours}:" if always_include_hours or hours > 0 else ""

    # Handle different format requirements (SRT vs VTT)
    if decimal_marker == ',':  # SRT format
        return f"{hours_marker}{minutes:02d}:{seconds:06.3f}".replace('.', decimal_marker)
    else:  # VTT format
        return f"{hours_marker}{minutes:02d}:{seconds:06.3f}"

def format_srt(segments: List[WhisperSegment]) -> str:
    """
    Format segments as SRT subtitle format

    Args:
        segments: List of transcription segments

    Returns:
        SRT formatted string
    """
    srt_content = ""
    for i, segment in enumerate(segments):
        segment_id = i + 1
        start = _format_timestamp(segment.start, always_include_hours=True, decimal_marker=',')
        end = _format_timestamp(segment.end, always_include_hours=True, decimal_marker=',')
        text = segment.text.strip().replace('-->', '->')

        # Format for SRT (with speaker if available)
        speaker_prefix = f"[{segment.speaker}] " if hasattr(segment, "speaker") and segment.speaker else ""
        srt_content += f"{segment_id}\n{start} --> {end}\n{speaker_prefix}{text}\n\n"

    return srt_content.strip()

def format_vtt(segments: List[WhisperSegment]) -> str:
    """
    Format segments as WebVTT subtitle format

    Args:
        segments: List of transcription segments

    Returns:
        WebVTT formatted string
    """
    vtt_content = "WEBVTT\n\n"
    for i, segment in enumerate(segments):
        start = _format_timestamp(segment.start, always_include_hours=True)
        end = _format_timestamp(segment.end, always_include_hours=True)
        text = segment.text.strip()

        # Format for VTT (with speaker if available)
        speaker_prefix = f"<v {segment.speaker}>" if hasattr(segment, "speaker") and segment.speaker else ""
        vtt_content += f"{start} --> {end}\n{speaker_prefix}{text}\n\n"

    return vtt_content.strip()

def transcribe_audio_chunk(model, audio_path: str, language: Optional[str] = None,
                          word_timestamps: bool = False) -> Tuple[str, List[WhisperSegment]]:
    """
    Transcribe a single audio chunk using the loaded NeMo ASR model

    Args:
        model: The loaded ASR model
        audio_path: Path to the audio file
        language: Optional language code
        word_timestamps: Whether to generate word-level timestamps

    Returns:
        Tuple of (transcription text, list of WhisperSegment objects)
    """
    try:
        # Use the NeMo model to transcribe audio
        with torch.no_grad():
            # Simply pass the audio path(s) as a list to the transcribe method
            transcription = model.transcribe(
                [audio_path],
                source_lang=language,
                target_lang=language,
                # taskname="asr",
                # batch_size=1,
                timestamps=True  # Always request timestamps for segmentation
            )
        logger.info(f"Transcription of chunk  {transcription}")

        # Extract the text from the result
        if not transcription or len(transcription) == 0:
            logger.warning(f"No transcription generated for {audio_path}")
            return "", []

        result = transcription[0]  # Get the first result
        text = result.text

        # Create segments from the timestamp information if available
        segments = []

        # Check if we have timestamp information
        if hasattr(result, 'timestamp') and 'segment' in result.timestamp:
            for i, stamp in enumerate(result.timestamp['segment']):
                segments.append(WhisperSegment(
                    id=i,
                    start=stamp['start'],
                    end=stamp['end'],
                    text=stamp['segment']
                ))
        else:
            # If no segments available, create a single segment for the entire chunk
            segments.append(WhisperSegment(
                id=0,
                start=0.0,
                end=len(text.split()) / 2.0,  # Rough estimate based on word count
                text=text
            ))

        return text, segments

    except Exception as e:
        logger.error(f"Error transcribing audio chunk: {str(e)}")
        return "", []
