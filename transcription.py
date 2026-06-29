# Originally from https://github.com/jfgonsalves/parakeet-diarized (commit 6abadfd)
# Copyright (c) jfgonsalves - MIT License
# Modified by meganoob1337 for the NoobScribe project
import logging
from typing import Any, List, Optional, Tuple

import torch

from config import use_cuda
from models import WhisperSegment
from services.asr_backend import load_onnx_asr_model, wrap_loaded_model
from services.asr_models import LoadedAsrModel, ResolvedAsrModel

logger = logging.getLogger(__name__)


def load_model(resolved: ResolvedAsrModel) -> LoadedAsrModel:
    """
    Load a local ASR model from a resolved registry entry.

    Args:
        resolved: Validated registry resolution (backend, source, path, quantization)

    Returns:
        LoadedAsrModel wrapper for backend-specific transcription
    """
    spec = resolved.spec
    try:
        if spec.backend == "nemo":
            from nemo.collections.asr.models import ASRModel

            hf_id = spec.source.hf_id
            assert hf_id is not None
            if resolved.effective_path:
                logger.info("Loading NeMo ASR model from local path: %s", resolved.effective_path)
                model = ASRModel.restore_from(restore_path=resolved.effective_path)
            else:
                logger.info("Loading NeMo ASR model %s (%s)", spec.id, hf_id)
                model = ASRModel.from_pretrained(model_name=hf_id)

            if use_cuda():
                model = model.cuda()
                logger.info("Model loaded on GPU: %s", torch.cuda.get_device_name(0))
            else:
                logger.warning("Running on CPU (will be slow)")

            return wrap_loaded_model(resolved, model)

        if spec.backend == "onnx":
            model = load_onnx_asr_model(resolved)
            return wrap_loaded_model(resolved, model)

        raise ValueError(f"Unsupported ASR backend: {spec.backend}")
    except Exception as e:
        logger.error("Error loading ASR model %s: %s", spec.id, e)
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


def _segments_from_onnx_result(result: Any) -> Tuple[str, List[WhisperSegment]]:
    """Map onnx-asr timestamped output to WhisperSegment list."""
    if isinstance(result, list):
        if not result:
            return "", []
        if all(hasattr(item, "start") and hasattr(item, "end") for item in result):
            segments = [
                WhisperSegment(id=i, start=float(item.start), end=float(item.end), text=str(item.text))
                for i, item in enumerate(result)
            ]
            text = " ".join(segment.text for segment in segments).strip()
            return text, segments
        result = result[0]

    if hasattr(result, "start") and hasattr(result, "end"):
        text = str(getattr(result, "text", "") or "")
        return text, [
            WhisperSegment(
                id=0,
                start=float(result.start),
                end=float(result.end),
                text=text,
            )
        ]

    text = str(getattr(result, "text", result) or "")
    if not text:
        return "", []
    return text, [
        WhisperSegment(
            id=0,
            start=0.0,
            end=max(len(text.split()) / 2.0, 0.1),
            text=text,
        )
    ]


def _transcribe_nemo_chunk(
    model: Any,
    audio_path: str,
    language: Optional[str],
) -> Tuple[str, List[WhisperSegment]]:
    with torch.no_grad():
        transcription = model.transcribe(
            [audio_path],
            source_lang=language,
            target_lang=language,
            timestamps=True,
        )
    logger.info("NeMo transcription of chunk %s", transcription)

    if not transcription:
        logger.warning("No transcription generated for %s", audio_path)
        return "", []

    result = transcription[0]
    text = result.text
    segments: List[WhisperSegment] = []

    if hasattr(result, "timestamp") and result.timestamp and "segment" in result.timestamp:
        for i, stamp in enumerate(result.timestamp["segment"]):
            segments.append(
                WhisperSegment(
                    id=i,
                    start=stamp["start"],
                    end=stamp["end"],
                    text=stamp["segment"],
                )
            )
    else:
        segments.append(
            WhisperSegment(
                id=0,
                start=0.0,
                end=len(text.split()) / 2.0,
                text=text,
            )
        )
    return text, segments


def _transcribe_onnx_chunk(
    model: Any,
    audio_path: str,
    language: Optional[str],
) -> Tuple[str, List[WhisperSegment]]:
    kwargs = {}
    if language:
        kwargs["language"] = language
        kwargs["target_language"] = language
    result = model.recognize(audio_path, **kwargs)
    text, segments = _segments_from_onnx_result(result)
    logger.info("ONNX transcription of chunk: %s", text[:120] if text else "")
    return text, segments


def transcribe_audio_chunk(
    loaded: LoadedAsrModel,
    audio_path: str,
    language: Optional[str] = None,
    word_timestamps: bool = False,
) -> Tuple[str, List[WhisperSegment]]:
    """
    Transcribe a single audio chunk using the loaded local ASR model.

    Args:
        loaded: LoadedAsrModel from load_model()
        audio_path: Path to the audio file
        language: Optional language code
        word_timestamps: Whether to generate word-level timestamps (NeMo only today)

    Returns:
        Tuple of (transcription text, list of WhisperSegment objects)
    """
    _ = word_timestamps
    try:
        if loaded.backend == "nemo":
            return _transcribe_nemo_chunk(loaded.model, audio_path, language)
        if loaded.backend == "onnx":
            return _transcribe_onnx_chunk(loaded.model, audio_path, language)
        raise ValueError(f"Unsupported ASR backend: {loaded.backend}")
    except Exception as e:
        logger.error("Error transcribing audio chunk: %s", e)
        return "", []
