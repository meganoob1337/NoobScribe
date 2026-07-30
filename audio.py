# Originally from https://github.com/jfgonsalves/parakeet-diarized (commit 6abadfd)
# Copyright (c) jfgonsalves - MIT License
# Modified by meganoob1337 for the NoobScribe project
import os
import re
import tempfile
import logging
import subprocess
import wave
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_MIN_DURATION = 15.0
DEFAULT_CHUNK_MAX_DURATION = 30.0
DEFAULT_CHUNK_SILENCE_DB = -40.0
DEFAULT_CHUNK_SILENCE_MIN_DURATION = 0.3

_SILENCE_START_RE = re.compile(r"silence_start:\s*([-0-9.]+)")
_SILENCE_END_RE = re.compile(r"silence_end:\s*([-0-9.]+)")


@dataclass
class AudioChunk:
    """One ASR chunk cut from a longer WAV, with absolute start time."""

    path: str
    start_sec: float


def get_wav_duration_seconds(audio_path: str) -> Optional[float]:
    """
    Return duration in seconds for a WAV file, or None if unreadable.
    """
    try:
        with wave.open(audio_path, "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate <= 0:
                return None
            return frames / float(rate)
    except Exception as e:
        logger.debug("Could not read WAV duration for %s: %s", audio_path, e)
        return None


def detect_silence_intervals(
    audio_path: str,
    *,
    silence_db: float = DEFAULT_CHUNK_SILENCE_DB,
    silence_min_duration: float = DEFAULT_CHUNK_SILENCE_MIN_DURATION,
) -> List[Tuple[float, float]]:
    """
    Run ffmpeg silencedetect and return list of (start, end) silence intervals.
    """
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-i",
        audio_path,
        "-af",
        f"silencedetect=noise={silence_db}dB:d={silence_min_duration}",
        "-f",
        "null",
        "-",
    ]
    logger.debug("Running silence detection: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    # silencedetect writes to stderr; non-zero can still contain useful output
    stderr = result.stderr or ""

    intervals: List[Tuple[float, float]] = []
    current_start: Optional[float] = None
    for line in stderr.splitlines():
        start_match = _SILENCE_START_RE.search(line)
        if start_match:
            current_start = float(start_match.group(1))
            continue
        end_match = _SILENCE_END_RE.search(line)
        if end_match and current_start is not None:
            end = float(end_match.group(1))
            if end > current_start:
                intervals.append((current_start, end))
            current_start = None

    if current_start is not None:
        duration = get_wav_duration_seconds(audio_path)
        if duration is not None and duration > current_start:
            intervals.append((current_start, duration))

    return intervals


def choose_chunk_boundaries(
    duration: float,
    silences: List[Tuple[float, float]],
    *,
    min_duration: float = DEFAULT_CHUNK_MIN_DURATION,
    max_duration: float = DEFAULT_CHUNK_MAX_DURATION,
) -> List[float]:
    """
    Return cut times (absolute seconds), including 0 and duration.

    Prefers the longest silence whose midpoint falls in
    [cursor + min_duration, cursor + max_duration]; otherwise force-cuts at max.
    """
    if duration <= 0:
        return [0.0, 0.0]
    if min_duration <= 0 or max_duration < min_duration:
        raise ValueError(
            f"Invalid chunk duration bounds: min={min_duration}, max={max_duration}"
        )
    if duration <= max_duration:
        return [0.0, duration]

    cuts = [0.0]
    cursor = 0.0
    while cursor + max_duration < duration:
        window_start = cursor + min_duration
        window_end = cursor + max_duration
        best: Optional[Tuple[float, float, float]] = None  # (length, mid, cut)
        for silence_start, silence_end in silences:
            mid = (silence_start + silence_end) / 2.0
            if window_start <= mid <= window_end:
                length = silence_end - silence_start
                if best is None or length > best[0] or (
                    length == best[0] and mid < best[1]
                ):
                    best = (length, mid, mid)
        if best is not None:
            cut = best[2]
        else:
            cut = window_end
        if cut <= cursor:
            cut = window_end
        cuts.append(cut)
        cursor = cut

    cuts.append(duration)
    return cuts


def _extract_chunk(
    audio_path: str,
    start_sec: float,
    duration_sec: float,
    output_path: str,
) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start_sec),
        "-i",
        audio_path,
        "-t",
        str(duration_sec),
        "-c:a",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        output_path,
    ]
    logger.debug("Running ffmpeg chunk extract: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("Error splitting audio chunk: %s", result.stderr)
        raise Exception(f"Failed to split audio: {result.stderr}")


def split_audio_into_chunks(
    audio_path: str,
    *,
    min_duration: float = DEFAULT_CHUNK_MIN_DURATION,
    max_duration: float = DEFAULT_CHUNK_MAX_DURATION,
    silence_db: float = DEFAULT_CHUNK_SILENCE_DB,
    silence_min_duration: float = DEFAULT_CHUNK_SILENCE_MIN_DURATION,
) -> List[AudioChunk]:
    """
    Split a long audio file at speaking pauses into chunks of roughly
    ``min_duration``–``max_duration`` seconds.

    Returns:
        List of AudioChunk (path + absolute start time). Short files are
        returned as a single chunk pointing at the original path.
    """
    try:
        duration = get_wav_duration_seconds(audio_path)
        if duration is None:
            raise Exception(f"Could not read duration for {audio_path}")

        logger.info("Audio duration: %.2f seconds", duration)

        if duration <= max_duration:
            logger.info("Audio is within max chunk duration, no splitting needed")
            return [AudioChunk(path=audio_path, start_sec=0.0)]

        silences = detect_silence_intervals(
            audio_path,
            silence_db=silence_db,
            silence_min_duration=silence_min_duration,
        )
        logger.info("Detected %s silence intervals for chunking", len(silences))

        boundaries = choose_chunk_boundaries(
            duration,
            silences,
            min_duration=min_duration,
            max_duration=max_duration,
        )
        num_chunks = len(boundaries) - 1
        logger.info("Splitting audio into %s pause-aware chunks", num_chunks)

        temp_dir = tempfile.mkdtemp()
        chunks: List[AudioChunk] = []

        for i in range(num_chunks):
            start_time = boundaries[i]
            end_time = boundaries[i + 1]
            chunk_len = end_time - start_time
            output_path = os.path.join(temp_dir, f"chunk_{i}.wav")
            _extract_chunk(audio_path, start_time, chunk_len, output_path)
            chunks.append(AudioChunk(path=output_path, start_sec=start_time))

        return chunks

    except Exception as e:
        logger.error("Error splitting audio: %s", e)
        return [AudioChunk(path=audio_path, start_sec=0.0)]


def convert_audio_to_wav(audio_path: str) -> str:
    """
    Convert any audio format to WAV format (16kHz, mono, 16-bit PCM)
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        Path to the converted WAV file
    """
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_file.close()
    output_path = temp_file.name
    
    try:
        # Use ffmpeg to convert audio
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output files
            "-i", audio_path,  # Input file
            "-c:a", "pcm_s16le",  # Audio codec (16-bit PCM)
            "-ar", "16000",  # Sample rate (16kHz)
            "-ac", "1",  # Mono audio
            output_path
        ]
        
        logger.debug(f"Running ffmpeg command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Error converting audio: {result.stderr}")
            raise Exception(f"Failed to convert audio: {result.stderr}")
            
        return output_path
        
    except Exception as e:
        logger.error(f"Error converting audio: {str(e)}")
        # Clean up temporary file
        if os.path.exists(output_path):
            try:
                os.unlink(output_path)
            except:
                pass
        raise e


def cut_audio_segment(
    audio_path: str,
    start_sec: float,
    end_sec: float,
    *,
    max_duration_sec: float = 300.0,
) -> str:
    """
    Extract [start_sec, end_sec] to a temporary 16 kHz mono WAV via ffmpeg.

    Returns:
        Path to the new WAV file (caller should delete when done).

    Raises:
        ValueError: invalid time range or duration exceeds max_duration_sec.
        Exception: ffmpeg failure.
    """
    if end_sec <= start_sec:
        raise ValueError("end_sec must be greater than start_sec")
    duration = end_sec - start_sec
    if duration > max_duration_sec:
        raise ValueError(
            f"Segment duration {duration:.1f}s exceeds maximum {max_duration_sec:.0f}s"
        )

    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_file.close()
    output_path = temp_file.name

    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start_sec),
        "-i",
        audio_path,
        "-t",
        str(duration),
        "-c:a",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        output_path,
    ]
    logger.debug("Running ffmpeg cut: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        try:
            if os.path.exists(output_path):
                os.unlink(output_path)
        except OSError:
            pass
        logger.error("Error cutting audio segment: %s", result.stderr)
        raise Exception(f"Failed to cut audio segment: {result.stderr}")

    return output_path
