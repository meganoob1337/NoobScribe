# Originally from https://github.com/jfgonsalves/parakeet-diarized (commit 6abadfd)
# Copyright (c) jfgonsalves - MIT License
# Modified by meganoob1337 for the NoobScribe project
import os
import tempfile
import logging
import subprocess
import math
import wave
from typing import List, Optional, Dict, Any, Union
from pathlib import Path

logger = logging.getLogger(__name__)


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


def split_audio_into_chunks(audio_path: str, chunk_duration: int = 300) -> List[str]:
    """
    Split a long audio file into smaller chunks for processing.
    
    Args:
        audio_path: Path to the audio file
        chunk_duration: Duration of each chunk in seconds (default: 5 minutes)
        
    Returns:
        List of paths to the chunked audio files
    """
    try:
        # Check audio duration using wave module
        with wave.open(audio_path, 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration = frames / rate
            
        logger.info(f"Audio duration: {duration:.2f} seconds")
        
        # If duration is less than chunk_duration, no need to split
        if duration <= chunk_duration:
            logger.info("Audio is shorter than chunk duration, no splitting needed")
            return [audio_path]
            
        # Calculate number of chunks
        num_chunks = math.ceil(duration / chunk_duration)
        logger.info(f"Splitting audio into {num_chunks} chunks")
        
        # Create temporary directory for chunks
        temp_dir = tempfile.mkdtemp()
        chunk_paths = []
        
        # Process each chunk
        for i in range(num_chunks):
            start_time = i * chunk_duration
            output_path = os.path.join(temp_dir, f"chunk_{i}.wav")
            
            # Use ffmpeg to extract chunk
            cmd = [
                "ffmpeg",
                "-y",  # Overwrite output files
                "-ss", str(start_time),  # Start time
                "-i", audio_path,  # Input file
                "-t", str(chunk_duration),  # Duration to extract
                "-c:a", "pcm_s16le",  # Audio codec
                "-ar", "16000",  # Sample rate
                "-ac", "1",  # Mono audio
                output_path
            ]
            
            logger.debug(f"Running ffmpeg command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Error splitting audio chunk {i}: {result.stderr}")
                raise Exception(f"Failed to split audio: {result.stderr}")
                
            chunk_paths.append(output_path)
            
        return chunk_paths
        
    except Exception as e:
        logger.error(f"Error splitting audio: {str(e)}")
        # If there's an error, return the original file
        return [audio_path]

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