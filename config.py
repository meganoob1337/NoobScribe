# Originally from https://github.com/jfgonsalves/parakeet-diarized (commit 6abadfd)
# Copyright (c) jfgonsalves - MIT License
# Modified by meganoob1337 for the NoobScribe project
# Configuration settings for NoobScribe
import os
import logging
from typing import Dict, Optional, Any
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    # Load environment variables from .env file if it exists
    load_dotenv()
    logger.info("Loaded environment variables from .env file")
except ImportError:
    logger.warning("dotenv package not installed. Environment variables will only be loaded from system.")

# API settings
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
DEBUG_MODE = os.environ.get("DEBUG", "0") == "1"

# Model settings
DEFAULT_MODEL_ID = "nvidia/canary-1b-v2"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_CHUNK_DURATION = 20  # seconds per chunk for long audio

# Hugging Face configuration
HF_TOKEN = os.environ.get("HUGGINGFACE_ACCESS_TOKEN")

# Diarization settings
DEFAULT_DIARIZE = True
DEFAULT_NUM_SPEAKERS = None  # None means auto-detection
DEFAULT_INCLUDE_DIARIZATION_IN_TEXT = True  # Whether to include speaker labels in the text


def force_cpu_from_env() -> bool:
    """True when FORCE_CPU is set (force CPU even if CUDA is available)."""
    return os.environ.get("FORCE_CPU", "").lower() in ("1", "true", "yes")


def use_cuda() -> bool:
    """Use GPU for inference only when CUDA is available and FORCE_CPU is not set."""
    if force_cpu_from_env():
        return False
    import torch

    return torch.cuda.is_available()


class Config:
    """Global configuration for NoobScribe"""

    # Singleton instance
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize configuration with default values"""
        # API settings
        self.host = os.environ.get("HOST", DEFAULT_HOST)
        self.port = int(os.environ.get("PORT", DEFAULT_PORT))
        self.debug = DEBUG_MODE
        self.force_cpu = force_cpu_from_env()

        # Model settings
        self.model_id = os.environ.get("MODEL_ID", DEFAULT_MODEL_ID)
        self.temperature = float(os.environ.get("TEMPERATURE", DEFAULT_TEMPERATURE))
        self.chunk_duration = int(os.environ.get("CHUNK_DURATION", DEFAULT_CHUNK_DURATION))

        # Diarization settings
        self.hf_token = HF_TOKEN
        self.enable_diarization = os.environ.get("ENABLE_DIARIZATION", str(DEFAULT_DIARIZE)).lower() == "true"
        self.include_diarization_in_text = os.environ.get("INCLUDE_DIARIZATION_IN_TEXT", str(DEFAULT_INCLUDE_DIARIZATION_IN_TEXT)).lower() == "true"
        self.default_num_speakers = DEFAULT_NUM_SPEAKERS

        # Speaker embedding settings
        self.speaker_similarity_threshold = float(os.environ.get("SPEAKER_SIMILARITY_THRESHOLD", "0.7"))
        self.chromadb_path = os.environ.get("CHROMADB_PATH", "./data/speakers")
        Path(self.chromadb_path).mkdir(parents=True, exist_ok=True)

        # Local model paths (optional; offline / no HuggingFace download)
        self.model_path = os.environ.get("MODEL_PATH", "").strip() or None
        self.diarization_model_path = os.environ.get("DIARIZATION_MODEL_PATH", "").strip() or None

        # Recordings + SQLite
        self.recordings_path = os.environ.get("RECORDINGS_PATH", "./data/recordings")
        Path(self.recordings_path).mkdir(parents=True, exist_ok=True)
        self.database_url = os.environ.get("DATABASE_URL", "sqlite+aiosqlite:///./data/noobscribe.db")

        # File paths
        self.temp_dir = os.environ.get("TEMP_DIR", "/tmp/noobscribe")
        Path(self.temp_dir).mkdir(parents=True, exist_ok=True)

        # Spoken language ID (SpeechBrain) when API ``language`` is omitted
        self.language_id_model_id = os.environ.get(
            "LANGUAGE_ID_MODEL_ID", "speechbrain/lang-id-voxlingua107-ecapa"
        )
        _lid_savedir = os.environ.get("LANGUAGE_ID_SAVEDIR", "").strip()
        self.language_id_savedir = _lid_savedir or str(Path(self.temp_dir) / "speechbrain_lang_id")
        Path(self.language_id_savedir).mkdir(parents=True, exist_ok=True)
        self.language_id_max_audio_seconds = int(os.environ.get("LANGUAGE_ID_MAX_AUDIO_SECONDS", "30"))

        logger.debug(f"Initialized configuration: debug={self.debug}, model={self.model_id}")

    def update_hf_token(self, token: str) -> None:
        """Update the HuggingFace token"""
        self.hf_token = token
        logger.info("Updated HuggingFace token")

    def get_hf_token(self) -> Optional[str]:
        """Get the HuggingFace token"""
        return self.hf_token

    def as_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary (for API responses)"""
        return {
            "host": self.host,
            "port": self.port,
            "debug": self.debug,
            "force_cpu": self.force_cpu,
            "model_id": self.model_id,
            "temperature": self.temperature,
            "chunk_duration": self.chunk_duration,
            "enable_diarization": self.enable_diarization,
            "include_diarization_in_text": self.include_diarization_in_text,
            "has_hf_token": self.hf_token is not None,
            "speaker_similarity_threshold": self.speaker_similarity_threshold,
            "chromadb_path": self.chromadb_path,
            "model_path": self.model_path,
            "diarization_model_path": self.diarization_model_path,
            "recordings_path": self.recordings_path,
            "database_url": self.database_url,
            "temp_dir": self.temp_dir,
            "language_id_model_id": self.language_id_model_id,
            "language_id_savedir": self.language_id_savedir,
            "language_id_max_audio_seconds": self.language_id_max_audio_seconds,
        }


# Create a global instance
config = Config()

def get_config() -> Config:
    """Get the global configuration instance"""
    return config
