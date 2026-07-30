"""Load ASR model and speaker DB for RQ workers."""
from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

from config import get_config, use_cuda
from database.speakers import SpeakerDB
from transcription import load_model

logger = logging.getLogger(__name__)

_asr_model: Any = None
_speaker_db: Optional[SpeakerDB] = None
_resources_initialized = False


def init_worker_resources() -> Tuple[Any, Optional[SpeakerDB]]:
    """Load models once per worker process (unless LOAD_MODELS_ON_DEMAND)."""
    global _asr_model, _speaker_db, _resources_initialized
    if _resources_initialized:
        return _asr_model, _speaker_db

    cfg = get_config()
    try:
        if use_cuda():
            import torch

            logger.info("CUDA available: %s", torch.cuda.get_device_name(0))
        else:
            logger.warning("Worker using CPU for inference (this will be slow)")

        if getattr(cfg, "use_api", False):
            _asr_model = None
            logger.info(
                "Worker ASR model skipped (USE_API=1); local NeMo/Canary will not be loaded"
            )
            logger.info(
                "Worker using remote STT API at %s (model=%s)",
                cfg.stt_base_url,
                cfg.model_id,
            )
        elif getattr(cfg, "load_models_on_demand", False):
            _asr_model = None
            logger.info(
                "Worker ASR model deferred (LOAD_MODELS_ON_DEMAND=1); "
                "model loads during transcription and unloads afterward"
            )
        else:
            _asr_model = load_model(cfg.resolved_asr_model)
            logger.info(
                "Worker ASR model loaded (%s, backend=%s)",
                cfg.model_path or cfg.asr_model,
                cfg.resolved_asr_model.spec.backend,
            )
    except Exception as e:
        logger.error("Worker failed to load ASR model: %s", e)
        raise

    try:
        _speaker_db = SpeakerDB(
            db_path=cfg.chromadb_path,
            similarity_threshold=cfg.speaker_similarity_threshold,
        )
        logger.info("Worker speaker database initialized at %s", cfg.chromadb_path)
    except Exception as e:
        logger.warning("Worker speaker database init failed: %s", e)
        _speaker_db = None

    _resources_initialized = True
    return _asr_model, _speaker_db


def get_worker_asr_model() -> Any:
    """Return the warm ASR model, or None when USE_API / LOAD_MODELS_ON_DEMAND."""
    if not _resources_initialized:
        init_worker_resources()
    return _asr_model


def get_worker_speaker_db() -> Optional[SpeakerDB]:
    if _speaker_db is None:
        init_worker_resources()
    return _speaker_db
