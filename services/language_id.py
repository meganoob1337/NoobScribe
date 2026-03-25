"""
Spoken language identification (SpeechBrain VoxLingua107 ECAPA) when ASR language is omitted.

Model: https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Any, Optional

from config import use_cuda

logger = logging.getLogger(__name__)

_classifier: Any = None
_classifier_lock = threading.Lock()

# Label quirks in VoxLingua107 / SpeechBrain defaults (see model card on Hugging Face)
_LANG_CODE_FIXES = {"iw": "he", "jw": "jv"}


def normalize_language_param(language: Optional[str]) -> Optional[str]:
    """Treat empty/whitespace-only form values as missing."""
    if language is None:
        return None
    s = language.strip()
    return s if s else None


def _parse_classifier_labels(pred: tuple) -> Optional[str]:
    if not pred or len(pred) < 4:
        return None
    labels = pred[3]
    if not labels:
        return None
    raw = labels[0] if isinstance(labels, (list, tuple)) else labels
    if isinstance(raw, str):
        code = raw.split(":")[0].strip().lower()
        return code if code else None
    if raw is not None:
        return str(raw).lower()
    return None


def _get_classifier(config: Any) -> Any:
    global _classifier
    if os.environ.get("DISABLE_LANGUAGE_ID", "").lower() in ("1", "true", "yes"):
        return None
    with _classifier_lock:
        if _classifier is not None:
            return _classifier
        try:
            from speechbrain.inference.classifiers import EncoderClassifier
        except ImportError:
            logger.warning("speechbrain is not installed; language auto-detection is disabled")
            return None
        device = "cuda" if use_cuda() else "cpu"
        savedir = getattr(config, "language_id_savedir", None) or os.path.join(
            getattr(config, "temp_dir", "/tmp/noobscribe"), "speechbrain_lang_id"
        )
        source = getattr(config, "language_id_model_id", "speechbrain/lang-id-voxlingua107-ecapa")
        logger.info("Loading language identification model %s (device=%s)", source, device)
        _classifier = EncoderClassifier.from_hparams(
            source=source,
            savedir=savedir,
            run_opts={"device": device},
        )
        return _classifier


def detect_spoken_language(wav_path: str, config: Any) -> Optional[str]:
    """
    Return ISO 639-1-style language code for the start of ``wav_path`` (16 kHz mono WAV expected).

    Uses at most ``config.language_id_max_audio_seconds`` of audio from the beginning of the file.
    """
    classifier = _get_classifier(config)
    if classifier is None:
        return None
    max_sec = int(getattr(config, "language_id_max_audio_seconds", 30))
    max_samples = max(1, max_sec * 16000)
    try:
        with _classifier_lock:
            signal = classifier.load_audio(wav_path)
            if signal.dim() == 2:
                n = signal.shape[-1]
                if n > max_samples:
                    signal = signal[..., :max_samples]
            elif signal.dim() == 1:
                n = signal.shape[0]
                if n > max_samples:
                    signal = signal[:max_samples]
            pred = classifier.classify_batch(signal)
        code = _parse_classifier_labels(pred)
        if code:
            code = _LANG_CODE_FIXES.get(code, code)
        logger.info("Language identification result: %s", code)
        return code
    except Exception as e:
        logger.warning("Language identification failed: %s", e)
        return None


def resolve_transcription_language(language: Optional[str], wav_path: str, config: Any) -> Optional[str]:
    """
    If ``language`` is unset or blank, detect from ``wav_path``; otherwise return stripped user value.
    """
    normalized = normalize_language_param(language)
    if normalized is not None:
        return normalized
    return detect_spoken_language(wav_path, config)
