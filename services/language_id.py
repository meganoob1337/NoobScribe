"""
Spoken language identification (SpeechBrain VoxLingua107 ECAPA) when ASR language is omitted.

Model: https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa
"""
from __future__ import annotations

import logging
import os
import threading
from collections import Counter
from typing import Any, List, Optional

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


_SAMPLE_RATE = 16000
# Skip extra windows when the timeline cannot fit a meaningfully distinct second sample.
_MIN_EXTRA_START_SEC = 1.5
# Ignore tail windows shorter than this when other samples exist (SpeechBrain still gets full file on short input).
_MIN_CLASSIFY_SEC = 1.0


def _segment_start_seconds(duration_sec: float, segment_sec: int) -> List[float]:
    """
    Start times for start, middle, and 75% samples.

    Short files use a single window from 0 (entire clip, up to ``segment_sec``).
    """
    if duration_sec <= 0:
        return []

    if duration_sec <= segment_sec:
        return [0.0]

    max_start = duration_sec - segment_sec
    if max_start < _MIN_EXTRA_START_SEC:
        return [0.0]

    raw_starts = [
        0.0,
        min(max_start, 0.5 * duration_sec),
        min(max_start, 0.75 * duration_sec),
    ]
    min_gap = min(_MIN_EXTRA_START_SEC, segment_sec * 0.05)
    unique: List[float] = []
    for start in raw_starts:
        if not any(abs(start - existing) < min_gap for existing in unique):
            unique.append(start)
    return unique or [0.0]


def _plan_language_id_segments(
    total_samples: int,
    segment_sec: int,
    *,
    sample_rate: int = _SAMPLE_RATE,
) -> List[tuple[float, int, int]]:
    """
    Return ``(start_sec, start_sample, num_samples)`` windows to classify.

    Falls back to one full clip when every planned window would be too short.
    """
    if total_samples <= 0:
        return []

    segment_samples = max(1, segment_sec * sample_rate)
    min_classify_samples = max(1, int(_MIN_CLASSIFY_SEC * sample_rate))
    duration_sec = total_samples / sample_rate
    starts_sec = _segment_start_seconds(duration_sec, segment_sec)

    planned: List[tuple[float, int, int]] = []
    for start_sec in starts_sec:
        start_sample = max(0, min(int(start_sec * sample_rate), total_samples))
        end_sample = min(start_sample + segment_samples, total_samples)
        num_samples = end_sample - start_sample
        if num_samples <= 0:
            continue
        if num_samples < min_classify_samples and len(starts_sec) > 1:
            continue
        planned.append((start_sec, start_sample, num_samples))

    if planned:
        return planned

    num_samples = min(total_samples, segment_samples)
    if num_samples <= 0:
        return []
    return [(0.0, 0, num_samples)]


def _signal_num_samples(signal: Any) -> int:
    if signal.dim() == 2:
        return int(signal.shape[-1])
    if signal.dim() == 1:
        return int(signal.shape[0])
    return 0


def _slice_signal(signal: Any, start_sample: int, num_samples: int) -> Any:
    end_sample = start_sample + num_samples
    if signal.dim() == 2:
        return signal[..., start_sample:end_sample]
    if signal.dim() == 1:
        return signal[start_sample:end_sample]
    return signal


def _normalize_lang_code(code: Optional[str]) -> Optional[str]:
    if not code:
        return None
    return _LANG_CODE_FIXES.get(code, code)


def _classify_signal_segment(classifier: Any, signal: Any) -> Optional[str]:
    pred = classifier.classify_batch(signal)
    return _normalize_lang_code(_parse_classifier_labels(pred))


def _pick_majority_language(segment_codes: List[Optional[str]]) -> Optional[str]:
    """Majority vote; on tie use the first segment result (start sample)."""
    valid = [c for c in segment_codes if c]
    if not valid:
        return None
    counts = Counter(valid)
    top_count = counts.most_common(1)[0][1]
    winners = [code for code, count in counts.items() if count == top_count]
    if len(winners) == 1:
        return winners[0]
    for code in segment_codes:
        if code:
            return code
    return None


def detect_spoken_language(wav_path: str, config: Any) -> Optional[str]:
    """
    Return ISO 639-1-style language code for ``wav_path`` (16 kHz mono WAV expected).

    Classifies up to three ``config.language_id_max_audio_seconds`` segments from the
    start, middle, and 75% mark; returns the majority label (first segment on a tie).
    """
    classifier = _get_classifier(config)
    if classifier is None:
        return None
    segment_sec = int(getattr(config, "language_id_max_audio_seconds", 30))
    try:
        with _classifier_lock:
            signal = classifier.load_audio(wav_path)
            total_samples = _signal_num_samples(signal)
            if total_samples <= 0:
                return None
            duration_sec = total_samples / _SAMPLE_RATE
            segments = _plan_language_id_segments(total_samples, segment_sec)
            if not segments:
                return None
            if len(segments) == 1 and duration_sec <= segment_sec:
                logger.info(
                    "Language identification using single sample (audio %.1fs <= %ss segment)",
                    duration_sec,
                    segment_sec,
                )
            elif len(segments) < 3:
                logger.info(
                    "Language identification using %s sample(s) for %.1fs audio",
                    len(segments),
                    duration_sec,
                )
            segment_codes: List[Optional[str]] = []
            for start_sec, start_sample, num_samples in segments:
                chunk = _slice_signal(signal, start_sample, num_samples)
                if _signal_num_samples(chunk) <= 0:
                    segment_codes.append(None)
                    continue
                code = _classify_signal_segment(classifier, chunk)
                segment_codes.append(code)
                logger.info(
                    "Language identification segment at %.1fs (%.1fs): %s",
                    start_sec,
                    num_samples / _SAMPLE_RATE,
                    code,
                )
        code = _pick_majority_language(segment_codes)
        logger.info("Language identification result (majority): %s from %s", code, segment_codes)
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
