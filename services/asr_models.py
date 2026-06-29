"""Curated local ASR model registry (config/asr_models.yaml)."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)

DEFAULT_ASR_MODEL = "canary-1b-v2"
DEFAULT_REMOTE_MODEL_ID = "whisper-1"

AsrBackend = Literal["nemo", "onnx"]

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_REGISTRY_PATH = _REPO_ROOT / "config" / "asr_models.yaml"


@dataclass(frozen=True)
class AsrModelSource:
    hf_id: Optional[str] = None
    onnx_asr_name: Optional[str] = None
    hf_repo: Optional[str] = None


@dataclass(frozen=True)
class AsrModelSpec:
    id: str
    display_name: str
    backend: AsrBackend
    source: AsrModelSource
    quantization: Optional[str] = None
    allow_path_override: bool = False


@dataclass(frozen=True)
class ResolvedAsrModel:
    spec: AsrModelSpec
    effective_path: Optional[str] = None
    effective_quantization: Optional[str] = None


@dataclass
class LoadedAsrModel:
    """Wrapper returned by transcription.load_model for backend dispatch."""

    backend: AsrBackend
    model: Any
    spec: AsrModelSpec


def registry_path() -> Path:
    override = os.environ.get("ASR_MODELS_CONFIG", "").strip()
    if override:
        return Path(override)
    return _DEFAULT_REGISTRY_PATH


@lru_cache(maxsize=1)
def _load_registry_raw() -> Dict[str, Any]:
    path = registry_path()
    if not path.is_file():
        raise FileNotFoundError(f"ASR model registry not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    models = data.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError(f"ASR model registry at {path} must define a non-empty 'models' map")
    return models


def clear_registry_cache() -> None:
    _load_registry_raw.cache_clear()


def _parse_spec(model_id: str, raw: Dict[str, Any]) -> AsrModelSpec:
    backend = raw.get("backend")
    if backend not in ("nemo", "onnx"):
        raise ValueError(f"Model {model_id!r}: backend must be 'nemo' or 'onnx'")

    source_raw = raw.get("source") or {}
    source = AsrModelSource(
        hf_id=source_raw.get("hf_id"),
        onnx_asr_name=source_raw.get("onnx_asr_name"),
        hf_repo=source_raw.get("hf_repo"),
    )

    if backend == "nemo" and not source.hf_id:
        raise ValueError(f"Model {model_id!r}: nemo backend requires source.hf_id")
    if backend == "onnx" and not source.onnx_asr_name:
        raise ValueError(f"Model {model_id!r}: onnx backend requires source.onnx_asr_name")

    quantization = raw.get("quantization")
    if quantization is not None and quantization != "int8":
        raise ValueError(f"Model {model_id!r}: unsupported quantization {quantization!r}")

    return AsrModelSpec(
        id=model_id,
        display_name=str(raw.get("display_name") or model_id),
        backend=backend,
        source=source,
        quantization=quantization,
        allow_path_override=bool(raw.get("allow_path_override", False)),
    )


def list_local_asr_models() -> List[AsrModelSpec]:
    return [_parse_spec(mid, raw) for mid, raw in sorted(_load_registry_raw().items())]


def list_local_asr_model_keys() -> List[str]:
    return sorted(_load_registry_raw().keys())


def resolve_local_asr_model(key: str) -> AsrModelSpec:
    models = _load_registry_raw()
    if key not in models:
        available = ", ".join(sorted(models.keys()))
        raise ValueError(
            f"Unknown local ASR model {key!r}. Configured models: {available}"
        )
    return _parse_spec(key, models[key])


def _resolve_onnx_quantization(spec: AsrModelSpec, model_path: Optional[str]) -> Optional[str]:
    if spec.quantization != "int8":
        return spec.quantization
    if not model_path:
        return "int8"
    path = Path(model_path)
    if (path / "encoder-model.int8.onnx").exists():
        return "int8"
    logger.warning(
        "int8 ONNX files not found in %s; falling back to fp32 ONNX weights",
        model_path,
    )
    return None


def validate_local_asr_config(
    asr_model_key: str,
    model_path: Optional[str],
) -> Tuple[ResolvedAsrModel, Optional[str]]:
    """
    Resolve registry entry and validate MODEL_PATH override rules.

    Returns (resolved spec, effective model path for loading).
    """
    spec = resolve_local_asr_model(asr_model_key)
    effective_path: Optional[str] = None

    if model_path:
        if not spec.allow_path_override:
            raise ValueError(
                f"ASR model {asr_model_key!r} does not allow MODEL_PATH override"
            )
        effective_path = model_path

    effective_quantization = (
        _resolve_onnx_quantization(spec, effective_path)
        if spec.backend == "onnx"
        else None
    )

    return (
        ResolvedAsrModel(
            spec=spec,
            effective_path=effective_path,
            effective_quantization=effective_quantization,
        ),
        effective_path,
    )
