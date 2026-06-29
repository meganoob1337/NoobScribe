"""ONNX Runtime provider selection and ONNX ASR loading."""
from __future__ import annotations

import logging
from typing import Any, List

from config import use_cuda
from services.asr_models import LoadedAsrModel, ResolvedAsrModel

logger = logging.getLogger(__name__)


def resolve_onnx_providers() -> List[str]:
    """Pick ONNX Runtime execution providers from environment and installed packages."""
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise ImportError(
            "onnx-asr backend requires onnxruntime. "
            "Install with: pip install onnx-asr[cpu,hub] or onnx-asr[gpu,hub]"
        ) from e

    version = getattr(ort, "__version__", "")
    if version == "1.24.1":
        logger.warning(
            "onnxruntime 1.24.1 has known compatibility issues with onnx-asr; "
            "upgrade to 1.24.2+ or use another version"
        )

    available = set(ort.get_available_providers())
    if use_cuda():
        if "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            logger.info("ONNX Runtime providers: %s", providers)
            return providers
        logger.warning(
            "CUDA requested but CUDAExecutionProvider is unavailable "
            "(install onnx-asr[gpu,hub] / onnxruntime-gpu); using CPU"
        )
    providers = ["CPUExecutionProvider"]
    logger.info("ONNX Runtime providers: %s", providers)
    return providers


def load_onnx_asr_model(resolved: ResolvedAsrModel) -> Any:
    """Load an onnx-asr model from a resolved registry entry."""
    try:
        import onnx_asr
    except ImportError as e:
        raise ImportError(
            "ONNX ASR backend requires onnx-asr. "
            "Install with: pip install onnx-asr[cpu,hub] or onnx-asr[gpu,hub]"
        ) from e

    spec = resolved.spec
    onnx_name = spec.source.onnx_asr_name
    assert onnx_name is not None

    logger.info(
        "Loading ONNX ASR model %s (%s, quantization=%s, path=%s)",
        spec.id,
        onnx_name,
        resolved.effective_quantization,
        resolved.effective_path or "hub",
    )

    model = onnx_asr.load_model(
        onnx_name,
        path=resolved.effective_path,
        quantization=resolved.effective_quantization,
        providers=resolve_onnx_providers(),
    )
    return model.with_timestamps()


def wrap_loaded_model(resolved: ResolvedAsrModel, model: Any) -> LoadedAsrModel:
    return LoadedAsrModel(backend=resolved.spec.backend, model=model, spec=resolved.spec)
