"""Tests for curated local ASR model registry and ONNX provider selection."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services import asr_models
from services.asr_backend import resolve_onnx_providers
from services.asr_models import (
    DEFAULT_ASR_MODEL,
    LoadedAsrModel,
    ResolvedAsrModel,
    validate_local_asr_config,
)
from transcription import _segments_from_onnx_result, transcribe_audio_chunk


class TestAsrModelRegistry(unittest.TestCase):
    def setUp(self):
        asr_models.clear_registry_cache()

    def test_default_model_resolves_nemo_backend(self):
        resolved, path = validate_local_asr_config(DEFAULT_ASR_MODEL, None)
        self.assertIsNone(path)
        self.assertEqual(resolved.spec.backend, "nemo")
        self.assertEqual(resolved.spec.source.hf_id, "nvidia/canary-1b-v2")

    def test_onnx_int8_entry(self):
        resolved, _ = validate_local_asr_config("canary-1b-v2-onnx-int8", None)
        self.assertEqual(resolved.spec.backend, "onnx")
        self.assertEqual(resolved.spec.source.onnx_asr_name, "nemo-canary-1b-v2")
        self.assertEqual(resolved.effective_quantization, "int8")

    def test_unknown_model_lists_available_keys(self):
        with self.assertRaisesRegex(ValueError, "Unknown local ASR model 'missing'"):
            validate_local_asr_config("missing", None)
        with self.assertRaisesRegex(ValueError, "canary-1b-v2-onnx-int8"):
            validate_local_asr_config("missing", None)

    def test_model_path_override_disallowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            custom = Path(tmp) / "custom.yaml"
            custom.write_text(
                "models:\n"
                "  locked:\n"
                "    display_name: Locked\n"
                "    backend: nemo\n"
                "    source:\n"
                "      hf_id: nvidia/canary-1b-v2\n"
                "    allow_path_override: false\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"ASR_MODELS_CONFIG": str(custom)}):
                asr_models.clear_registry_cache()
                with self.assertRaisesRegex(ValueError, "does not allow MODEL_PATH override"):
                    validate_local_asr_config("locked", "/tmp/model.nemo")

    def test_int8_fallback_when_local_path_missing_int8_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "encoder-model.onnx").write_text("x", encoding="utf-8")
            resolved, path = validate_local_asr_config("canary-1b-v2-onnx-int8", tmp)
            self.assertEqual(path, tmp)
            self.assertIsNone(resolved.effective_quantization)


class TestOnnxProviders(unittest.TestCase):
    @patch("services.asr_backend.use_cuda", return_value=False)
    def test_cpu_only_providers(self, _mock_cuda):
        mock_ort = MagicMock()
        mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]
        mock_ort.__version__ = "1.24.2"
        with patch.dict(sys.modules, {"onnxruntime": mock_ort}):
            self.assertEqual(resolve_onnx_providers(), ["CPUExecutionProvider"])

    @patch("services.asr_backend.use_cuda", return_value=True)
    def test_cuda_when_available(self, _mock_cuda):
        mock_ort = MagicMock()
        mock_ort.get_available_providers.return_value = [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        mock_ort.__version__ = "1.24.2"
        with patch.dict(sys.modules, {"onnxruntime": mock_ort}):
            self.assertEqual(
                resolve_onnx_providers(),
                ["CUDAExecutionProvider", "CPUExecutionProvider"],
            )


class TestOnnxTranscriptionMapping(unittest.TestCase):
    def test_segment_list_mapping(self):
        seg = MagicMock()
        seg.start = 0.0
        seg.end = 1.5
        seg.text = "hello"
        text, segments = _segments_from_onnx_result([seg])
        self.assertEqual(text, "hello")
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].text, "hello")

    def test_transcribe_onnx_chunk_uses_loaded_wrapper(self):
        resolved = validate_local_asr_config("canary-1b-v2-onnx-int8", None)[0]
        mock_model = MagicMock()
        item = MagicMock(start=0.0, end=1.0, text="hi")
        mock_model.recognize.return_value = [item]
        loaded = LoadedAsrModel(backend="onnx", model=mock_model, spec=resolved.spec)
        text, segments = transcribe_audio_chunk(loaded, "chunk.wav", language="en")
        self.assertEqual(text, "hi")
        self.assertEqual(segments[0].text, "hi")
        mock_model.recognize.assert_called_once_with(
            "chunk.wav", language="en", target_language="en"
        )


class TestConfigLocalValidation(unittest.TestCase):
    def test_config_rejects_unknown_local_model(self):
        import config as config_module

        config_module.Config._instance = None
        with patch.dict(
            os.environ,
            {"USE_API": "0", "ASR_MODEL": "not-a-real-model"},
            clear=False,
        ):
            with self.assertRaises(ValueError):
                config_module.Config()
        config_module.Config._instance = None


if __name__ == "__main__":
    unittest.main()
