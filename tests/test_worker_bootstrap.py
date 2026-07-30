"""Worker bootstrap: ASR model loading and USE_API mode."""
from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if "chromadb" not in sys.modules:
    sys.modules["chromadb"] = MagicMock()
    sys.modules["chromadb.config"] = MagicMock()


def _reset_worker_bootstrap():
    import services.worker_bootstrap as wb

    wb._asr_model = None
    wb._speaker_db = None
    wb._resources_initialized = False
    return wb


def _local_cfg(**overrides):
    cfg = MagicMock()
    cfg.use_api = False
    cfg.load_models_on_demand = False
    cfg.model_path = None
    cfg.asr_model = "canary-1b-v2"
    cfg.resolved_asr_model = MagicMock()
    cfg.resolved_asr_model.spec.backend = "nemo"
    cfg.chromadb_path = "/tmp/noobscribe-test-speakers"
    cfg.speaker_similarity_threshold = 0.7
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


class TestWorkerBootstrapApiMode(unittest.TestCase):
    def setUp(self):
        _reset_worker_bootstrap()

    def tearDown(self):
        _reset_worker_bootstrap()

    @patch("services.worker_bootstrap.get_config")
    @patch("services.worker_bootstrap.SpeakerDB")
    @patch("services.worker_bootstrap.load_model")
    def test_api_mode_skips_load_model(self, mock_load_model, mock_speaker_db, mock_get_config):
        import services.worker_bootstrap as wb

        cfg = _local_cfg(use_api=True, stt_base_url="https://api.example.com/v1", model_id="whisper-1")
        mock_get_config.return_value = cfg
        mock_speaker_db.return_value = MagicMock()

        asr_model, speaker_db = wb.init_worker_resources()

        self.assertIsNone(asr_model)
        mock_load_model.assert_not_called()
        self.assertIsNotNone(speaker_db)
        self.assertTrue(wb._resources_initialized)

    @patch("services.worker_bootstrap.get_config")
    @patch("services.worker_bootstrap.SpeakerDB")
    @patch("services.worker_bootstrap.load_model")
    def test_get_worker_asr_model_initializes_in_api_mode(self, mock_load_model, mock_speaker_db, mock_get_config):
        import services.worker_bootstrap as wb

        cfg = _local_cfg(use_api=True, stt_base_url="https://api.example.com/v1", model_id="whisper-1")
        mock_get_config.return_value = cfg
        mock_speaker_db.return_value = MagicMock()

        model = wb.get_worker_asr_model()

        self.assertIsNone(model)
        mock_load_model.assert_not_called()
        self.assertTrue(wb._resources_initialized)

    @patch("services.worker_bootstrap.get_config")
    @patch("services.worker_bootstrap.SpeakerDB")
    @patch("services.worker_bootstrap.load_model")
    def test_on_demand_skips_preload(self, mock_load_model, mock_speaker_db, mock_get_config):
        import services.worker_bootstrap as wb

        cfg = _local_cfg(load_models_on_demand=True)
        mock_get_config.return_value = cfg
        mock_speaker_db.return_value = MagicMock()

        asr_model, speaker_db = wb.init_worker_resources()

        self.assertIsNone(asr_model)
        mock_load_model.assert_not_called()
        self.assertIsNotNone(speaker_db)
        self.assertIsNone(wb.get_worker_asr_model())


if __name__ == "__main__":
    unittest.main()
