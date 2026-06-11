"""Diarizer GPU unload behavior."""
from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diarization import Diarizer, diarizer_session


class TestDiarizerUnload(unittest.TestCase):
    @patch("diarization.use_cuda", return_value=True)
    @patch("diarization.torch.cuda.empty_cache")
    @patch("diarization.gc.collect")
    def test_unload_clears_pipeline_and_frees_cache(self, mock_gc, mock_empty_cache, _mock_cuda):
        diarizer = Diarizer.__new__(Diarizer)
        pipeline = MagicMock()
        diarizer.pipeline = pipeline
        diarizer.device = "cuda"

        diarizer.unload()

        pipeline.to.assert_called_once()
        self.assertIsNone(diarizer.pipeline)
        mock_gc.assert_called_once()
        mock_empty_cache.assert_called_once()

    @patch.object(Diarizer, "unload")
    @patch.object(Diarizer, "_initialize")
    def test_diarizer_session_unloads_on_exit(self, _mock_init, mock_unload):
        with diarizer_session(access_token="token") as diarizer:
            self.assertIsInstance(diarizer, Diarizer)
        mock_unload.assert_called_once()


if __name__ == "__main__":
    unittest.main()
