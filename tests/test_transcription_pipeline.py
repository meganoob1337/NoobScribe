"""Transcription pipeline ordering and GPU model lifecycle."""
from __future__ import annotations

import os
import sys
import unittest
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diarization import DiarizationResult, SpeakerSegment
from models import WhisperSegment


class TestTranscriptionPipelineOrdering(unittest.TestCase):
    @patch("services.transcription_pipeline.transcribe_api_chunk")
    @patch("services.transcription_pipeline.split_audio_into_chunks", return_value=["chunk.wav"])
    @patch("services.transcription_pipeline.resolve_transcription_language", return_value="en")
    def test_diarizer_unloaded_before_transcription(self, _mock_lang, _mock_split, mock_transcribe):
        from services.transcription_pipeline import run_transcription_pipeline

        events: list[str] = []

        @contextmanager
        def tracking_session(**_kwargs):
            diarizer = MagicMock()
            diarizer.pipeline = MagicMock()
            diarizer.diarize.return_value = DiarizationResult(
                segments=[SpeakerSegment(start=0.0, end=1.0, speaker="speaker_SPEAKER_00")],
                num_speakers=1,
                embeddings=None,
            )
            events.append("diarizer_enter")
            try:
                yield diarizer
            finally:
                events.append("diarizer_exit")

        def track_transcribe(*_args, **_kwargs):
            events.append("transcribe")
            return "hello", [WhisperSegment(id=0, start=0.0, end=1.0, text="hello")]

        mock_transcribe.side_effect = track_transcribe
        config = MagicMock()
        config.use_api = True
        config.chunk_duration = 10
        config.diarization_model_path = "/models/diarization"
        config.get_hf_token.return_value = None
        config.include_diarization_in_text = False
        config.model_id = "whisper-1"
        config.speaker_similarity_threshold = 0.7

        with patch("services.transcription_pipeline.diarizer_session", tracking_session):
            run_transcription_pipeline(
                "audio.wav",
                None,
                config,
                None,
                diarize=True,
            )

        self.assertEqual(events, ["diarizer_enter", "diarizer_exit", "transcribe"])


if __name__ == "__main__":
    unittest.main()
