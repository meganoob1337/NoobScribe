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


def _base_config(**overrides):
    config = MagicMock()
    config.use_api = True
    config.load_models_on_demand = False
    config.chunk_min_duration = 15
    config.chunk_max_duration = 30
    config.chunk_silence_db = -40.0
    config.chunk_silence_min_duration = 0.3
    config.diarization_model_path = "/models/diarization"
    config.get_hf_token.return_value = None
    config.include_diarization_in_text = False
    config.get_transcription_model_name.return_value = "whisper-1"
    config.speaker_similarity_threshold = 0.7
    config.resolved_asr_model = MagicMock()
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


class TestTranscriptionPipelineOrdering(unittest.TestCase):
    @patch("services.transcription_pipeline.transcribe_api_chunk")
    @patch("services.transcription_pipeline.split_audio_into_chunks")
    @patch("services.transcription_pipeline.resolve_transcription_language", return_value="en")
    def test_diarizer_unloaded_before_transcription(self, _mock_lang, mock_split, mock_transcribe):
        from audio import AudioChunk
        from services.transcription_pipeline import run_transcription_pipeline

        mock_split.return_value = [AudioChunk(path="chunk.wav", start_sec=0.0)]
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
        config = _base_config()

        with patch("services.transcription_pipeline.diarizer_session", tracking_session):
            run_transcription_pipeline(
                "audio.wav",
                None,
                config,
                None,
                diarize=True,
            )

        self.assertEqual(events, ["diarizer_enter", "diarizer_exit", "transcribe"])

    @patch("services.transcription_pipeline.transcribe_api_chunk")
    @patch("services.transcription_pipeline.split_audio_into_chunks")
    @patch("services.transcription_pipeline.resolve_transcription_language", return_value="en")
    def test_segment_timestamps_use_chunk_start(self, _mock_lang, mock_split, mock_transcribe):
        from audio import AudioChunk
        from services.transcription_pipeline import run_transcription_pipeline

        mock_split.return_value = [
            AudioChunk(path="c0.wav", start_sec=0.0),
            AudioChunk(path="c1.wav", start_sec=23.0),
        ]
        mock_transcribe.side_effect = [
            ("a", [WhisperSegment(id=0, start=0.0, end=1.0, text="a")]),
            ("b", [WhisperSegment(id=0, start=0.5, end=1.5, text="b")]),
        ]
        config = _base_config()

        result = run_transcription_pipeline(
            "audio.wav",
            None,
            config,
            None,
            diarize=False,
            timestamps=True,
        )
        segments = result.all_segments
        self.assertEqual(len(segments), 2)
        self.assertAlmostEqual(segments[0].start, 0.0)
        self.assertAlmostEqual(segments[1].start, 23.5)
        self.assertAlmostEqual(segments[1].end, 24.5)

    @patch("services.transcription_pipeline.unload_model")
    @patch("services.transcription_pipeline.load_model")
    @patch("services.transcription_pipeline.transcribe_audio_chunk")
    @patch("services.transcription_pipeline.split_audio_into_chunks")
    @patch("services.transcription_pipeline.resolve_transcription_language", return_value="en")
    def test_on_demand_loads_after_diarization_and_unloads(
        self, _mock_lang, mock_split, mock_transcribe, mock_load, mock_unload
    ):
        from audio import AudioChunk
        from services.transcription_pipeline import run_transcription_pipeline

        mock_split.return_value = [AudioChunk(path="chunk.wav", start_sec=0.0)]
        loaded = MagicMock(name="asr")
        mock_load.return_value = loaded
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

        def track_load(_resolved):
            events.append("asr_load")
            return loaded

        def track_transcribe(model, *_args, **_kwargs):
            events.append("transcribe")
            self.assertIs(model, loaded)
            return "hello", [WhisperSegment(id=0, start=0.0, end=1.0, text="hello")]

        def track_unload(model):
            events.append("asr_unload")
            self.assertIs(model, loaded)

        mock_load.side_effect = track_load
        mock_transcribe.side_effect = track_transcribe
        mock_unload.side_effect = track_unload

        config = _base_config(
            use_api=False,
            load_models_on_demand=True,
        )

        with patch("services.transcription_pipeline.diarizer_session", tracking_session):
            run_transcription_pipeline(
                "audio.wav",
                None,
                config,
                None,
                diarize=True,
            )

        self.assertEqual(
            events,
            ["diarizer_enter", "diarizer_exit", "asr_load", "transcribe", "asr_unload"],
        )
        mock_load.assert_called_once_with(config.resolved_asr_model)
        mock_unload.assert_called_once_with(loaded)

    @patch("services.transcription_pipeline.transcribe_api_chunk")
    @patch("services.transcription_pipeline.split_audio_into_chunks")
    @patch("services.transcription_pipeline.resolve_transcription_language", return_value="en")
    def test_progress_cancel_after_chunk_skips_remaining(self, _mock_lang, mock_split, mock_transcribe):
        from audio import AudioChunk
        from services.transcription_pipeline import run_transcription_pipeline

        mock_split.return_value = [
            AudioChunk(path="c0.wav", start_sec=0.0),
            AudioChunk(path="c1.wav", start_sec=23.0),
        ]
        mock_transcribe.side_effect = [
            ("a", [WhisperSegment(id=0, start=0.0, end=1.0, text="a")]),
            ("b", [WhisperSegment(id=0, start=0.5, end=1.5, text="b")]),
        ]
        transcribing_hits = []

        def on_progress(stage, current, total):
            if stage != "transcribing":
                return
            transcribing_hits.append(current)
            if current == 1 and transcribing_hits.count(1) == 2:
                raise RuntimeError("cancelled")

        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            run_transcription_pipeline(
                "audio.wav",
                None,
                _base_config(),
                None,
                diarize=False,
                on_progress=on_progress,
            )

        self.assertEqual(mock_transcribe.call_count, 1)


if __name__ == "__main__":
    unittest.main()
