# From https://github.com/jfgonsalves/parakeet-diarized (commit 6abadfd)
# Copyright (c) jfgonsalves - MIT License
# Modified for pause-based chunking
import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import wave
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from audio import (
    AudioChunk,
    choose_chunk_boundaries,
    detect_silence_intervals,
    split_audio_into_chunks,
)


class TestChooseChunkBoundaries(unittest.TestCase):
    def test_short_audio_no_internal_cuts(self):
        cuts = choose_chunk_boundaries(20.0, [], min_duration=15.0, max_duration=30.0)
        self.assertEqual(cuts, [0.0, 20.0])

    def test_force_cut_without_silence(self):
        cuts = choose_chunk_boundaries(70.0, [], min_duration=15.0, max_duration=30.0)
        self.assertEqual(cuts, [0.0, 30.0, 60.0, 70.0])

    def test_prefers_longest_silence_in_window(self):
        # Window for first cut: [15, 30]. Short silence at 18, longer at 22.
        silences = [(17.5, 18.5), (20.0, 24.0)]
        cuts = choose_chunk_boundaries(50.0, silences, min_duration=15.0, max_duration=30.0)
        self.assertAlmostEqual(cuts[1], 22.0)
        self.assertEqual(cuts[0], 0.0)
        self.assertEqual(cuts[-1], 50.0)

    def test_silence_outside_window_ignored(self):
        silences = [(5.0, 6.0), (40.0, 41.0)]  # too early / after first force window
        cuts = choose_chunk_boundaries(50.0, silences, min_duration=15.0, max_duration=30.0)
        self.assertEqual(cuts[1], 30.0)


class TestAudioChunking(unittest.TestCase):
    def create_test_wav(self, duration_seconds=10, sample_rate=16000, silent_ranges=None):
        """Create a test WAV; optional silent_ranges are (start, end) zeroed regions."""
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_file.close()

        n = int(duration_seconds * sample_rate)
        t = np.linspace(0, duration_seconds, n, False)
        data = (np.sin(2 * np.pi * 440 * t) * 20000).astype(np.int16)
        if silent_ranges:
            for start, end in silent_ranges:
                s = int(start * sample_rate)
                e = int(end * sample_rate)
                data[s:e] = 0

        with wave.open(temp_file.name, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(data.tobytes())

        return temp_file.name

    def test_no_chunking_for_short_audio(self):
        test_file = self.create_test_wav(duration_seconds=10)
        try:
            result = split_audio_into_chunks(
                test_file, min_duration=15.0, max_duration=30.0
            )
            self.assertEqual(len(result), 1)
            self.assertIsInstance(result[0], AudioChunk)
            self.assertEqual(result[0].path, test_file)
            self.assertEqual(result[0].start_sec, 0.0)
        finally:
            if os.path.exists(test_file):
                os.unlink(test_file)

    @patch('audio._extract_chunk')
    @patch('audio.detect_silence_intervals')
    def test_chunking_uses_silence_and_offsets(self, mock_detect, mock_extract):
        mock_detect.return_value = [(22.0, 24.0)]
        test_file = self.create_test_wav(duration_seconds=50)

        try:
            result = split_audio_into_chunks(
                test_file, min_duration=15.0, max_duration=30.0
            )
            self.assertEqual(len(result), 2)
            self.assertAlmostEqual(result[0].start_sec, 0.0)
            self.assertAlmostEqual(result[1].start_sec, 23.0)
            self.assertEqual(mock_extract.call_count, 2)
            # First extract: start 0, length 23
            self.assertAlmostEqual(mock_extract.call_args_list[0][0][1], 0.0)
            self.assertAlmostEqual(mock_extract.call_args_list[0][0][2], 23.0)
            # Second extract: start 23, length 27
            self.assertAlmostEqual(mock_extract.call_args_list[1][0][1], 23.0)
            self.assertAlmostEqual(mock_extract.call_args_list[1][0][2], 27.0)
        finally:
            for chunk in result:
                if chunk.path != test_file and os.path.exists(chunk.path):
                    try:
                        os.unlink(chunk.path)
                    except OSError:
                        pass
            if os.path.exists(test_file):
                os.unlink(test_file)

    @patch('audio._extract_chunk')
    @patch('audio.detect_silence_intervals', return_value=[])
    def test_force_split_at_max_without_silence(self, _mock_detect, mock_extract):
        test_file = self.create_test_wav(duration_seconds=65)
        try:
            result = split_audio_into_chunks(
                test_file, min_duration=15.0, max_duration=30.0
            )
            self.assertEqual(len(result), 3)
            self.assertEqual([c.start_sec for c in result], [0.0, 30.0, 60.0])
            self.assertEqual(mock_extract.call_count, 3)
        finally:
            if os.path.exists(test_file):
                os.unlink(test_file)

    def test_detect_silence_intervals_parses_ffmpeg(self):
        stderr = (
            "[silencedetect @ 0x1] silence_start: 22.0\n"
            "[silencedetect @ 0x1] silence_end: 24.0 | silence_duration: 2.0\n"
        )
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stderr = stderr
        with patch('subprocess.run', return_value=mock_proc):
            intervals = detect_silence_intervals("dummy.wav")
        self.assertEqual(intervals, [(22.0, 24.0)])


if __name__ == '__main__':
    unittest.main()
