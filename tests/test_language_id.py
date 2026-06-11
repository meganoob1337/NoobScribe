"""Language ID segment planning and model unload."""
from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.language_id import (
    _plan_language_id_segments,
    _segment_start_seconds,
    detect_spoken_language,
)
import services.language_id as language_id_module


class TestLanguageIdSegmentPlanning(unittest.TestCase):
    def test_empty_duration(self):
        self.assertEqual(_segment_start_seconds(0, 30), [])
        self.assertEqual(_plan_language_id_segments(0, 30), [])

    def test_shorter_than_segment_uses_single_start(self):
        self.assertEqual(_segment_start_seconds(12.5, 30), [0.0])
        planned = _plan_language_id_segments(int(12.5 * 16000), 30)
        self.assertEqual(planned, [(0.0, 0, int(12.5 * 16000))])

    def test_exactly_segment_length(self):
        self.assertEqual(_segment_start_seconds(30, 30), [0.0])
        planned = _plan_language_id_segments(30 * 16000, 30)
        self.assertEqual(planned, [(0.0, 0, 30 * 16000)])

    def test_slightly_longer_than_segment_stays_single(self):
        self.assertEqual(_segment_start_seconds(31, 30), [0.0])

    def test_long_audio_uses_three_windows(self):
        self.assertEqual(_segment_start_seconds(120, 30), [0.0, 60.0, 90.0])
        planned = _plan_language_id_segments(120 * 16000, 30)
        self.assertEqual(
            planned,
            [
                (0.0, 0, 30 * 16000),
                (60.0, 60 * 16000, 30 * 16000),
                (90.0, 90 * 16000, 30 * 16000),
            ],
        )

    def test_medium_audio_dedupes_overlapping_anchors(self):
        starts = _segment_start_seconds(35, 30)
        self.assertEqual(starts, [0.0, 5.0])
        planned = _plan_language_id_segments(35 * 16000, 30)
        self.assertEqual(len(planned), 2)
        self.assertEqual(planned[0][0], 0.0)
        self.assertEqual(planned[1][0], 5.0)

    def test_very_short_clip_still_classifies_once(self):
        planned = _plan_language_id_segments(int(0.8 * 16000), 30)
        self.assertEqual(planned, [(0.0, 0, int(0.8 * 16000))])


class TestLanguageIdUnload(unittest.TestCase):
    def setUp(self):
        language_id_module._classifier = None

    def tearDown(self):
        language_id_module._classifier = None

    @patch("services.language_id.unload_language_id_classifier")
    @patch("services.language_id._get_classifier")
    def test_detect_spoken_language_unloads_classifier(self, mock_get_classifier, mock_unload):
        classifier = MagicMock()
        classifier.load_audio.return_value = MagicMock(dim=MagicMock(return_value=1), shape=[16000])
        mock_get_classifier.return_value = classifier

        with patch("services.language_id._signal_num_samples", return_value=16000), patch(
            "services.language_id._classify_signal_segment", return_value="en"
        ):
            result = detect_spoken_language("audio.wav", MagicMock(language_id_max_audio_seconds=30))

        self.assertEqual(result, "en")
        mock_unload.assert_called_once()


if __name__ == "__main__":
    unittest.main()
