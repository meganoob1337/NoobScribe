"""Language ID segment planning for short and long audio."""
import unittest

from services.language_id import (
    _plan_language_id_segments,
    _segment_start_seconds,
)


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


if __name__ == "__main__":
    unittest.main()
