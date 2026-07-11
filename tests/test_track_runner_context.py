from __future__ import annotations

import unittest
from types import SimpleNamespace

from src.runners.tracknet_realtime_runner import (
    _frame_step_seconds,
    _reset_filter_state_preserving_debug,
)
from src.runners.unified_runner import _pose_bboxes


class _FakeTrackFilter:
    def __init__(self) -> None:
        self.debug_records = [{"frame_index": 1}]
        self.reset_called = False

    def reset(self) -> None:
        self.reset_called = True
        self.debug_records.clear()


class TrackRunnerContextTest(unittest.TestCase):
    def test_realtime_step_uses_capture_elapsed_time(self) -> None:
        self.assertAlmostEqual(_frame_step_seconds(10.18, 10.00, 60.0), 0.18)
        self.assertAlmostEqual(_frame_step_seconds(10.00, None, 50.0), 0.02)

    def test_filter_reset_preserves_accumulated_debug_records(self) -> None:
        track_filter = _FakeTrackFilter()

        _reset_filter_state_preserving_debug(track_filter)

        self.assertTrue(track_filter.reset_called)
        self.assertEqual(track_filter.debug_records, [{"frame_index": 1}])

    def test_unified_runner_forwards_valid_pose_boxes(self) -> None:
        poses = [
            SimpleNamespace(bbox=[10.0, 20.0, 80.0, 180.0]),
            SimpleNamespace(bbox=[30.0, 40.0, 20.0, 90.0]),
            SimpleNamespace(bbox=[]),
        ]

        self.assertEqual(_pose_bboxes(poses), [(10.0, 20.0, 80.0, 180.0)])


if __name__ == "__main__":
    unittest.main()
