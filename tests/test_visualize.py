from __future__ import annotations

import unittest

import numpy as np

from src.utils.structures import FrameResult, TrackResult
from src.utils.visualize import TrackTrailRenderer


def _frame_result(frame_id: int, x: float, y: float) -> FrameResult:
    track = TrackResult(ball_xy=[x, y], visible=1, score=0.85, heatmap_shape=[288, 512])
    return FrameResult(frame_id=frame_id, pose=[], track=track)


def _missing_result(frame_id: int) -> FrameResult:
    track = TrackResult(ball_xy=[-1.0, -1.0], visible=0, score=0.0, heatmap_shape=[288, 512])
    return FrameResult(frame_id=frame_id, pose=[], track=track)


class TrackTrailRendererTest(unittest.TestCase):
    def test_hit_marker_stays_solid_red_for_two_seconds_then_disappears(self) -> None:
        renderer = TrackTrailRenderer(fps=25.0)
        frame = np.zeros((160, 180, 3), dtype=np.uint8)

        renderer.draw_on(frame.copy(), _frame_result(0, 40.0, 80.0), timestamp_ms=0)
        renderer.draw_on(frame.copy(), _frame_result(1, 70.0, 80.0), timestamp_ms=40)
        hit_canvas = renderer.draw_on(frame.copy(), _frame_result(2, 70.0, 45.0), timestamp_ms=80)
        before_expiry = renderer.draw_on(frame.copy(), _missing_result(3), timestamp_ms=2079)
        after_expiry = renderer.draw_on(frame.copy(), _missing_result(4), timestamp_ms=2080)

        red = np.array([0, 0, 255], dtype=np.uint8)
        self.assertTrue(np.array_equal(hit_canvas[80, 70], red))
        self.assertFalse(np.array_equal(hit_canvas[45, 70], red))
        self.assertTrue(np.array_equal(before_expiry[80, 70], red))
        self.assertFalse(np.array_equal(after_expiry[80, 70], red))

    def test_does_not_mark_straight_fast_motion_as_hit(self) -> None:
        renderer = TrackTrailRenderer(fps=25.0)
        frame = np.zeros((160, 180, 3), dtype=np.uint8)

        renderer.draw_on(frame.copy(), _frame_result(0, 40.0, 80.0))
        renderer.draw_on(frame.copy(), _frame_result(1, 70.0, 80.0))
        canvas = renderer.draw_on(frame.copy(), _frame_result(2, 100.0, 80.0))

        self.assertFalse(np.array_equal(canvas[80, 100], np.array([0, 0, 255], dtype=np.uint8)))


if __name__ == "__main__":
    unittest.main()
