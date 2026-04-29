from __future__ import annotations

import unittest

from src.postprocess.track_filter import BallTrackFilter
from src.utils.structures import TrackResult


def _track(x: float, y: float, score: float = 0.72, visible: int = 1) -> TrackResult:
    return TrackResult(ball_xy=[x, y], visible=visible, score=score, heatmap_shape=[288, 512])


def _missing(score: float = 0.05) -> TrackResult:
    return TrackResult(ball_xy=[-1.0, -1.0], visible=0, score=score, heatmap_shape=[288, 512])


class BallTrackFilterTest(unittest.TestCase):
    def test_rejects_isolated_bootstrap_outlier_before_locking(self) -> None:
        tracker = BallTrackFilter(fps=25.0)

        outputs = [
            tracker.update(_track(850.0, 5.0, 0.55)),
            tracker.update(_track(210.0, 440.0, 0.62)),
            tracker.update(_track(212.0, 439.0, 0.64)),
            tracker.update(_track(214.0, 438.0, 0.66)),
        ]

        self.assertFalse(outputs[0].visible)
        self.assertFalse(outputs[1].visible)
        self.assertFalse(outputs[2].visible)
        self.assertTrue(outputs[3].visible)
        self.assertAlmostEqual(outputs[3].ball_xy[0], 214.0)

    def test_hides_outlier_detection_without_drifting_render_point(self) -> None:
        tracker = BallTrackFilter(fps=25.0)

        self.assertTrue(tracker.update(_track(100.0, 100.0, 0.92)).visible)
        self.assertTrue(tracker.update(_track(145.0, 100.0, 0.88)).visible)
        self.assertTrue(tracker.update(_track(190.0, 100.0, 0.86)).visible)

        outlier = tracker.update(_track(780.0, 420.0, 0.95))
        recovered = tracker.update(_track(235.0, 100.0, 0.89))

        self.assertFalse(outlier.visible)
        self.assertEqual(outlier.ball_xy, [-1.0, -1.0])
        self.assertTrue(recovered.visible)
        self.assertAlmostEqual(recovered.ball_xy[0], 235.0)

    def test_short_missing_gap_can_coast_when_motion_is_stable(self) -> None:
        tracker = BallTrackFilter(fps=25.0)

        tracker.update(_track(100.0, 100.0, 0.92))
        tracker.update(_track(140.0, 100.0, 0.9))
        tracker.update(_track(180.0, 100.0, 0.9))

        coasted = tracker.update(_missing())

        self.assertTrue(coasted.visible)
        self.assertGreater(coasted.ball_xy[0], 180.0)
        self.assertLess(coasted.score, 0.05)

    def test_missing_gap_is_filled_from_parabolic_motion(self) -> None:
        tracker = BallTrackFilter(fps=25.0)

        for i in range(5):
            tracker.update(_track(100.0 + 20.0 * i, 200.0 - 18.0 * i + 2.0 * i * i, 0.92))

        coasted = tracker.update(_missing())

        self.assertTrue(coasted.visible)
        self.assertAlmostEqual(coasted.ball_xy[0], 200.0, delta=4.0)
        self.assertAlmostEqual(coasted.ball_xy[1], 160.0, delta=4.0)
        self.assertLess(coasted.score, 0.05)

    def test_detection_outside_parabola_is_replaced_by_prediction(self) -> None:
        tracker = BallTrackFilter(fps=25.0)

        for i in range(5):
            tracker.update(_track(100.0 + 20.0 * i, 200.0 - 18.0 * i + 2.0 * i * i, 0.92))

        corrected = tracker.update(_track(200.0, 235.0, 0.95))

        self.assertTrue(corrected.visible)
        self.assertAlmostEqual(corrected.ball_xy[0], 200.0, delta=4.0)
        self.assertAlmostEqual(corrected.ball_xy[1], 160.0, delta=4.0)
        self.assertLess(corrected.score, 0.95)

    def test_top_exit_hides_prediction_and_ignores_in_frame_hallucination(self) -> None:
        tracker = BallTrackFilter(fps=25.0)
        frame_shape = (180, 320, 3)

        for i in range(4):
            tracker.update(
                _track(80.0 + 20.0 * i, 120.0 - 34.0 * i, 0.92),
                frame_shape=frame_shape,
            )

        outside = tracker.update(_missing(), frame_shape=frame_shape)
        hallucination = tracker.update(_track(160.0, 8.0, 0.96), frame_shape=frame_shape)

        self.assertFalse(outside.visible)
        self.assertEqual(outside.ball_xy, [-1.0, -1.0])
        self.assertFalse(hallucination.visible)
        self.assertEqual(hallucination.ball_xy, [-1.0, -1.0])

    def test_relocks_after_stable_new_cluster(self) -> None:
        tracker = BallTrackFilter(fps=25.0)

        tracker.update(_track(100.0, 100.0, 0.92))
        tracker.update(_track(140.0, 100.0, 0.9))
        tracker.update(_track(180.0, 100.0, 0.9))

        first = tracker.update(_track(700.0, 350.0, 0.72))
        second = tracker.update(_track(706.0, 354.0, 0.74))
        third = tracker.update(_track(712.0, 358.0, 0.76))

        self.assertFalse(first.visible)
        self.assertFalse(second.visible)
        self.assertTrue(third.visible)
        self.assertAlmostEqual(third.ball_xy[0], 712.0)


if __name__ == "__main__":
    unittest.main()
