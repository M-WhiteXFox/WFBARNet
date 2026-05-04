from __future__ import annotations

import unittest

from src.postprocess.track_correction import RealtimeKalmanTrackCorrector, RealtimeKalmanTrackCorrectorConfig
from src.postprocess.track_filter import BallTrackFilter
from src.utils.structures import TrackResult


def _track(x: float, y: float, score: float = 0.82, visible: int = 1) -> TrackResult:
    return TrackResult(ball_xy=[x, y], visible=visible, score=score, heatmap_shape=[288, 512])


def _missing(score: float = 0.05) -> TrackResult:
    return TrackResult(ball_xy=[-1.0, -1.0], visible=0, score=score, heatmap_shape=[288, 512])


def _corrector(*, fixed_lag_frames: int = 0) -> RealtimeKalmanTrackCorrector:
    return RealtimeKalmanTrackCorrector(
        RealtimeKalmanTrackCorrectorConfig(fps=25.0, fixed_lag_frames=fixed_lag_frames),
        debug_enabled=True,
    )


class RealtimeKalmanTrackCorrectorTest(unittest.TestCase):
    def test_prefers_motion_candidate_over_higher_score_noise(self) -> None:
        corrector = _corrector()

        corrector.update(_track(100.0, 100.0, 0.9), dt=0.04, frame_shape=(400, 600, 3))
        corrector.update(_track(130.0, 100.0, 0.9), dt=0.04, frame_shape=(400, 600, 3))
        corrector.update(_track(160.0, 100.0, 0.9), dt=0.04, frame_shape=(400, 600, 3))

        corrected = corrector.update_candidates(
            [
                _track(520.0, 300.0, 0.98),
                _track(190.0, 100.0, 0.56),
            ],
            dt=0.04,
            frame_shape=(400, 600, 3),
        )

        self.assertTrue(corrected.visible)
        self.assertLess(abs(corrected.ball_xy[0] - 190.0), abs(corrected.ball_xy[0] - 520.0))
        self.assertAlmostEqual(corrected.ball_xy[1], 100.0, delta=20.0)
        self.assertEqual(corrector.debug_records[-1]["selected_candidate_index"], 1)
        self.assertEqual(corrector.debug_records[-1]["action"], "accept")

    def test_coasts_through_person_occlusion_when_candidate_is_inside_bbox(self) -> None:
        corrector = _corrector()
        frame_shape = (400, 600, 3)
        person_bboxes = [(175.0, 70.0, 230.0, 140.0)]

        corrector.update(_track(100.0, 100.0, 0.9), dt=0.04, frame_shape=frame_shape)
        corrector.update(_track(130.0, 100.0, 0.9), dt=0.04, frame_shape=frame_shape)
        corrector.update(_track(160.0, 100.0, 0.9), dt=0.04, frame_shape=frame_shape)

        coasted = corrector.update_candidates(
            [_track(198.0, 104.0, 0.55)],
            dt=0.04,
            frame_shape=frame_shape,
            person_bboxes=person_bboxes,
        )

        self.assertTrue(coasted.visible)
        self.assertEqual(corrector.debug_records[-1]["action"], "coast")
        self.assertEqual(corrector.debug_records[-1]["reason"], "occlusion_prediction")
        self.assertGreater(coasted.ball_xy[0], 160.0)

    def test_rejects_instead_of_coasting_when_candidate_fails_gate_without_occlusion(self) -> None:
        corrector = _corrector()
        frame_shape = (400, 600, 3)

        corrector.update(_track(100.0, 100.0, 0.9), dt=0.04, frame_shape=frame_shape)
        corrector.update(_track(130.0, 100.0, 0.9), dt=0.04, frame_shape=frame_shape)
        corrector.update(_track(160.0, 100.0, 0.9), dt=0.04, frame_shape=frame_shape)
        corrected = corrector.update(_track(500.0, 300.0, 0.56), dt=0.04, frame_shape=frame_shape)

        self.assertFalse(corrected.visible)
        self.assertEqual(corrector.debug_records[-1]["action"], "reject")
        self.assertEqual(corrector.debug_records[-1]["reason"], "no_candidate_inside_gate")

    def test_out_of_frame_state_suppresses_edge_noise(self) -> None:
        corrector = _corrector()
        frame_shape = (100, 200, 3)

        corrector.update(_track(100.0, 70.0, 0.9), dt=0.04, frame_shape=frame_shape)
        corrector.update(_track(100.0, 40.0, 0.9), dt=0.04, frame_shape=frame_shape)
        corrector.update(_track(100.0, 12.0, 0.9), dt=0.04, frame_shape=frame_shape)
        outside = corrector.update(_missing(), dt=0.04, frame_shape=frame_shape)
        edge_noise = corrector.update(_track(105.0, 4.0, 0.96), dt=0.04, frame_shape=frame_shape)

        self.assertFalse(outside.visible)
        self.assertFalse(edge_noise.visible)
        self.assertEqual(corrector.debug_records[-1]["action"], "out_of_frame")
        self.assertEqual(corrector.debug_records[-1]["reason"], "suppress_edge_noise")

    def test_out_of_frame_state_unlocks_after_long_exit(self) -> None:
        corrector = RealtimeKalmanTrackCorrector(
            RealtimeKalmanTrackCorrectorConfig(
                fps=25.0,
                max_missed_frames=2,
                out_of_frame_suppression_frames=4,
                relock_confidence=0.95,
            ),
            debug_enabled=True,
        )
        frame_shape = (100, 200, 3)

        corrector.update(_track(100.0, 70.0, 0.9), dt=0.04, frame_shape=frame_shape)
        corrector.update(_track(100.0, 40.0, 0.9), dt=0.04, frame_shape=frame_shape)
        corrector.update(_track(100.0, 12.0, 0.9), dt=0.04, frame_shape=frame_shape)
        for _ in range(5):
            corrector.update(_missing(), dt=0.04, frame_shape=frame_shape)

        relocked = corrector.update(_track(120.0, 50.0, 0.72), dt=0.04, frame_shape=frame_shape)

        self.assertTrue(relocked.visible)
        self.assertEqual(corrector.debug_records[-1]["action"], "bootstrap_accept")

    def test_low_score_candidate_after_miss_does_not_reacquire(self) -> None:
        corrector = _corrector()
        frame_shape = (400, 600, 3)

        corrector.update(_track(100.0, 100.0, 0.9), dt=0.04, frame_shape=frame_shape)
        corrector.update(_track(130.0, 100.0, 0.9), dt=0.04, frame_shape=frame_shape)
        corrector.update(_missing(), dt=0.04, frame_shape=frame_shape)
        corrected = corrector.update(_track(160.0, 100.0, 0.40), dt=0.04, frame_shape=frame_shape)

        self.assertFalse(corrected.visible)
        self.assertEqual(corrector.debug_records[-1]["action"], "reject")

    def test_fixed_lag_outputs_delayed_smoothed_point(self) -> None:
        corrector = _corrector(fixed_lag_frames=1)

        first = corrector.update(_track(10.0, 20.0, 0.9), dt=0.04, frame_shape=(200, 300, 3))
        second = corrector.update(_track(40.0, 20.0, 0.9), dt=0.04, frame_shape=(200, 300, 3))

        self.assertTrue(first.visible)
        self.assertTrue(second.visible)
        self.assertLess(second.ball_xy[0], 40.0)
        self.assertGreaterEqual(second.ball_xy[0], first.ball_xy[0])

    def test_realtime_default_has_no_fixed_lag_display_delay(self) -> None:
        corrector = RealtimeKalmanTrackCorrector(debug_enabled=True)

        corrector.update(_track(10.0, 20.0, 0.9), dt=0.04, frame_shape=(200, 300, 3))
        second = corrector.update(_track(40.0, 20.0, 0.9), dt=0.04, frame_shape=(200, 300, 3))

        self.assertEqual(corrector.config.fixed_lag_frames, 0)
        self.assertGreater(second.ball_xy[0], 30.0)

    def test_maneuver_snap_reduces_lag_after_direction_change(self) -> None:
        corrector = _corrector()
        frame_shape = (400, 600, 3)

        corrector.update(_track(100.0, 50.0, 0.9), dt=0.04, frame_shape=frame_shape)
        corrector.update(_track(100.0, 90.0, 0.9), dt=0.04, frame_shape=frame_shape)
        corrector.update(_track(100.0, 130.0, 0.9), dt=0.04, frame_shape=frame_shape)
        corrected = corrector.update(_track(115.0, 70.0, 0.7), dt=0.04, frame_shape=frame_shape)

        self.assertTrue(corrected.visible)
        self.assertLess(abs(corrected.ball_xy[1] - 70.0), 15.0)

    def test_can_be_plugged_into_ball_track_filter_interface(self) -> None:
        corrector = _corrector()
        track_filter = BallTrackFilter(algorithm=corrector)

        output = track_filter.update_candidates([_track(120.0, 130.0, 0.9)], dt=0.04, frame_shape=(400, 600, 3))

        self.assertTrue(output.visible)
        self.assertIs(track_filter.debug_records, corrector.debug_records)
        self.assertEqual(track_filter.last_debug_record()["action"], "bootstrap_accept")


if __name__ == "__main__":
    unittest.main()
