from __future__ import annotations

import unittest

from src.postprocess.track_filter import BallTrackFilter
from src.utils.structures import TrackResult


def _track(x: float, y: float, score: float = 0.72, visible: int = 1) -> TrackResult:
    return TrackResult(ball_xy=[x, y], visible=visible, score=score, heatmap_shape=[288, 512])


def _missing(score: float = 0.05) -> TrackResult:
    return TrackResult(ball_xy=[-1.0, -1.0], visible=0, score=score, heatmap_shape=[288, 512])


def _court() -> dict[str, object]:
    return {
        "valid": True,
        "image_to_court_h": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    }


def _air_court() -> dict[str, object]:
    return {
        "valid": True,
        "corners": [[200.0, 300.0], [800.0, 300.0], [900.0, 900.0], [100.0, 900.0]],
        "image_to_court_h": [[1.0, 0.0, 0.0], [0.0, 1.0, -500.0], [0.0, 0.0, 1.0]],
    }


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

    def test_update_candidates_prefers_lower_score_point_on_motion_path(self) -> None:
        tracker = BallTrackFilter(fps=25.0, debug_enabled=True)

        self.assertTrue(tracker.update(_track(100.0, 100.0, 0.92)).visible)
        self.assertTrue(tracker.update(_track(145.0, 100.0, 0.88)).visible)
        self.assertTrue(tracker.update(_track(190.0, 100.0, 0.86)).visible)

        selected = tracker.update_candidates(
            [
                _track(780.0, 420.0, 0.95),
                _track(235.0, 100.0, 0.58),
            ]
        )

        self.assertTrue(selected.visible)
        self.assertAlmostEqual(selected.ball_xy[0], 235.0)
        self.assertAlmostEqual(selected.ball_xy[1], 100.0)
        self.assertEqual(len(tracker.debug_records), 4)
        self.assertEqual(tracker.debug_records[-1]["candidate_count"], 2)
        self.assertEqual(tracker.debug_records[-1]["selected_candidate_index"], 1)
        self.assertEqual(tracker.debug_records[-1]["action"], "accept")

    def test_repeated_static_candidate_is_suppressed_as_hotspot(self) -> None:
        tracker = BallTrackFilter(fps=60.0, debug_enabled=True)

        outputs = [
            tracker.update_candidates([_track(900.0, 220.0, 0.88)], frame_shape=(720, 1280, 3))
            for _ in range(6)
        ]

        self.assertFalse(outputs[-1].visible)
        self.assertGreaterEqual(tracker.debug_records[-1]["static_hotspot_count"], 1)
        self.assertGreaterEqual(tracker.debug_records[-1]["static_filtered_count"], 1)

    def test_static_hotspot_filter_keeps_moving_candidate_available(self) -> None:
        tracker = BallTrackFilter(fps=60.0, debug_enabled=True)

        for _ in range(5):
            tracker.update_candidates([_track(900.0, 220.0, 0.88)], frame_shape=(720, 1280, 3))

        selected = tracker.update_candidates(
            [
                _track(900.0, 220.0, 0.92),
                _track(180.0, 300.0, 0.62),
            ],
            frame_shape=(720, 1280, 3),
        )

        self.assertFalse(selected.visible)
        self.assertAlmostEqual(tracker.debug_records[-1]["input_x"], 180.0)
        self.assertEqual(tracker.debug_records[-1]["static_filtered_count"], 1)

    def test_slow_edge_drift_candidate_is_suppressed(self) -> None:
        tracker = BallTrackFilter(fps=60.0, debug_enabled=True)

        outputs = [
            tracker.update_candidates([_track(1240.0, 200.0 + 3.0 * index, 0.86)], frame_shape=(720, 1280, 3))
            for index in range(6)
        ]

        self.assertFalse(outputs[-1].visible)
        self.assertGreaterEqual(tracker.debug_records[-1]["static_filtered_count"], 1)

    def test_court_filter_prefers_main_court_candidate_over_other_court_peak(self) -> None:
        tracker = BallTrackFilter(fps=25.0, debug_enabled=True)

        selected = tracker.update_candidates(
            [
                _track(950.0, 300.0, 0.95),
                _track(300.0, 300.0, 0.62),
            ],
            court_prediction=_court(),
        )

        self.assertFalse(selected.visible)
        self.assertEqual(tracker.debug_records[-1]["candidate_count"], 1)
        self.assertAlmostEqual(tracker.debug_records[-1]["input_x"], 300.0)

    def test_court_filter_does_not_relock_to_other_court_candidate(self) -> None:
        tracker = BallTrackFilter(fps=25.0, debug_enabled=True)
        court = _court()

        tracker.update(_track(100.0, 100.0, 0.92), court_prediction=court)
        tracker.update(_track(130.0, 110.0, 0.9), court_prediction=court)
        tracker.update(_track(160.0, 120.0, 0.9), court_prediction=court)
        outside = tracker.update_candidates(
            [_track(950.0, 300.0, 0.95)],
            court_prediction=court,
        )

        self.assertNotEqual(outside.ball_xy, [950.0, 300.0])
        self.assertEqual(tracker.debug_records[-1]["input_visible"], 0)

    def test_court_filter_keeps_ball_above_projected_court_lines(self) -> None:
        tracker = BallTrackFilter(fps=25.0, debug_enabled=True)

        selected = tracker.update_candidates(
            [_track(500.0, 150.0, 0.95)],
            frame_shape=(1000, 1000, 3),
            court_prediction=_air_court(),
        )

        self.assertTrue(selected.visible)
        self.assertAlmostEqual(selected.ball_xy[0], 500.0)
        self.assertAlmostEqual(selected.ball_xy[1], 150.0)

    def test_court_filter_rejects_lateral_ball_above_other_court(self) -> None:
        tracker = BallTrackFilter(fps=25.0, debug_enabled=True)

        selected = tracker.update_candidates(
            [_track(40.0, 150.0, 0.95)],
            frame_shape=(1000, 1000, 3),
            court_prediction=_air_court(),
        )

        self.assertFalse(selected.visible)
        self.assertEqual(tracker.debug_records[-1]["input_visible"], 0)

    def test_short_missing_gap_can_coast_when_motion_is_stable(self) -> None:
        tracker = BallTrackFilter(fps=25.0)

        tracker.update(_track(100.0, 100.0, 0.92))
        tracker.update(_track(140.0, 100.0, 0.9))
        tracker.update(_track(180.0, 100.0, 0.9))

        coasted = tracker.update(_missing())

        self.assertTrue(coasted.visible)
        self.assertGreater(coasted.ball_xy[0], 180.0)
        self.assertLess(coasted.score, 0.05)

    def test_person_occlusion_candidate_is_replaced_by_prediction(self) -> None:
        tracker = BallTrackFilter(fps=25.0, debug_enabled=True)
        frame_shape = (300, 500, 3)
        person_bboxes = [(205.0, 80.0, 270.0, 170.0)]

        tracker.update(_track(100.0, 100.0, 0.92), frame_shape=frame_shape)
        tracker.update(_track(140.0, 100.0, 0.9), frame_shape=frame_shape)
        tracker.update(_track(180.0, 100.0, 0.9), frame_shape=frame_shape)

        corrected = tracker.update_candidates(
            [_track(242.0, 136.0, 0.97)],
            frame_shape=frame_shape,
            person_bboxes=person_bboxes,
        )

        self.assertTrue(corrected.visible)
        self.assertLess(corrected.ball_xy[0], 230.0)
        self.assertAlmostEqual(corrected.ball_xy[1], 100.0, delta=4.0)
        self.assertEqual(tracker.debug_records[-1]["action"], "coast")
        self.assertEqual(tracker.debug_records[-1]["reason"], "person_occlusion_prediction")

    def test_person_occlusion_extends_missing_gap_coasting(self) -> None:
        tracker = BallTrackFilter(fps=25.0)
        frame_shape = (300, 500, 3)
        person_bboxes = [(190.0, 60.0, 420.0, 170.0)]

        tracker.update(_track(100.0, 100.0, 0.92), frame_shape=frame_shape)
        tracker.update(_track(140.0, 100.0, 0.9), frame_shape=frame_shape)
        tracker.update(_track(180.0, 100.0, 0.9), frame_shape=frame_shape)

        outputs = [
            tracker.update(_missing(), frame_shape=frame_shape, person_bboxes=person_bboxes)
            for _ in range(5)
        ]

        self.assertTrue(all(output.visible for output in outputs))
        self.assertGreater(outputs[-1].ball_xy[0], 280.0)
        self.assertAlmostEqual(outputs[-1].ball_xy[1], 100.0, delta=4.0)

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

    def test_high_score_point_close_to_prediction_survives_inertia_break(self) -> None:
        tracker = BallTrackFilter(fps=60.0)

        tracker.update(_track(100.0, 420.0, 0.92))
        tracker.update(_track(110.0, 360.0, 0.92))
        tracker.update(_track(120.0, 300.0, 0.92))

        recovered = tracker.update(_track(130.0, 210.0, 0.66))

        self.assertTrue(recovered.visible)
        self.assertAlmostEqual(recovered.ball_xy[0], 130.0)
        self.assertAlmostEqual(recovered.ball_xy[1], 210.0)

    def test_low_confidence_point_on_motion_path_can_keep_lock(self) -> None:
        tracker = BallTrackFilter(fps=25.0)

        tracker.update(_track(100.0, 100.0, 0.92))
        tracker.update(_track(130.0, 110.0, 0.9))
        tracker.update(_track(160.0, 120.0, 0.9))

        weak = tracker.update(_track(190.0, 130.0, 0.29))

        self.assertTrue(weak.visible)
        self.assertAlmostEqual(weak.ball_xy[0], 190.0)
        self.assertAlmostEqual(weak.ball_xy[1], 130.0)

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

    def test_top_exit_does_not_coast_downward_after_missing_detection(self) -> None:
        tracker = BallTrackFilter(fps=25.0)
        frame_shape = (180, 320, 3)
        points = [(80.0, 90.0), (100.0, 45.0), (120.0, 18.0), (140.0, 14.0)]

        for x, y in points:
            self.assertTrue(tracker.update(_track(x, y, 0.92), frame_shape=frame_shape).visible)

        missing = tracker.update(_missing(), frame_shape=frame_shape)
        hallucination = tracker.update(_track(160.0, 29.0, 0.96), frame_shape=frame_shape)

        self.assertFalse(missing.visible)
        self.assertEqual(missing.ball_xy, [-1.0, -1.0])
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

    def test_relocks_early_when_confirmed_candidate_reverses_direction(self) -> None:
        tracker = BallTrackFilter(fps=60.0)

        tracker.update(_track(100.0, 100.0, 0.92))
        tracker.update(_track(105.0, 150.0, 0.9))
        tracker.update(_track(110.0, 200.0, 0.9))

        first = tracker.update(_track(112.0, 60.0, 0.62))
        second = tracker.update(_track(114.0, 35.0, 0.64))

        self.assertFalse(first.visible)
        self.assertTrue(second.visible)
        self.assertAlmostEqual(second.ball_xy[0], 114.0)
        self.assertAlmostEqual(second.ball_xy[1], 35.0)


if __name__ == "__main__":
    unittest.main()
