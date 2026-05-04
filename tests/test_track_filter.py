from __future__ import annotations

import unittest

from src.postprocess.track_filter import BallTrackFilter, LegacyBallTrackFilterAlgorithm, _TrajectoryPoint
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


class _RecordingTrackFilterAlgorithm:
    def __init__(self) -> None:
        self.debug_records: list[dict[str, object]] = []
        self.calls: list[dict[str, object]] = []
        self.reset_called = False

    def reset(self) -> None:
        self.reset_called = True
        self.debug_records.clear()

    def update(self, track: TrackResult, **kwargs: object) -> TrackResult:
        self.calls.append({"method": "update", "track": track, "kwargs": kwargs})
        self.debug_records.append({"action": "custom_update"})
        return _track(33.0, 44.0, 0.66)

    def update_candidates(self, tracks: object, **kwargs: object) -> TrackResult:
        self.calls.append({"method": "update_candidates", "tracks": tracks, "kwargs": kwargs})
        self.debug_records.append({"action": "custom_candidates"})
        return _track(42.0, 43.0, 0.77)

    def last_debug_record(self) -> dict[str, object] | None:
        if not self.debug_records:
            return None
        return dict(self.debug_records[-1])


class BallTrackFilterTest(unittest.TestCase):
    def test_delegates_to_custom_algorithm_interface(self) -> None:
        algorithm = _RecordingTrackFilterAlgorithm()
        tracker = BallTrackFilter(algorithm=algorithm)

        selected = tracker.update_candidates(
            [_track(10.0, 20.0, 0.4)],
            dt=0.04,
            frame_shape=(720, 1280, 3),
            court_prediction={"valid": False},
            person_bboxes=[(1.0, 2.0, 3.0, 4.0)],
        )
        updated = tracker.update(_track(11.0, 21.0, 0.5), dt=0.05)

        self.assertEqual(selected.ball_xy, [42.0, 43.0])
        self.assertEqual(updated.ball_xy, [33.0, 44.0])
        self.assertEqual([call["method"] for call in algorithm.calls], ["update_candidates", "update"])
        self.assertIs(tracker.debug_records, algorithm.debug_records)
        self.assertEqual(tracker.last_debug_record(), {"action": "custom_update"})

        tracker.reset()
        self.assertTrue(algorithm.reset_called)
        self.assertEqual(tracker.debug_records, [])

    def test_legacy_algorithm_name_keeps_original_behavior(self) -> None:
        tracker = LegacyBallTrackFilterAlgorithm(fps=25.0, debug_enabled=True)

        output = tracker.update(_track(210.0, 440.0, 0.73), frame_shape=(720, 1280, 3))

        self.assertTrue(output.visible)
        self.assertEqual(tracker.debug_records[-1]["action"], "bootstrap_accept")

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

    def test_high_score_bootstrap_locks_immediately(self) -> None:
        tracker = BallTrackFilter(fps=25.0, debug_enabled=True)

        output = tracker.update(_track(210.0, 440.0, 0.73), frame_shape=(720, 1280, 3))

        self.assertTrue(output.visible)
        self.assertEqual(tracker.debug_records[-1]["action"], "bootstrap_accept")
        self.assertEqual(tracker.debug_records[-1]["reason"], "strong_confidence")

    def test_high_score_bootstrap_near_top_edge_still_requires_confirmation(self) -> None:
        tracker = BallTrackFilter(fps=25.0, debug_enabled=True)

        output = tracker.update(_track(725.0, 18.0, 0.73), frame_shape=(1080, 1920, 3))

        self.assertFalse(output.visible)
        self.assertEqual(tracker.debug_records[-1]["action"], "bootstrap_wait")

    def test_low_score_bootstrap_candidate_confirmation_keeps_waiting(self) -> None:
        tracker = BallTrackFilter(fps=25.0, debug_enabled=True)

        outputs = [
            tracker.update(_track(210.0, 440.0, 0.45)),
            tracker.update(_track(212.0, 439.0, 0.47)),
            tracker.update(_track(214.0, 438.0, 0.50)),
            tracker.update(_track(216.0, 437.0, 0.53)),
        ]
        recovered = tracker.update(_track(218.0, 436.0, 0.56))

        self.assertTrue(all(not output.visible for output in outputs))
        self.assertEqual(tracker.debug_records[-2]["action"], "bootstrap_wait")
        self.assertTrue(recovered.visible)
        self.assertEqual(tracker.debug_records[-1]["action"], "bootstrap_accept")
        self.assertEqual(tracker.debug_records[-1]["reason"], "candidate_confirmed")

    def test_static_bootstrap_candidate_cluster_keeps_waiting(self) -> None:
        tracker = BallTrackFilter(fps=25.0, debug_enabled=True)

        outputs = [
            tracker.update(_track(1071.0, 219.0, 0.676)),
            tracker.update(_track(1072.0, 210.0, 0.676)),
            tracker.update(_track(1071.0, 202.0, 0.676)),
        ]

        self.assertTrue(all(not output.visible for output in outputs))
        self.assertEqual(tracker.debug_records[-1]["action"], "bootstrap_wait")
        self.assertEqual(tracker.debug_records[-1]["reason"], "static_bootstrap_candidate")

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

    def test_stable_motion_can_coast_across_ten_frame_gap(self) -> None:
        tracker = BallTrackFilter(fps=30.0, debug_enabled=True)

        tracker.update(_track(100.0, 100.0, 0.92))
        tracker.update(_track(220.0, 100.0, 0.9))
        tracker.update(_track(340.0, 100.0, 0.9))

        outputs = [tracker.update(_missing()) for _ in range(11)]

        self.assertTrue(all(output.visible for output in outputs[:10]))
        self.assertFalse(outputs[10].visible)
        self.assertEqual(tracker.debug_records[3]["action"], "coast")
        self.assertEqual(tracker.debug_records[12]["action"], "coast")
        self.assertEqual(tracker.debug_records[13]["action"], "reject")
        self.assertTrue(tracker.debug_records[13]["locked_after"])

    def test_person_occlusion_candidate_is_replaced_by_prediction(self) -> None:
        tracker = BallTrackFilter(fps=25.0, debug_enabled=True)
        frame_shape = (300, 500, 3)
        person_bboxes = [(205.0, 80.0, 270.0, 170.0)]

        tracker.update(_track(100.0, 100.0, 0.92), frame_shape=frame_shape)
        tracker.update(_track(140.0, 100.0, 0.9), frame_shape=frame_shape)
        tracker.update(_track(180.0, 100.0, 0.9), frame_shape=frame_shape)

        corrected = tracker.update_candidates(
            [_track(242.0, 136.0, 0.54)],
            frame_shape=frame_shape,
            person_bboxes=person_bboxes,
        )

        self.assertTrue(corrected.visible)
        self.assertLess(corrected.ball_xy[0], 230.0)
        self.assertAlmostEqual(corrected.ball_xy[1], 100.0, delta=4.0)
        self.assertEqual(tracker.debug_records[-1]["action"], "coast")
        self.assertEqual(tracker.debug_records[-1]["reason"], "person_occlusion_prediction")

    def test_high_score_person_occlusion_candidate_stays_invisible(self) -> None:
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

        self.assertFalse(corrected.visible)
        self.assertEqual(corrected.ball_xy, [-1.0, -1.0])
        self.assertEqual(tracker.debug_records[-1]["action"], "reject")
        self.assertEqual(tracker.debug_records[-1]["reason"], "person_occlusion_candidate_high_score")

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

    def test_parabola_fill_does_not_override_opposite_y_direction_candidate(self) -> None:
        tracker = BallTrackFilter(fps=60.0, debug_enabled=True)

        tracker.update(_track(1194.0, 370.0, 0.92))
        tracker.update(_track(1193.4, 402.4, 0.9))
        tracker.update(_track(1192.7, 430.9, 0.9))
        tracker.update(_track(1193.2, 463.9, 0.9))

        corrected = tracker.update(_track(1177.8, 411.3, 0.62))

        self.assertFalse(corrected.visible)
        self.assertEqual(corrected.ball_xy, [-1.0, -1.0])
        self.assertEqual(tracker.debug_records[-1]["action"], "reject")
        self.assertEqual(tracker.debug_records[-1]["reason"], "candidate_failed_motion_gate")

    def test_far_outlier_does_not_draw_parabola_fill_point(self) -> None:
        tracker = BallTrackFilter(fps=25.0)

        for i in range(5):
            tracker.update(_track(100.0 + 20.0 * i, 200.0 - 18.0 * i + 2.0 * i * i, 0.92))

        corrected = tracker.update(_track(260.0, 520.0, 0.95))

        self.assertFalse(corrected.visible)
        self.assertEqual(corrected.ball_xy, [-1.0, -1.0])

    def test_high_score_point_close_to_prediction_survives_inertia_break(self) -> None:
        tracker = BallTrackFilter(fps=60.0)

        tracker.update(_track(100.0, 420.0, 0.92))
        tracker.update(_track(110.0, 360.0, 0.92))
        tracker.update(_track(120.0, 300.0, 0.92))

        recovered = tracker.update(_track(130.0, 210.0, 0.66))

        self.assertTrue(recovered.visible)
        self.assertAlmostEqual(recovered.ball_xy[0], 130.0)
        self.assertAlmostEqual(recovered.ball_xy[1], 210.0)

    def test_high_score_candidate_inside_base_gate_bypasses_inertia(self) -> None:
        tracker = BallTrackFilter(fps=60.0, debug_enabled=True)
        tracker._locked = True
        tracker._last_point = (1066.5353695543097, 206.3616917138888)
        tracker._render_point = tracker._last_point
        tracker._velocity = (-489.9986454449422, -820.8701774028916)
        tracker._frame_index = 690
        history = [
            (679, 1075.8901448126026, 172.95252401747692),
            (680, 1076.3744563644889, 177.04372903175627),
            (681, 1076.4854369968448, 179.5780570269667),
            (682, 1078.0051818629001, 183.8507139797569),
            (683, 1078.2924906762812, 191.92828437532467),
            (684, 1077.997617126353, 195.51949284985687),
            (685, 1079.3519821563373, 205.51319421038346),
            (686, 1080.1486459609794, 211.09571730744992),
            (687, 1080.8598092794048, 222.78462830653143),
            (688, 1081.2334592506968, 229.31283374924462),
            (689, 1077.8636445286556, 228.01518214693175),
            (690, 1066.5353695543097, 206.3616917138888),
        ]
        for frame_index, x, y in history:
            tracker._history.append(_TrajectoryPoint(frame_index, (x, y)))
        tracker._real_detections_since_relock = len(history)

        recovered = tracker.update(
            _track(1050.5739815295533, 150.9508812632277, 0.6683239936828613),
            frame_shape=(1080, 1920, 3),
        )

        self.assertTrue(recovered.visible)
        self.assertAlmostEqual(recovered.ball_xy[0], 1050.5739815295533)
        self.assertAlmostEqual(recovered.ball_xy[1], 150.9508812632277)
        self.assertEqual(tracker.debug_records[-1]["action"], "accept")

    def test_close_prediction_motion_break_drops_conflicting_parabola(self) -> None:
        tracker = BallTrackFilter(fps=25.0, debug_enabled=True)

        for i in range(4):
            tracker.update(_track(100.0, 300.0 - 30.0 * i, 0.92))

        recovered = tracker.update(_track(100.0, 252.0, 0.66))
        coasted = tracker.update(_missing())

        self.assertTrue(recovered.visible)
        self.assertEqual(tracker.debug_records[-2]["reason"], "passes_motion_gate")
        self.assertTrue(coasted.visible)
        self.assertGreater(coasted.ball_xy[1], 252.0)
        self.assertEqual(tracker.debug_records[-1]["reason"], "velocity_prediction")

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

        first = tracker.update(_track(700.0, 350.0, 0.62))
        second = tracker.update(_track(706.0, 354.0, 0.64))
        third = tracker.update(_track(712.0, 358.0, 0.66))

        self.assertFalse(first.visible)
        self.assertFalse(second.visible)
        self.assertTrue(third.visible)
        self.assertAlmostEqual(third.ball_xy[0], 712.0)

    def test_high_score_candidate_relocks_after_second_consistent_failure(self) -> None:
        tracker = BallTrackFilter(fps=25.0, debug_enabled=True)

        tracker.update(_track(100.0, 100.0, 0.92))
        tracker.update(_track(140.0, 100.0, 0.9))
        tracker.update(_track(180.0, 100.0, 0.9))

        first = tracker.update(_track(700.0, 350.0, 0.73))
        second = tracker.update(_track(706.0, 354.0, 0.74))

        self.assertFalse(first.visible)
        self.assertTrue(second.visible)
        self.assertAlmostEqual(second.ball_xy[0], 706.0)
        self.assertEqual(tracker.debug_records[-1]["reason"], "high_score_fast_relock")

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

    def test_impact_relock_ignores_far_hallucinated_reversal(self) -> None:
        tracker = BallTrackFilter(fps=60.0)
        frame_shape = (1080, 1920, 3)

        tracker.update(_track(600.0, 400.0, 0.92), frame_shape=frame_shape)
        tracker.update(_track(630.0, 420.0, 0.9), frame_shape=frame_shape)
        tracker.update(_track(660.0, 440.0, 0.9), frame_shape=frame_shape)

        first = tracker.update(_track(760.0, 15.0, 0.72), frame_shape=frame_shape)
        second = tracker.update(_track(763.0, 16.0, 0.74), frame_shape=frame_shape)
        third = tracker.update(_track(766.0, 12.0, 0.76), frame_shape=frame_shape)

        for output in (first, second, third):
            if output.visible:
                self.assertGreater(output.ball_xy[1], 100.0)


if __name__ == "__main__":
    unittest.main()
