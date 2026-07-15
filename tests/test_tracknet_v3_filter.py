from __future__ import annotations

import unittest

from src.postprocess.track_filter import BallTrackFilter, BallTrackFilterConfig
from src.postprocess.trajectory_events import RealtimeTrajectoryEventDetector, TrajectoryEventDetectorConfig
from src.postprocess.tracknet_v3_filter import (
    TrackNetV3TrajectoryFilter,
    TrackNetV3TrajectoryFilterConfig,
    create_tracknet_v3_ball_track_filter,
    generate_tracknet_v3_inpaint_mask,
    linear_interpolate_masked_values,
)
from src.utils.structures import FrameResult, TrackResult


def _track(x: float, y: float, score: float = 0.72, visible: int = 1) -> TrackResult:
    return TrackResult(ball_xy=[x, y], visible=visible, score=score, heatmap_shape=[288, 512])


def _missing(score: float = 0.05) -> TrackResult:
    return TrackResult(ball_xy=[-1.0, -1.0], visible=0, score=score, heatmap_shape=[288, 512])


def _court_prediction() -> dict[str, object]:
    return {
        "valid": True,
        "corners": [
            [300.0, 200.0],
            [980.0, 200.0],
            [980.0, 650.0],
            [300.0, 650.0],
        ],
    }


class TrackNetV3TrajectoryFilterTest(unittest.TestCase):
    def test_generates_inpaint_mask_for_middle_disappearance(self) -> None:
        mask = generate_tracknet_v3_inpaint_mask(
            [90.0, 100.0, -1.0, -1.0, 130.0],
            [1, 1, 0, 0, 1],
            height_threshold=30.0,
        )

        self.assertEqual(mask, [0, 0, 1, 1, 0])

    def test_does_not_inpaint_top_exit_disappearance(self) -> None:
        mask = generate_tracknet_v3_inpaint_mask(
            [20.0, -1.0, -1.0, 25.0],
            [1, 0, 0, 1],
            height_threshold=30.0,
        )

        self.assertEqual(mask, [0, 0, 0, 0])

    def test_linear_interpolation_matches_tracknet_v3_rule(self) -> None:
        interpolated = linear_interpolate_masked_values(
            [10.0, -1.0, -1.0, 40.0],
            [0, 1, 1, 0],
        )

        self.assertEqual(interpolated, [10.0, 20.0, 30.0, 40.0])

    def test_keeps_post_hit_direction_change_without_kalman_coast(self) -> None:
        tracker = TrackNetV3TrajectoryFilter(debug_enabled=True)
        frame_shape = (1080, 1920, 3)

        before_hit = tracker.update(_track(649.8, 413.5, 0.56), dt=1.0 / 60.0, frame_shape=frame_shape)
        after_hit = tracker.update(_track(680.7, 313.2, 0.74), dt=1.0 / 60.0, frame_shape=frame_shape)
        rising = tracker.update(_track(753.6, 49.3, 0.58), dt=1.0 / 60.0, frame_shape=frame_shape)

        self.assertEqual(before_hit.ball_xy, [649.8, 413.5])
        self.assertEqual(after_hit.ball_xy, [680.7, 313.2])
        self.assertEqual(rising.ball_xy, [753.6, 49.3])
        self.assertEqual([record["action"] for record in tracker.debug_records], ["accept", "accept", "accept"])
        self.assertTrue(all(record["coast_after"] == 0 for record in tracker.debug_records))

    def test_prefers_motion_consistent_candidate_over_far_high_score_spike(self) -> None:
        tracker = TrackNetV3TrajectoryFilter(debug_enabled=True)
        frame_shape = (720, 1280, 3)

        tracker.update(_track(100.0, 200.0, 0.78), dt=1.0 / 60.0, frame_shape=frame_shape)
        tracker.update(_track(112.0, 206.0, 0.76), dt=1.0 / 60.0, frame_shape=frame_shape)
        tracker.update(_track(124.0, 212.0, 0.74), dt=1.0 / 60.0, frame_shape=frame_shape)

        stable = tracker.update_candidates(
            [
                _track(780.0, 520.0, 0.96),
                _track(136.0, 218.0, 0.58),
            ],
            dt=1.0 / 60.0,
            frame_shape=frame_shape,
        )

        self.assertTrue(stable.visible)
        self.assertEqual(stable.ball_xy, [136.0, 218.0])
        self.assertEqual(tracker.debug_records[-1]["selected_candidate_index"], 1)

    def test_directional_drift_guard_rejects_single_far_wrong_way_candidate(self) -> None:
        tracker = TrackNetV3TrajectoryFilter(debug_enabled=True)
        frame_shape = (720, 1280, 3)
        tracker.update(_track(100.0, 200.0, 0.80), dt=1.0 / 30.0, frame_shape=frame_shape)
        tracker.update(_track(112.0, 208.0, 0.78), dt=1.0 / 30.0, frame_shape=frame_shape)

        guarded = tracker.update_candidates(
            [_track(40.0, 520.0, 0.96)],
            dt=1.0 / 30.0,
            frame_shape=frame_shape,
        )

        self.assertTrue(guarded.visible)
        self.assertEqual(guarded.ball_xy, [124.0, 216.0])
        self.assertEqual(tracker.debug_records[-1]["selected_candidate_index"], -1)
        self.assertEqual(tracker.debug_records[-1]["reason"], "tracknetv2_short_gap_bridge")

    def test_directional_drift_guard_keeps_likely_upward_impact(self) -> None:
        tracker = TrackNetV3TrajectoryFilter(debug_enabled=True)
        frame_shape = (720, 1280, 3)
        tracker.update(_track(670.0, 360.0, 0.80), dt=1.0 / 30.0, frame_shape=frame_shape)
        tracker.update(_track(684.0, 492.0, 0.78), dt=1.0 / 30.0, frame_shape=frame_shape)

        impact = tracker.update_candidates(
            [_track(687.0, 385.0, 0.96)],
            dt=1.0 / 30.0,
            frame_shape=frame_shape,
        )

        self.assertTrue(impact.visible)
        self.assertEqual(impact.ball_xy, [687.0, 385.0])
        self.assertEqual(tracker.debug_records[-1]["selected_candidate_index"], 0)
        self.assertEqual(tracker.debug_records[-1]["reason"], "tracknet_v3_candidate")

    def test_court_filter_rejects_other_court_candidate(self) -> None:
        tracker = TrackNetV3TrajectoryFilter(debug_enabled=True)
        frame_shape = (720, 1280, 3)
        court = _court_prediction()
        tracker.update(_track(500.0, 400.0, 0.80), dt=1.0 / 30.0, frame_shape=frame_shape, court_prediction=court)
        tracker.update(_track(510.0, 410.0, 0.78), dt=1.0 / 30.0, frame_shape=frame_shape, court_prediction=court)

        guarded = tracker.update_candidates(
            [_track(120.0, 430.0, 0.96)],
            dt=1.0 / 30.0,
            frame_shape=frame_shape,
            court_prediction=court,
        )

        self.assertTrue(guarded.visible)
        self.assertEqual(guarded.ball_xy, [520.0, 420.0])
        self.assertEqual(tracker.debug_records[-1]["candidate_count"], 0)
        self.assertEqual(tracker.debug_records[-1]["court_filter_active"], 1)
        self.assertEqual(tracker.debug_records[-1]["court_filtered_count"], 1)
        self.assertEqual(tracker.debug_records[-1]["reason"], "tracknetv2_short_gap_bridge")

    def test_court_filter_keeps_high_air_ball_above_current_court(self) -> None:
        tracker = TrackNetV3TrajectoryFilter(debug_enabled=True)
        frame_shape = (720, 1280, 3)

        accepted = tracker.update_candidates(
            [_track(640.0, 40.0, 0.82)],
            dt=1.0 / 30.0,
            frame_shape=frame_shape,
            court_prediction=_court_prediction(),
        )

        self.assertTrue(accepted.visible)
        self.assertEqual(accepted.ball_xy, [640.0, 40.0])
        self.assertEqual(tracker.debug_records[-1]["court_filtered_count"], 0)
        self.assertEqual(tracker.debug_records[-1]["reason"], "tracknet_v3_candidate")

    def test_can_linearly_repair_missing_span_when_lag_allows_future_endpoint(self) -> None:
        tracker = TrackNetV3TrajectoryFilter(
            TrackNetV3TrajectoryFilterConfig(fps=25.0, fixed_lag_frames=5),
            debug_enabled=True,
        )
        frame_shape = (400, 600, 3)

        tracker.update(_track(10.0, 100.0, 0.8), dt=0.04, frame_shape=frame_shape)
        tracker.update(_track(20.0, 110.0, 0.8), dt=0.04, frame_shape=frame_shape)
        tracker.update(_missing(), dt=0.04, frame_shape=frame_shape)
        tracker.update(_missing(), dt=0.04, frame_shape=frame_shape)
        tracker.update(_track(50.0, 140.0, 0.8), dt=0.04, frame_shape=frame_shape)
        tracker.update(_track(60.0, 150.0, 0.8), dt=0.04, frame_shape=frame_shape)
        tracker.update(_track(70.0, 160.0, 0.8), dt=0.04, frame_shape=frame_shape)
        repaired = tracker.update(_track(80.0, 170.0, 0.8), dt=0.04, frame_shape=frame_shape)

        self.assertTrue(repaired.visible)
        self.assertEqual(repaired.ball_xy, [30.0, 120.0])
        self.assertEqual(tracker.debug_records[-1]["action"], "inpaint")

    def test_default_short_gap_bridge_uses_recent_velocity_without_latency(self) -> None:
        tracker = TrackNetV3TrajectoryFilter(debug_enabled=True)
        frame_shape = (720, 1280, 3)
        tracker.update(_track(10.0, 100.0, 0.8), dt=1.0 / 30.0, frame_shape=frame_shape)
        tracker.update(_track(20.0, 110.0, 0.8), dt=1.0 / 30.0, frame_shape=frame_shape)

        bridged = tracker.update(_missing(), dt=1.0 / 30.0, frame_shape=frame_shape)

        self.assertTrue(bridged.visible)
        self.assertEqual(bridged.ball_xy, [30.0, 120.0])
        self.assertEqual(tracker.debug_records[-1]["action"], "inpaint")
        self.assertEqual(tracker.debug_records[-1]["reason"], "tracknetv2_short_gap_bridge")

    def test_short_gap_bridge_does_not_extend_top_exit(self) -> None:
        tracker = TrackNetV3TrajectoryFilter(debug_enabled=True)
        frame_shape = (720, 1280, 3)
        tracker.update(_track(200.0, 60.0, 0.8), dt=1.0 / 30.0, frame_shape=frame_shape)
        tracker.update(_track(205.0, 30.0, 0.8), dt=1.0 / 30.0, frame_shape=frame_shape)

        missing = tracker.update(_missing(), dt=1.0 / 30.0, frame_shape=frame_shape)

        self.assertFalse(missing.visible)
        self.assertEqual(tracker.debug_records[-1]["reason"], "missing_or_low_confidence")

    def test_can_be_plugged_into_ball_track_filter_interface(self) -> None:
        algorithm = TrackNetV3TrajectoryFilter(debug_enabled=True)
        track_filter = BallTrackFilter(algorithm=algorithm)

        output = track_filter.update_candidates([_track(120.0, 130.0, 0.9)], dt=0.04, frame_shape=(400, 600, 3))

        self.assertTrue(output.visible)
        self.assertEqual(output.ball_xy, [120.0, 130.0])
        self.assertIs(track_filter.debug_records, algorithm.debug_records)
        self.assertEqual(track_filter.last_debug_record()["reason"], "tracknet_v3_candidate")

    def test_factory_builds_tracknet_v3_runtime_filter(self) -> None:
        track_filter = create_tracknet_v3_ball_track_filter(fps=60.0, debug_enabled=True)

        output = track_filter.update_candidates([_track(120.0, 130.0, 0.9)], dt=1.0 / 60.0, frame_shape=(400, 600, 3))

        self.assertTrue(output.visible)
        self.assertEqual(output.ball_xy, [120.0, 130.0])
        self.assertEqual(track_filter.last_debug_record()["reason"], "strong_confidence")

    def test_runtime_filter_ranks_candidates_for_current_parabola_frame(self) -> None:
        track_filter = create_tracknet_v3_ball_track_filter(fps=20.0, debug_enabled=True)
        frame_shape = (720, 1280, 3)
        for x, y in [(100.0, 100.0), (110.0, 120.0), (120.0, 145.0), (130.0, 175.0)]:
            track_filter.update_candidates(
                [_track(x, y, 0.8)],
                dt=0.05,
                frame_shape=frame_shape,
            )

        output = track_filter.update_candidates(
            [
                _track(140.0, 175.0, 0.7),
                _track(140.0, 210.0, 0.7),
            ],
            dt=0.05,
            frame_shape=frame_shape,
        )

        self.assertTrue(output.visible)
        self.assertEqual(output.ball_xy, [140.0, 210.0])
        self.assertEqual(track_filter.last_debug_record()["selected_candidate_index"], 1)

    def test_runtime_filter_parabola_uses_real_time_spacing(self) -> None:
        track_filter = create_tracknet_v3_ball_track_filter(fps=30.0, debug_enabled=True)
        frame_shape = (720, 1280, 3)

        def point_at(time_seconds: float) -> tuple[float, float]:
            return (
                100.0 + 300.0 * time_seconds,
                120.0 + 500.0 * time_seconds + 900.0 * time_seconds * time_seconds,
            )

        time_seconds = 0.0
        for step_dt in [0.05, 0.10, 0.04, 0.08]:
            time_seconds += step_dt
            x, y = point_at(time_seconds)
            track_filter.update_candidates(
                [_track(x, y, 0.8)],
                dt=step_dt,
                frame_shape=frame_shape,
            )

        expected_x, expected_y = point_at(time_seconds + 0.09)
        output = track_filter.update_candidates(
            [
                _track(194.5, 377.13, 0.7),
                _track(expected_x, expected_y, 0.7),
            ],
            dt=0.09,
            frame_shape=frame_shape,
        )

        self.assertTrue(output.visible)
        self.assertAlmostEqual(output.ball_xy[0], expected_x, places=3)
        self.assertAlmostEqual(output.ball_xy[1], expected_y, places=3)
        self.assertEqual(track_filter.last_debug_record()["selected_candidate_index"], 1)

    def test_runtime_filter_prefers_consistent_soft_candidate_over_far_hard_candidate(self) -> None:
        track_filter = create_tracknet_v3_ball_track_filter(fps=30.0, debug_enabled=True)
        frame_shape = (720, 1280, 3)
        track_filter.update_candidates([_track(100.0, 100.0, 0.9)], dt=1.0 / 30.0, frame_shape=frame_shape)
        track_filter.update_candidates([_track(110.0, 100.0, 0.9)], dt=1.0 / 30.0, frame_shape=frame_shape)

        output = track_filter.update_candidates(
            [
                _track(120.0, 100.0, 0.32),
                _track(700.0, 500.0, 0.50),
            ],
            dt=1.0 / 30.0,
            frame_shape=frame_shape,
        )

        self.assertTrue(output.visible)
        self.assertEqual(output.ball_xy, [120.0, 100.0])
        self.assertEqual(track_filter.last_debug_record()["selected_candidate_index"], 0)
        self.assertEqual(track_filter.last_debug_record()["reason"], "soft_confidence_motion_gate")

    def test_runtime_filter_fast_relock_requires_current_high_score(self) -> None:
        track_filter = create_tracknet_v3_ball_track_filter(fps=30.0, debug_enabled=True)
        frame_shape = (720, 1280, 3)
        track_filter.update_candidates([_track(100.0, 100.0, 0.9)], dt=1.0 / 30.0, frame_shape=frame_shape)
        track_filter.update_candidates([_track(110.0, 100.0, 0.9)], dt=1.0 / 30.0, frame_shape=frame_shape)

        first = track_filter.update_candidates(
            [_track(800.0, 500.0, 0.90)],
            dt=1.0 / 30.0,
            frame_shape=frame_shape,
        )
        second = track_filter.update_candidates(
            [_track(810.0, 505.0, 0.36)],
            dt=1.0 / 30.0,
            frame_shape=frame_shape,
        )

        self.assertFalse(first.visible)
        self.assertFalse(second.visible)
        self.assertNotEqual(track_filter.last_debug_record()["reason"], "high_score_fast_relock")

    def test_runtime_filter_rejects_far_candidate_after_missing_until_confirmed(self) -> None:
        track_filter = create_tracknet_v3_ball_track_filter(fps=30.0, debug_enabled=True)
        frame_shape = (720, 1280, 3)
        track_filter.update_candidates([_track(100.0, 300.0, 0.8)], dt=1.0 / 30.0, frame_shape=frame_shape)
        track_filter.update_candidates([_track(110.0, 300.0, 0.8)], dt=1.0 / 30.0, frame_shape=frame_shape)
        track_filter.update_candidates([_missing()], dt=1.0 / 30.0, frame_shape=frame_shape)

        first = track_filter.update_candidates(
            [_track(900.0, 600.0, 0.7)],
            dt=1.0 / 30.0,
            frame_shape=frame_shape,
        )
        second = track_filter.update_candidates(
            [_track(905.0, 602.0, 0.7)],
            dt=1.0 / 30.0,
            frame_shape=frame_shape,
        )
        confirmed = track_filter.update_candidates(
            [_track(910.0, 604.0, 0.7)],
            dt=1.0 / 30.0,
            frame_shape=frame_shape,
        )

        self.assertFalse(first.visible)
        self.assertFalse(second.visible)
        self.assertTrue(confirmed.visible)
        self.assertEqual(confirmed.ball_xy, [910.0, 604.0])
        self.assertEqual(track_filter.last_debug_record()["reason"], "stable_new_candidate")

    def test_runtime_filter_rejects_single_large_upward_spike(self) -> None:
        track_filter = create_tracknet_v3_ball_track_filter(fps=30.0, debug_enabled=True)
        frame_shape = (720, 1280, 3)
        track_filter.update_candidates([_track(500.0, 500.0, 0.8)], dt=1.0 / 30.0, frame_shape=frame_shape)
        track_filter.update_candidates([_track(510.0, 500.0, 0.8)], dt=1.0 / 30.0, frame_shape=frame_shape)

        guarded = track_filter.update_candidates(
            [_track(300.0, 80.0, 0.7)],
            dt=1.0 / 30.0,
            frame_shape=frame_shape,
        )

        self.assertFalse(guarded.visible)
        self.assertEqual(track_filter.last_debug_record()["reason"], "candidate_failed_motion_gate")

    def test_runtime_filter_does_not_coast_when_new_relock_candidate_is_pending(self) -> None:
        track_filter = create_tracknet_v3_ball_track_filter(fps=60.0, debug_enabled=True)
        frame_shape = (1080, 1920, 3)
        sequence = [
            ([_track(1105.0, 380.0, 0.8)], 0.017),
            ([_track(1105.0, 400.0, 0.7)], 0.017),
            ([_track(1105.0, 420.0, 0.7)], 0.017),
            ([_track(1105.0, 443.0, 0.63)], 0.016),
            ([_missing(0.068)], 0.034),
            ([_track(1038.0, 442.0, 0.402)], 0.016),
        ]
        for candidates, dt in sequence:
            track_filter.update_candidates(candidates, dt=dt, frame_shape=frame_shape)

        pending = track_filter.update_candidates(
            [_track(926.0, 391.0, 0.236)],
            dt=0.05,
            frame_shape=frame_shape,
        )

        self.assertFalse(pending.visible)
        self.assertEqual(track_filter.last_debug_record()["reason"], "low_confidence_candidate_conflict")

    def test_runtime_filter_stops_occlusion_coast_when_candidate_reappears(self) -> None:
        track_filter = create_tracknet_v3_ball_track_filter(fps=60.0, debug_enabled=True)
        frame_shape = (1080, 1920, 3)
        full_frame_person = [(0.0, 0.0, 1920.0, 1080.0)]
        track_filter.update_candidates([_track(718.0, 300.0, 0.8)], dt=0.017, frame_shape=frame_shape)
        track_filter.update_candidates([_track(740.0, 330.0, 0.8)], dt=0.017, frame_shape=frame_shape)
        track_filter.update_candidates([_track(769.0, 368.0, 0.8)], dt=0.017, frame_shape=frame_shape)
        track_filter.update_candidates(
            [_missing()],
            dt=0.05,
            frame_shape=frame_shape,
            person_bboxes=full_frame_person,
        )

        reappeared = track_filter.update_candidates(
            [_track(711.0, 343.0, 0.36)],
            dt=0.017,
            frame_shape=frame_shape,
            person_bboxes=full_frame_person,
        )

        self.assertFalse(reappeared.visible)
        self.assertEqual(track_filter.last_debug_record()["reason"], "person_occlusion_candidate_high_score")

    def test_runtime_filter_bridges_reliable_low_speed_single_gap(self) -> None:
        track_filter = create_tracknet_v3_ball_track_filter(fps=30.0, debug_enabled=True)
        frame_shape = (720, 1280, 3)
        for point in [(500.0, 300.0), (501.0, 296.0), (502.0, 291.0)]:
            track_filter.update_candidates(
                [_track(*point, 0.85)],
                dt=1.0 / 30.0,
                frame_shape=frame_shape,
            )

        bridged = track_filter.update_candidates(
            [_missing(0.10)],
            dt=1.0 / 30.0,
            frame_shape=frame_shape,
        )

        self.assertTrue(bridged.visible)
        self.assertEqual(track_filter.last_debug_record()["action"], "coast")

    def test_runtime_filter_masks_low_speed_downward_branch_outlier(self) -> None:
        track_filter = BallTrackFilter(
            BallTrackFilterConfig(fps=30.0, static_hotspot_enabled=False),
            debug_enabled=True,
        )
        frame_shape = (720, 1280, 3)
        for point in [(500.0, 300.0), (501.0, 298.0), (502.0, 297.0), (503.0, 296.0)]:
            track_filter.update_candidates(
                [_track(*point, 0.85)],
                dt=1.0 / 30.0,
                frame_shape=frame_shape,
            )

        guarded = track_filter.update_candidates(
            [_track(510.0, 340.0, 0.92)],
            dt=1.0 / 30.0,
            frame_shape=frame_shape,
        )

        self.assertFalse(guarded.visible)
        self.assertEqual(track_filter.last_debug_record()["reason"], "literature_prediction_outlier")

        recovered = track_filter.update_candidates(
            [_track(511.0, 339.0, 0.92), _track(504.0, 295.0, 0.45)],
            dt=1.0 / 30.0,
            frame_shape=frame_shape,
        )

        self.assertTrue(recovered.visible)
        self.assertEqual(recovered.ball_xy, [504.0, 295.0])

    def test_runtime_filter_resets_motion_model_after_occlusion_reversal(self) -> None:
        track_filter = create_tracknet_v3_ball_track_filter(fps=30.0, debug_enabled=True)
        frame_shape = (720, 1280, 3)
        full_frame_person = [(0.0, 0.0, 1280.0, 720.0)]
        for point in [(500.0, 300.0), (502.0, 320.0), (504.0, 342.0)]:
            track_filter.update_candidates(
                [_track(*point, 0.90)],
                dt=1.0 / 30.0,
                frame_shape=frame_shape,
                person_bboxes=full_frame_person,
            )
        track_filter.update_candidates(
            [_missing(0.10)],
            dt=1.0 / 30.0,
            frame_shape=frame_shape,
            person_bboxes=full_frame_person,
        )

        reset = track_filter.update_candidates(
            [_track(470.0, 250.0, 0.93)],
            dt=1.0 / 30.0,
            frame_shape=frame_shape,
            person_bboxes=full_frame_person,
        )

        self.assertTrue(reset.visible)
        self.assertEqual(reset.ball_xy, [470.0, 250.0])
        self.assertEqual(track_filter.last_debug_record()["reason"], "literature_occlusion_model_reset")

    def test_runtime_filter_accepts_motion_consistent_ball_inside_person_bbox(self) -> None:
        track_filter = create_tracknet_v3_ball_track_filter(fps=30.0, debug_enabled=True)
        frame_shape = (720, 1280, 3)
        full_frame_person = [(0.0, 0.0, 1280.0, 720.0)]
        track_filter.update_candidates(
            [_track(100.0, 200.0, 0.90)],
            dt=1.0 / 30.0,
            frame_shape=frame_shape,
        )
        track_filter.update_candidates(
            [_track(120.0, 220.0, 0.90)],
            dt=1.0 / 30.0,
            frame_shape=frame_shape,
        )

        accepted = track_filter.update_candidates(
            [_track(140.0, 240.0, 0.90)],
            dt=1.0 / 30.0,
            frame_shape=frame_shape,
            person_bboxes=full_frame_person,
        )

        self.assertTrue(accepted.visible)
        self.assertEqual(accepted.ball_xy, [140.0, 240.0])
        self.assertEqual(track_filter.last_debug_record()["reason"], "person_occlusion_motion_gate")

    def test_runtime_filter_requires_current_impact_score_for_relock(self) -> None:
        track_filter = create_tracknet_v3_ball_track_filter(fps=60.0, debug_enabled=True)
        frame_shape = (1080, 1920, 3)
        sequence = [
            ([_track(630.0, 550.0, 0.8)], 0.034),
            ([_track(632.0, 600.0, 0.8)], 0.034),
            ([_track(633.0, 644.0, 0.8)], 0.034),
            ([_missing(0.06)], 0.017),
            ([_missing(0.06)], 0.017),
            ([_track(677.0, 582.0, 0.657)], 0.033),
            ([_missing(0.16)], 0.034),
            ([_missing(0.11)], 0.016),
        ]
        for candidates, dt in sequence:
            track_filter.update_candidates(candidates, dt=dt, frame_shape=frame_shape)

        low_score_impact = track_filter.update_candidates(
            [_track(733.0, 554.0, 0.307)],
            dt=0.034,
            frame_shape=frame_shape,
        )

        self.assertFalse(low_score_impact.visible)
        self.assertEqual(track_filter.last_debug_record()["reason"], "missing_or_low_confidence")

    def test_runtime_filter_prefers_candidate_continuing_pending_impact_relock(self) -> None:
        track_filter = BallTrackFilter(
            BallTrackFilterConfig(fps=30.0, impact_relock_confirm_frames=2),
            debug_enabled=True,
        )
        frame_shape = (720, 1280, 3)
        for x, y in [(100.0, 300.0), (100.0, 350.0), (100.0, 400.0)]:
            track_filter.update_candidates(
                [_track(x, y, 0.90)],
                dt=1.0 / 30.0,
                frame_shape=frame_shape,
            )

        pending = track_filter.update_candidates(
            [_track(80.0, 280.0, 0.62)],
            dt=1.0 / 30.0,
            frame_shape=frame_shape,
        )
        accepted = track_filter.update_candidates(
            [
                _track(60.0, 200.0, 0.90),
                _track(110.0, 450.0, 0.32),
            ],
            dt=1.0 / 30.0,
            frame_shape=frame_shape,
        )

        self.assertFalse(pending.visible)
        self.assertTrue(accepted.visible)
        self.assertEqual(accepted.ball_xy, [60.0, 200.0])
        self.assertEqual(track_filter.last_debug_record()["selected_candidate_index"], 0)
        self.assertEqual(track_filter.last_debug_record()["reason"], "high_score_fast_relock")

    def test_runtime_filter_default_impact_relock_accepts_first_strong_reversal(self) -> None:
        track_filter = create_tracknet_v3_ball_track_filter(fps=30.0, debug_enabled=True)
        frame_shape = (720, 1280, 3)
        for x, y in [(100.0, 300.0), (100.0, 350.0), (100.0, 400.0)]:
            track_filter.update_candidates(
                [_track(x, y, 0.90)],
                dt=1.0 / 30.0,
                frame_shape=frame_shape,
            )
        track_filter.update_candidates(
            [_missing()],
            dt=1.0 / 30.0,
            frame_shape=frame_shape,
        )

        accepted = track_filter.update_candidates(
            [_track(80.0, 280.0, 0.62)],
            dt=1.0 / 30.0,
            frame_shape=frame_shape,
        )

        self.assertTrue(accepted.visible)
        self.assertEqual(accepted.ball_xy, [80.0, 280.0])
        self.assertEqual(track_filter.last_debug_record()["reason"], "impact_direction_change")

    def test_runtime_filter_motion_consistent_person_candidate_beats_far_outside_peak(self) -> None:
        track_filter = create_tracknet_v3_ball_track_filter(fps=30.0, debug_enabled=True)
        frame_shape = (720, 1280, 3)
        person = [(100.0, 180.0, 180.0, 300.0)]
        track_filter.update_candidates(
            [_track(100.0, 200.0, 0.90)],
            dt=1.0 / 30.0,
            frame_shape=frame_shape,
        )
        track_filter.update_candidates(
            [_track(120.0, 220.0, 0.90)],
            dt=1.0 / 30.0,
            frame_shape=frame_shape,
        )

        accepted = track_filter.update_candidates(
            [
                _track(700.0, 500.0, 0.90),
                _track(140.0, 240.0, 0.55),
            ],
            dt=1.0 / 30.0,
            frame_shape=frame_shape,
            person_bboxes=person,
        )

        self.assertTrue(accepted.visible)
        self.assertEqual(accepted.ball_xy, [140.0, 240.0])
        self.assertEqual(track_filter.last_debug_record()["selected_candidate_index"], 1)
        self.assertEqual(track_filter.last_debug_record()["reason"], "person_occlusion_motion_gate")

    def test_runtime_filter_slow_moving_track_claims_static_hotspot(self) -> None:
        track_filter = create_tracknet_v3_ball_track_filter(fps=30.0, debug_enabled=True)
        frame_shape = (720, 1280, 3)
        outputs = []
        for frame_index in range(8):
            outputs.append(
                track_filter.update_candidates(
                    [_track(500.0 + frame_index, 400.0 + frame_index * 2.0, 0.90)],
                    dt=1.0 / 30.0,
                    frame_shape=frame_shape,
                )
            )

        self.assertTrue(all(output.visible for output in outputs))
        self.assertEqual(track_filter.last_debug_record()["static_filtered_count"], 0)

    def test_runtime_filter_stops_low_confidence_ground_bounce_tail(self) -> None:
        track_filter = create_tracknet_v3_ball_track_filter(fps=20.0, debug_enabled=True)
        frame_shape = (500, 500, 3)
        samples = [
            (100.0, 300.0, 0.80),
            (102.0, 350.0, 0.80),
            (104.0, 410.0, 0.80),
            (106.0, 450.0, 0.70),
            (108.0, 440.0, 0.55),
            (110.0, 420.0, 0.32),
        ]
        for x, y, score in samples:
            result = track_filter.update_candidates(
                [_track(x, y, score)],
                dt=0.05,
                frame_shape=frame_shape,
            )
            self.assertTrue(result.visible)

        stopped = track_filter.update_candidates(
            [_track(112.0, 400.0, 0.33)],
            dt=0.05,
            frame_shape=frame_shape,
        )
        suppressed = track_filter.update_candidates(
            [_track(114.0, 380.0, 0.80)],
            dt=0.05,
            frame_shape=frame_shape,
        )

        self.assertFalse(stopped.visible)
        self.assertEqual(track_filter.debug_records[-2]["reason"], "low_confidence_ground_bounce_tail")
        self.assertFalse(suppressed.visible)
        self.assertEqual(track_filter.last_debug_record()["reason"], "active_ground_bounce_suppression")

    def test_runtime_filter_stops_compact_high_confidence_ground_bounce(self) -> None:
        track_filter = create_tracknet_v3_ball_track_filter(fps=20.0, debug_enabled=True)
        frame_shape = (500, 500, 3)
        far_person = [(350.0, 100.0, 450.0, 400.0)]
        samples = [
            (100.0, 300.0, 0.90),
            (102.0, 350.0, 0.90),
            (104.0, 410.0, 0.90),
            (106.0, 450.0, 0.85),
            (105.0, 444.0, 0.88),
        ]
        for x, y, score in samples:
            result = track_filter.update_candidates(
                [_track(x, y, score)],
                dt=0.05,
                frame_shape=frame_shape,
                person_bboxes=far_person,
            )
            self.assertTrue(result.visible)

        stopped = track_filter.update_candidates(
            [_track(104.0, 439.0, 0.92)],
            dt=0.05,
            frame_shape=frame_shape,
            person_bboxes=far_person,
        )
        suppressed = track_filter.update_candidates(
            [_missing(0.90)],
            dt=0.05,
            frame_shape=frame_shape,
        )

        self.assertFalse(stopped.visible)
        self.assertEqual(track_filter.debug_records[-2]["reason"], "compact_ground_bounce")
        self.assertFalse(suppressed.visible)
        self.assertEqual(track_filter.last_debug_record()["reason"], "active_ground_bounce_suppression")

    def test_runtime_filter_keeps_compact_high_confidence_reversal_near_player(self) -> None:
        track_filter = create_tracknet_v3_ball_track_filter(fps=20.0, debug_enabled=True)
        frame_shape = (500, 500, 3)
        nearby_person = [(80.0, 250.0, 150.0, 490.0)]
        samples = [
            (100.0, 300.0, 0.90),
            (102.0, 350.0, 0.90),
            (104.0, 410.0, 0.90),
            (106.0, 450.0, 0.85),
            (105.0, 444.0, 0.88),
        ]
        for x, y, score in samples:
            track_filter.update_candidates(
                [_track(x, y, score)],
                dt=0.05,
                frame_shape=frame_shape,
                person_bboxes=nearby_person,
            )

        continued = track_filter.update_candidates(
            [_track(104.0, 439.0, 0.92)],
            dt=0.05,
            frame_shape=frame_shape,
            person_bboxes=nearby_person,
        )

        self.assertTrue(continued.visible)
        self.assertNotEqual(track_filter.last_debug_record()["reason"], "compact_ground_bounce")

    def test_runtime_filter_keeps_high_confidence_upward_hit_near_ground(self) -> None:
        track_filter = create_tracknet_v3_ball_track_filter(fps=20.0, debug_enabled=True)
        frame_shape = (500, 500, 3)
        samples = [
            (100.0, 300.0, 0.80),
            (102.0, 350.0, 0.80),
            (104.0, 410.0, 0.80),
            (106.0, 450.0, 0.75),
            (112.0, 430.0, 0.75),
            (120.0, 400.0, 0.75),
        ]
        for x, y, score in samples:
            track_filter.update_candidates(
                [_track(x, y, score)],
                dt=0.05,
                frame_shape=frame_shape,
            )

        continued = track_filter.update_candidates(
            [_track(130.0, 360.0, 0.75)],
            dt=0.05,
            frame_shape=frame_shape,
        )

        self.assertTrue(continued.visible)
        self.assertNotEqual(track_filter.last_debug_record()["reason"], "low_confidence_ground_bounce_tail")

    def test_runtime_filter_keeps_upward_track_with_only_one_low_score(self) -> None:
        track_filter = create_tracknet_v3_ball_track_filter(fps=20.0, debug_enabled=True)
        frame_shape = (500, 500, 3)
        samples = [
            (100.0, 300.0, 0.80),
            (102.0, 350.0, 0.80),
            (104.0, 410.0, 0.80),
            (106.0, 450.0, 0.75),
            (112.0, 430.0, 0.75),
            (120.0, 400.0, 0.75),
        ]
        for x, y, score in samples:
            track_filter.update_candidates(
                [_track(x, y, score)],
                dt=0.05,
                frame_shape=frame_shape,
            )

        continued = track_filter.update_candidates(
            [_track(130.0, 360.0, 0.32)],
            dt=0.05,
            frame_shape=frame_shape,
        )

        self.assertTrue(continued.visible)
        self.assertNotEqual(track_filter.last_debug_record()["reason"], "low_confidence_ground_bounce_tail")

    def test_ground_bounce_guard_still_allows_bounce_end_event(self) -> None:
        track_filter = create_tracknet_v3_ball_track_filter(fps=20.0, debug_enabled=True)
        event_detector = RealtimeTrajectoryEventDetector(
            TrajectoryEventDetectorConfig(
                fps=20.0,
                visibility_drop_missing_frames=3,
                rally_end_missing_frames=5,
                tracking_lost_end_seconds=0.20,
            )
        )
        frame_shape = (500, 500, 3)
        raw_tracks = [
            _track(100.0, 300.0, 0.80),
            _track(102.0, 350.0, 0.80),
            _track(104.0, 410.0, 0.80),
            _track(106.0, 450.0, 0.70),
            _track(108.0, 440.0, 0.55),
            _track(110.0, 420.0, 0.30),
            _track(112.0, 400.0, 0.32),
            *[_missing() for _ in range(12)],
        ]

        events = []
        for frame_id, raw_track in enumerate(raw_tracks):
            filtered = track_filter.update_candidates(
                [raw_track],
                dt=0.05,
                frame_shape=frame_shape,
            )
            event = event_detector.update(
                FrameResult(frame_id=frame_id, pose=[], track=filtered),
                timestamp_ms=frame_id * 50,
                frame_shape=frame_shape,
            )
            if event is not None:
                events.append(event)

        self.assertTrue(
            any(event.get("rule") == "tracking_lost_bounce_end" for event in events)
        )

    def test_runtime_filter_debug_records_court_filtering(self) -> None:
        track_filter = create_tracknet_v3_ball_track_filter(fps=30.0, debug_enabled=True)

        output = track_filter.update_candidates(
            [_track(120.0, 430.0, 0.96)],
            dt=1.0 / 30.0,
            frame_shape=(720, 1280, 3),
            court_prediction=_court_prediction(),
        )

        self.assertFalse(output.visible)
        self.assertEqual(track_filter.last_debug_record()["court_filter_active"], 1)
        self.assertEqual(track_filter.last_debug_record()["court_filtered_count"], 1)

    def test_runtime_filter_expands_high_air_court_laterally_but_rejects_far_side(self) -> None:
        frame_shape = (720, 1280, 3)
        accepted_filter = create_tracknet_v3_ball_track_filter(fps=30.0, debug_enabled=True)
        accepted = accepted_filter.update_candidates(
            [_track(150.0, 100.0, 0.82)],
            dt=1.0 / 30.0,
            frame_shape=frame_shape,
            court_prediction=_court_prediction(),
        )

        rejected_filter = create_tracknet_v3_ball_track_filter(fps=30.0, debug_enabled=True)
        rejected = rejected_filter.update_candidates(
            [_track(40.0, 100.0, 0.82)],
            dt=1.0 / 30.0,
            frame_shape=frame_shape,
            court_prediction=_court_prediction(),
        )

        self.assertTrue(accepted.visible)
        self.assertEqual(accepted_filter.last_debug_record()["court_filtered_count"], 0)
        self.assertFalse(rejected.visible)
        self.assertEqual(rejected_filter.last_debug_record()["court_filtered_count"], 1)

    def test_explicit_fixed_lag_keeps_tracknet_v3_adapter(self) -> None:
        track_filter = create_tracknet_v3_ball_track_filter(
            fps=60.0,
            debug_enabled=True,
            fixed_lag_frames=0,
        )

        output = track_filter.update_candidates(
            [_track(120.0, 130.0, 0.9)],
            dt=1.0 / 60.0,
            frame_shape=(400, 600, 3),
        )

        self.assertTrue(output.visible)
        self.assertEqual(track_filter.last_debug_record()["reason"], "tracknet_v3_candidate")


if __name__ == "__main__":
    unittest.main()
