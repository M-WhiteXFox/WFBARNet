from __future__ import annotations

import unittest

from src.postprocess.track_filter import BallTrackFilter
from src.postprocess.tracknet_v3_filter import (
    TrackNetV3TrajectoryFilter,
    TrackNetV3TrajectoryFilterConfig,
    create_tracknet_v3_ball_track_filter,
    generate_tracknet_v3_inpaint_mask,
    linear_interpolate_masked_values,
)
from src.utils.structures import TrackResult


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
        self.assertEqual(track_filter.last_debug_record()["reason"], "tracknet_v3_candidate")


if __name__ == "__main__":
    unittest.main()
