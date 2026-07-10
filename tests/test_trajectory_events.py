from __future__ import annotations

import unittest

from src.postprocess.trajectory_events import (
    RealtimeTrajectoryEventDetector,
    TrajectoryEventCandidateGenerator,
    TrajectoryEventDetectorConfig,
)
from src.utils.structures import FrameResult, TrackResult


def _frame(frame_id: int, x: float, y: float, visible: int = 1, score: float = 0.8) -> FrameResult:
    return FrameResult(
        frame_id=frame_id,
        pose=[],
        track=TrackResult(ball_xy=[x, y] if visible else [-1.0, -1.0], visible=visible, score=score),
    )


class TrajectoryEventCandidateGeneratorTest(unittest.TestCase):
    def test_detects_hit_from_vertical_velocity_reversal(self) -> None:
        generator = TrajectoryEventCandidateGenerator()

        candidates = generator.generate(
            [100.0, 100.0, 100.0, 100.0, 105.0, 110.0],
            [100.0, 120.0, 140.0, 160.0, 120.0, 80.0],
            [1, 1, 1, 1, 1, 1],
            include_trajectory_end=False,
        )

        self.assertTrue(candidates)
        self.assertEqual(candidates[0]["event_type"], "hit")
        self.assertEqual(candidates[0]["rule"], "vy_reversal")
        self.assertEqual(candidates[0]["frame"], 3)

    def test_detects_hit_across_short_visibility_gap(self) -> None:
        generator = TrajectoryEventCandidateGenerator()

        candidates = generator.generate(
            [100.0, 100.0, 100.0, 100.0, -1.0, -1.0, 100.0, 100.0, -1.0, -1.0, 105.0, 110.0],
            [200.0, 220.0, 240.0, 260.0, -1.0, -1.0, 300.0, 320.0, -1.0, -1.0, 250.0, 200.0],
            [1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
            include_trajectory_end=False,
        )

        hit = next(item for item in candidates if item["event_type"] == "hit")
        self.assertEqual(hit["rule"], "vy_reversal")
        self.assertEqual(hit["frame"], 7)

    def test_does_not_use_missing_gap_as_hit_reversal_velocity(self) -> None:
        generator = TrajectoryEventCandidateGenerator()

        candidates = generator.generate(
            [100.0, 100.0, -1.0, -1.0, 100.0, 100.0, 100.0, 100.0],
            [420.0, 360.0, -1.0, -1.0, 260.0, 180.0, 110.0, 60.0],
            [1, 1, 0, 0, 1, 1, 1, 1],
            include_trajectory_end=False,
        )

        self.assertFalse(any(item["event_type"] == "hit" for item in candidates))

    def test_does_not_use_missing_gap_as_landing_speed_drop(self) -> None:
        generator = TrajectoryEventCandidateGenerator()

        candidates = generator.generate(
            [1071.6, 1070.1, -1.0, -1.0, 1072.9, 1074.0, 1074.2, 1075.5, 1075.9],
            [161.4, 161.5, -1.0, -1.0, 161.9, 162.9, 166.4, 169.0, 173.0],
            [1, 1, 0, 0, 1, 1, 1, 1, 1],
            img_width=1920,
            img_height=1080,
            include_trajectory_end=False,
        )

        self.assertFalse(any(item["event_type"] == "landing" for item in candidates))

    def test_detects_landing_from_speed_step(self) -> None:
        generator = TrajectoryEventCandidateGenerator()

        candidates = generator.generate(
            [0.0, 10.0, 20.0, 30.0, 31.0, 31.0, 31.0, 31.0],
            [100.0] * 8,
            [1] * 8,
            include_trajectory_end=False,
        )

        landing = next(item for item in candidates if item["event_type"] == "landing")
        self.assertEqual(landing["rule"], "speed_step")
        self.assertEqual(landing["frame"], 4)

    def test_does_not_mark_top_apex_low_speed_as_landing(self) -> None:
        generator = TrajectoryEventCandidateGenerator()

        candidates = generator.generate(
            [0.0, 8.0, 16.0, 18.0, 19.0, 20.0, 21.0, 22.0],
            [20.0] * 8,
            [1] * 8,
            img_width=500,
            img_height=200,
            include_trajectory_end=True,
        )

        self.assertFalse(any(item["event_type"] == "landing" for item in candidates))

    def test_does_not_mark_upper_court_upward_slowdown_as_landing(self) -> None:
        generator = TrajectoryEventCandidateGenerator()

        candidates = generator.generate(
            [1071.5, 1071.7, 1071.8, 1071.8, 1071.6, 1071.2, 1071.6, 1070.1],
            [200.6, 186.9, 173.7, 168.9, 164.3, 161.9, 161.4, 161.5],
            [1] * 8,
            img_width=1920,
            img_height=1080,
            include_trajectory_end=False,
        )

        self.assertFalse(any(item["event_type"] == "landing" for item in candidates))

    def test_does_not_mark_slow_high_clear_apex_as_landing(self) -> None:
        generator = TrajectoryEventCandidateGenerator()

        candidates = generator.generate(
            [909.5, 906.1, 903.0, 899.8, 897.7, 894.3, 892.5, 891.0, 889.2, 888.5],
            [308.9, 305.1, 299.8, 298.6, 296.7, 295.1, 297.1, 297.9, 299.0, 301.1],
            [1] * 10,
            img_width=1920,
            img_height=1080,
            include_trajectory_end=False,
        )

        self.assertFalse(any(item["event_type"] == "landing" for item in candidates))

    def test_detects_trajectory_end_only_from_low_speed_tail(self) -> None:
        generator = TrajectoryEventCandidateGenerator()

        candidates = generator.generate(
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [100.0] * 7,
            [1] * 7,
            include_trajectory_end=True,
        )

        landing = next(item for item in candidates if item.get("rule") == "trajectory_end")
        self.assertEqual(landing["event_type"], "landing")
        self.assertEqual(landing["features"]["landing_type"], "tail_low_speed_start")

    def test_does_not_use_high_speed_tail_as_trajectory_end_landing(self) -> None:
        generator = TrajectoryEventCandidateGenerator()

        candidates = generator.generate(
            [0.0, 30.0, 60.0, 90.0, 120.0, -1.0, -1.0, -1.0, -1.0, -1.0],
            [100.0, 100.0, 100.0, 100.0, 100.0, -1.0, -1.0, -1.0, -1.0, -1.0],
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            include_trajectory_end=True,
        )

        self.assertFalse(any(item.get("rule") == "trajectory_end" for item in candidates))

    def test_marks_edge_visibility_drop_as_out_of_frame(self) -> None:
        generator = TrajectoryEventCandidateGenerator(
            TrajectoryEventDetectorConfig(visibility_drop_missing_frames=3)
        )

        candidates = generator.generate(
            [100.0, 80.0, 50.0, 10.0, -1.0, -1.0, -1.0],
            [100.0, 100.0, 100.0, 100.0, -1.0, -1.0, -1.0],
            [1, 1, 1, 1, 0, 0, 0],
            img_width=200,
            img_height=200,
            include_trajectory_end=False,
        )

        self.assertTrue(candidates)
        self.assertEqual(candidates[0]["event_type"], "out_of_frame")
        self.assertEqual(candidates[0]["rule"], "visibility_drop_edge")

    def test_does_not_mark_one_frame_dropout_as_out_of_frame(self) -> None:
        generator = TrajectoryEventCandidateGenerator()

        candidates = generator.generate(
            [100.0, 100.0, 100.0, 100.0, -1.0, 102.0],
            [200.0, 220.0, 240.0, 260.0, -1.0, 280.0],
            [1, 1, 1, 1, 0, 1],
            img_width=500,
            img_height=500,
            include_trajectory_end=False,
        )

        self.assertFalse(any(item["event_type"] == "out_of_frame" for item in candidates))

    def test_does_not_use_missing_point_as_upward_motion(self) -> None:
        generator = TrajectoryEventCandidateGenerator()

        candidates = generator.generate(
            [100.0, 100.0, 100.0, 100.0, -1.0, -1.0, -1.0],
            [200.0, 220.0, 240.0, 260.0, -1.0, -1.0, -1.0],
            [1, 1, 1, 1, 0, 0, 0],
            img_width=500,
            img_height=500,
            include_trajectory_end=False,
        )

        self.assertFalse(any(item["event_type"] == "out_of_frame" for item in candidates))

    def test_marks_confirmed_top_exit_as_out_of_frame(self) -> None:
        generator = TrajectoryEventCandidateGenerator(
            TrajectoryEventDetectorConfig(visibility_drop_missing_frames=3)
        )

        candidates = generator.generate(
            [100.0, 100.0, 100.0, 100.0, -1.0, -1.0, -1.0],
            [80.0, 60.0, 35.0, 8.0, -1.0, -1.0, -1.0],
            [1, 1, 1, 1, 0, 0, 0],
            img_width=500,
            img_height=500,
            include_trajectory_end=False,
        )

        self.assertTrue(candidates)
        self.assertEqual(candidates[0]["event_type"], "out_of_frame")
        self.assertEqual(candidates[0]["rule"], "visibility_drop_edge")

    def test_marks_generic_visibility_drop_as_tracking_loss_not_landing(self) -> None:
        generator = TrajectoryEventCandidateGenerator(
            TrajectoryEventDetectorConfig(visibility_drop_missing_frames=3)
        )

        candidates = generator.generate(
            [100.0, 120.0, 140.0, 160.0, -1.0, -1.0, -1.0],
            [200.0, 230.0, 260.0, 300.0, -1.0, -1.0, -1.0],
            [1, 1, 1, 1, 0, 0, 0],
            img_width=500,
            img_height=500,
            include_trajectory_end=False,
        )

        self.assertTrue(candidates)
        self.assertEqual(candidates[0]["event_type"], "out_of_frame")
        self.assertEqual(candidates[0]["rule"], "visibility_drop_tracking_lost")

    def test_marks_long_tracking_loss_as_rally_end_landing(self) -> None:
        generator = TrajectoryEventCandidateGenerator(
            TrajectoryEventDetectorConfig(
                visibility_drop_missing_frames=3,
                rally_end_missing_frames=5,
            )
        )

        candidates = generator.generate(
            [100.0, 110.0, 120.0, 130.0, -1.0, -1.0, -1.0, -1.0, -1.0],
            [300.0, 320.0, 340.0, 370.0, -1.0, -1.0, -1.0, -1.0, -1.0],
            [1, 1, 1, 1, 0, 0, 0, 0, 0],
            img_width=500,
            img_height=500,
            include_trajectory_end=False,
        )

        self.assertTrue(candidates)
        self.assertEqual(candidates[0]["event_type"], "landing")
        self.assertEqual(candidates[0]["rule"], "tracking_lost_rally_end")

    def test_does_not_mark_tracking_loss_from_jump_outlier_as_rally_end(self) -> None:
        generator = TrajectoryEventCandidateGenerator(
            TrajectoryEventDetectorConfig(
                visibility_drop_missing_frames=3,
                rally_end_missing_frames=5,
                tracking_lost_end_max_speed=120.0,
            )
        )

        candidates = generator.generate(
            [900.0, 910.0, 920.0, 1826.0, -1.0, -1.0, -1.0, -1.0, -1.0],
            [450.0, 452.0, 449.0, 966.0, -1.0, -1.0, -1.0, -1.0, -1.0],
            [1, 1, 1, 1, 0, 0, 0, 0, 0],
            img_width=1920,
            img_height=1080,
            include_trajectory_end=False,
        )

        self.assertTrue(candidates)
        self.assertEqual(candidates[0]["event_type"], "out_of_frame")
        self.assertEqual(candidates[0]["rule"], "visibility_drop_tracking_lost")

    def test_does_not_emit_hit_from_smooth_acceleration_peaks_only(self) -> None:
        generator = TrajectoryEventCandidateGenerator()

        candidates = generator.generate(
            [
                1167.1,
                1169.0,
                1172.7,
                1174.7,
                1175.7,
                1177.4,
                1181.0,
                1185.0,
                1189.8,
                1195.1,
            ],
            [
                5.8,
                5.5,
                7.5,
                10.7,
                13.7,
                17.0,
                25.8,
                48.6,
                86.0,
                196.0,
            ],
            [1] * 10,
            img_width=1920,
            img_height=1080,
            include_trajectory_end=False,
        )

        self.assertFalse(any(item["event_type"] == "hit" for item in candidates))


class RealtimeTrajectoryEventDetectorTest(unittest.TestCase):
    def test_emits_confirmed_event_with_original_frame_metadata(self) -> None:
        detector = RealtimeTrajectoryEventDetector(TrajectoryEventDetectorConfig(fps=25.0))
        event = None
        points = [
            (100.0, 100.0),
            (100.0, 120.0),
            (100.0, 140.0),
            (100.0, 160.0),
            (105.0, 120.0),
            (110.0, 80.0),
        ]

        for frame_id, (x, y) in enumerate(points):
            event = detector.update(_frame(frame_id, x, y), timestamp_ms=frame_id * 40, frame_shape=(300, 500, 3))

        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(event["event_type"], "hit")
        self.assertEqual(event["frame_id"], 3)
        self.assertEqual(event["timestamp_ms"], 120)
        self.assertEqual(event["ball_xy"], [100.0, 160.0])

    def test_suppresses_top_band_reversal_hits(self) -> None:
        detector = RealtimeTrajectoryEventDetector(TrajectoryEventDetectorConfig(fps=60.0))
        event = None
        points = [
            (100.0, 10.0),
            (100.0, 20.0),
            (100.0, 30.0),
            (100.0, 70.0),
            (105.0, 20.0),
            (110.0, 10.0),
        ]

        for frame_id, (x, y) in enumerate(points):
            event = detector.update(_frame(frame_id, x, y), timestamp_ms=frame_id * 16, frame_shape=(1080, 1920, 3))

        self.assertIsNone(event)

    def test_suppresses_reversal_hit_with_low_score_neighbor(self) -> None:
        detector = RealtimeTrajectoryEventDetector(TrajectoryEventDetectorConfig(fps=25.0))
        event = None
        points = [
            (100.0, 100.0, 0.8),
            (100.0, 120.0, 0.8),
            (100.0, 140.0, 0.8),
            (100.0, 160.0, 0.8),
            (105.0, 120.0, 0.2),
            (110.0, 80.0, 0.8),
        ]

        for frame_id, (x, y, score) in enumerate(points):
            event = detector.update(
                _frame(frame_id, x, y, score=score),
                timestamp_ms=frame_id * 40,
                frame_shape=(300, 500, 3),
            )

        self.assertIsNone(event)

    def test_emits_reversal_hit_with_moderate_current_score(self) -> None:
        detector = RealtimeTrajectoryEventDetector(TrajectoryEventDetectorConfig(fps=25.0))
        event = None
        points = [
            (100.0, 100.0, 0.8),
            (100.0, 120.0, 0.8),
            (100.0, 140.0, 0.8),
            (100.0, 160.0, 0.49),
            (105.0, 120.0, 0.8),
            (110.0, 80.0, 0.8),
        ]

        for frame_id, (x, y, score) in enumerate(points):
            event = detector.update(
                _frame(frame_id, x, y, score=score),
                timestamp_ms=frame_id * 40,
                frame_shape=(300, 500, 3),
            )

        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(event["event_type"], "hit")
        self.assertEqual(event["frame_id"], 3)

    def test_emits_reversal_hit_with_moderate_score_neighbor(self) -> None:
        detector = RealtimeTrajectoryEventDetector(TrajectoryEventDetectorConfig(fps=25.0))
        event = None
        points = [
            (100.0, 100.0, 0.8),
            (100.0, 120.0, 0.8),
            (100.0, 140.0, 0.8),
            (100.0, 160.0, 0.8),
            (105.0, 120.0, 0.36),
            (110.0, 80.0, 0.8),
        ]

        for frame_id, (x, y, score) in enumerate(points):
            event = detector.update(
                _frame(frame_id, x, y, score=score),
                timestamp_ms=frame_id * 40,
                frame_shape=(300, 500, 3),
            )

        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(event["event_type"], "hit")
        self.assertEqual(event["frame_id"], 3)

    def test_emits_high_reversal_hit_outside_narrow_top_band(self) -> None:
        detector = RealtimeTrajectoryEventDetector(TrajectoryEventDetectorConfig(fps=60.0))
        event = None
        points = [
            (100.0, 50.0),
            (100.0, 80.0),
            (100.0, 100.0),
            (100.0, 120.0),
            (105.0, 80.0),
            (110.0, 40.0),
        ]

        for frame_id, (x, y) in enumerate(points):
            event = detector.update(_frame(frame_id, x, y), timestamp_ms=frame_id * 16, frame_shape=(1080, 1920, 3))

        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(event["event_type"], "hit")
        self.assertEqual(event["frame_id"], 3)

    def test_short_tracking_dropout_does_not_end_rally(self) -> None:
        detector = RealtimeTrajectoryEventDetector(TrajectoryEventDetectorConfig(fps=60.0))
        ending_events = []
        samples = [
            (0, 0.0, 100.0, 1),
            (1, 20.0, 100.0, 1),
            (2, 40.0, 100.0, 1),
            (3, 60.0, 100.0, 1),
            (4, 80.0, 100.0, 1),
            (5, -1.0, -1.0, 0),
            (6, -1.0, -1.0, 0),
            (7, -1.0, -1.0, 0),
            (8, 100.0, 100.0, 1),
            (9, 120.0, 100.0, 1),
            (10, 140.0, 100.0, 1),
        ]

        for frame_id, x, y, visible in samples:
            event = detector.update(
                _frame(frame_id, x, y, visible=visible),
                timestamp_ms=int(round(frame_id * 1000 / 60)),
                frame_shape=(300, 500, 3),
            )
            if isinstance(event, dict) and event.get("event_type") in {"landing", "out_of_frame"}:
                ending_events.append(event)

        self.assertEqual(ending_events, [])

    def test_emits_delayed_rally_end_after_long_tracking_loss(self) -> None:
        detector = RealtimeTrajectoryEventDetector(
            TrajectoryEventDetectorConfig(
                fps=60.0,
                visibility_drop_missing_frames=3,
                rally_end_missing_frames=5,
                tracking_lost_end_seconds=0.0,
            )
        )
        event = None
        samples = [
            (0, 100.0, 300.0, 1),
            (1, 110.0, 320.0, 1),
            (2, 120.0, 340.0, 1),
            (3, 130.0, 370.0, 1),
            (4, -1.0, -1.0, 0),
            (5, -1.0, -1.0, 0),
            (6, -1.0, -1.0, 0),
            (7, -1.0, -1.0, 0),
            (8, -1.0, -1.0, 0),
            (9, -1.0, -1.0, 0),
        ]

        for frame_id, x, y, visible in samples:
            current = detector.update(
                _frame(frame_id, x, y, visible=visible),
                timestamp_ms=int(round(frame_id * 1000 / 60)),
                frame_shape=(500, 500, 3),
            )
            if isinstance(current, dict) and current.get("event_type") == "landing":
                event = current

        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(event["frame_id"], 3)
        self.assertEqual(event["rule"], "tracking_lost_rally_end")

    def test_dense_tracking_loss_waits_for_real_elapsed_time(self) -> None:
        detector = RealtimeTrajectoryEventDetector(
            TrajectoryEventDetectorConfig(
                fps=60.0,
                visibility_drop_missing_frames=3,
                rally_end_missing_frames=5,
                tracking_lost_end_seconds=0.75,
            )
        )
        samples = [
            (0, 100.0, 300.0, 1),
            (1, 110.0, 320.0, 1),
            (2, 120.0, 340.0, 1),
            (3, 130.0, 370.0, 1),
            *[(frame_id, -1.0, -1.0, 0) for frame_id in range(4, 14)],
        ]

        events = []
        for frame_id, x, y, visible in samples:
            event = detector.update(
                _frame(frame_id, x, y, visible=visible),
                timestamp_ms=frame_id * 20,
                frame_shape=(500, 500, 3),
            )
            if event is not None:
                events.append(event)

        self.assertFalse(any(event["event_type"] == "landing" for event in events))

    def test_tracking_loss_ends_rally_after_real_time_confirmation(self) -> None:
        detector = RealtimeTrajectoryEventDetector(
            TrajectoryEventDetectorConfig(
                fps=60.0,
                visibility_drop_missing_frames=3,
                rally_end_missing_frames=5,
                tracking_lost_end_seconds=0.75,
            )
        )
        samples = [
            (0, 100.0, 300.0, 1),
            (1, 110.0, 320.0, 1),
            (2, 120.0, 340.0, 1),
            (3, 130.0, 370.0, 1),
            *[(frame_id, -1.0, -1.0, 0) for frame_id in range(4, 24)],
        ]

        events = []
        for frame_id, x, y, visible in samples:
            event = detector.update(
                _frame(frame_id, x, y, visible=visible),
                timestamp_ms=frame_id * 50,
                frame_shape=(500, 500, 3),
            )
            if event is not None:
                events.append(event)

        self.assertTrue(
            any(
                event["event_type"] == "landing" and event["rule"] == "tracking_lost_rally_end"
                for event in events
            )
        )

    def test_upward_tracking_loss_does_not_end_rally(self) -> None:
        detector = RealtimeTrajectoryEventDetector(
            TrajectoryEventDetectorConfig(
                fps=60.0,
                visibility_drop_missing_frames=3,
                rally_end_missing_frames=5,
                tracking_lost_end_seconds=0.20,
            )
        )
        samples = [
            (0, 100.0, 400.0, 1),
            (1, 110.0, 380.0, 1),
            (2, 120.0, 360.0, 1),
            (3, 130.0, 340.0, 1),
            *[(frame_id, -1.0, -1.0, 0) for frame_id in range(4, 24)],
        ]

        events = []
        for frame_id, x, y, visible in samples:
            event = detector.update(
                _frame(frame_id, x, y, visible=visible),
                timestamp_ms=frame_id * 40,
                frame_shape=(500, 500, 3),
            )
            if event is not None:
                events.append(event)

        self.assertFalse(any(event["event_type"] == "landing" for event in events))

    def test_ground_bounce_then_tracking_loss_ends_rally(self) -> None:
        detector = RealtimeTrajectoryEventDetector(
            TrajectoryEventDetectorConfig(
                fps=20.0,
                visibility_drop_missing_frames=3,
                rally_end_missing_frames=5,
                tracking_lost_end_seconds=0.20,
            )
        )
        samples = [
            (0, 100.0, 300.0, 1, 0.8),
            (1, 102.0, 350.0, 1, 0.8),
            (2, 104.0, 410.0, 1, 0.8),
            (3, 106.0, 450.0, 1, 0.7),
            (4, 108.0, 440.0, 1, 0.20),
            (5, 110.0, 420.0, 1, 0.10),
            (6, 112.0, 400.0, 1, 0.05),
            *[(frame_id, -1.0, -1.0, 0, 0.0) for frame_id in range(7, 19)],
        ]

        events = []
        for frame_id, x, y, visible, score in samples:
            event = detector.update(
                _frame(frame_id, x, y, visible=visible, score=score),
                timestamp_ms=frame_id * 50,
                frame_shape=(500, 500, 3),
            )
            if event is not None:
                events.append(event)

        landing = next(
            event
            for event in events
            if event["event_type"] == "landing" and event["rule"] == "tracking_lost_bounce_end"
        )
        self.assertEqual(landing["frame_id"], 3)
        self.assertEqual(landing["timestamp_ms"], 150)
        self.assertEqual(landing["ball_xy"], [106.0, 450.0])

    def test_high_confidence_upward_flight_after_reversal_is_not_bounce_end(self) -> None:
        detector = RealtimeTrajectoryEventDetector(
            TrajectoryEventDetectorConfig(
                fps=20.0,
                visibility_drop_missing_frames=3,
                rally_end_missing_frames=5,
                tracking_lost_end_seconds=0.20,
            )
        )
        samples = [
            (0, 100.0, 300.0, 1),
            (1, 102.0, 360.0, 1),
            (2, 104.0, 430.0, 1),
            (3, 108.0, 400.0, 1),
            (4, 116.0, 350.0, 1),
            (5, 128.0, 300.0, 1),
            *[(frame_id, -1.0, -1.0, 0) for frame_id in range(6, 18)],
        ]

        events = []
        for frame_id, x, y, visible in samples:
            event = detector.update(
                _frame(frame_id, x, y, visible=visible, score=0.75),
                timestamp_ms=frame_id * 50,
                frame_shape=(500, 500, 3),
            )
            if event is not None:
                events.append(event)

        self.assertFalse(any(event.get("rule") == "tracking_lost_bounce_end" for event in events))

    def test_low_speed_landing_is_cancelled_when_motion_resumes(self) -> None:
        detector = RealtimeTrajectoryEventDetector(
            TrajectoryEventDetectorConfig(
                fps=20.0,
                landing_confirmation_seconds=0.35,
            )
        )
        points = [
            (0.0, 120.0),
            (10.0, 120.0),
            (20.0, 120.0),
            (30.0, 120.0),
            (31.0, 120.0),
            (31.5, 120.0),
            (32.0, 120.0),
            (32.5, 120.0),
            (70.0, 120.0),
            (90.0, 120.0),
            (110.0, 120.0),
        ]

        events = []
        for frame_id, (x, y) in enumerate(points):
            event = detector.update(
                _frame(frame_id, x, y),
                timestamp_ms=frame_id * 50,
                frame_shape=(300, 500, 3),
            )
            if event is not None:
                events.append(event)

        self.assertFalse(any(event["event_type"] == "landing" for event in events))

    def test_confirmed_low_speed_landing_is_emitted(self) -> None:
        detector = RealtimeTrajectoryEventDetector(
            TrajectoryEventDetectorConfig(
                fps=20.0,
                landing_confirmation_seconds=0.35,
            )
        )
        points = [
            (0.0, 120.0),
            (10.0, 120.0),
            (20.0, 120.0),
            (30.0, 120.0),
            (31.0, 120.0),
            (31.5, 120.0),
            (32.0, 120.0),
            (32.2, 120.0),
            (32.3, 120.0),
            (32.4, 120.0),
            (32.5, 120.0),
            (32.6, 120.0),
        ]

        events = []
        for frame_id, (x, y) in enumerate(points):
            event = detector.update(
                _frame(frame_id, x, y),
                timestamp_ms=frame_id * 50,
                frame_shape=(300, 500, 3),
            )
            if event is not None:
                events.append(event)

        self.assertTrue(any(event["event_type"] == "landing" for event in events))

    def test_landing_outside_calibrated_court_is_rejected(self) -> None:
        detector = RealtimeTrajectoryEventDetector(
            TrajectoryEventDetectorConfig(
                fps=20.0,
                landing_confirmation_seconds=0.20,
            )
        )
        court_prediction = {
            "corners": [[400.0, 100.0], [100.0, 100.0], [80.0, 280.0], [420.0, 280.0]],
        }
        points = [
            (0.0, 200.0),
            (10.0, 200.0),
            (20.0, 200.0),
            (30.0, 200.0),
            (31.0, 200.0),
            (31.5, 200.0),
            (32.0, 200.0),
            (32.2, 200.0),
            (32.3, 200.0),
        ]

        events = []
        for frame_id, (x, y) in enumerate(points):
            event = detector.update(
                _frame(frame_id, x, y),
                timestamp_ms=frame_id * 50,
                frame_shape=(300, 500, 3),
                court_prediction=court_prediction,
            )
            if event is not None:
                events.append(event)

        self.assertFalse(any(event["event_type"] == "landing" for event in events))


if __name__ == "__main__":
    unittest.main()
