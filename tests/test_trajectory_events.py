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

    def test_marks_edge_visibility_drop_as_out_of_frame(self) -> None:
        generator = TrajectoryEventCandidateGenerator()

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
        generator = TrajectoryEventCandidateGenerator()

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


if __name__ == "__main__":
    unittest.main()
