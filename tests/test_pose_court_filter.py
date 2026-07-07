from __future__ import annotations

import unittest

from src.postprocess.pose import CourtPoseTargetTracker, filter_pose_results_by_court_halves
from src.utils.structures import PersonPoseResult


def _pose(person_id: int, bbox: list[float], score: float) -> PersonPoseResult:
    return PersonPoseResult(
        person_id=person_id,
        bbox=bbox,
        keypoints=[],
        scores=[],
        person_score=score,
    )


class PoseCourtFilterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.court = {
            "valid": True,
            "image_to_court_h": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        }

    def test_keeps_one_person_per_court_half(self) -> None:
        outside = _pose(0, [-110.0, 100.0, -50.0, 450.0], 0.99)
        top = _pose(1, [250.0, 120.0, 350.0, 520.0], 0.45)
        bottom = _pose(2, [245.0, 720.0, 355.0, 1030.0], 0.40)

        result = filter_pose_results_by_court_halves([outside, top, bottom], self.court)

        self.assertEqual([pose.person_score for pose in result], [0.45, 0.40])
        self.assertEqual([pose.person_id for pose in result], [0, 1])

    def test_picks_best_candidate_inside_each_half(self) -> None:
        top_low = _pose(0, [250.0, 200.0, 350.0, 500.0], 0.30)
        top_high = _pose(1, [260.0, 220.0, 360.0, 510.0], 0.82)
        bottom = _pose(2, [240.0, 760.0, 340.0, 1060.0], 0.55)

        result = filter_pose_results_by_court_halves([top_low, top_high, bottom], self.court)

        self.assertEqual([pose.person_score for pose in result], [0.82, 0.55])

    def test_invalid_court_caps_candidates_to_two(self) -> None:
        poses = [
            _pose(4, [0.0, 0.0, 20.0, 20.0], 0.1),
            _pose(5, [30.0, 0.0, 50.0, 20.0], 0.2),
            _pose(6, [60.0, 0.0, 80.0, 20.0], 0.3),
        ]

        result = filter_pose_results_by_court_halves(poses, {"valid": False})

        self.assertEqual([pose.person_score for pose in result], [0.1, 0.2])
        self.assertEqual([pose.person_id for pose in result], [0, 1])

    def test_tracker_predicts_short_missing_gap(self) -> None:
        tracker = CourtPoseTargetTracker(
            max_missing_frames=3,
            detection_smoothing=1.0,
            velocity_smoothing=0.0,
        )

        first = tracker.update([_pose(0, [250.0, 140.0, 350.0, 500.0], 0.8)], self.court)
        second = tracker.update([_pose(1, [260.0, 140.0, 360.0, 500.0], 0.8)], self.court)
        predicted = tracker.update([], self.court)

        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 1)
        self.assertEqual(len(predicted), 1)
        self.assertAlmostEqual(predicted[0].bbox[0], 270.0)

    def test_tracker_can_hold_missing_gap_without_motion_prediction(self) -> None:
        tracker = CourtPoseTargetTracker(
            max_missing_frames=3,
            detection_smoothing=1.0,
            velocity_smoothing=0.0,
            predict_missing_motion=False,
        )

        tracker.update([_pose(0, [250.0, 140.0, 350.0, 500.0], 0.8)], self.court)
        second = tracker.update([_pose(1, [260.0, 140.0, 360.0, 500.0], 0.8)], self.court)
        predicted = tracker.update([], self.court)

        self.assertEqual(len(predicted), 1)
        self.assertAlmostEqual(predicted[0].bbox[0], second[0].bbox[0])

    def test_tracker_can_use_scaled_missing_motion_prediction(self) -> None:
        tracker = CourtPoseTargetTracker(
            max_missing_frames=3,
            detection_smoothing=1.0,
            velocity_smoothing=0.0,
            motion_prediction_scale=0.5,
        )

        tracker.update([_pose(0, [250.0, 140.0, 350.0, 500.0], 0.8)], self.court)
        tracker.update([_pose(1, [260.0, 140.0, 360.0, 500.0], 0.8)], self.court)
        predicted = tracker.update([], self.court)

        self.assertEqual(len(predicted), 1)
        self.assertAlmostEqual(predicted[0].bbox[0], 265.0)

    def test_tracker_accepts_reasonable_fast_motion_in_same_half(self) -> None:
        tracker = CourtPoseTargetTracker(
            max_missing_frames=2,
            detection_smoothing=1.0,
            velocity_smoothing=0.0,
        )

        tracker.update([_pose(0, [280.0, 120.0, 330.0, 260.0], 0.8)], self.court)
        result = tracker.update([_pose(1, [350.0, 120.0, 400.0, 260.0], 0.86)], self.court)

        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0].bbox[0], 350.0)

    def test_tracker_rejects_large_jump_candidate_in_same_half(self) -> None:
        tracker = CourtPoseTargetTracker(
            max_missing_frames=2,
            detection_smoothing=1.0,
            velocity_smoothing=0.0,
        )

        tracker.update([_pose(0, [280.0, 120.0, 330.0, 260.0], 0.8)], self.court)
        result = tracker.update([_pose(1, [520.0, 120.0, 570.0, 260.0], 0.99)], self.court)

        self.assertEqual(len(result), 1)
        self.assertLess(result[0].bbox[0], 330.0)

    def test_tracker_expires_after_missing_limit(self) -> None:
        tracker = CourtPoseTargetTracker(max_missing_frames=1)

        self.assertEqual(len(tracker.update([_pose(0, [250.0, 140.0, 350.0, 500.0], 0.8)], self.court)), 1)
        self.assertEqual(len(tracker.update([], self.court)), 1)
        self.assertEqual(len(tracker.update([], self.court)), 0)

    def test_tracker_ignores_outside_detection(self) -> None:
        tracker = CourtPoseTargetTracker(max_missing_frames=2)

        tracker.update([_pose(0, [250.0, 140.0, 350.0, 500.0], 0.8)], self.court)
        result = tracker.update([_pose(1, [-130.0, 100.0, -60.0, 500.0], 0.99)], self.court)

        self.assertEqual(len(result), 1)
        self.assertGreater(result[0].bbox[0], 200.0)

    def test_tracker_can_require_valid_court(self) -> None:
        tracker = CourtPoseTargetTracker(max_missing_frames=2, court_required=True)

        result = tracker.update([_pose(0, [250.0, 140.0, 350.0, 500.0], 0.8)], {"valid": False})

        self.assertEqual(result, [])

    def test_tracker_degrades_without_court_when_not_required(self) -> None:
        tracker = CourtPoseTargetTracker(max_missing_frames=2, court_required=False)

        result = tracker.update(
            [
                _pose(0, [250.0, 140.0, 350.0, 500.0], 0.8),
                _pose(1, [260.0, 720.0, 360.0, 1040.0], 0.7),
            ],
            {"valid": False},
        )

        self.assertEqual(len(result), 2)
        self.assertEqual([pose.person_id for pose in result], [0, 1])

    def test_tracker_clears_prediction_outside_court_margin(self) -> None:
        tracker = CourtPoseTargetTracker(
            max_missing_frames=3,
            court_margin=0.0,
            detection_smoothing=1.0,
            velocity_smoothing=0.0,
        )

        tracker.update([_pose(0, [500.0, 720.0, 580.0, 900.0], 0.8)], self.court)
        tracker.update([_pose(1, [550.0, 720.0, 630.0, 900.0], 0.8)], self.court)
        result = tracker.update([], self.court)

        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
