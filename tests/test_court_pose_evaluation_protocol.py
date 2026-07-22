from __future__ import annotations

import unittest

import numpy as np

from scripts.evaluate_court_pose_accuracy import _predict_refined_sample


class _ProtocolDetector:
    def __init__(self) -> None:
        self.reset_count = 0
        self.predict_count = 0

    def reset(self) -> None:
        self.reset_count += 1

    def predict(
        self,
        frame: np.ndarray,
        frame_id: int,
        timestamp_ms: int,
        *,
        force: bool = False,
    ) -> object:
        self.predict_count += 1
        return object()


class CourtPoseEvaluationProtocolTest(unittest.TestCase):
    def test_cold_start_resets_detector_for_each_sample(self) -> None:
        detector = _ProtocolDetector()
        frame = np.zeros((12, 16, 3), dtype=np.uint8)

        _predict_refined_sample(detector, frame, frame_id=10, timestamp_ms=400, stateful=False)
        _predict_refined_sample(detector, frame, frame_id=20, timestamp_ms=800, stateful=False)

        self.assertEqual(detector.reset_count, 2)
        self.assertEqual(detector.predict_count, 2)

    def test_stateful_evaluation_preserves_detector_state(self) -> None:
        detector = _ProtocolDetector()
        frame = np.zeros((12, 16, 3), dtype=np.uint8)

        _predict_refined_sample(detector, frame, frame_id=10, timestamp_ms=400, stateful=True)
        _predict_refined_sample(detector, frame, frame_id=20, timestamp_ms=800, stateful=True)

        self.assertEqual(detector.reset_count, 0)
        self.assertEqual(detector.predict_count, 2)


if __name__ == "__main__":
    unittest.main()
