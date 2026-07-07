from __future__ import annotations

import unittest

import numpy as np

from src.court.batch_court import BatchCourtPredictor
from src.court.opencv_court_detector import CourtLinePrediction


def _prediction(*, valid: bool, scheme: str) -> CourtLinePrediction:
    return CourtLinePrediction(
        frame_id=0,
        timestamp_ms=0,
        source_size=(160, 120),
        valid=valid,
        attempted=True,
        updated=valid,
        update_type=scheme,
        status=scheme,
        confidence=1.0 if valid else 0.0,
        candidate_confidence=1.0 if valid else 0.0,
        reason=scheme,
        scheme=scheme,
        corners=[],
        keypoints=[],
        court_to_image_h=[],
        image_to_court_h=[],
        projected_lines={},
        metrics={},
        detect_ms=0.0,
        rejected_count=0,
    )


class _FakeDetector:
    def __init__(self, prediction: CourtLinePrediction) -> None:
        self.prediction = prediction

    def reset(self) -> None:
        return

    def latest_prediction(self) -> CourtLinePrediction | None:
        return self.prediction

    def predict(
        self,
        frame: np.ndarray,
        frame_id: int,
        timestamp_ms: int,
        *,
        force: bool = False,
    ) -> CourtLinePrediction:
        return self.prediction


class BatchCourtPredictorTest(unittest.TestCase):
    def test_falls_back_when_first_backend_raises(self) -> None:
        calls: list[str] = []

        def factory(backend: str) -> _FakeDetector:
            calls.append(backend)
            if backend == "shuttlecourt_seg":
                raise FileNotFoundError("missing weight")
            return _FakeDetector(_prediction(valid=True, scheme=backend))

        predictor = BatchCourtPredictor(
            backends=("shuttlecourt_seg", "monotrack", "opencv"),
            detector_factory=factory,
        )

        result = predictor.predict(np.zeros((120, 160, 3), dtype=np.uint8), 0, 0, force=True)

        self.assertIsNotNone(result)
        self.assertTrue(result.valid)
        self.assertEqual(predictor.active_backend, "monotrack")
        self.assertEqual(calls, ["shuttlecourt_seg", "monotrack"])
        self.assertIn("shuttlecourt_seg", predictor.errors[0])

    def test_uses_next_backend_when_first_prediction_is_invalid(self) -> None:
        predictions = {
            "monotrack": _prediction(valid=False, scheme="monotrack"),
            "opencv": _prediction(valid=True, scheme="opencv"),
        }

        predictor = BatchCourtPredictor(
            backends=("monotrack", "opencv"),
            detector_factory=lambda backend: _FakeDetector(predictions[backend]),
        )

        result = predictor.predict(np.zeros((120, 160, 3), dtype=np.uint8), 0, 0, force=True)

        self.assertIsNotNone(result)
        self.assertTrue(result.valid)
        self.assertEqual(predictor.active_backend, "opencv")


if __name__ == "__main__":
    unittest.main()
