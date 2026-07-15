from __future__ import annotations

import unittest

import numpy as np

from src.court.batch_court import (
    BatchCourtPredictor,
    is_trusted_automatic_court_prediction,
)
from src.court.opencv_court_detector import CourtLinePrediction


def _prediction(
    *,
    valid: bool,
    scheme: str,
    confidence: float | None = None,
    metrics: dict[str, object] | None = None,
    corners: list[list[float]] | None = None,
) -> CourtLinePrediction:
    resolved_confidence = (1.0 if valid else 0.0) if confidence is None else float(confidence)
    resolved_corners = (
        corners
        if corners is not None
        else [[20.0, 15.0], [140.0, 15.0], [145.0, 105.0], [15.0, 105.0]]
        if valid
        else []
    )
    return CourtLinePrediction(
        frame_id=0,
        timestamp_ms=0,
        source_size=(160, 120),
        valid=valid,
        attempted=True,
        updated=valid,
        update_type=scheme,
        status=scheme,
        confidence=resolved_confidence,
        candidate_confidence=resolved_confidence,
        reason=scheme,
        scheme=scheme,
        corners=resolved_corners,
        keypoints=[],
        court_to_image_h=[],
        image_to_court_h=[],
        projected_lines={"doubles_outer": resolved_corners} if resolved_corners else {},
        metrics=metrics or {},
        detect_ms=0.0,
        rejected_count=0,
    )


class _FakeDetector:
    def __init__(self, prediction: CourtLinePrediction) -> None:
        self.prediction = prediction
        self.reset_count = 0
        self.predict_count = 0

    def reset(self) -> None:
        self.reset_count += 1

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
        self.predict_count += 1
        return self.prediction


class _SequenceDetector(_FakeDetector):
    def __init__(self, predictions: list[CourtLinePrediction]) -> None:
        super().__init__(predictions[-1])
        self.predictions = list(predictions)
        self.predict_count = 0

    def predict(
        self,
        frame: np.ndarray,
        frame_id: int,
        timestamp_ms: int,
        *,
        force: bool = False,
    ) -> CourtLinePrediction:
        index = min(self.predict_count, len(self.predictions) - 1)
        self.predict_count += 1
        return self.predictions[index]


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
            fallback_confirm_frames=1,
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
            fallback_confirm_frames=1,
        )

        result = predictor.predict(np.zeros((120, 160, 3), dtype=np.uint8), 0, 0, force=True)

        self.assertIsNotNone(result)
        self.assertTrue(result.valid)
        self.assertEqual(predictor.active_backend, "opencv")

    def test_active_backend_invalid_result_triggers_fallback(self) -> None:
        detectors = {
            "court_pose": _SequenceDetector(
                [
                    _prediction(valid=True, scheme="court_pose_white_line", confidence=0.92),
                    _prediction(valid=False, scheme="court_pose_white_line"),
                ]
            ),
            "opencv": _SequenceDetector([_prediction(valid=True, scheme="8", confidence=0.70)]),
        }
        predictor = BatchCourtPredictor(
            backends=("court_pose", "opencv"),
            detector_factory=lambda backend: detectors[backend],
            fallback_confirm_frames=1,
        )
        frame = np.zeros((120, 160, 3), dtype=np.uint8)

        first = predictor.predict(frame, 0, 0, force=True)
        second = predictor.predict(frame, 1, 40, force=True)

        self.assertTrue(first.valid)
        self.assertEqual(first.scheme, "court_pose_white_line")
        self.assertTrue(second.valid)
        self.assertEqual(second.scheme, "8")
        self.assertEqual(predictor.active_backend, "opencv")

    def test_all_invalid_results_do_not_lock_the_first_backend(self) -> None:
        factory_calls: list[str] = []
        detectors: dict[str, _FakeDetector] = {}

        def factory(backend: str) -> _FakeDetector:
            factory_calls.append(backend)
            detector = _FakeDetector(_prediction(valid=False, scheme=backend))
            detectors[backend] = detector
            return detector

        predictor = BatchCourtPredictor(
            backends=("monotrack", "opencv"),
            detector_factory=factory,
        )
        frame = np.zeros((120, 160, 3), dtype=np.uint8)

        first = predictor.predict(frame, 0, 0, force=True)
        second = predictor.predict(frame, 1, 40, force=True)

        self.assertFalse(first.valid)
        self.assertFalse(second.valid)
        self.assertEqual(predictor.active_backend, "")
        self.assertEqual(factory_calls, ["monotrack", "opencv"])
        self.assertEqual(detectors["monotrack"].predict_count, 2)
        self.assertEqual(detectors["opencv"].predict_count, 2)

    def test_invalid_detector_state_accumulates_across_frames_until_valid(self) -> None:
        detector = _SequenceDetector(
            [
                _prediction(valid=False, scheme="court_pose_coarse", confidence=0.58),
                _prediction(valid=False, scheme="court_pose_coarse", confidence=0.59),
                _prediction(valid=True, scheme="court_pose_coarse", confidence=0.60),
            ]
        )
        factory_calls: list[str] = []

        def factory(backend: str) -> _SequenceDetector:
            factory_calls.append(backend)
            return detector

        predictor = BatchCourtPredictor(
            backends=("court_pose",),
            detector_factory=factory,
        )
        frame = np.zeros((120, 160, 3), dtype=np.uint8)

        first = predictor.predict(frame, 0, 0, force=True)
        second = predictor.predict(frame, 1, 40, force=True)
        third = predictor.predict(frame, 2, 80, force=True)

        self.assertFalse(first.valid)
        self.assertFalse(second.valid)
        self.assertTrue(third.valid)
        self.assertEqual(predictor.active_backend, "court_pose")
        self.assertEqual(factory_calls, ["court_pose"])
        self.assertEqual(detector.predict_count, 3)

    def test_lower_priority_fallback_is_rechecked_until_court_pose_becomes_valid(self) -> None:
        detectors = {
            "court_pose": _SequenceDetector(
                [
                    _prediction(valid=False, scheme="court_pose_coarse", confidence=0.58),
                    _prediction(valid=False, scheme="court_pose_coarse", confidence=0.59),
                    _prediction(valid=True, scheme="court_pose_white_line", confidence=0.96),
                ]
            ),
            "shuttlecourt_seg": _FakeDetector(
                _prediction(valid=True, scheme="shuttlecourt_seg", confidence=0.94)
            ),
        }
        predictor = BatchCourtPredictor(
            backends=("court_pose", "shuttlecourt_seg"),
            detector_factory=lambda backend: detectors[backend],
        )
        frame = np.zeros((120, 160, 3), dtype=np.uint8)

        first = predictor.predict(frame, 0, 0, force=True)
        second = predictor.predict(frame, 1, 40, force=True)
        third = predictor.predict(frame, 2, 80, force=True)

        self.assertEqual(first.scheme, "shuttlecourt_seg")
        self.assertEqual(second.scheme, "shuttlecourt_seg")
        self.assertEqual(third.scheme, "court_pose_white_line")
        self.assertEqual(predictor.active_backend, "court_pose")
        self.assertEqual(detectors["court_pose"].predict_count, 3)

    def test_lower_priority_recheck_is_throttled_between_selection_intervals(self) -> None:
        detectors = {
            "court_pose": _FakeDetector(
                _prediction(valid=False, scheme="court_pose_coarse", confidence=0.58)
            ),
            "shuttlecourt_seg": _FakeDetector(
                _prediction(valid=True, scheme="shuttlecourt_seg", confidence=0.94)
            ),
        }
        predictor = BatchCourtPredictor(
            backends=("court_pose", "shuttlecourt_seg"),
            detector_factory=lambda backend: detectors[backend],
            fallback_recheck_interval_ms=750,
        )
        frame = np.zeros((120, 160, 3), dtype=np.uint8)

        predictor.predict(frame, 0, 0, force=True)
        predictor.predict(frame, 1, 100, force=False)
        predictor.predict(frame, 2, 800, force=False)

        self.assertEqual(detectors["court_pose"].predict_count, 2)
        self.assertEqual(detectors["shuttlecourt_seg"].predict_count, 3)
        self.assertEqual(predictor.active_backend, "shuttlecourt_seg")

    def test_lower_priority_fallback_requires_consistent_confirmation(self) -> None:
        detectors = {
            "court_pose": _FakeDetector(
                _prediction(valid=False, scheme="court_pose_coarse", confidence=0.58)
            ),
            "shuttlecourt_seg": _FakeDetector(
                _prediction(valid=True, scheme="shuttlecourt_seg", confidence=0.94)
            ),
        }
        predictor = BatchCourtPredictor(
            backends=("court_pose", "shuttlecourt_seg"),
            detector_factory=lambda backend: detectors[backend],
            fallback_confirm_frames=3,
        )
        frame = np.zeros((120, 160, 3), dtype=np.uint8)

        first = predictor.predict(frame, 0, 0, force=True)
        second = predictor.predict(frame, 1, 40, force=True)
        third = predictor.predict(frame, 2, 80, force=True)

        self.assertFalse(first.valid)
        self.assertEqual(first.status, "fallback confirmation 1/3")
        self.assertEqual(len(first.corners), 4)
        self.assertIn("doubles_outer", first.projected_lines)
        self.assertFalse(second.valid)
        self.assertEqual(second.status, "fallback confirmation 2/3")
        self.assertTrue(third.valid)
        self.assertEqual(third.scheme, "shuttlecourt_seg")
        self.assertEqual(predictor.active_backend, "shuttlecourt_seg")

    def test_fallback_confirmation_compares_every_frame_to_initial_anchor(self) -> None:
        base = [[20.0, 20.0], [140.0, 20.0], [145.0, 105.0], [15.0, 105.0]]
        shifted_three = [[x + 3.0, y] for x, y in base]
        shifted_six = [[x + 6.0, y] for x, y in base]
        detectors = {
            "court_pose": _FakeDetector(
                _prediction(valid=False, scheme="court_pose_coarse", confidence=0.58)
            ),
            "monotrack": _SequenceDetector(
                [
                    _prediction(valid=True, scheme="monotrack", corners=base),
                    _prediction(valid=True, scheme="monotrack", corners=shifted_three),
                    _prediction(valid=True, scheme="monotrack", corners=shifted_six),
                ]
            ),
        }
        predictor = BatchCourtPredictor(
            backends=("court_pose", "monotrack"),
            detector_factory=lambda backend: detectors[backend],
            fallback_confirm_frames=3,
            fallback_max_corner_shift_ratio=0.035,
        )
        frame = np.zeros((120, 160, 3), dtype=np.uint8)

        first = predictor.predict(frame, 0, 0, force=True)
        second = predictor.predict(frame, 1, 40, force=True)
        third = predictor.predict(frame, 2, 80, force=True)

        self.assertEqual(first.status, "fallback confirmation 1/3")
        self.assertEqual(second.status, "fallback confirmation 2/3")
        self.assertFalse(third.valid)
        self.assertEqual(third.status, "fallback confirmation 1/3")

    def test_confirmed_fallback_geometry_is_locked_against_later_drift(self) -> None:
        base = [[20.0, 20.0], [140.0, 20.0], [145.0, 105.0], [15.0, 105.0]]
        drifted = [[x + 40.0, y] for x, y in base]
        detectors = {
            "court_pose": _FakeDetector(
                _prediction(valid=False, scheme="court_pose_coarse", confidence=0.58)
            ),
            "monotrack": _SequenceDetector(
                [
                    _prediction(valid=True, scheme="monotrack", corners=base),
                    _prediction(valid=True, scheme="monotrack", corners=base),
                    _prediction(valid=True, scheme="monotrack", corners=base),
                    _prediction(valid=True, scheme="monotrack", corners=drifted),
                ]
            ),
        }
        predictor = BatchCourtPredictor(
            backends=("court_pose", "monotrack"),
            detector_factory=lambda backend: detectors[backend],
            fallback_confirm_frames=3,
            lock_confirmed_fallback_geometry=True,
        )
        frame = np.zeros((120, 160, 3), dtype=np.uint8)

        predictor.predict(frame, 0, 0, force=True)
        predictor.predict(frame, 1, 40, force=True)
        trusted = predictor.predict(frame, 2, 80, force=True)
        locked = predictor.predict(frame, 3, 120, force=True)

        self.assertTrue(trusted.valid)
        self.assertTrue(locked.valid)
        self.assertEqual(locked.status, "locked trusted calibration")
        self.assertEqual(locked.corners, base)
        self.assertEqual(detectors["monotrack"].predict_count, 3)

    def test_locked_fallback_can_only_be_upgraded_by_court_pose(self) -> None:
        base = [[20.0, 20.0], [140.0, 20.0], [145.0, 105.0], [15.0, 105.0]]
        detectors = {
            "court_pose": _FakeDetector(
                _prediction(valid=False, scheme="court_pose_coarse", confidence=0.58)
            ),
            "monotrack": _FakeDetector(
                _prediction(valid=True, scheme="monotrack", confidence=0.99)
            ),
            "opencv": _SequenceDetector(
                [
                    _prediction(valid=True, scheme="8", corners=base),
                    _prediction(valid=True, scheme="8", corners=base),
                    _prediction(valid=True, scheme="8", corners=base),
                ]
            ),
        }
        predictor = BatchCourtPredictor(
            backends=("court_pose", "monotrack", "opencv"),
            detector_factory=lambda backend: detectors[backend],
            prediction_acceptor=lambda backend, prediction: backend == "opencv",
            fallback_confirm_frames=3,
            lock_confirmed_fallback_geometry=True,
        )
        frame = np.zeros((120, 160, 3), dtype=np.uint8)

        predictor.predict(frame, 0, 0, force=True)
        predictor.predict(frame, 1, 40, force=True)
        trusted = predictor.predict(frame, 2, 80, force=True)
        locked = predictor.predict(frame, 3, 120, force=True)

        self.assertTrue(trusted.valid)
        self.assertTrue(locked.valid)
        self.assertEqual(locked.scheme, "8")
        self.assertEqual(locked.status, "locked trusted calibration")
        self.assertEqual(detectors["monotrack"].predict_count, 3)
        self.assertEqual(detectors["opencv"].predict_count, 3)

    def test_cached_fallback_reuse_does_not_advance_confirmation(self) -> None:
        fresh = _prediction(valid=True, scheme="shuttlecourt_seg", confidence=0.94)
        cached = _prediction(valid=True, scheme="shuttlecourt_seg", confidence=0.94)
        cached.attempted = False
        cached.updated = False
        detectors = {
            "court_pose": _FakeDetector(
                _prediction(valid=False, scheme="court_pose_coarse", confidence=0.58)
            ),
            "shuttlecourt_seg": _SequenceDetector([fresh, cached]),
        }
        predictor = BatchCourtPredictor(
            backends=("court_pose", "shuttlecourt_seg"),
            detector_factory=lambda backend: detectors[backend],
            fallback_confirm_frames=3,
        )
        frame = np.zeros((120, 160, 3), dtype=np.uint8)

        first = predictor.predict(frame, 0, 0, force=True)
        second = predictor.predict(frame, 1, 40, force=False)

        self.assertEqual(first.status, "fallback confirmation 1/3")
        self.assertEqual(second.status, "fallback confirmation 1/3")
        self.assertFalse(second.valid)

    def test_valid_but_unaccepted_backend_does_not_block_lower_fallback(self) -> None:
        predictions = {
            "shuttlecourt_seg": _prediction(
                valid=True,
                scheme="shuttlecourt_seg",
                confidence=0.94,
            ),
            "monotrack": _prediction(valid=True, scheme="monotrack", confidence=0.72),
        }
        predictor = BatchCourtPredictor(
            backends=("shuttlecourt_seg", "monotrack"),
            detector_factory=lambda backend: _FakeDetector(predictions[backend]),
            prediction_acceptor=lambda backend, prediction: backend == "monotrack",
            reset_unaccepted_detector_state=True,
            fallback_confirm_frames=1,
        )

        result = predictor.predict(np.zeros((120, 160, 3), dtype=np.uint8), 0, 0, force=True)

        self.assertTrue(result.valid)
        self.assertEqual(result.scheme, "monotrack")
        self.assertEqual(predictor.active_backend, "monotrack")
        self.assertEqual(predictor._detectors["shuttlecourt_seg"].reset_count, 1)

    def test_unaccepted_candidate_is_returned_as_displayable_provisional(self) -> None:
        prediction = _prediction(valid=True, scheme="shuttlecourt_seg", confidence=0.94)
        predictor = BatchCourtPredictor(
            backends=("shuttlecourt_seg",),
            detector_factory=lambda backend: _FakeDetector(prediction),
            prediction_acceptor=lambda backend, candidate: False,
            fallback_confirm_frames=1,
        )

        result = predictor.predict(np.zeros((120, 160, 3), dtype=np.uint8), 0, 0, force=True)

        self.assertFalse(result.valid)
        self.assertEqual(result.update_type, "provisional candidate")
        self.assertEqual(len(result.corners), 4)
        self.assertIn("doubles_outer", result.projected_lines)
        self.assertEqual(result.metrics["provisional_candidate"], 1)

    def test_provisional_selection_uses_backend_priority_not_raw_confidence(self) -> None:
        predictions = {
            "court_pose": _prediction(
                valid=True,
                scheme="court_pose_coarse",
                confidence=0.42,
            ),
            "monotrack": _prediction(valid=True, scheme="monotrack", confidence=0.99),
        }
        predictor = BatchCourtPredictor(
            backends=("court_pose", "monotrack"),
            detector_factory=lambda backend: _FakeDetector(predictions[backend]),
            prediction_acceptor=lambda backend, prediction: False,
            fallback_confirm_frames=1,
        )

        result = predictor.predict(np.zeros((120, 160, 3), dtype=np.uint8), 0, 0, force=True)

        self.assertFalse(result.valid)
        self.assertEqual(result.scheme, "court_pose_coarse")

    def test_plausible_provisional_replaces_earlier_implausible_geometry(self) -> None:
        predictions = {
            "court_pose": _prediction(
                valid=True,
                scheme="court_pose_coarse",
                confidence=0.99,
                corners=[[20.0, 2.0], [140.0, 2.0], [145.0, 60.0], [15.0, 60.0]],
            ),
            "monotrack": _prediction(
                valid=True,
                scheme="monotrack",
                confidence=0.40,
                corners=[[30.0, 30.0], [130.0, 30.0], [145.0, 110.0], [15.0, 110.0]],
            ),
        }
        predictor = BatchCourtPredictor(
            backends=("court_pose", "monotrack"),
            detector_factory=lambda backend: _FakeDetector(predictions[backend]),
            prediction_acceptor=lambda backend, prediction: False,
            fallback_confirm_frames=1,
        )

        result = predictor.predict(np.zeros((120, 160, 3), dtype=np.uint8), 0, 0, force=True)

        self.assertFalse(result.valid)
        self.assertEqual(result.scheme, "monotrack")

    def test_shared_trust_gate_rejects_weak_segmentation_and_accepts_white_line_pose(self) -> None:
        weak = _prediction(
            valid=True,
            scheme="shuttlecourt_seg",
            metrics={"components": {"seg_line_fit": 0.0}},
        )
        trusted = _prediction(valid=True, scheme="court_pose_white_line")

        self.assertFalse(is_trusted_automatic_court_prediction(weak))
        self.assertTrue(is_trusted_automatic_court_prediction(trusted))

    def test_shared_trust_gate_requires_outer_lines_and_plausible_court_geometry(self) -> None:
        plausible_corners = [[30.0, 30.0], [130.0, 30.0], [145.0, 110.0], [15.0, 110.0]]
        strong_metrics = {
            "components": {
                "seg_line_fit": 1.0,
                "singles_min_support": 0.40,
                "singles_support_ratio": 0.80,
                "outer_min_support": 0.30,
            }
        }
        strong = _prediction(
            valid=True,
            scheme="shuttlecourt_seg",
            metrics=strong_metrics,
            corners=plausible_corners,
        )
        missing_outer = _prediction(
            valid=True,
            scheme="shuttlecourt_seg",
            metrics={
                "components": {
                    **strong_metrics["components"],
                    "outer_min_support": 0.0,
                }
            },
            corners=plausible_corners,
        )
        implausible = _prediction(
            valid=True,
            scheme="shuttlecourt_seg",
            metrics=strong_metrics,
            corners=[[30.0, 5.0], [130.0, 5.0], [140.0, 60.0], [20.0, 60.0]],
        )

        self.assertTrue(is_trusted_automatic_court_prediction(strong))
        self.assertFalse(is_trusted_automatic_court_prediction(missing_outer))
        self.assertFalse(is_trusted_automatic_court_prediction(implausible))

        monotrack_without_outer = _prediction(
            valid=True,
            scheme="monotrack",
            metrics={
                "components": {
                    "singles_min_support": 0.40,
                    "singles_support_ratio": 0.80,
                    "outer_min_support": 0.0,
                }
            },
            corners=plausible_corners,
        )
        weak_opencv_shape = _prediction(
            valid=True,
            scheme="8",
            metrics={
                "components": {
                    "singles_min_support": 0.40,
                    "singles_support_ratio": 0.80,
                    "outer_min_support": 0.40,
                    "shape": 0.55,
                    "quad": 0.30,
                }
            },
            corners=plausible_corners,
        )
        strong_opencv = _prediction(
            valid=True,
            scheme="8",
            metrics={
                "components": {
                    "singles_min_support": 0.40,
                    "singles_support_ratio": 0.80,
                    "outer_min_support": 0.40,
                    "shape": 0.90,
                    "quad": 0.80,
                }
            },
            corners=plausible_corners,
        )

        self.assertTrue(is_trusted_automatic_court_prediction(monotrack_without_outer))
        self.assertFalse(is_trusted_automatic_court_prediction(weak_opencv_shape))
        self.assertTrue(is_trusted_automatic_court_prediction(strong_opencv))

    def test_court_pose_coarse_remains_provisional_below_lower_white_line(self) -> None:
        predictions = {
            "court_pose": _prediction(valid=True, scheme="court_pose_coarse", confidence=0.99),
            "monotrack": _prediction(valid=True, scheme="monotrack", confidence=0.72),
        }
        predictor = BatchCourtPredictor(
            backends=("court_pose", "monotrack"),
            detector_factory=lambda backend: _FakeDetector(predictions[backend]),
            fallback_confirm_frames=1,
        )

        result = predictor.predict(np.zeros((120, 160, 3), dtype=np.uint8), 0, 0, force=True)

        self.assertTrue(result.valid)
        self.assertEqual(result.scheme, "monotrack")
        self.assertEqual(predictor.active_backend, "monotrack")

    def test_backend_order_is_not_overridden_by_incomparable_confidence(self) -> None:
        predictions = {
            "monotrack": _prediction(valid=True, scheme="monotrack", confidence=0.66),
            "opencv": _prediction(valid=True, scheme="8", confidence=0.84),
        }
        predictor = BatchCourtPredictor(
            backends=("monotrack", "opencv"),
            detector_factory=lambda backend: _FakeDetector(predictions[backend]),
        )

        result = predictor.predict(np.zeros((120, 160, 3), dtype=np.uint8), 0, 0, force=True)

        self.assertTrue(result.valid)
        self.assertEqual(result.scheme, "monotrack")
        self.assertEqual(predictor.active_backend, "monotrack")

    def test_high_quality_court_pose_white_line_stops_fallback_early(self) -> None:
        factory_calls: list[str] = []
        predictions = {
            "court_pose": _prediction(
                valid=True,
                scheme="court_pose_white_line",
                confidence=0.95,
                metrics={"components": {"pose_white_line_refined": 1.0}},
            ),
            "monotrack": _prediction(valid=True, scheme="monotrack", confidence=0.99),
        }

        def factory(backend: str) -> _FakeDetector:
            factory_calls.append(backend)
            return _FakeDetector(predictions[backend])

        predictor = BatchCourtPredictor(
            backends=("court_pose", "monotrack"),
            detector_factory=factory,
        )

        result = predictor.predict(np.zeros((120, 160, 3), dtype=np.uint8), 0, 0, force=True)

        self.assertTrue(result.valid)
        self.assertEqual(result.scheme, "court_pose_white_line")
        self.assertEqual(factory_calls, ["court_pose"])

    def test_reset_clears_active_detector_latest_prediction_and_source_errors(self) -> None:
        created_detectors: list[_FakeDetector] = []

        def factory(backend: str) -> _FakeDetector:
            detector = _FakeDetector(_prediction(valid=True, scheme="monotrack", confidence=0.80))
            created_detectors.append(detector)
            return detector

        predictor = BatchCourtPredictor(
            backends=("monotrack",),
            detector_factory=factory,
        )

        result = predictor.predict(np.zeros((120, 160, 3), dtype=np.uint8), 0, 0, force=True)
        predictor.errors.append("old source diagnostic")

        self.assertIs(predictor.latest_prediction(), result)
        predictor.reset()

        self.assertEqual(created_detectors[0].reset_count, 1)
        self.assertIsNone(predictor.latest_prediction())
        self.assertEqual(predictor.active_backend, "")
        self.assertEqual(predictor.errors, [])

        predictor.predict(np.zeros((120, 160, 3), dtype=np.uint8), 1, 40, force=True)
        self.assertEqual(len(created_detectors), 2)


if __name__ == "__main__":
    unittest.main()
