from __future__ import annotations

import importlib.util
import os
from time import monotonic, sleep
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

if importlib.util.find_spec("PyQt6") is None:
    raise unittest.SkipTest("PyQt6 is not installed in this test environment.")

from PyQt6.QtCore import QCoreApplication

from apps.pyqt6.services.court_detection_service import CourtDetectionService
from apps.pyqt6.services.manual_court_calibration_service import manual_court_prediction_from_corners


def _prediction(*, frame_id: int, valid: bool):
    prediction = manual_court_prediction_from_corners(
        [[10.0, 10.0], [90.0, 10.0], [90.0, 70.0], [10.0, 70.0]],
        source_size=(100, 80),
        frame_id=frame_id,
        timestamp_ms=frame_id * 40,
    )
    prediction.scheme = "fake"
    prediction.valid = valid
    prediction.updated = valid
    return prediction


class _SequenceDetector:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = list(outcomes)
        self.reset_count = 0
        self.forces: list[bool] = []

    def reset(self) -> None:
        self.reset_count += 1

    def predict(self, frame, frame_id, timestamp_ms, *, force=False):
        self.forces.append(bool(force))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class CourtDetectionServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.app = QCoreApplication.instance() or QCoreApplication([])

    def _wait_until(self, predicate, timeout_s: float = 2.0) -> bool:
        deadline = monotonic() + timeout_s
        while monotonic() < deadline:
            self.app.processEvents()
            if predicate():
                return True
            sleep(0.01)
        self.app.processEvents()
        return bool(predicate())

    def _submit_eventually(self, service: CourtDetectionService, frame_id: int) -> bool:
        frame = np.zeros((80, 100, 3), dtype=np.uint8)
        deadline = monotonic() + 2.0
        while monotonic() < deadline:
            if service.submit_frame(frame, frame_id, frame_id * 40):
                return True
            self.app.processEvents()
            sleep(0.01)
        return False

    def test_invalid_and_failed_attempts_keep_monitoring_future_frames(self) -> None:
        detector = _SequenceDetector(
            [
                _prediction(frame_id=1, valid=False),
                RuntimeError("temporary detector failure"),
                _prediction(frame_id=3, valid=True),
            ]
        )
        service = CourtDetectionService(
            detector_factory=lambda: detector,
            submit_interval_s=0.1,
        )
        results: list[dict] = []
        failures: list[str] = []
        service.resultReady.connect(results.append)
        service.failed.connect(failures.append)
        service.start()
        try:
            service.request_prediction()
            self.assertTrue(self._submit_eventually(service, 1))
            self.assertTrue(self._wait_until(lambda: len(results) == 1))
            self.assertFalse(results[0]["valid"])

            self.assertTrue(self._submit_eventually(service, 2))
            self.assertTrue(self._wait_until(lambda: failures == ["temporary detector failure"]))

            self.assertTrue(self._submit_eventually(service, 3))
            self.assertTrue(self._wait_until(lambda: len(results) == 2))
            self.assertTrue(results[-1]["valid"])
            self.assertEqual(detector.forces, [True, False, False])
        finally:
            service.stop()

    def test_reset_discards_result_already_queued_from_old_generation(self) -> None:
        old_prediction = _prediction(frame_id=1, valid=True)
        detector = _SequenceDetector([old_prediction])
        service = CourtDetectionService(
            detector_factory=lambda: detector,
            submit_interval_s=0.1,
        )
        results: list[dict] = []
        service.resultReady.connect(results.append)
        service.start()
        try:
            service.request_prediction()
            self.assertTrue(self._submit_eventually(service, 1))
            deadline = monotonic() + 2.0
            while monotonic() < deadline and service.latest_prediction() is not old_prediction:
                sleep(0.01)
            self.assertIs(service.latest_prediction(), old_prediction)

            service.reset()
            self.assertTrue(self._wait_until(lambda: detector.reset_count == 1))
            self.app.processEvents()

            self.assertEqual(results, [])
            self.assertIsNone(service.latest_prediction())
        finally:
            service.stop()

    def test_bootstrap_sequence_stops_after_first_valid_prediction(self) -> None:
        detector = _SequenceDetector(
            [
                _prediction(frame_id=1, valid=False),
                _prediction(frame_id=2, valid=True),
                _prediction(frame_id=3, valid=True),
            ]
        )
        service = CourtDetectionService(
            detector_factory=lambda: detector,
            submit_interval_s=0.1,
        )
        results: list[dict] = []
        service.resultReady.connect(results.append)
        service.start()
        frame = np.zeros((80, 100, 3), dtype=np.uint8)
        try:
            submitted = service.submit_bootstrap_frames(
                [
                    (frame, 1, 40),
                    (frame, 2, 80),
                    (frame, 3, 120),
                ]
            )

            self.assertEqual(submitted, 3)
            self.assertTrue(self._wait_until(lambda: len(results) == 2))
            sleep(0.1)
            self.app.processEvents()
            self.assertEqual([result["frame_id"] for result in results], [1, 2])
            self.assertEqual(detector.forces, [True, True])
        finally:
            service.stop()

    def test_bootstrap_sequence_marks_last_invalid_result_as_exhausted(self) -> None:
        detector = _SequenceDetector(
            [
                _prediction(frame_id=1, valid=False),
                _prediction(frame_id=2, valid=False),
            ]
        )
        service = CourtDetectionService(
            detector_factory=lambda: detector,
            submit_interval_s=0.1,
        )
        results: list[dict] = []
        service.resultReady.connect(results.append)
        service.start()
        frame = np.zeros((80, 100, 3), dtype=np.uint8)
        try:
            submitted = service.submit_bootstrap_frames(
                [
                    (frame, 1, 40),
                    (frame, 2, 80),
                ]
            )

            self.assertEqual(submitted, 2)
            self.assertTrue(self._wait_until(lambda: len(results) == 2))
            self.assertNotIn("bootstrap_exhausted", results[0]["metrics"])
            self.assertEqual(results[1]["metrics"]["bootstrap_exhausted"], 1)
            self.assertEqual(results[1]["status"], "automatic court scan exhausted")
        finally:
            service.stop()


if __name__ == "__main__":
    unittest.main()
