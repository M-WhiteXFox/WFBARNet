from __future__ import annotations

import importlib.util
import unittest

import numpy as np

if importlib.util.find_spec("PyQt6") is None:
    raise unittest.SkipTest("PyQt6 is not installed in this test environment.")

from PyQt6.QtCore import QObject, pyqtSignal

from apps.pyqt6.services.automatic_court_calibration_service import AutomaticCourtCalibrationService
from apps.pyqt6.services.manual_court_calibration_service import manual_court_prediction_from_corners


def _prediction(*, scheme: str, frame_id: int = 1):
    prediction = manual_court_prediction_from_corners(
        [[100.0, 40.0], [420.0, 48.0], [470.0, 320.0], [70.0, 310.0]],
        source_size=(520, 360),
        frame_id=frame_id,
        timestamp_ms=40,
    )
    prediction.scheme = scheme
    prediction.update_type = scheme
    prediction.status = scheme
    prediction.confidence = 0.92
    return prediction


class _FakeAutomaticService(QObject):
    resultReady = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self.latest = None
        self.reset_count = 0
        self.request_count = 0
        self.clear_pending_count = 0
        self.submit_count = 0

    def start(self) -> None:
        return

    def stop(self) -> None:
        return

    def reset(self) -> None:
        self.reset_count += 1
        self.latest = None

    def request_prediction(self) -> None:
        self.request_count += 1

    def clear_pending(self) -> None:
        self.clear_pending_count += 1

    def submit_frame(self, frame: np.ndarray, frame_id: int, timestamp_ms: int) -> bool:
        self.submit_count += 1
        return True

    def latest_prediction(self):
        return self.latest

    def emit_prediction(self, prediction) -> None:
        self.latest = prediction
        self.resultReady.emit(prediction.to_dict())


class AutomaticCourtCalibrationServiceTest(unittest.TestCase):
    def test_automatic_prediction_is_used_until_manual_override(self) -> None:
        automatic = _FakeAutomaticService()
        service = AutomaticCourtCalibrationService(automatic_service=automatic)
        emitted: list[object] = []
        service.resultReady.connect(emitted.append)

        automatic.emit_prediction(_prediction(scheme="court_pose_white_line"))

        self.assertEqual(service.latest_prediction_dict()["scheme"], "court_pose_white_line")
        self.assertEqual(len(emitted), 1)

        manual = service.set_calibration(
            [[102.0, 42.0], [418.0, 49.0], [468.0, 318.0], [72.0, 308.0]],
            source_size=(520, 360),
            frame_id=2,
            timestamp_ms=80,
        )
        automatic.emit_prediction(_prediction(scheme="court_pose_white_line", frame_id=3))

        self.assertIs(service.latest_prediction(), manual)
        self.assertEqual(service.latest_prediction_dict()["scheme"], "manual")
        self.assertEqual(len(emitted), 2)

    def test_reset_preserves_manual_correction_and_clear_allows_new_auto_result(self) -> None:
        automatic = _FakeAutomaticService()
        service = AutomaticCourtCalibrationService(automatic_service=automatic)
        service.set_calibration(
            [[100.0, 40.0], [420.0, 48.0], [470.0, 320.0], [70.0, 310.0]],
            source_size=(520, 360),
        )

        service.reset()

        self.assertEqual(service.latest_prediction_dict()["scheme"], "manual")
        self.assertEqual(automatic.reset_count, 0)

        service.clear_calibration()
        service.request_prediction()
        submitted = service.submit_frame(np.zeros((32, 48, 3), dtype=np.uint8), 4, 120)
        automatic.emit_prediction(_prediction(scheme="court_pose_white_line", frame_id=4))

        self.assertTrue(submitted)
        self.assertEqual(automatic.reset_count, 1)
        self.assertEqual(automatic.request_count, 1)
        self.assertEqual(service.latest_prediction_dict()["scheme"], "court_pose_white_line")


if __name__ == "__main__":
    unittest.main()
