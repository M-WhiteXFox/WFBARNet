from __future__ import annotations

import importlib.util
import unittest

import numpy as np

if importlib.util.find_spec("PyQt6") is None:
    raise unittest.SkipTest("PyQt6 is not installed in this test environment.")

from PyQt6.QtCore import QObject, pyqtSignal

from apps.pyqt6.services.automatic_court_calibration_service import AutomaticCourtCalibrationService
from apps.pyqt6.services.manual_court_calibration_service import manual_court_prediction_from_corners
from src.court.batch_court import BatchCourtPredictor


def _prediction(*, scheme: str, frame_id: int = 1):
    prediction = manual_court_prediction_from_corners(
        [[100.0, 80.0], [420.0, 88.0], [470.0, 320.0], [70.0, 310.0]],
        source_size=(520, 360),
        frame_id=frame_id,
        timestamp_ms=40,
    )
    prediction.scheme = scheme
    prediction.update_type = scheme
    prediction.status = scheme
    prediction.confidence = 0.92
    if scheme == "court_pose_white_line":
        prediction.metrics = {"components": {"pose_white_line_refined": 1.0}}
    elif scheme == "shuttlecourt_seg":
        prediction.metrics = {
            "components": {
                "seg_line_fit": 1.0,
                "singles_min_support": 0.80,
                "singles_support_ratio": 1.0,
                "outer_min_support": 0.80,
            }
        }
    elif scheme == "monotrack":
        prediction.metrics = {
            "components": {
                "singles_min_support": 0.50,
                "singles_support_ratio": 0.75,
                "outer_min_support": 0.60,
            }
        }
    elif scheme == "courtkeynet":
        prediction.metrics = {
            "components": {
                "courtkeynet_combined_confidence": 0.92,
                "courtkeynet_confidence_threshold": 0.50,
                "courtkeynet_confirmation_complete": 1.0,
            }
        }
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

    def test_existing_automatic_prediction_can_be_refreshed(self) -> None:
        automatic = _FakeAutomaticService()
        service = AutomaticCourtCalibrationService(automatic_service=automatic)
        automatic.emit_prediction(_prediction(scheme="court_pose_white_line", frame_id=1))

        service.request_prediction()
        submitted = service.submit_frame(np.zeros((32, 48, 3), dtype=np.uint8), 2, 80)
        automatic.emit_prediction(_prediction(scheme="shuttlecourt_seg", frame_id=2))

        self.assertEqual(automatic.request_count, 1)
        self.assertTrue(submitted)
        self.assertEqual(service.latest_prediction().frame_id, 2)
        self.assertEqual(service.latest_prediction().scheme, "shuttlecourt_seg")

    def test_reset_clears_cached_automatic_and_provisional_results(self) -> None:
        automatic = _FakeAutomaticService()
        service = AutomaticCourtCalibrationService(automatic_service=automatic)
        automatic.emit_prediction(_prediction(scheme="court_pose_white_line", frame_id=1))

        service.reset()

        self.assertIsNone(service.latest_prediction())
        self.assertIsNone(service.latest_display_prediction_dict())
        self.assertEqual(automatic.reset_count, 1)

    def test_invalid_refresh_keeps_last_valid_automatic_prediction(self) -> None:
        automatic = _FakeAutomaticService()
        service = AutomaticCourtCalibrationService(automatic_service=automatic)
        valid = _prediction(scheme="court_pose_white_line", frame_id=1)
        automatic.emit_prediction(valid)
        invalid = _prediction(scheme="court_pose_white_line", frame_id=2)
        invalid.valid = False

        automatic.emit_prediction(invalid)

        self.assertIs(service.latest_prediction(), valid)

    def test_coarse_prediction_remains_provisional_for_automatic_overlay(self) -> None:
        automatic = _FakeAutomaticService()
        service = AutomaticCourtCalibrationService(automatic_service=automatic)
        emitted: list[object] = []
        service.resultReady.connect(emitted.append)

        automatic.emit_prediction(_prediction(scheme="court_pose_coarse", frame_id=1))

        self.assertIsNone(service.latest_prediction())
        display = service.latest_display_prediction_dict()
        self.assertIsNotNone(display)
        self.assertTrue(display["provisional"])
        self.assertEqual(len(display["corners"]), 4)
        self.assertIn("doubles_outer", display["projected_lines"])
        self.assertEqual(len(emitted), 1)
        self.assertFalse(emitted[0]["valid"])
        self.assertTrue(emitted[0]["provisional"])
        self.assertEqual(emitted[0]["status"], "provisional automatic court; waiting for verified white-line evidence")

    def test_pending_courtkeynet_prediction_reports_native_confirmation_progress(self) -> None:
        automatic = _FakeAutomaticService()
        service = AutomaticCourtCalibrationService(automatic_service=automatic)
        pending = _prediction(scheme="courtkeynet", frame_id=2)
        pending.valid = False
        pending.updated = False
        pending.status = "courtkeynet confirmation 1/3"
        pending.metrics["components"]["courtkeynet_confirmation_count"] = 1.0
        pending.metrics["components"]["courtkeynet_confirmation_required"] = 3.0

        automatic.emit_prediction(pending)

        display = service.latest_display_prediction_dict()
        self.assertEqual(
            display["status"],
            "provisional CourtKeyNet court; confirmation 1/3",
        )

    def test_rejected_courtkeynet_candidate_does_not_report_zero_confirmation(self) -> None:
        automatic = _FakeAutomaticService()
        service = AutomaticCourtCalibrationService(automatic_service=automatic)
        rejected = _prediction(scheme="courtkeynet", frame_id=2)
        rejected.valid = False
        rejected.updated = False
        rejected.status = "courtkeynet confidence below threshold"
        rejected.metrics = {
            "components": {
                "courtkeynet_combined_confidence": 0.40,
                "courtkeynet_confidence_threshold": 0.50,
            }
        }

        automatic.emit_prediction(rejected)

        display = service.latest_display_prediction_dict()
        self.assertEqual(
            display["status"],
            "provisional automatic court; waiting for verified white-line evidence",
        )
        self.assertNotIn("0/0", display["status"])

    def test_low_evidence_monotrack_does_not_replace_verified_prediction(self) -> None:
        automatic = _FakeAutomaticService()
        service = AutomaticCourtCalibrationService(automatic_service=automatic)
        verified = _prediction(scheme="court_pose_white_line", frame_id=1)
        automatic.emit_prediction(verified)
        weak = _prediction(scheme="monotrack", frame_id=2)
        weak.metrics = {
            "components": {
                "singles_min_support": 0.10,
                "singles_support_ratio": 0.20,
            }
        }

        automatic.emit_prediction(weak)

        self.assertIs(service.latest_prediction(), verified)

    def test_manual_prediction_blocks_automatic_refresh(self) -> None:
        automatic = _FakeAutomaticService()
        service = AutomaticCourtCalibrationService(automatic_service=automatic)
        service.set_calibration(
            [[100.0, 40.0], [420.0, 48.0], [470.0, 320.0], [70.0, 310.0]],
            source_size=(520, 360),
        )

        service.request_prediction()
        submitted = service.submit_frame(np.zeros((32, 48, 3), dtype=np.uint8), 2, 80)

        self.assertEqual(automatic.request_count, 0)
        self.assertFalse(submitted)

    def test_default_automatic_detector_uses_full_backend_fallback_order(self) -> None:
        service = AutomaticCourtCalibrationService()

        detector = service._automatic._detector_factory()

        self.assertIsInstance(detector, BatchCourtPredictor)
        self.assertEqual(
            detector.backends,
            ("courtkeynet", "shuttlecourt_seg", "monotrack", "opencv"),
        )
        self.assertEqual(detector.authoritative_backends, ("courtkeynet",))
        self.assertEqual(detector.fallback_confirm_frames, 3)
        self.assertTrue(detector.reset_unaccepted_detector_state)
        self.assertTrue(detector.lock_confirmed_fallback_geometry)

    def test_verified_prediction_replaces_provisional_display_candidate(self) -> None:
        automatic = _FakeAutomaticService()
        service = AutomaticCourtCalibrationService(automatic_service=automatic)

        automatic.emit_prediction(_prediction(scheme="court_pose_coarse", frame_id=1))
        provisional = service.latest_display_prediction_dict()
        automatic.emit_prediction(_prediction(scheme="court_pose_white_line", frame_id=2))

        self.assertTrue(provisional["provisional"])
        self.assertEqual(service.latest_prediction().frame_id, 2)
        self.assertFalse(service.latest_display_prediction_dict().get("provisional", False))


if __name__ == "__main__":
    unittest.main()
