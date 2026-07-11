from __future__ import annotations

import os
from pathlib import Path
import time
import unittest

import cv2

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@unittest.skipUnless(
    os.environ.get("WFBARNET_RUN_AUTO_COURT_GUI") == "1",
    "set WFBARNET_RUN_AUTO_COURT_GUI=1 to run the automatic court GUI integration test",
)
class AutomaticCourtServiceIntegrationTest(unittest.TestCase):
    def test_real_frame_auto_calibrates_and_manual_correction_overrides(self) -> None:
        from PyQt6.QtCore import QCoreApplication

        from apps.pyqt6.services.automatic_court_calibration_service import (
            create_automatic_court_calibration_service,
        )

        app = QCoreApplication.instance() or QCoreApplication([])
        service = create_automatic_court_calibration_service()
        video_path = Path(__file__).resolve().parents[1] / "videos" / "MVI_0211.MP4"
        capture = cv2.VideoCapture(str(video_path))
        ok, frame = capture.read()
        capture.release()
        self.assertTrue(ok)
        self.assertIsNotNone(frame)
        service.request_prediction()
        self.assertTrue(service.submit_frame(frame, 0, 0))

        deadline = time.monotonic() + 30.0
        prediction = None
        while time.monotonic() < deadline:
            app.processEvents()
            prediction = service.latest_prediction()
            if prediction is not None and prediction.valid:
                break
            time.sleep(0.02)

        try:
            self.assertIsNotNone(prediction)
            self.assertTrue(prediction.valid)
            self.assertIn(prediction.scheme, {"court_pose_white_line", "court_pose_coarse"})
            self.assertEqual(len(prediction.corners), 4)
            original = prediction.corners[0]

            corrected_corners = [[float(x), float(y)] for x, y in prediction.corners]
            corrected_corners[0] = [original[0] + 2.0, original[1] + 2.0]
            service.set_calibration(
                corrected_corners,
                source_size=(frame.shape[1], frame.shape[0]),
                frame_id=0,
                timestamp_ms=0,
            )
            corrected = service.latest_prediction()

            self.assertIsNotNone(corrected)
            self.assertEqual(corrected.scheme, "manual")
            self.assertAlmostEqual(corrected.corners[0][0], original[0] + 2.0)
            self.assertAlmostEqual(corrected.corners[0][1], original[1] + 2.0)
            service.reset()
            self.assertEqual(service.latest_prediction().scheme, "manual")
        finally:
            service.stop()
            app.processEvents()


if __name__ == "__main__":
    unittest.main()
