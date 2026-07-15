from __future__ import annotations

import importlib.util
import unittest

if importlib.util.find_spec("PyQt6") is None:
    raise unittest.SkipTest("PyQt6 is not installed in this test environment.")

from apps.pyqt6.services.manual_court_calibration_service import (
    ManualCourtCalibrationService,
    manual_court_prediction_from_corners,
)


class ManualCourtCalibrationServiceTest(unittest.TestCase):
    def test_manual_corners_create_court_prediction(self) -> None:
        prediction = manual_court_prediction_from_corners(
            [
                [100.0, 40.0],
                [420.0, 48.0],
                [470.0, 320.0],
                [70.0, 310.0],
            ],
            source_size=(520, 360),
            frame_id=12,
            timestamp_ms=480,
        )

        self.assertTrue(prediction.valid)
        self.assertEqual(prediction.scheme, "manual")
        self.assertEqual(prediction.source_size, (520, 360))
        self.assertEqual(len(prediction.corners), 4)
        self.assertIn("doubles_outer", prediction.projected_lines)
        self.assertEqual(len(prediction.image_to_court_h), 3)

    def test_service_stores_and_clears_latest_prediction(self) -> None:
        service = ManualCourtCalibrationService()

        service.set_calibration(
            [
                [100.0, 40.0],
                [420.0, 48.0],
                [470.0, 320.0],
                [70.0, 310.0],
            ],
            source_size=(520, 360),
        )

        self.assertIsNotNone(service.latest_prediction())
        self.assertEqual(service.latest_prediction_dict()["scheme"], "manual")

        service.clear_calibration()

        self.assertIsNone(service.latest_prediction())
        self.assertIsNone(service.latest_prediction_dict())

    def test_manual_corners_are_normalized_to_tl_tr_br_bl(self) -> None:
        prediction = manual_court_prediction_from_corners(
            [
                [470.0, 320.0],
                [100.0, 40.0],
                [70.0, 310.0],
                [420.0, 48.0],
            ],
            source_size=(520, 360),
        )

        self.assertEqual(
            prediction.corners,
            [
                [100.0, 40.0],
                [420.0, 48.0],
                [470.0, 320.0],
                [70.0, 310.0],
            ],
        )

    def test_invalid_manual_corners_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            manual_court_prediction_from_corners(
                [
                    [100.0, 40.0],
                    [420.0, 48.0],
                    [420.0, 48.0],
                    [70.0, 310.0],
                ],
                source_size=(520, 360),
            )


if __name__ == "__main__":
    unittest.main()
