from __future__ import annotations

import unittest

import numpy as np

from src.court.opencv_court_detector import OpenCVCourtLineDetector


class OpenCVCourtLineDetectorTest(unittest.TestCase):
    def test_blank_frame_returns_stable_payload(self) -> None:
        detector = OpenCVCourtLineDetector()

        result = detector.predict(np.zeros((120, 160, 3), dtype=np.uint8), frame_id=0, timestamp_ms=0)
        payload = result.to_dict()

        self.assertEqual(payload["frame_id"], 0)
        self.assertEqual(payload["timestamp_ms"], 0)
        self.assertEqual(payload["source_size"], [160, 120])
        self.assertTrue(payload["attempted"])
        self.assertFalse(payload["valid"])
        self.assertIn("court_to_image_h", payload)
        self.assertIn("image_to_court_h", payload)
        self.assertIn("projected_lines", payload)


if __name__ == "__main__":
    unittest.main()
