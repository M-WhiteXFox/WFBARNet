from __future__ import annotations

import os
from pathlib import Path
import unittest

import cv2

from scripts.evaluate_court_pose_accuracy import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_WEIGHTS,
    EvaluationConfig,
    _default_frame_logs,
    assert_quality,
    evaluate_court_pose,
)
from src.court.monotrack_court_detector import MonoTrackCourtLineDetector

COURT_VIDEO = os.environ.get("WFBARNET_COURT_VIDEO", "").strip()


@unittest.skipUnless(COURT_VIDEO, "set WFBARNET_COURT_VIDEO to an external court video")
class CourtModelIntegrationTest(unittest.TestCase):
    def test_monotrack_detects_real_video_frame(self) -> None:
        video_path = Path(COURT_VIDEO)
        self.assertTrue(video_path.is_file(), video_path)
        capture = cv2.VideoCapture(str(video_path))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        middle = max(0, frame_count // 2)
        capture.set(cv2.CAP_PROP_POS_FRAMES, middle)
        ok, frame = capture.read()
        capture.release()
        self.assertTrue(ok)
        self.assertIsNotNone(frame)

        result = MonoTrackCourtLineDetector().predict(
            frame,
            frame_id=middle,
            timestamp_ms=0,
            force=True,
        )

        self.assertTrue(result.valid, result.to_dict())
        self.assertEqual(result.scheme, "monotrack")

    @unittest.skipUnless(
        os.environ.get("WFBARNET_RUN_COURT_POSE_ACCURACY") == "1",
        "set WFBARNET_RUN_COURT_POSE_ACCURACY=1 to run CourtPose accuracy validation",
    )
    def test_court_pose_accuracy(self) -> None:
        report = evaluate_court_pose(
            EvaluationConfig(
                weights=Path(os.environ.get("COURT_POSE_WEIGHTS", str(DEFAULT_WEIGHTS))),
                video=Path(COURT_VIDEO),
                frame_logs=_default_frame_logs(),
                output_dir=Path(
                    os.environ.get("COURT_POSE_OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR))
                ),
                samples=int(os.environ.get("COURT_POSE_SAMPLES", "24")),
                device=os.environ.get("COURT_POSE_DEVICE", "0"),
                refine_white_lines=os.environ.get("COURT_POSE_REFINE_WHITE_LINES") == "1",
                stateful=os.environ.get("COURT_POSE_STATEFUL") == "1",
            )
        )
        assert_quality(report)


if __name__ == "__main__":
    unittest.main()
