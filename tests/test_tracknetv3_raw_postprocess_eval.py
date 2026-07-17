from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from scripts.evaluate_tracknetv3_raw_postprocess import _candidate_oracle, _load_gt, _metrics


class TrackNetV3RawPostprocessEvaluationTests(unittest.TestCase):
    def test_load_gt_converts_normalized_coordinates_and_invisible_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            path.write_text(
                "Frame,Ball,x,y\n0,1,0.5,0.25\n1,0,-1,-1\n",
                encoding="utf-8",
            )

            rows = _load_gt(path, 1280, 720)

        self.assertEqual(rows[0]["point"], (640.0, 180.0))
        self.assertIsNone(rows[1]["point"])

    def test_metrics_and_oracle_count_missing_drift_and_false_positive(self) -> None:
        rows = [
            {
                "gt_point": (10.0, 10.0),
                "fixed_point": (12.0, 10.0),
                "candidate_points": [(12.0, 10.0)],
            },
            {
                "gt_point": (20.0, 20.0),
                "fixed_point": None,
                "candidate_points": [(21.0, 20.0)],
            },
            {
                "gt_point": (30.0, 30.0),
                "fixed_point": (60.0, 60.0),
                "candidate_points": [(60.0, 60.0)],
            },
            {
                "gt_point": None,
                "fixed_point": (40.0, 40.0),
                "candidate_points": [(40.0, 40.0)],
            },
        ]

        metrics = _metrics(rows, "fixed_point", 10.0)
        oracle = _candidate_oracle(rows, 10.0)

        self.assertEqual(metrics["correct"], 1)
        self.assertEqual(metrics["missing"], 1)
        self.assertEqual(metrics["drift"], 1)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(oracle["correct"], 2)


if __name__ == "__main__":
    unittest.main()
