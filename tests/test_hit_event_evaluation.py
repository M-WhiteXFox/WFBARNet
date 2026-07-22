from __future__ import annotations

import unittest

from scripts.render_hit_event_review import _evaluation_metrics, _match_hit_frames


class HitEventEvaluationTests(unittest.TestCase):
    def test_matching_maximizes_cardinality_before_offset(self) -> None:
        matches = _match_hit_frames([1, 3], [0, 2], tolerance=1)

        self.assertEqual(matches, [(1, 0, 1), (3, 2, 1)])

    def test_matching_is_one_to_one(self) -> None:
        matches = _match_hit_frames([10, 11], [10], tolerance=2)

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0], (10, 10, 0))

    def test_metrics_report_false_positives_and_misses(self) -> None:
        metrics = _evaluation_metrics([10, 21, 40], [10, 20, 30], tolerances=(0, 1))

        self.assertEqual(metrics["0"]["true_positives"], 1)
        self.assertEqual(metrics["0"]["false_positives"], 2)
        self.assertEqual(metrics["0"]["false_negatives"], 2)
        self.assertEqual(metrics["1"]["true_positives"], 2)
        self.assertEqual(metrics["1"]["matches"][1]["offset_frames"], 1)
        self.assertEqual(metrics["1"]["median_absolute_error_frames"], 0.5)
        self.assertEqual(metrics["1"]["unmatched_prediction_frames"], [40])
        self.assertEqual(metrics["1"]["unmatched_ground_truth_frames"], [30])


if __name__ == "__main__":
    unittest.main()
