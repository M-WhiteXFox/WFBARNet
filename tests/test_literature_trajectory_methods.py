from __future__ import annotations

import unittest

from scripts.evaluate_literature_trajectory_methods import (
    apply_fixed_lag_branch_recovery,
    apply_hit_aware_short_gap,
    apply_occlusion_relock,
)


def _row(
    frame_id: int,
    point: tuple[float, float] | None,
    *,
    score: float = 0.9,
    action: str = "accept",
    reason: str = "passes_motion_gate",
    candidates: list[tuple[float, float, float]] | None = None,
) -> dict[str, object]:
    return {
        "frame_id": frame_id,
        "point": point,
        "baseline_point": point,
        "score": score,
        "action": action,
        "reason": reason,
        "source": "baseline",
        "candidates": [
            {"point": (x, y), "score": candidate_score}
            for x, y, candidate_score in (candidates or [])
        ],
        "active": True,
        "gt_point": None,
    }


class LiteratureTrajectoryMethodsTests(unittest.TestCase):
    def test_short_gap_interpolates_smooth_motion(self) -> None:
        rows = [
            _row(0, (0.0, 0.0)),
            _row(1, (10.0, 0.0)),
            _row(2, None, action="reject"),
            _row(3, (30.0, 0.0)),
        ]

        repaired = apply_hit_aware_short_gap(rows)

        self.assertEqual(repaired[2]["point"], (20.0, 0.0))
        self.assertEqual(repaired[2]["source"], "hit_aware_short_gap")

    def test_short_gap_does_not_cross_direction_reversal(self) -> None:
        rows = [
            _row(0, (0.0, 0.0)),
            _row(1, (10.0, 0.0)),
            _row(2, None, action="reject"),
            _row(3, (-10.0, 0.0)),
        ]

        repaired = apply_hit_aware_short_gap(rows)

        self.assertIsNone(repaired[2]["point"])

    def test_occlusion_relock_uses_consistent_high_score_chain(self) -> None:
        rows = [
            _row(0, (0.0, 0.0), reason="person_occlusion_prediction"),
            _row(1, None, action="reject", candidates=[(20.0, -20.0, 0.92)]),
            _row(2, None, action="reject", candidates=[(40.0, -40.0, 0.95)]),
            _row(3, (60.0, -60.0), score=0.96, action="relock_accept"),
        ]

        repaired = apply_occlusion_relock(rows)

        self.assertEqual(repaired[1]["point"], (20.0, -20.0))
        self.assertEqual(repaired[2]["point"], (40.0, -40.0))

    def test_occlusion_relock_removes_unresolved_old_branch(self) -> None:
        rows = [
            _row(0, (0.0, 0.0)),
            _row(1, (10.0, 10.0), reason="person_occlusion_prediction"),
            _row(2, (20.0, 20.0), reason="person_occlusion_motion_gate"),
            _row(3, None, action="reject", candidates=[(50.0, -50.0, 0.92)]),
            _row(4, None, action="reject", candidates=[(70.0, -70.0, 0.95)]),
            _row(5, (90.0, -90.0), score=0.96, action="relock_accept"),
        ]

        repaired = apply_occlusion_relock(rows)

        self.assertIsNone(repaired[1]["point"])
        self.assertIsNone(repaired[2]["point"])
        self.assertEqual(repaired[3]["point"], (50.0, -50.0))

    def test_fixed_lag_recovers_future_branch_and_masks_outlier(self) -> None:
        rows = [
            _row(0, (0.0, 0.0)),
            _row(1, (5.0, 0.0)),
            _row(2, (10.0, 0.0)),
            _row(3, (15.0, 45.0), candidates=[(15.0, 45.0, 0.9)]),
            _row(4, (20.0, 45.0), candidates=[(20.0, 45.0, 0.9), (20.0, 0.0, 0.5)]),
            _row(5, (25.0, 45.0), candidates=[(25.0, 45.0, 0.9), (25.0, 0.0, 0.6)]),
            _row(6, (30.0, 0.0), candidates=[(30.0, 0.0, 0.9)]),
        ]

        repaired = apply_fixed_lag_branch_recovery(rows)

        self.assertEqual(repaired[3]["point"], (15.0, 0.0))
        self.assertEqual(repaired[4]["point"], (20.0, 0.0))
        self.assertEqual(repaired[5]["point"], (25.0, 0.0))


if __name__ == "__main__":
    unittest.main()
