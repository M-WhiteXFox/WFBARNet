from __future__ import annotations

import unittest

from src.postprocess.rally_start_backfill import apply_known_rally_start, fit_known_rally_start
from src.postprocess.track_output import TrackFrameOutput
from src.utils.structures import TrackResult


def _track(x: float, y: float) -> TrackResult:
    return TrackResult(ball_xy=[x, y], visible=1, score=0.9, heatmap_shape=[288, 512])


def _missing() -> TrackResult:
    return TrackResult(ball_xy=[-1.0, -1.0], visible=0, score=0.0, heatmap_shape=[288, 512])


class CandidateGraphRefinementEvaluationTests(unittest.TestCase):
    def test_lifecycle_backfill_is_estimate_only(self) -> None:
        tracks = [
            _missing() if frame_id < 2 else _track(100.0 + 5.0 * frame_id, 200.0)
            for frame_id in range(8)
        ]
        outputs = [
            TrackFrameOutput(
                frame_index=frame_id,
                measured_track=track,
                measured_source="contextual",
            )
            for frame_id, track in enumerate(tracks)
        ]

        resolved, debug = apply_known_rally_start(
            outputs,
            active_start=0,
            active_end=7,
            width=1280,
            height=720,
        )

        self.assertEqual(debug["filled_frames"], [0, 1])
        self.assertFalse(resolved[0].track.visible)
        self.assertTrue(resolved[0].render_track.visible)
        self.assertEqual(resolved[0].estimated_source, "rally_start_quadratic_backfill")

    def test_lifecycle_start_backfill_recovers_quadratic_track(self) -> None:
        current = []
        for frame_id in range(12):
            track = (
                _missing()
                if frame_id < 2
                else _track(
                    100.0 + 5.0 * frame_id,
                    200.0 + 2.0 * frame_id + frame_id**2,
                )
            )
            current.append({"track": track})

        result = fit_known_rally_start(
            [row["track"] for row in current],
            active_start=0,
            active_end=11,
            width=1280,
            height=720,
        )
        filled = result.points
        debug = result.debug

        self.assertEqual(debug["filled_frames"], [0, 1])
        self.assertAlmostEqual(filled[0][0], 100.0, places=6)
        self.assertAlmostEqual(filled[0][1], 200.0, places=6)
        self.assertAlmostEqual(filled[1][0], 105.0, places=6)
        self.assertAlmostEqual(filled[1][1], 203.0, places=6)

    def test_lifecycle_start_backfill_rejects_long_unknown_start(self) -> None:
        current = [{"track": _missing()} for _ in range(5)]
        current.extend({"track": _track(200.0 + frame_id, 300.0)} for frame_id in range(7))

        result = fit_known_rally_start(
            [row["track"] for row in current],
            active_start=0,
            active_end=11,
            width=1280,
            height=720,
        )
        filled = result.points
        debug = result.debug

        self.assertEqual(filled, {})
        self.assertEqual(debug["reason"], "start_gap_out_of_range")


if __name__ == "__main__":
    unittest.main()
