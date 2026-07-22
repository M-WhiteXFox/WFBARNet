from __future__ import annotations

import unittest

from src.postprocess.candidate_graph_track import CandidateGraphConfig, FixedLagCandidateGraph
from src.utils.structures import TrackResult


def _candidate(x: float, y: float, score: float) -> TrackResult:
    return TrackResult(
        ball_xy=[x, y],
        visible=1,
        score=score,
        heatmap_shape=[288, 512],
    )


class CandidateGraphTrackTests(unittest.TestCase):
    def test_rejects_an_isolated_high_score_candidate(self) -> None:
        frames = [[] for _ in range(11)]
        frames[5] = [_candidate(320.0, 180.0, 0.99)]

        decisions = FixedLagCandidateGraph(CandidateGraphConfig(fps=30.0)).select_sequence(frames)

        self.assertEqual(len(decisions), len(frames))
        self.assertFalse(decisions[5].track.visible)

    def test_chooses_smooth_candidate_over_higher_score_jump(self) -> None:
        frames = []
        for frame_id in range(12):
            candidates = [_candidate(100.0 + frame_id * 10.0, 200.0, 0.86)]
            if frame_id == 6:
                candidates.insert(0, _candidate(900.0, 40.0, 0.98))
            frames.append(candidates)

        decisions = FixedLagCandidateGraph(CandidateGraphConfig(fps=30.0)).select_sequence(frames)

        self.assertTrue(decisions[6].track.visible)
        self.assertEqual(decisions[6].track.ball_xy, [160.0, 200.0])
        self.assertEqual(decisions[6].candidate_rank, 2)

    def test_keeps_confirmed_direction_reversal_without_moving_coordinates(self) -> None:
        xs = [100.0, 120.0, 140.0, 115.0, 90.0, 65.0, 40.0]
        frames = [[_candidate(x, 220.0, 0.92)] for x in xs]

        decisions = FixedLagCandidateGraph(CandidateGraphConfig(fps=30.0)).select_sequence(frames)

        self.assertEqual([decision.track.ball_xy[0] for decision in decisions], xs)
        self.assertTrue(all(decision.source == "candidate_graph_candidate" for decision in decisions))

    def test_keeps_a_weak_candidate_inside_a_confirmed_track(self) -> None:
        frames = [
            [_candidate(200.0 + frame_id * 8.0, 300.0, 0.24 if frame_id == 5 else 0.93)]
            for frame_id in range(12)
        ]

        decisions = FixedLagCandidateGraph(CandidateGraphConfig(fps=30.0)).select_sequence(frames)

        self.assertTrue(decisions[5].track.visible)
        self.assertEqual(decisions[5].track.ball_xy, [240.0, 300.0])


if __name__ == "__main__":
    unittest.main()
