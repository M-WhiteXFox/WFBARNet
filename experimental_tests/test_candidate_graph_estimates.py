from __future__ import annotations

import unittest

from src.postprocess.candidate_graph_track import (
    CandidateGraphDecision,
    CandidateGraphRefinementConfig,
    refine_candidate_graph_sequence,
    select_candidate_graph_outputs,
)
from src.utils.structures import TrackResult


def _candidate(x: float, y: float, score: float) -> TrackResult:
    return TrackResult(
        ball_xy=[x, y],
        visible=1,
        score=score,
        heatmap_shape=[288, 512],
    )


def _decision(frame_id: int, track: TrackResult | None) -> CandidateGraphDecision:
    return CandidateGraphDecision(
        frame_index=frame_id,
        track=track or TrackResult(ball_xy=[-1.0, -1.0], visible=0, score=0.0),
        source="test",
        candidate_rank=1 if track is not None else 0,
    )


class CandidateGraphEstimateTests(unittest.TestCase):
    def test_continuous_output_never_overwrites_measured_visibility(self) -> None:
        frames = [
            [_candidate(100.0 + frame_id * 20.0, 200.0, 0.95)]
            for frame_id in range(11)
        ]
        frames.append([])

        outputs = select_candidate_graph_outputs(
            frames,
            width=1280,
            height=720,
            continuous_rendering=True,
        )

        self.assertFalse(outputs[-1].track.visible)
        self.assertTrue(outputs[-1].render_track.visible)
        self.assertEqual(outputs[-1].estimated_source, "candidate_graph_continuation")

    def test_refinement_bridges_supported_run_and_only_interpolates_empty_frame(self) -> None:
        frames = [
            [_candidate(100.0, 200.0, 0.9)],
            [_candidate(110.0, 200.0, 0.35)],
            [],
            [_candidate(130.0, 200.0, 0.42)],
            [_candidate(140.0, 200.0, 0.51)],
            [_candidate(150.0, 200.0, 0.9)],
        ]
        decisions = [
            _decision(0, frames[0][0]),
            _decision(1, None),
            _decision(2, None),
            _decision(3, None),
            _decision(4, None),
            _decision(5, frames[5][0]),
        ]

        refined = refine_candidate_graph_sequence(
            frames,
            decisions,
            width=1280,
            height=720,
            config=CandidateGraphRefinementConfig(
                raw_bridge_enabled=True,
                interpolation_max_gap_frames=1,
            ),
        )

        self.assertEqual(refined[1].track.ball_xy, [110.0, 200.0])
        self.assertEqual(refined[3].track.ball_xy, [130.0, 200.0])
        self.assertEqual(refined[4].track.ball_xy, [140.0, 200.0])
        self.assertEqual(refined[2].track.ball_xy, [120.0, 200.0])
        self.assertEqual(refined[2].source, "candidate_graph_interpolation")

    def test_refinement_does_not_restore_candidates_far_from_anchor_path(self) -> None:
        frames = [
            [_candidate(100.0, 200.0, 0.9)],
            [_candidate(900.0, 50.0, 0.95)],
            [_candidate(850.0, 60.0, 0.94)],
            [_candidate(130.0, 200.0, 0.9)],
        ]
        decisions = [
            _decision(0, frames[0][0]),
            _decision(1, None),
            _decision(2, None),
            _decision(3, frames[3][0]),
        ]

        refined = refine_candidate_graph_sequence(
            frames,
            decisions,
            width=1280,
            height=720,
            config=CandidateGraphRefinementConfig(raw_bridge_enabled=True),
        )

        self.assertFalse(refined[1].track.visible)
        self.assertFalse(refined[2].track.visible)

    def test_refinement_requires_full_candidate_support_for_two_frame_gap(self) -> None:
        frames = [
            [_candidate(100.0, 200.0, 0.9)],
            [_candidate(900.0, 50.0, 0.95)],
            [_candidate(130.0, 200.0, 0.45)],
            [_candidate(145.0, 200.0, 0.9)],
        ]
        decisions = [
            _decision(0, frames[0][0]),
            _decision(1, None),
            _decision(2, None),
            _decision(3, frames[3][0]),
        ]

        refined = refine_candidate_graph_sequence(
            frames,
            decisions,
            width=1280,
            height=720,
            config=CandidateGraphRefinementConfig(raw_bridge_enabled=True),
        )

        self.assertFalse(refined[1].track.visible)
        self.assertFalse(refined[2].track.visible)

    def test_refinement_recovers_one_boundary_candidate_before_exit(self) -> None:
        frames = [
            [_candidate(100.0, 24.0, 0.9)],
            [_candidate(90.0, 10.0, 0.9)],
            [_candidate(81.0, 1.0, 0.31)],
        ]
        decisions = [
            _decision(0, frames[0][0]),
            _decision(1, frames[1][0]),
            _decision(2, None),
        ]

        refined = refine_candidate_graph_sequence(
            frames,
            decisions,
            width=1280,
            height=720,
            config=CandidateGraphRefinementConfig(
                boundary_recovery_enabled=True,
                current_proposal_tolerance_px=0.0,
            ),
            current_proposals=[frame[0] for frame in frames],
            current_proposal_allowed=[True, True, True],
        )

        self.assertEqual(refined[2].track.ball_xy, [81.0, 1.0])
        self.assertEqual(refined[2].source, "candidate_graph_boundary_candidate")

    def test_refinement_only_continues_fast_stable_candidate_free_motion(self) -> None:
        fast_frames = [
            [_candidate(100.0, 200.0, 0.9)],
            [_candidate(120.0, 200.0, 0.9)],
            [_candidate(140.0, 200.0, 0.9)],
            [],
        ]
        slow_frames = [
            [_candidate(100.0, 200.0, 0.9)],
            [_candidate(105.0, 200.0, 0.9)],
            [_candidate(110.0, 200.0, 0.9)],
            [],
        ]
        fast_decisions = [
            _decision(frame_id, frame[0] if frame else None)
            for frame_id, frame in enumerate(fast_frames)
        ]
        slow_decisions = [
            _decision(frame_id, frame[0] if frame else None)
            for frame_id, frame in enumerate(slow_frames)
        ]

        refined_fast = refine_candidate_graph_sequence(
            fast_frames,
            fast_decisions,
            width=1280,
            height=720,
            config=CandidateGraphRefinementConfig(continuation_enabled=True),
        )
        refined_slow = refine_candidate_graph_sequence(
            slow_frames,
            slow_decisions,
            width=1280,
            height=720,
            config=CandidateGraphRefinementConfig(continuation_enabled=True),
        )

        self.assertEqual(refined_fast[3].track.ball_xy, [160.0, 200.0])
        self.assertEqual(refined_fast[3].source, "candidate_graph_continuation")
        self.assertFalse(refined_slow[3].track.visible)

    def test_refinement_uses_only_current_proposals_supported_by_graph_anchors(self) -> None:
        frames = [
            [_candidate(100.0, 200.0, 0.9)],
            [],
            [],
            [],
            [_candidate(180.0, 200.0, 0.9)],
        ]
        decisions = [
            _decision(0, frames[0][0]),
            _decision(1, None),
            _decision(2, None),
            _decision(3, None),
            _decision(4, frames[4][0]),
        ]
        supported = [
            frames[0][0],
            _candidate(120.0, 200.0, 0.4),
            _candidate(140.0, 200.0, 0.3),
            _candidate(160.0, 200.0, 0.4),
            frames[4][0],
        ]
        unsupported = [
            frames[0][0],
            _candidate(800.0, 40.0, 0.9),
            _candidate(820.0, 40.0, 0.9),
            _candidate(840.0, 40.0, 0.9),
            frames[4][0],
        ]

        refined_supported = refine_candidate_graph_sequence(
            frames,
            decisions,
            width=1280,
            height=720,
            config=CandidateGraphRefinementConfig(current_proposal_enabled=True),
            current_proposals=supported,
        )
        refined_unsupported = refine_candidate_graph_sequence(
            frames,
            decisions,
            width=1280,
            height=720,
            config=CandidateGraphRefinementConfig(current_proposal_enabled=True),
            current_proposals=unsupported,
        )

        self.assertTrue(all(refined_supported[index].track.visible for index in (1, 2, 3)))
        self.assertTrue(all(not refined_unsupported[index].track.visible for index in (1, 2, 3)))

    def test_refinement_rejects_single_synthetic_current_proposal(self) -> None:
        frames = [
            [_candidate(100.0, 200.0, 0.9)],
            [],
            [_candidate(140.0, 200.0, 0.9)],
        ]
        decisions = [
            _decision(0, frames[0][0]),
            _decision(1, None),
            _decision(2, frames[2][0]),
        ]
        proposals = [frames[0][0], _candidate(120.0, 200.0, 0.2), frames[2][0]]

        refined = refine_candidate_graph_sequence(
            frames,
            decisions,
            width=1280,
            height=720,
            config=CandidateGraphRefinementConfig(current_proposal_enabled=True),
            current_proposals=proposals,
        )

        self.assertFalse(refined[1].track.visible)

    def test_refinement_removes_isolated_zigzag_but_keeps_sustained_reversal(self) -> None:
        zigzag_frames = [
            [_candidate(100.0, 200.0, 0.9)],
            [_candidate(130.0, 200.0, 0.9)],
            [_candidate(90.0, 200.0, 0.9)],
            [_candidate(150.0, 200.0, 0.9)],
        ]
        zigzag_decisions = [
            _decision(frame_id, frame[0]) for frame_id, frame in enumerate(zigzag_frames)
        ]
        reversal_xs = [100.0, 130.0, 160.0, 125.0, 90.0]
        reversal_frames = [[_candidate(x, 220.0, 0.9)] for x in reversal_xs]
        reversal_decisions = [
            _decision(frame_id, frame[0]) for frame_id, frame in enumerate(reversal_frames)
        ]

        refined_zigzag = refine_candidate_graph_sequence(
            zigzag_frames,
            zigzag_decisions,
            width=1280,
            height=720,
            config=CandidateGraphRefinementConfig(zigzag_veto_enabled=True),
        )
        refined_reversal = refine_candidate_graph_sequence(
            reversal_frames,
            reversal_decisions,
            width=1280,
            height=720,
        )

        self.assertFalse(refined_zigzag[2].track.visible)
        self.assertTrue(all(decision.track.visible for decision in refined_reversal))

    def test_refinement_applies_only_explicit_static_veto_frames(self) -> None:
        frames = [[_candidate(100.0 + frame_id, 200.0, 0.9)] for frame_id in range(4)]
        decisions = [_decision(frame_id, frame[0]) for frame_id, frame in enumerate(frames)]

        refined = refine_candidate_graph_sequence(
            frames,
            decisions,
            width=1280,
            height=720,
            config=CandidateGraphRefinementConfig(),
            static_veto_frames=[2],
        )

        self.assertTrue(refined[1].track.visible)
        self.assertTrue(refined[2].track.visible)
        enabled = refine_candidate_graph_sequence(
            frames,
            decisions,
            width=1280,
            height=720,
            config=CandidateGraphRefinementConfig(static_veto_enabled=True),
            static_veto_frames=[2],
        )
        self.assertFalse(enabled[2].track.visible)
        self.assertEqual(enabled[2].source, "candidate_graph_static_veto")
        self.assertTrue(refined[3].track.visible)


if __name__ == "__main__":
    unittest.main()
