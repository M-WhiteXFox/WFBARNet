from __future__ import annotations

import unittest

from src.postprocess.fixed_lag_track import FixedLagTrackConfig, FixedLagTrackPostProcessor
from src.postprocess.track_filter import filter_track_results
from src.utils.structures import TrackResult


def _track(x: float, y: float, score: float = 0.8) -> TrackResult:
    return TrackResult(ball_xy=[x, y], visible=1, score=score)


def _missing(score: float = 0.0) -> TrackResult:
    return TrackResult(ball_xy=[-1.0, -1.0], visible=0, score=score)


def _debug(
    action: str = "accept",
    reason: str = "passes_motion_gate",
    **values: object,
) -> dict[str, object]:
    return {"action": action, "reason": reason, **values}


class FixedLagTrackPostProcessorTests(unittest.TestCase):
    def test_sequence_helper_flushes_all_delayed_results(self) -> None:
        tracks = [_track(float(index), 20.0) for index in range(12)]

        filtered = filter_track_results(tracks, fps=30.0)

        self.assertEqual(len(filtered), len(tracks))

    def test_delay_is_clamped_and_converted_from_fps(self) -> None:
        self.assertEqual(FixedLagTrackConfig(fps=25.0, delay_ms=300).delay_frames, 8)
        self.assertEqual(FixedLagTrackConfig(fps=30.0, delay_ms=300).delay_frames, 9)
        self.assertEqual(FixedLagTrackConfig(fps=30.0, delay_ms=2000).delay_frames, 30)

    def test_push_delays_frames_and_flush_preserves_order(self) -> None:
        processor = FixedLagTrackPostProcessor(FixedLagTrackConfig(fps=10.0, delay_ms=300))

        self.assertIsNone(processor.push(_track(0.0, 10.0), debug_record=_debug(), payload="f0"))
        self.assertIsNone(processor.push(_track(1.0, 10.0), debug_record=_debug(), payload="f1"))
        self.assertIsNone(processor.push(_track(2.0, 10.0), debug_record=_debug(), payload="f2"))
        emitted = processor.push(_track(3.0, 10.0), debug_record=_debug(), payload="f3")

        self.assertIsNotNone(emitted)
        self.assertEqual(emitted.payload, "f0")
        self.assertEqual([frame.payload for frame in processor.flush()], ["f1", "f2", "f3"])

    def test_single_frame_smoothing_repairs_missing_and_low_confidence_outlier(self) -> None:
        processor = FixedLagTrackPostProcessor(FixedLagTrackConfig(fps=10.0, delay_ms=500))
        rows = [
            (_track(0.0, 0.0), _debug()),
            (_track(10.0, 0.0), _debug()),
            (_track(20.0, 0.0), _debug()),
            (_track(30.0, 0.0), _debug()),
            (_missing(0.9), _debug("reject", "literature_prediction_outlier")),
            (_track(50.0, 0.0), _debug()),
        ]
        emitted = []
        for track, debug in rows:
            frame = processor.push(track, debug_record=debug)
            if frame is not None:
                emitted.append(frame)
        emitted.extend(processor.flush())

        self.assertEqual(emitted[4].track.ball_xy, [40.0, 0.0])
        self.assertEqual(emitted[4].source, "fixed_lag_single_smooth")

    def test_two_frame_gap_uses_hermite_reconstruction(self) -> None:
        processor = FixedLagTrackPostProcessor(FixedLagTrackConfig(fps=10.0, delay_ms=800))
        rows = [
            (_track(0.0, 0.0), _debug()),
            (_track(10.0, 0.0), _debug()),
            (_track(20.0, 0.0), _debug()),
            (_track(30.0, 0.0), _debug()),
            (_track(40.0, 0.0), _debug()),
            (_missing(0.5), _debug("reject", "candidate_failed_motion_gate")),
            (_missing(0.5), _debug("reject", "candidate_failed_motion_gate")),
            (_track(70.0, 0.0, 0.4), _debug()),
            (_track(80.0, 0.0), _debug()),
        ]
        emitted = []
        for track, debug in rows:
            frame = processor.push(track, debug_record=debug)
            if frame is not None:
                emitted.append(frame)
        emitted.extend(processor.flush())

        self.assertAlmostEqual(emitted[5].track.ball_xy[0], 50.0, places=5)
        self.assertAlmostEqual(emitted[6].track.ball_xy[0], 60.0, places=5)
        self.assertEqual(emitted[5].source, "fixed_lag_hermite")

    def test_impact_relock_backcasts_two_low_score_coast_frames(self) -> None:
        processor = FixedLagTrackPostProcessor(FixedLagTrackConfig(fps=10.0, delay_ms=500))
        rows = [
            (_track(0.0, 0.0), _debug()),
            (_track(10.0, 0.0), _debug()),
            (_track(20.0, 0.0, 0.10), _debug("coast", "parabola_prediction")),
            (_track(30.0, 0.0, 0.08), _debug("coast", "person_occlusion_prediction")),
            (_track(50.0, 0.0, 0.85), _debug("relock_accept", "impact_direction_change")),
            (_track(60.0, 0.0, 0.88), _debug()),
        ]
        emitted = []
        for track, debug in rows:
            frame = processor.push(track, debug_record=debug)
            if frame is not None:
                emitted.append(frame)
        emitted.extend(processor.flush())

        self.assertEqual(emitted[2].track.ball_xy, [30.0, 0.0])
        self.assertEqual(emitted[3].track.ball_xy, [40.0, 0.0])
        self.assertEqual(emitted[2].source, "fixed_lag_impact_backcast")

    def test_confirmed_occlusion_reset_removes_stale_branch(self) -> None:
        processor = FixedLagTrackPostProcessor(FixedLagTrackConfig(fps=10.0, delay_ms=800))
        rows = [
            (_track(0.0, 0.0, 0.2), _debug("coast", "person_occlusion_prediction")),
            (_track(1.0, 0.0, 0.1), _debug("coast", "person_occlusion_prediction")),
            (_track(2.0, 0.0, 0.6), _debug("accept", "person_occlusion_motion_gate")),
            (_track(3.0, 0.0, 0.1), _debug("coast", "person_occlusion_prediction")),
            (_track(50.0, 0.0, 0.9), _debug("relock_accept", "literature_occlusion_model_reset")),
            (_track(60.0, 0.0, 0.9), _debug()),
            (_track(70.0, 0.0, 0.9), _debug()),
        ]
        emitted = []
        for track, debug in rows:
            frame = processor.push(track, debug_record=debug)
            if frame is not None:
                emitted.append(frame)
        emitted.extend(processor.flush())

        self.assertTrue(all(not frame.track.visible for frame in emitted[:4]))
        self.assertTrue(all(frame.source == "fixed_lag_stale_branch_removed" for frame in emitted[:4]))

    def test_confirmed_bootstrap_recovers_matching_previous_candidate_only(self) -> None:
        processor = FixedLagTrackPostProcessor(FixedLagTrackConfig(fps=10.0, delay_ms=1000))
        processor.push(
            _missing(),
            candidates=[_track(9.0, 0.0, 0.3), _track(100.0, 100.0, 0.9)],
            debug_record=_debug("reject", "missing_or_low_confidence"),
        )
        rows = [
            (_track(10.0, 0.0), _debug("bootstrap_accept", "strong_confidence")),
            *[(_track(10.0 + index, 0.0), _debug()) for index in range(1, 8)],
        ]
        emitted = []
        for track, debug in rows:
            frame = processor.push(track, debug_record=debug)
            if frame is not None:
                emitted.append(frame)
        emitted.extend(processor.flush())

        self.assertTrue(emitted[0].track.visible)
        self.assertEqual(emitted[0].track.ball_xy, [9.0, 0.0])
        self.assertEqual(emitted[0].source, "fixed_lag_bootstrap_candidate")

    def test_top_exit_candidate_chain_recovers_confirmed_apex_only(self) -> None:
        processor = FixedLagTrackPostProcessor(FixedLagTrackConfig(fps=10.0, delay_ms=1000))
        rows = [
            (_track(100.0, 180.0), [], _debug(frame_w=640, frame_h=480)),
            (_track(110.0, 150.0), [], _debug(frame_w=640, frame_h=480)),
            (_track(120.0, 125.0), [], _debug(frame_w=640, frame_h=480)),
            (_track(130.0, 105.0), [], _debug(frame_w=640, frame_h=480)),
            (_missing(), [], _debug("top_exit_enter", "likely_top_exit", frame_w=640, frame_h=480)),
            (
                _missing(),
                [_track(145.0, 90.0, 0.25), _track(300.0, 350.0, 0.90)],
                _debug("top_exit_suppress", "active_top_exit_suppression", frame_w=640, frame_h=480),
            ),
            (
                _missing(),
                [_track(150.0, 84.0, 0.55)],
                _debug("top_exit_suppress", "active_top_exit_suppression", frame_w=640, frame_h=480),
            ),
            (_missing(), [], _debug("reject", "missing_or_low_confidence", frame_w=640, frame_h=480)),
            (
                _missing(),
                [_track(160.0, 88.0, 0.40)],
                _debug("bootstrap_wait", "waiting_for_candidate_confirmation", frame_w=640, frame_h=480),
            ),
            (
                _missing(),
                [_track(165.0, 98.0, 0.50)],
                _debug("bootstrap_wait", "waiting_for_candidate_confirmation", frame_w=640, frame_h=480),
            ),
        ]
        emitted = []
        for track, candidates, debug in rows:
            frame = processor.push(track, candidates=candidates, debug_record=debug)
            if frame is not None:
                emitted.append(frame)
        emitted.extend(processor.flush())

        self.assertTrue(all(frame.track.visible for frame in emitted[4:10]))
        self.assertEqual(emitted[5].track.ball_xy, [145.0, 90.0])
        self.assertEqual(emitted[7].source, "fixed_lag_top_apex_interpolation")
        self.assertNotEqual(emitted[5].track.ball_xy, [300.0, 350.0])

    def test_impact_relock_recovers_only_continuous_candidate_branch(self) -> None:
        processor = FixedLagTrackPostProcessor(FixedLagTrackConfig(fps=10.0, delay_ms=1000))
        rows = [
            (_track(50.0, 50.0), [] , _debug()),
            (_track(60.0, 50.0), [] , _debug()),
            (_missing(), [_track(130.0, 140.0, 0.60)], _debug("reject", "candidate_failed_motion_gate")),
            (_missing(), [], _debug("reject", "candidate_failed_motion_gate")),
            (_missing(), [_track(130.0, 82.0, 0.35)], _debug("reject", "candidate_failed_motion_gate")),
            (
                _missing(),
                [_track(150.0, 86.0, 0.40), _track(20.0, 400.0, 0.95)],
                _debug("reject", "candidate_failed_motion_gate"),
            ),
            (_missing(), [_track(170.0, 91.0, 0.45)], _debug("reject", "candidate_failed_motion_gate")),
            (_missing(), [_track(190.0, 96.0, 0.50)], _debug("reject", "candidate_failed_motion_gate")),
            (_track(210.0, 101.0, 0.80), [], _debug("relock_accept", "impact_direction_change")),
            (_track(230.0, 106.0, 0.85), [], _debug()),
        ]
        emitted = []
        for track, candidates, debug in rows:
            frame = processor.push(track, candidates=candidates, debug_record=debug)
            if frame is not None:
                emitted.append(frame)
        emitted.extend(processor.flush())

        self.assertFalse(emitted[2].track.visible)
        self.assertFalse(emitted[3].track.visible)
        self.assertEqual(
            [frame.track.ball_xy for frame in emitted[4:8]],
            [[130.0, 82.0], [150.0, 86.0], [170.0, 91.0], [190.0, 96.0]],
        )
        self.assertTrue(all(frame.source == "fixed_lag_relock_candidate" for frame in emitted[4:8]))

    def test_future_measurements_recover_matching_high_speed_candidate(self) -> None:
        processor = FixedLagTrackPostProcessor(FixedLagTrackConfig(fps=30.0, delay_ms=300))
        rows = [
            (_track(20.0, 20.0), [], _debug()),
            (
                _missing(),
                [_track(100.0, 50.0, 0.80), _track(300.0, 300.0, 0.95)],
                _debug("reject", "candidate_failed_motion_gate"),
            ),
            (_track(120.0, 55.0, 0.85), [], _debug()),
            (_track(140.0, 60.0, 0.88), [], _debug()),
        ]
        emitted = []
        for track, candidates, debug in rows:
            frame = processor.push(track, candidates=candidates, debug_record=debug)
            if frame is not None:
                emitted.append(frame)
        emitted.extend(processor.flush())

        self.assertEqual(emitted[1].track.ball_xy, [100.0, 50.0])
        self.assertEqual(emitted[1].source, "fixed_lag_future_candidate")

    def test_future_measurements_do_not_recover_inconsistent_candidate(self) -> None:
        processor = FixedLagTrackPostProcessor(FixedLagTrackConfig(fps=30.0, delay_ms=300))
        rows = [
            (_track(20.0, 20.0), [], _debug()),
            (
                _missing(),
                [_track(70.0, 150.0, 0.95)],
                _debug("reject", "candidate_failed_motion_gate"),
            ),
            (_track(120.0, 55.0, 0.85), [], _debug()),
            (_track(140.0, 60.0, 0.88), [], _debug()),
        ]
        emitted = []
        for track, candidates, debug in rows:
            frame = processor.push(track, candidates=candidates, debug_record=debug)
            if frame is not None:
                emitted.append(frame)
        emitted.extend(processor.flush())

        self.assertFalse(emitted[1].track.visible)

    def test_future_branch_recovers_low_confidence_post_impact_candidates(self) -> None:
        processor = FixedLagTrackPostProcessor(FixedLagTrackConfig(fps=30.0, delay_ms=300))
        rows = [
            (_track(0.0, 100.0, 0.9), [], _debug()),
            (_track(10.0, 100.0, 0.9), [], _debug()),
            (_track(20.0, 100.0, 0.9), [], _debug()),
            (
                _missing(0.48),
                [_track(31.0, 65.0, 0.48), _track(250.0, 250.0, 0.95)],
                _debug("reject", "candidate_failed_motion_gate"),
            ),
            (
                _missing(0.49),
                [_track(34.0, 35.0, 0.49)],
                _debug("reject", "candidate_failed_motion_gate"),
            ),
            (_track(37.0, 12.0, 0.90), [], _debug("relock_accept", "high_score_fast_relock")),
            (_track(40.0, 1.0, 0.85), [], _debug()),
        ]
        emitted = []
        for track, candidates, debug in rows:
            frame = processor.push(track, candidates=candidates, debug_record=debug)
            if frame is not None:
                emitted.append(frame)
        emitted.extend(processor.flush())

        self.assertEqual(emitted[3].track.ball_xy, [31.0, 65.0])
        self.assertEqual(emitted[4].track.ball_xy, [34.0, 35.0])
        self.assertTrue(
            all(
                frame.source == "fixed_lag_rejected_branch_candidate"
                for frame in emitted[3:5]
            )
        )

    def test_two_sided_short_visibility_gap_prefers_weak_raw_candidate(self) -> None:
        processor = FixedLagTrackPostProcessor(FixedLagTrackConfig(fps=30.0, delay_ms=300))
        rows = [
            (_track(0.0, 50.0, 0.8), [], _debug()),
            (_track(10.0, 50.0, 0.8), [], _debug()),
            (_track(20.0, 50.0, 0.8), [], _debug()),
            (
                _missing(0.24),
                [_track(31.0, 50.0, 0.24), _track(200.0, 200.0, 0.9)],
                _debug("reject", "missing_or_low_confidence"),
            ),
            (_missing(0.15), [], _debug("reject", "missing_or_low_confidence")),
            (_track(50.0, 50.0, 0.8), [], _debug()),
            (_track(60.0, 50.0, 0.8), [], _debug()),
        ]
        emitted = []
        for track, candidates, debug in rows:
            frame = processor.push(track, candidates=candidates, debug_record=debug)
            if frame is not None:
                emitted.append(frame)
        emitted.extend(processor.flush())

        self.assertEqual(emitted[3].track.ball_xy, [31.0, 50.0])
        self.assertEqual(emitted[3].source, "fixed_lag_weak_visibility_candidate")
        self.assertEqual(emitted[4].track.ball_xy, [40.0, 50.0])
        self.assertEqual(emitted[4].source, "fixed_lag_short_visibility_gap")

    def test_future_candidate_does_not_seed_neighbor_interpolation(self) -> None:
        processor = FixedLagTrackPostProcessor(FixedLagTrackConfig(fps=30.0, delay_ms=300))
        rows = [
            (_track(0.0, 50.0), [], _debug()),
            (_track(10.0, 50.0), [], _debug()),
            (_track(20.0, 50.0), [], _debug()),
            (_missing(), [], _debug("reject", "missing_or_low_confidence")),
            (
                _missing(),
                [_track(100.0, 50.0, 0.80)],
                _debug("reject", "candidate_failed_motion_gate"),
            ),
            (_track(120.0, 55.0, 0.85), [], _debug()),
            (_track(140.0, 60.0, 0.88), [], _debug()),
        ]
        emitted = []
        for track, candidates, debug in rows:
            frame = processor.push(track, candidates=candidates, debug_record=debug)
            if frame is not None:
                emitted.append(frame)
        emitted.extend(processor.flush())

        self.assertFalse(emitted[3].track.visible)
        self.assertEqual(emitted[4].source, "fixed_lag_future_candidate")


if __name__ == "__main__":
    unittest.main()
