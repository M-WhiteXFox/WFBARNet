from __future__ import annotations

import unittest
from types import SimpleNamespace

from src.postprocess.adaptive_track import (
    CANDIDATE_GRAPH_TRACK_ROUTE,
    CONTEXTUAL_TRACK_ROUTE,
    AdaptiveTrackPostProcessor,
    resolve_track_route,
)
from src.postprocess.fixed_lag_track import FixedLagTrackConfig
from src.runners.tracknet_realtime_runner import (
    _frame_step_seconds,
    _reset_filter_state_preserving_debug,
)
from src.runners.unified_runner import _pose_bboxes
from src.utils.structures import TrackResult


class _FakeTrackFilter:
    def __init__(self) -> None:
        self.debug_records = [{"frame_index": 1}]
        self.reset_called = False

    def reset(self) -> None:
        self.reset_called = True
        self.debug_records.clear()


class _FakeContextTrackFilter:
    def __init__(self) -> None:
        self.debug_records: list[dict[str, object]] = []
        self.person_bboxes = None

    def reset(self) -> None:
        self.debug_records.clear()

    def update_candidates(self, tracks, **values):  # noqa: ANN001
        self.person_bboxes = values.get("person_bboxes")
        self.debug_records.append({"action": "accept", "reason": "test"})
        return tracks[0]

    def last_debug_record(self) -> dict[str, object]:
        return self.debug_records[-1]


class TrackRunnerContextTest(unittest.TestCase):
    def test_auto_route_is_stable_for_available_context(self) -> None:
        self.assertEqual(
            resolve_track_route("auto", reliable_context=False),
            CANDIDATE_GRAPH_TRACK_ROUTE,
        )
        self.assertEqual(
            resolve_track_route("auto", reliable_context=True),
            CONTEXTUAL_TRACK_ROUTE,
        )

    def test_candidate_graph_route_preserves_payload_order(self) -> None:
        processor = AdaptiveTrackPostProcessor(fps=10.0, reliable_context=False)
        emitted = []
        for frame_id in range(5):
            emitted.extend(
                processor.push(
                    [
                        TrackResult(
                            ball_xy=[100.0 + frame_id * 10.0, 200.0],
                            visible=1,
                            score=0.95,
                        )
                    ],
                    payload={"frame_id": frame_id},
                )
            )
        emitted.extend(processor.flush())

        self.assertEqual(processor.route, CANDIDATE_GRAPH_TRACK_ROUTE)
        self.assertEqual([item.payload["frame_id"] for item in emitted], list(range(5)))
        self.assertTrue(all(item.track.visible for item in emitted))

    def test_context_route_forwards_person_boxes(self) -> None:
        track_filter = _FakeContextTrackFilter()
        processor = AdaptiveTrackPostProcessor(
            fps=25.0,
            reliable_context=True,
            track_filter=track_filter,
            fixed_lag_config=FixedLagTrackConfig(delay_ms=0),
        )
        boxes = [(10.0, 20.0, 80.0, 180.0)]

        emitted = processor.push(
            [TrackResult(ball_xy=[30.0, 40.0], visible=1, score=0.9)],
            person_bboxes=boxes,
            payload={"frame_id": 0},
        )

        self.assertEqual(processor.route, CONTEXTUAL_TRACK_ROUTE)
        self.assertEqual(track_filter.person_bboxes, boxes)
        self.assertEqual(emitted[0].track.ball_xy, [30.0, 40.0])

    def test_realtime_step_uses_capture_elapsed_time(self) -> None:
        self.assertAlmostEqual(_frame_step_seconds(10.18, 10.00, 60.0), 0.18)
        self.assertAlmostEqual(_frame_step_seconds(10.00, None, 50.0), 0.02)

    def test_filter_reset_preserves_accumulated_debug_records(self) -> None:
        track_filter = _FakeTrackFilter()

        _reset_filter_state_preserving_debug(track_filter)

        self.assertTrue(track_filter.reset_called)
        self.assertEqual(track_filter.debug_records, [{"frame_index": 1}])

    def test_unified_runner_forwards_valid_pose_boxes(self) -> None:
        poses = [
            SimpleNamespace(bbox=[10.0, 20.0, 80.0, 180.0]),
            SimpleNamespace(bbox=[30.0, 40.0, 20.0, 90.0]),
            SimpleNamespace(bbox=[]),
        ]

        self.assertEqual(_pose_bboxes(poses), [(10.0, 20.0, 80.0, 180.0)])


if __name__ == "__main__":
    unittest.main()
