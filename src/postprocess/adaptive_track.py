from __future__ import annotations

from dataclasses import replace
from typing import Any, Sequence

from src.postprocess.candidate_graph_track import (
    CandidateGraphConfig,
    CandidateGraphDecision,
    FixedLagCandidateGraph,
)
from src.postprocess.fixed_lag_track import (
    FixedLagTrackConfig,
    FixedLagTrackFrame,
    FixedLagTrackPostProcessor,
)
from src.postprocess.track_filter import FrameShape, PersonBBoxes, TrackFilterAlgorithm
from src.postprocess.track_output import TrackFrameOutput, copy_track_result
from src.postprocess.tracknet_v3_filter import create_tracknet_v3_ball_track_filter
from src.utils.structures import TrackResult


AUTO_TRACK_ROUTE = "auto"
CANDIDATE_GRAPH_TRACK_ROUTE = "candidate_graph"
CONTEXTUAL_TRACK_ROUTE = "contextual"
TRACK_ROUTES = {
    AUTO_TRACK_ROUTE,
    CANDIDATE_GRAPH_TRACK_ROUTE,
    CONTEXTUAL_TRACK_ROUTE,
}


def resolve_track_route(route: str, *, reliable_context: bool) -> str:
    if route not in TRACK_ROUTES:
        raise ValueError(f"unknown track postprocess route: {route}")
    if route == AUTO_TRACK_ROUTE:
        return CONTEXTUAL_TRACK_ROUTE if reliable_context else CANDIDATE_GRAPH_TRACK_ROUTE
    return route


class AdaptiveTrackPostProcessor:
    """Route a complete stream through one stable measured-track pipeline."""

    def __init__(
        self,
        *,
        fps: float,
        route: str = AUTO_TRACK_ROUTE,
        reliable_context: bool = False,
        debug_enabled: bool = True,
        track_filter: TrackFilterAlgorithm | None = None,
        fixed_lag_config: FixedLagTrackConfig | None = None,
        graph_config: CandidateGraphConfig | None = None,
    ) -> None:
        self.route = resolve_track_route(route, reliable_context=reliable_context)
        self._track_filter: TrackFilterAlgorithm | None = None
        self._fixed_lag: FixedLagTrackPostProcessor | None = None
        self._graph: FixedLagCandidateGraph | None = None
        if self.route == CANDIDATE_GRAPH_TRACK_ROUTE:
            config = (
                replace(graph_config, fps=float(fps))
                if graph_config is not None
                else CandidateGraphConfig(fps=float(fps))
            )
            self._graph = FixedLagCandidateGraph(config)
            return
        self._track_filter = track_filter or create_tracknet_v3_ball_track_filter(
            fps=fps,
            debug_enabled=debug_enabled,
        )
        config = (
            replace(fixed_lag_config, fps=float(fps))
            if fixed_lag_config is not None
            else FixedLagTrackConfig(fps=float(fps))
        )
        self._fixed_lag = FixedLagTrackPostProcessor(config)

    @property
    def delay_frames(self) -> int:
        if self._graph is not None:
            return self._graph.delay_frames
        assert self._fixed_lag is not None
        return self._fixed_lag.delay_frames

    @property
    def pending_count(self) -> int:
        if self._graph is not None:
            return self._graph.pending_count
        assert self._fixed_lag is not None
        return self._fixed_lag.pending_count

    def reset(self) -> None:
        if self._graph is not None:
            self._graph.reset()
            return
        assert self._track_filter is not None and self._fixed_lag is not None
        self._track_filter.reset()
        self._fixed_lag.reset()

    def push(
        self,
        candidates: Sequence[TrackResult],
        *,
        dt: float | None = None,
        frame_shape: FrameShape = None,
        court_prediction: Any | None = None,
        person_bboxes: PersonBBoxes = None,
        payload: Any = None,
    ) -> list[TrackFrameOutput]:
        if self._graph is not None:
            return [
                self._from_graph(decision)
                for decision in self._graph.push(candidates, payload=payload)
            ]

        assert self._track_filter is not None and self._fixed_lag is not None
        track = self._track_filter.update_candidates(
            candidates,
            dt=dt,
            frame_shape=frame_shape,
            court_prediction=court_prediction,
            person_bboxes=person_bboxes,
        )
        lagged = self._fixed_lag.push(
            track,
            candidates=candidates,
            debug_record=self._track_filter.last_debug_record(),
            payload=payload,
        )
        self._track_filter.debug_records.clear()
        return [self._from_context(lagged)] if lagged is not None else []

    def flush_one(self) -> TrackFrameOutput | None:
        if self._graph is not None:
            decision = self._graph.flush_one()
            return self._from_graph(decision) if decision is not None else None
        assert self._fixed_lag is not None
        lagged = self._fixed_lag.flush_one()
        return self._from_context(lagged) if lagged is not None else None

    def flush(self) -> list[TrackFrameOutput]:
        if self._graph is not None:
            return [self._from_graph(decision) for decision in self._graph.flush()]
        assert self._fixed_lag is not None
        return [self._from_context(frame) for frame in self._fixed_lag.flush()]

    @staticmethod
    def _from_graph(decision: CandidateGraphDecision) -> TrackFrameOutput:
        return TrackFrameOutput(
            frame_index=int(decision.frame_index),
            measured_track=copy_track_result(decision.track),
            measured_source=str(decision.source),
            candidate_rank=int(decision.candidate_rank),
            payload=decision.payload,
            debug_record=(
                dict(decision.debug_record)
                if isinstance(decision.debug_record, dict)
                else None
            ),
        )

    @staticmethod
    def _from_context(frame: FixedLagTrackFrame) -> TrackFrameOutput:
        return TrackFrameOutput(
            frame_index=int(frame.frame_index),
            measured_track=copy_track_result(frame.track),
            measured_source=str(frame.source),
            payload=frame.payload,
            debug_record=(
                dict(frame.debug_record)
                if isinstance(frame.debug_record, dict)
                else None
            ),
        )
