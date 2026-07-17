from __future__ import annotations

import argparse
from collections import Counter, deque
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_labeled_trajectory_filter import (
    _load_cache,
    _point_from_track,
    _track_from_cache,
)
from scripts.evaluate_tracknetv3_raw_postprocess import _candidate_oracle, _load_gt, _metrics
from src.utils.structures import TrackResult


Point = tuple[float, float]


@dataclass(slots=True)
class CandidateGraphConfig:
    fps: float = 25.0
    delay_ms: int = 300
    beam_width: int = 64
    min_candidate_score: float = 0.20
    score_center: float = 0.72
    score_weight: float = 10.0
    rank_cost: float = 0.08
    start_cost: float = 4.0
    stop_cost: float = 2.0
    continue_reward: float = 1.5
    reconnect_cost: float = 1.2
    gap_frame_cost: float = 0.5
    max_reconnect_gap_frames: int = 2
    speed_free_px: float = 45.0
    speed_scale_px: float = 45.0
    speed_cost_cap: float = 3.0
    acceleration_free_px: float = 30.0
    acceleration_scale_px: float = 45.0
    impact_reset_cost: float = 1.2

    @property
    def delay_frames(self) -> int:
        fps = self.fps if self.fps > 0.0 else 25.0
        return max(0, int(round(fps * max(0, int(self.delay_ms)) / 1000.0)))


@dataclass(slots=True)
class CandidateGraphDecision:
    frame_index: int
    track: TrackResult
    source: str
    candidate_rank: int


@dataclass(slots=True)
class CandidateGraphRefinementConfig:
    max_bridge_frames: int = 9
    bridge_candidate_min_score: float = 0.20
    bridge_candidate_tolerance_px: float = 30.0
    bridge_min_support_ratio: float = 0.50
    bridge_max_endpoint_step_px: float = 100.0
    interpolation_max_gap_frames: int = 0
    interpolation_max_step_px: float = 45.0
    continuation_min_step_px: float = 12.0
    continuation_max_step_px: float = 100.0
    continuation_max_acceleration_px: float = 24.0
    continuation_min_direction_cosine: float = 0.50
    continuation_score_scale: float = 0.50
    current_proposal_tolerance_px: float = 30.0
    current_proposal_min_support_ratio: float = 0.50
    edge_candidate_tolerance_px: float = 30.0
    edge_boundary_ratio: float = 0.025
    zigzag_min_step_px: float = 18.0
    zigzag_min_deviation_px: float = 24.0
    zigzag_reverse_cosine: float = -0.15
    zigzag_resume_cosine: float = 0.50
    static_veto_enabled: bool = False
    raw_bridge_enabled: bool = False
    continuation_enabled: bool = False
    current_proposal_enabled: bool = False
    boundary_recovery_enabled: bool = False
    boundary_requires_current_proposal: bool = True
    zigzag_veto_enabled: bool = False


@dataclass(slots=True)
class _Frame:
    frame_index: int
    candidates: tuple[TrackResult, ...]


@dataclass(slots=True)
class _BeamPath:
    cost: float
    choices: tuple[int, ...]
    visible_history: tuple[tuple[int, Point], ...]
    previous_visible: bool


def _point(track: TrackResult | None) -> Point | None:
    if track is None or not track.visible or len(track.ball_xy) < 2:
        return None
    x, y = float(track.ball_xy[0]), float(track.ball_xy[1])
    if not (math.isfinite(x) and math.isfinite(y)) or x < 0.0 or y < 0.0:
        return None
    return x, y


def _copy_track(track: TrackResult) -> TrackResult:
    return TrackResult(
        ball_xy=[float(value) for value in track.ball_xy[:2]],
        visible=int(bool(track.visible)),
        score=float(track.score),
        heatmap_shape=[int(value) for value in track.heatmap_shape],
    )


def _invisible_track(candidates: Sequence[TrackResult]) -> TrackResult:
    best_score = max((float(candidate.score) for candidate in candidates), default=0.0)
    heatmap_shape = list(candidates[0].heatmap_shape) if candidates else []
    return TrackResult(
        ball_xy=[-1.0, -1.0],
        visible=0,
        score=best_score,
        heatmap_shape=heatmap_shape,
    )


class FixedLagCandidateGraph:
    """Select raw candidates with a bounded-future, soft-cost trajectory graph."""

    def __init__(self, config: CandidateGraphConfig | None = None) -> None:
        self.config = config or CandidateGraphConfig()
        self._buffer: deque[_Frame] = deque()
        self._committed_visible_history: tuple[tuple[int, Point], ...] = ()
        self._committed_previous_visible = False
        self._next_frame_index = 0

    def select_sequence(
        self,
        frames: Sequence[Sequence[TrackResult]],
    ) -> list[CandidateGraphDecision]:
        decisions: list[CandidateGraphDecision] = []
        for candidates in frames:
            decisions.extend(self.push(candidates))
        decisions.extend(self.flush())
        return decisions

    def push(self, candidates: Sequence[TrackResult]) -> list[CandidateGraphDecision]:
        visible_candidates = tuple(
            _copy_track(candidate)
            for candidate in candidates
            if _point(candidate) is not None
            and float(candidate.score) >= float(self.config.min_candidate_score)
        )
        self._buffer.append(
            _Frame(
                frame_index=self._next_frame_index,
                candidates=visible_candidates,
            )
        )
        self._next_frame_index += 1
        if len(self._buffer) <= self.config.delay_frames:
            return []
        return [self._emit_oldest(final=False)]

    def flush(self) -> list[CandidateGraphDecision]:
        decisions: list[CandidateGraphDecision] = []
        while self._buffer:
            decisions.append(self._emit_oldest(final=True))
        return decisions

    def _emit_oldest(self, *, final: bool) -> CandidateGraphDecision:
        frames = list(self._buffer)
        choices = self._decode(frames, final=final)
        frame = self._buffer.popleft()
        choice = choices[0]
        if choice < 0:
            self._committed_previous_visible = False
            return CandidateGraphDecision(
                frame_index=frame.frame_index,
                track=_invisible_track(frame.candidates),
                source="candidate_graph_null",
                candidate_rank=0,
            )

        candidate = frame.candidates[choice]
        point = _point(candidate)
        assert point is not None
        self._committed_visible_history = (
            *self._committed_visible_history,
            (frame.frame_index, point),
        )[-2:]
        self._committed_previous_visible = True
        return CandidateGraphDecision(
            frame_index=frame.frame_index,
            track=_copy_track(candidate),
            source="candidate_graph_candidate",
            candidate_rank=choice + 1,
        )

    def _decode(self, frames: Sequence[_Frame], *, final: bool) -> tuple[int, ...]:
        beam = [
            _BeamPath(
                cost=0.0,
                choices=(),
                visible_history=self._committed_visible_history,
                previous_visible=self._committed_previous_visible,
            )
        ]
        beam_width = max(1, int(self.config.beam_width))
        for frame in frames:
            extended: list[_BeamPath] = []
            choices = range(-1, len(frame.candidates))
            for path in beam:
                for choice in choices:
                    candidate = frame.candidates[choice] if choice >= 0 else None
                    point = _point(candidate) if candidate is not None else None
                    cost = path.cost + self._node_cost(candidate, choice)
                    cost += self._transition_cost(
                        path,
                        frame_index=frame.frame_index,
                        point=point,
                    )
                    history = path.visible_history
                    if point is not None:
                        history = (*history, (frame.frame_index, point))[-2:]
                    extended.append(
                        _BeamPath(
                            cost=cost,
                            choices=(*path.choices, choice),
                            visible_history=history,
                            previous_visible=point is not None,
                        )
                    )
            extended.sort(key=lambda path: (path.cost, path.choices))
            beam = extended[:beam_width]

        if final:
            for path in beam:
                if path.previous_visible:
                    path.cost += float(self.config.stop_cost)
            beam.sort(key=lambda path: (path.cost, path.choices))
        return beam[0].choices

    def _node_cost(self, candidate: TrackResult | None, choice: int) -> float:
        if candidate is None:
            return 0.0
        return (
            float(self.config.score_weight)
            * (float(self.config.score_center) - float(candidate.score))
            + float(self.config.rank_cost) * max(0, choice)
        )

    def _transition_cost(
        self,
        path: _BeamPath,
        *,
        frame_index: int,
        point: Point | None,
    ) -> float:
        if point is None:
            return float(self.config.stop_cost) if path.previous_visible else 0.0

        if path.previous_visible:
            return -float(self.config.continue_reward) + self._motion_cost(
                path.visible_history,
                frame_index,
                point,
            )

        if path.visible_history:
            last_frame = path.visible_history[-1][0]
            gap_frames = frame_index - last_frame - 1
            if gap_frames <= max(0, int(self.config.max_reconnect_gap_frames)):
                return (
                    float(self.config.reconnect_cost)
                    + max(0, gap_frames) * float(self.config.gap_frame_cost)
                    + self._motion_cost(path.visible_history, frame_index, point)
                )
        return float(self.config.start_cost)

    def _motion_cost(
        self,
        history: tuple[tuple[int, Point], ...],
        frame_index: int,
        point: Point,
    ) -> float:
        if not history:
            return 0.0
        last_frame, last_point = history[-1]
        frame_gap = max(1, frame_index - last_frame)
        velocity = (
            (point[0] - last_point[0]) / frame_gap,
            (point[1] - last_point[1]) / frame_gap,
        )
        speed = math.hypot(*velocity)
        speed_cost = min(
            max(0.0, speed - float(self.config.speed_free_px))
            / max(1.0, float(self.config.speed_scale_px)),
            max(0.0, float(self.config.speed_cost_cap)),
        )
        if len(history) < 2:
            return speed_cost

        earlier_frame, earlier_point = history[-2]
        earlier_gap = max(1, last_frame - earlier_frame)
        earlier_velocity = (
            (last_point[0] - earlier_point[0]) / earlier_gap,
            (last_point[1] - earlier_point[1]) / earlier_gap,
        )
        acceleration = math.dist(velocity, earlier_velocity)
        acceleration_cost = min(
            max(0.0, acceleration - float(self.config.acceleration_free_px))
            / max(1.0, float(self.config.acceleration_scale_px)),
            max(0.0, float(self.config.impact_reset_cost)),
        )
        return speed_cost + acceleration_cost


def refine_candidate_graph_sequence(
    frames: Sequence[Sequence[TrackResult]],
    decisions: Sequence[CandidateGraphDecision],
    *,
    width: int,
    height: int,
    config: CandidateGraphRefinementConfig | None = None,
    static_veto_frames: Sequence[int] = (),
    current_proposals: Sequence[TrackResult] = (),
    current_proposal_allowed: Sequence[bool] = (),
) -> list[CandidateGraphDecision]:
    """Repair supported null runs, then remove context-confirmed false points."""

    if len(frames) != len(decisions):
        raise ValueError(f"frame/decision mismatch: {len(frames)} != {len(decisions)}")
    resolved = [
        CandidateGraphDecision(
            frame_index=decision.frame_index,
            track=_copy_track(decision.track),
            source=decision.source,
            candidate_rank=decision.candidate_rank,
        )
        for decision in decisions
    ]
    cfg = config or CandidateGraphRefinementConfig()
    if current_proposals and len(current_proposals) != len(frames):
        raise ValueError(
            f"frame/current proposal mismatch: {len(frames)} != {len(current_proposals)}"
        )
    if current_proposal_allowed and len(current_proposal_allowed) != len(frames):
        raise ValueError(
            "frame/current proposal eligibility mismatch: "
            f"{len(frames)} != {len(current_proposal_allowed)}"
        )
    allowed = (
        list(current_proposal_allowed)
        if current_proposal_allowed
        else [True] * len(frames)
    )
    scale = max(float(width) / 1280.0, float(height) / 720.0, 1e-6)
    if cfg.raw_bridge_enabled:
        _bridge_anchored_null_runs(frames, resolved, cfg, scale)
    if cfg.current_proposal_enabled and current_proposals:
        _recover_current_proposal_runs(
            frames,
            resolved,
            current_proposals,
            allowed,
            cfg,
            scale,
        )
    if cfg.boundary_recovery_enabled:
        _recover_boundary_continuations(
            frames,
            resolved,
            cfg,
            scale,
            width,
            height,
            current_proposals,
            allowed,
        )
    if cfg.continuation_enabled:
        _recover_fast_candidate_free_continuations(
            frames, resolved, cfg, scale, width, height
        )
    if cfg.static_veto_enabled:
        for frame_index in static_veto_frames:
            if (
                0 <= int(frame_index) < len(resolved)
                and _point(resolved[int(frame_index)].track) is not None
            ):
                resolved[int(frame_index)] = CandidateGraphDecision(
                    frame_index=int(frame_index),
                    track=_invisible_track(frames[int(frame_index)]),
                    source="candidate_graph_static_veto",
                    candidate_rank=0,
                )
    if cfg.zigzag_veto_enabled:
        _remove_isolated_zigzags(frames, resolved, cfg, scale)
    return resolved


def _bridge_anchored_null_runs(
    frames: Sequence[Sequence[TrackResult]],
    decisions: list[CandidateGraphDecision],
    config: CandidateGraphRefinementConfig,
    scale: float,
) -> None:
    index = 0
    while index < len(decisions):
        if _point(decisions[index].track) is not None:
            index += 1
            continue
        start = index
        while index < len(decisions) and _point(decisions[index].track) is None:
            index += 1
        end = index - 1
        run_length = end - start + 1
        left_index = start - 1
        right_index = end + 1
        if (
            run_length > max(0, int(config.max_bridge_frames))
            or left_index < 0
            or right_index >= len(decisions)
        ):
            continue
        left = _point(decisions[left_index].track)
        right = _point(decisions[right_index].track)
        if left is None or right is None:
            continue
        span = right_index - left_index
        if math.dist(left, right) / span > float(config.bridge_max_endpoint_step_px) * scale:
            continue

        matches: dict[int, tuple[int, TrackResult]] = {}
        tolerance = float(config.bridge_candidate_tolerance_px) * scale
        for frame_index in range(start, right_index):
            t = (frame_index - left_index) / span
            expected = (
                left[0] + (right[0] - left[0]) * t,
                left[1] + (right[1] - left[1]) * t,
            )
            candidates = [
                (rank, candidate)
                for rank, candidate in enumerate(frames[frame_index], start=1)
                if float(candidate.score) >= float(config.bridge_candidate_min_score)
                and _point(candidate) is not None
                and math.dist(_point(candidate) or expected, expected) <= tolerance
            ]
            if candidates:
                matches[frame_index] = min(
                    candidates,
                    key=lambda item: (
                        math.dist(_point(item[1]) or expected, expected),
                        -float(item[1].score),
                        item[0],
                    ),
                )

        required_support = (
            run_length
            if run_length <= 2
            else max(1, int(math.ceil(run_length * float(config.bridge_min_support_ratio))))
        )
        no_measurement_short_gap = (
            run_length <= max(0, int(config.interpolation_max_gap_frames))
            and all(not any(_point(candidate) is not None for candidate in frames[pos]) for pos in range(start, right_index))
            and _outer_directions_compatible(decisions, left_index, right_index)
        )
        if len(matches) < required_support and not no_measurement_short_gap:
            continue
        for frame_index, (rank, candidate) in matches.items():
            decisions[frame_index] = CandidateGraphDecision(
                frame_index=frame_index,
                track=_copy_track(candidate),
                source="candidate_graph_bridge_candidate",
                candidate_rank=rank,
            )
        _interpolate_candidate_free_gaps(
            frames,
            decisions,
            start=start,
            end=end,
            config=config,
            scale=scale,
        )


def _interpolate_candidate_free_gaps(
    frames: Sequence[Sequence[TrackResult]],
    decisions: list[CandidateGraphDecision],
    *,
    start: int,
    end: int,
    config: CandidateGraphRefinementConfig,
    scale: float,
) -> None:
    index = start
    while index <= end:
        if _point(decisions[index].track) is not None:
            index += 1
            continue
        gap_start = index
        while index <= end and _point(decisions[index].track) is None:
            index += 1
        gap_end = index - 1
        gap_length = gap_end - gap_start + 1
        left_index = gap_start - 1
        right_index = gap_end + 1
        if (
            gap_length > max(0, int(config.interpolation_max_gap_frames))
            or left_index < 0
            or right_index >= len(decisions)
            or any(any(_point(candidate) is not None for candidate in frames[pos]) for pos in range(gap_start, right_index))
        ):
            continue
        left = _point(decisions[left_index].track)
        right = _point(decisions[right_index].track)
        if left is None or right is None:
            continue
        span = right_index - left_index
        if math.dist(left, right) / span > float(config.interpolation_max_step_px) * scale:
            continue
        for frame_index in range(gap_start, right_index):
            t = (frame_index - left_index) / span
            point = (
                left[0] + (right[0] - left[0]) * t,
                left[1] + (right[1] - left[1]) * t,
            )
            decisions[frame_index] = CandidateGraphDecision(
                frame_index=frame_index,
                track=TrackResult(
                    ball_xy=[point[0], point[1]],
                    visible=1,
                    score=min(
                        float(decisions[left_index].track.score),
                        float(decisions[right_index].track.score),
                    )
                    * 0.5,
                    heatmap_shape=list(decisions[left_index].track.heatmap_shape),
                ),
                source="candidate_graph_interpolation",
                candidate_rank=0,
            )


def _recover_current_proposal_runs(
    frames: Sequence[Sequence[TrackResult]],
    decisions: list[CandidateGraphDecision],
    current_proposals: Sequence[TrackResult],
    current_proposal_allowed: Sequence[bool],
    config: CandidateGraphRefinementConfig,
    scale: float,
) -> None:
    index = 0
    while index < len(decisions):
        if _point(decisions[index].track) is not None:
            index += 1
            continue
        start = index
        while index < len(decisions) and _point(decisions[index].track) is None:
            index += 1
        end = index - 1
        run_length = end - start + 1
        if run_length > max(0, int(config.max_bridge_frames)):
            continue
        expected = _proposal_expectations(decisions, start, end)
        if not expected:
            continue
        tolerance = float(config.current_proposal_tolerance_px) * scale
        matches: list[int] = []
        for frame_index in range(start, end + 1):
            if not current_proposal_allowed[frame_index]:
                continue
            proposal = current_proposals[frame_index]
            proposal_point = _point(proposal)
            expected_point = expected.get(frame_index)
            if (
                proposal_point is not None
                and expected_point is not None
                and math.dist(proposal_point, expected_point) <= tolerance
            ):
                matches.append(frame_index)
        required = max(
            1,
            int(
                math.ceil(
                    run_length * float(config.current_proposal_min_support_ratio)
                )
            ),
        )
        if len(matches) < required:
            continue
        for frame_index in matches:
            proposal = current_proposals[frame_index]
            measured = any(
                (candidate_point := _point(candidate)) is not None
                and (proposal_point := _point(proposal)) is not None
                and math.dist(candidate_point, proposal_point) <= 1e-6
                for candidate in frames[frame_index]
            )
            if run_length == 1 and not measured:
                continue
            decisions[frame_index] = CandidateGraphDecision(
                frame_index=frame_index,
                track=_copy_track(proposal),
                source="candidate_graph_current_proposal",
                candidate_rank=0,
            )


def _proposal_expectations(
    decisions: Sequence[CandidateGraphDecision],
    start: int,
    end: int,
) -> dict[int, Point]:
    left_index = start - 1
    right_index = end + 1
    left = _point(decisions[left_index].track) if left_index >= 0 else None
    right = _point(decisions[right_index].track) if right_index < len(decisions) else None
    expected: dict[int, Point] = {}
    if left is not None and right is not None:
        span = right_index - left_index
        for frame_index in range(start, right_index):
            t = (frame_index - left_index) / span
            expected[frame_index] = (
                left[0] + (right[0] - left[0]) * t,
                left[1] + (right[1] - left[1]) * t,
            )
        return expected
    if right is not None and right_index + 1 < len(decisions):
        following = _point(decisions[right_index + 1].track)
        if following is not None:
            velocity = (following[0] - right[0], following[1] - right[1])
            for frame_index in range(start, right_index):
                distance = right_index - frame_index
                expected[frame_index] = (
                    right[0] - velocity[0] * distance,
                    right[1] - velocity[1] * distance,
                )
            return expected
    if left is not None and left_index >= 1:
        earlier = _point(decisions[left_index - 1].track)
        if earlier is not None:
            velocity = (left[0] - earlier[0], left[1] - earlier[1])
            for frame_index in range(start, end + 1):
                distance = frame_index - left_index
                expected[frame_index] = (
                    left[0] + velocity[0] * distance,
                    left[1] + velocity[1] * distance,
                )
    return expected


def _outer_directions_compatible(
    decisions: Sequence[CandidateGraphDecision],
    left_index: int,
    right_index: int,
) -> bool:
    if left_index < 1 or right_index + 1 >= len(decisions):
        return False
    before = _point(decisions[left_index - 1].track)
    left = _point(decisions[left_index].track)
    right = _point(decisions[right_index].track)
    after = _point(decisions[right_index + 1].track)
    if before is None or left is None or right is None or after is None:
        return False
    incoming = (left[0] - before[0], left[1] - before[1])
    span = (right[0] - left[0], right[1] - left[1])
    outgoing = (after[0] - right[0], after[1] - right[1])
    return _cosine(incoming, span) >= 0.25 and _cosine(span, outgoing) >= 0.25


def _recover_boundary_continuations(
    frames: Sequence[Sequence[TrackResult]],
    decisions: list[CandidateGraphDecision],
    config: CandidateGraphRefinementConfig,
    scale: float,
    width: int,
    height: int,
    current_proposals: Sequence[TrackResult],
    current_proposal_allowed: Sequence[bool],
) -> None:
    snapshot = [decision.track for decision in decisions]
    tolerance = float(config.edge_candidate_tolerance_px) * scale
    boundary_x = float(config.edge_boundary_ratio) * width
    boundary_y = float(config.edge_boundary_ratio) * height
    for frame_index in range(2, len(decisions)):
        if _point(snapshot[frame_index]) is not None:
            continue
        earlier = _point(snapshot[frame_index - 2])
        previous = _point(snapshot[frame_index - 1])
        if earlier is None or previous is None:
            continue
        predicted = (
            previous[0] + (previous[0] - earlier[0]),
            previous[1] + (previous[1] - earlier[1]),
        )
        predicted_outside = not (0.0 <= predicted[0] < width and 0.0 <= predicted[1] < height)
        candidates = [
            (rank, candidate)
            for rank, candidate in enumerate(frames[frame_index], start=1)
            if float(candidate.score) >= float(config.bridge_candidate_min_score)
            and (point := _point(candidate)) is not None
            and (
                point[0] <= boundary_x
                or point[0] >= width - boundary_x
                or point[1] <= boundary_y
                or point[1] >= height - boundary_y
            )
            and math.dist(point, predicted) <= tolerance
        ]
        if not predicted_outside or not candidates:
            continue
        rank, candidate = min(
            candidates,
            key=lambda item: (math.dist(_point(item[1]) or predicted, predicted), -float(item[1].score), item[0]),
        )
        if config.boundary_requires_current_proposal:
            if not current_proposals or not current_proposal_allowed[frame_index]:
                continue
            proposal_point = _point(current_proposals[frame_index])
            candidate_point = _point(candidate)
            if (
                proposal_point is None
                or candidate_point is None
                or math.dist(proposal_point, candidate_point) > 1e-6
            ):
                continue
        decisions[frame_index] = CandidateGraphDecision(
            frame_index=frame_index,
            track=_copy_track(candidate),
            source="candidate_graph_boundary_candidate",
            candidate_rank=rank,
        )


def _remove_isolated_zigzags(
    frames: Sequence[Sequence[TrackResult]],
    decisions: list[CandidateGraphDecision],
    config: CandidateGraphRefinementConfig,
    scale: float,
) -> None:
    points = [_point(decision.track) for decision in decisions]
    for frame_index in range(2, len(points) - 1):
        before, previous, current, following = points[frame_index - 2 : frame_index + 2]
        if before is None or previous is None or current is None or following is None:
            continue
        prior_velocity = (previous[0] - before[0], previous[1] - before[1])
        incoming = (current[0] - previous[0], current[1] - previous[1])
        outgoing = (following[0] - current[0], following[1] - current[1])
        min_step = float(config.zigzag_min_step_px) * scale
        if min(math.hypot(*prior_velocity), math.hypot(*incoming), math.hypot(*outgoing)) < min_step:
            continue
        midpoint = ((previous[0] + following[0]) * 0.5, (previous[1] + following[1]) * 0.5)
        if math.dist(current, midpoint) < float(config.zigzag_min_deviation_px) * scale:
            continue
        if (
            _cosine(prior_velocity, incoming) <= float(config.zigzag_reverse_cosine)
            and _cosine(incoming, outgoing) <= float(config.zigzag_reverse_cosine)
            and _cosine(prior_velocity, outgoing) >= float(config.zigzag_resume_cosine)
        ):
            decisions[frame_index] = CandidateGraphDecision(
                frame_index=frame_index,
                track=_invisible_track(frames[frame_index]),
                source="candidate_graph_zigzag_veto",
                candidate_rank=0,
            )


def _recover_fast_candidate_free_continuations(
    frames: Sequence[Sequence[TrackResult]],
    decisions: list[CandidateGraphDecision],
    config: CandidateGraphRefinementConfig,
    scale: float,
    width: int,
    height: int,
) -> None:
    points = [_point(decision.track) for decision in decisions]
    for frame_index in range(3, len(points)):
        if points[frame_index] is not None:
            continue
        if any(_point(candidate) is not None for candidate in frames[frame_index]):
            continue
        earlier, previous, current = points[frame_index - 3 : frame_index]
        if earlier is None or previous is None or current is None:
            continue
        prior_velocity = (previous[0] - earlier[0], previous[1] - earlier[1])
        velocity = (current[0] - previous[0], current[1] - previous[1])
        speed = math.hypot(*velocity)
        if not (
            float(config.continuation_min_step_px) * scale
            <= speed
            <= float(config.continuation_max_step_px) * scale
        ):
            continue
        if math.dist(prior_velocity, velocity) > float(config.continuation_max_acceleration_px) * scale:
            continue
        if _cosine(prior_velocity, velocity) < float(config.continuation_min_direction_cosine):
            continue
        predicted = (current[0] + velocity[0], current[1] + velocity[1])
        if not (0.0 <= predicted[0] < width and 0.0 <= predicted[1] < height):
            continue
        decisions[frame_index] = CandidateGraphDecision(
            frame_index=frame_index,
            track=TrackResult(
                ball_xy=[predicted[0], predicted[1]],
                visible=1,
                score=float(decisions[frame_index - 1].track.score)
                * float(config.continuation_score_scale),
                heatmap_shape=list(decisions[frame_index - 1].track.heatmap_shape),
            ),
            source="candidate_graph_continuation",
            candidate_rank=0,
        )


def _cosine(first: Point, second: Point) -> float:
    denominator = math.hypot(*first) * math.hypot(*second)
    if denominator <= 1e-9:
        return 1.0
    return (first[0] * second[0] + first[1] * second[1]) / denominator


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a fixed-lag candidate graph on cached TrackNetV3 candidates.",
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", action="append", choices=("raw_data", "raw_data2"))
    parser.add_argument("--threshold", action="append", type=float)
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument(
        "--baseline-summary",
        type=Path,
        default=ROOT / "outputs" / "tracknetv3_raw_postprocess" / "post_hit_optimized" / "summary.json",
    )
    return parser.parse_args()


def _apply_overrides(config: CandidateGraphConfig, values: Sequence[str]) -> None:
    for value in values:
        try:
            name, raw_value = value.split("=", maxsplit=1)
        except ValueError as exc:
            raise ValueError(f"Invalid --set value: {value!r}") from exc
        if not hasattr(config, name):
            raise ValueError(f"Unknown CandidateGraphConfig field: {name}")
        current = getattr(config, name)
        if isinstance(current, int):
            parsed: int | float = int(raw_value)
        elif isinstance(current, float):
            parsed = float(raw_value)
        else:
            raise ValueError(f"Unsupported config field type for {name}: {type(current).__name__}")
        setattr(config, name, parsed)


def _evaluate_video(
    *,
    split: str,
    name: str,
    metadata: dict[str, Any],
    config_values: Sequence[str],
) -> list[dict[str, Any]]:
    fps = float(metadata["fps"])
    width = int(metadata["width"])
    height = int(metadata["height"])
    cache_rows = _load_cache(Path(metadata["cache"]))
    gt_rows = _load_gt(Path(metadata["annotation"]), width, height)
    if len(cache_rows) != len(gt_rows):
        raise ValueError(f"Frame count mismatch for {split}/{name}: {len(cache_rows)} != {len(gt_rows)}")

    frames = [
        [_track_from_cache(item) for item in cache_row.get("candidates", [])]
        for cache_row in cache_rows
    ]
    config = CandidateGraphConfig(fps=fps)
    _apply_overrides(config, config_values)
    decisions = FixedLagCandidateGraph(config).select_sequence(frames)
    if len(decisions) != len(frames):
        raise RuntimeError(f"Candidate graph frame mismatch for {split}/{name}")

    rows: list[dict[str, Any]] = []
    previous_gt: Point | None = None
    for cache_row, gt_row, candidates, decision in zip(cache_rows, gt_rows, frames, decisions):
        frame_id = int(cache_row["frame_id"])
        if decision.frame_index != frame_id:
            raise RuntimeError(f"Candidate graph order mismatch for {split}/{name}: {decision.frame_index} != {frame_id}")
        candidate_points = [
            point for candidate in candidates if (point := _point_from_track(candidate)) is not None
        ]
        raw_point = candidate_points[0] if candidate_points else None
        gt_point = gt_row["point"]
        gt_speed = (
            math.dist(previous_gt, gt_point)
            if previous_gt is not None and gt_point is not None
            else None
        )
        previous_gt = gt_point
        rows.append(
            {
                "split": split,
                "video": name,
                "frame_id": frame_id,
                "gt_point": gt_point,
                "gt_speed_px_per_frame": gt_speed,
                "candidate_points": candidate_points,
                "raw_point": raw_point,
                "graph_point": _point_from_track(decision.track),
                "graph_score": float(decision.track.score),
                "graph_source": decision.source,
                "graph_candidate_rank": decision.candidate_rank,
            }
        )
    return rows


def _preservation(rows: Sequence[dict[str, Any]], threshold: float) -> dict[str, int]:
    raw_correct = graph_correct = harmed = rescued = coordinate_mismatch = 0
    for row in rows:
        gt = row["gt_point"]
        raw = row["raw_point"]
        graph = row["graph_point"]
        raw_ok = gt is not None and raw is not None and math.dist(gt, raw) <= threshold
        graph_ok = gt is not None and graph is not None and math.dist(gt, graph) <= threshold
        raw_correct += int(raw_ok)
        graph_correct += int(graph_ok)
        harmed += int(raw_ok and not graph_ok)
        rescued += int(not raw_ok and graph_ok)
        if graph is not None and not any(math.dist(graph, candidate) <= 1e-9 for candidate in row["candidate_points"]):
            coordinate_mismatch += 1
    return {
        "raw_correct": raw_correct,
        "graph_correct": graph_correct,
        "harmed_correct_raw_top1": harmed,
        "rescued_when_raw_top1_wrong": rescued,
        "measured_coordinate_mismatch": coordinate_mismatch,
    }


def _write_per_frame(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields = [
        "split",
        "video",
        "frame_id",
        "gt_visible",
        "gt_x",
        "gt_y",
        "gt_speed_px_per_frame",
        "raw_visible",
        "raw_x",
        "raw_y",
        "graph_visible",
        "graph_x",
        "graph_y",
        "graph_score",
        "graph_source",
        "graph_candidate_rank",
        "candidate_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            gt = row["gt_point"]
            raw = row["raw_point"]
            graph = row["graph_point"]
            writer.writerow(
                {
                    "split": row["split"],
                    "video": row["video"],
                    "frame_id": row["frame_id"],
                    "gt_visible": int(gt is not None),
                    "gt_x": gt[0] if gt else -1.0,
                    "gt_y": gt[1] if gt else -1.0,
                    "gt_speed_px_per_frame": row["gt_speed_px_per_frame"],
                    "raw_visible": int(raw is not None),
                    "raw_x": raw[0] if raw else -1.0,
                    "raw_y": raw[1] if raw else -1.0,
                    "graph_visible": int(graph is not None),
                    "graph_x": graph[0] if graph else -1.0,
                    "graph_y": graph[1] if graph else -1.0,
                    "graph_score": row["graph_score"],
                    "graph_source": row["graph_source"],
                    "graph_candidate_rank": row["graph_candidate_rank"],
                    "candidate_count": len(row["candidate_points"]),
                }
            )


def _baseline_metrics(path: Path, split: str, threshold: float) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    summary = json.loads(path.read_text(encoding="utf-8"))
    return summary.get("splits", {}).get(split, {}).get("metrics", {}).get(f"{threshold:g}px", {}).get("fixed")


def _speed_p90(rows: Sequence[dict[str, Any]]) -> float:
    values = sorted(
        float(row["gt_speed_px_per_frame"])
        for row in rows
        if row["gt_speed_px_per_frame"] is not None
    )
    return values[int(0.90 * (len(values) - 1))] if values else 0.0


def main() -> None:
    args = _parse_args()
    metadata = json.loads((args.cache_dir / "cache_meta.json").read_text(encoding="utf-8"))
    splits = args.split or ["raw_data", "raw_data2"]
    thresholds = args.threshold or [10.0, 20.0]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    probe_config = CandidateGraphConfig()
    _apply_overrides(probe_config, args.set)
    summary: dict[str, Any] = {
        "dataset_root": str(args.dataset_root.resolve()),
        "cache_dir": str(args.cache_dir.resolve()),
        "baseline_summary": str(args.baseline_summary.resolve()),
        "config": asdict(probe_config),
        "fps_source": "per-video cache metadata",
        "config_overrides": list(args.set),
        "thresholds": thresholds,
        "splits": {},
    }

    for split in splits:
        split_rows: list[dict[str, Any]] = []
        per_video: dict[str, Any] = {}
        for name, video_meta in sorted(metadata["splits"].get(split, {}).items()):
            rows = _evaluate_video(
                split=split,
                name=name,
                metadata=video_meta,
                config_values=args.set,
            )
            split_rows.extend(rows)
            per_video[name] = {
                f"{threshold:g}px": {
                    "raw": _metrics(rows, "raw_point", threshold),
                    "graph": _metrics(rows, "graph_point", threshold),
                    "oracle": _candidate_oracle(rows, threshold),
                    "preservation": _preservation(rows, threshold),
                }
                for threshold in thresholds
            }

        speed_p90 = _speed_p90(split_rows)
        high_speed_rows = [
            row
            for row in split_rows
            if row["gt_speed_px_per_frame"] is not None
            and float(row["gt_speed_px_per_frame"]) >= speed_p90
        ]
        summary["splits"][split] = {
            "videos": len(per_video),
            "frames": len(split_rows),
            "speed_p90_px_per_frame": speed_p90,
            "metrics": {
                f"{threshold:g}px": {
                    "raw": _metrics(split_rows, "raw_point", threshold),
                    "current_fixed": _baseline_metrics(args.baseline_summary, split, threshold),
                    "graph": _metrics(split_rows, "graph_point", threshold),
                    "oracle": _candidate_oracle(split_rows, threshold),
                    "high_speed_raw": _metrics(high_speed_rows, "raw_point", threshold),
                    "high_speed_graph": _metrics(high_speed_rows, "graph_point", threshold),
                    "preservation": _preservation(split_rows, threshold),
                }
                for threshold in thresholds
            },
            "graph_sources": dict(Counter(row["graph_source"] for row in split_rows)),
            "candidate_ranks": dict(Counter(str(row["graph_candidate_rank"]) for row in split_rows)),
            "per_video": per_video,
        }
        split_dir = args.output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        _write_per_frame(split_dir / "per_frame.csv", split_rows)

    output_path = args.output_dir / "summary.json"
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output_path.resolve())


if __name__ == "__main__":
    main()
