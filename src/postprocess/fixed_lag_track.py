from __future__ import annotations

import statistics
from collections import deque
from dataclasses import dataclass, field
from math import hypot, isfinite
from typing import Any, Sequence

from src.utils.structures import TrackResult


Point = tuple[float, float]
MEASURED_ACTIONS = {"accept", "bootstrap_accept", "relock_accept"}
BRANCH_REJECTION_REASONS = {"candidate_failed_motion_gate", "low_confidence_candidate_conflict"}
SHORT_VISIBILITY_GAP_REASONS = {"missing_or_low_confidence", "low_confidence_candidate_conflict"}


@dataclass(slots=True)
class FixedLagTrackConfig:
    fps: float = 25.0
    delay_ms: int = 300
    max_delay_ms: int = 1000
    history_frames: int = 12
    min_retained_history_frames: int = 4
    candidate_chain_max_interpolation_frames: int = 2
    candidate_chain_interpolation_score_scale: float = 0.5
    top_apex_min_y_px: float = 160.0
    top_apex_height_ratio: float = 0.20
    top_apex_default_y_px: float = 216.0
    top_apex_candidate_min_score: float = 0.20
    top_apex_min_chain_candidates: int = 4
    top_apex_min_span_frames: int = 3
    top_apex_strong_score: float = 0.45
    top_apex_min_step_px: float = 45.0
    top_apex_max_step_px: float = 100.0
    top_apex_step_px_per_sec: float = 3000.0
    top_apex_max_prediction_error_px: float = 70.0
    top_apex_transition_cost_weight: float = 0.25
    top_apex_score_cost_weight: float = 12.0
    relock_anchor_min_score: float = 0.60
    relock_following_min_score: float = 0.55
    relock_candidate_min_score: float = 0.20
    relock_min_step_px: float = 45.0
    relock_max_step_px: float = 120.0
    relock_step_px_per_sec: float = 3600.0
    relock_prediction_error_ratio: float = 0.60
    relock_velocity_change_min_px: float = 24.0
    relock_velocity_change_ratio: float = 0.22
    relock_transition_cost_weight: float = 0.15
    relock_score_cost_weight: float = 10.0
    relock_min_chain_candidates: int = 3
    relock_min_mean_score: float = 0.30
    rejected_branch_min_frames: int = 2
    rejected_branch_max_frames: int = 5
    rejected_branch_future_min_score: float = 0.65
    rejected_branch_candidate_min_score: float = 0.35
    rejected_branch_min_step_px: float = 50.0
    rejected_branch_max_step_px: float = 140.0
    rejected_branch_step_px_per_sec: float = 4200.0
    rejected_branch_prediction_error_ratio: float = 0.60
    rejected_branch_velocity_change_min_px: float = 48.0
    rejected_branch_velocity_change_ratio: float = 0.45
    rejected_branch_transition_cost_weight: float = 0.15
    rejected_branch_score_cost_weight: float = 10.0
    rejected_branch_min_mean_score: float = 0.30
    rejected_branch_strong_score: float = 0.45
    short_visibility_max_gap_frames: int = 2
    short_visibility_anchor_min_score: float = 0.45
    short_visibility_min_step_px: float = 50.0
    short_visibility_max_step_px: float = 140.0
    short_visibility_step_px_per_sec: float = 4200.0
    short_visibility_stationary_max_px: float = 12.0
    short_visibility_min_direction_px: float = 2.0
    short_visibility_min_direction_cosine: float = 0.25
    short_visibility_candidate_min_score: float = 0.20
    short_visibility_tolerance_min_px: float = 18.0
    short_visibility_tolerance_max_px: float = 42.0
    short_visibility_tolerance_speed_ratio: float = 0.45
    short_visibility_tolerance_base_px: float = 12.0
    short_visibility_score_scale: float = 0.5
    future_candidate_min_score: float = 0.76
    future_candidate_anchor_min_score: float = 0.55
    future_candidate_min_step_px: float = 50.0
    future_candidate_max_step_px: float = 140.0
    future_candidate_step_px_per_sec: float = 4200.0
    future_candidate_backcast_min_error_px: float = 24.0
    future_candidate_backcast_error_ratio: float = 0.25
    bootstrap_confirm_frames: int = 8
    bootstrap_velocity_points: int = 5
    bootstrap_backfill_frames: int = 3
    bootstrap_candidate_min_score: float = 0.20
    bootstrap_candidate_max_error_px: float = 20.0
    hermite_max_gap_frames: int = 2
    hermite_anchor_min_score: float = 0.23
    hermite_history_frames: int = 4
    hermite_max_average_step_px: float = 35.0
    hermite_max_repaired_step_px: float = 52.5
    hermite_score_scale: float = 0.5
    outer_reliable_search_frames: int = 4
    outer_reliable_min_score: float = 0.55
    single_smooth_context_frames: int = 3
    single_smooth_anchor_min_score: float = 0.45
    single_smooth_max_span_px: float = 80.0
    single_smooth_missing_deviation_px: float = 20.0
    single_smooth_measured_max_score: float = 0.65
    single_smooth_measured_deviation_px: float = 18.0
    single_smooth_score_scale: float = 0.5
    impact_backcast_anchor_min_score: float = 0.75
    impact_backcast_frames: int = 2
    impact_backcast_max_coast_score: float = 0.15
    impact_backcast_score_scale: float = 0.5
    occlusion_reset_history_frames: int = 4
    occlusion_reset_future_frames: int = 2
    occlusion_reset_future_min_score: float = 0.75
    occlusion_reset_min_coast_frames: int = 3

    @property
    def delay_frames(self) -> int:
        fps = self.fps if self.fps > 0 else 25.0
        delay_ms = min(max(0, int(self.delay_ms)), max(0, int(self.max_delay_ms)))
        return max(0, int(round(fps * delay_ms / 1000.0)))


@dataclass(slots=True)
class FixedLagTrackFrame:
    frame_index: int
    track: TrackResult
    payload: Any = None
    debug_record: dict[str, object] | None = None
    source: str = "causal"


@dataclass(slots=True)
class _Sample:
    frame_index: int
    baseline: TrackResult
    candidates: tuple[TrackResult, ...]
    action: str
    reason: str
    decision_score: float
    payload: Any
    debug_record: dict[str, object] | None
    repaired: TrackResult = field(init=False)
    source: str = "causal"
    emitted: bool = False

    def __post_init__(self) -> None:
        self.repaired = _copy_track(self.baseline)


class FixedLagTrackPostProcessor:
    """Repair causal shuttle tracks with a bounded future-frame window."""

    def __init__(self, config: FixedLagTrackConfig | None = None) -> None:
        self.config = config or FixedLagTrackConfig()
        self._samples: deque[_Sample] = deque()
        self._frame_index = -1

    @property
    def delay_frames(self) -> int:
        return self.config.delay_frames

    @property
    def pending_count(self) -> int:
        return sum(not sample.emitted for sample in self._samples)

    def reset(self) -> None:
        self._samples.clear()
        self._frame_index = -1

    def push(
        self,
        track: TrackResult,
        *,
        candidates: Sequence[TrackResult] = (),
        debug_record: dict[str, object] | None = None,
        payload: Any = None,
    ) -> FixedLagTrackFrame | None:
        self._frame_index += 1
        debug = dict(debug_record) if isinstance(debug_record, dict) else None
        self._samples.append(
            _Sample(
                frame_index=self._frame_index,
                baseline=_copy_track(track),
                candidates=tuple(_copy_track(candidate) for candidate in candidates),
                action=str((debug or {}).get("action", "")),
                reason=str((debug or {}).get("reason", "")),
                decision_score=float(track.score),
                payload=payload,
                debug_record=debug,
            )
        )
        self._recompute_pending()
        if self.pending_count <= self.delay_frames:
            return None
        return self._emit_oldest()

    def flush_one(self) -> FixedLagTrackFrame | None:
        if self.pending_count <= 0:
            return None
        self._recompute_pending()
        return self._emit_oldest()

    def flush(self) -> list[FixedLagTrackFrame]:
        frames: list[FixedLagTrackFrame] = []
        while True:
            frame = self.flush_one()
            if frame is None:
                return frames
            frames.append(frame)

    def _recompute_pending(self) -> None:
        samples = list(self._samples)
        self._apply_confirmed_bootstrap(samples)
        self._apply_top_apex_candidate_chain(samples)
        self._apply_confirmed_relock_candidate_chain(samples)
        self._apply_future_confirmed_rejected_branch(samples)
        self._apply_short_hermite_gaps(samples)
        self._apply_two_sided_short_visibility_gap(samples)
        self._apply_single_frame_smoothing(samples)
        self._apply_relock_backtracking(samples)
        self._apply_future_confirmed_candidate(samples)

    def _apply_top_apex_candidate_chain(self, samples: list[_Sample]) -> None:
        for top_index, sample in enumerate(samples):
            if sample.reason not in {"likely_top_exit", "measurement_reverses_after_top_exit"}:
                continue
            left_index = top_index - 1
            if left_index < 0:
                continue
            left_point = _point(samples[left_index].repaired)
            if left_point is None:
                continue
            height = float((sample.debug_record or {}).get("frame_h", 0.0) or 0.0)
            top_limit = (
                max(float(self.config.top_apex_min_y_px), height * float(self.config.top_apex_height_ratio))
                if height > 0
                else float(self.config.top_apex_default_y_px)
            )
            end_limit = len(samples) - 1
            for pos in range(top_index, end_limit + 1):
                if pos > top_index and samples[pos].action in MEASURED_ACTIONS:
                    end_limit = pos
                    break

            best: tuple[tuple[int, int, float, float], list[tuple[int, TrackResult]]] | None = None
            for end_index in range(top_index, end_limit + 1):
                endpoints = [
                    candidate
                    for candidate in samples[end_index].candidates
                    if float(candidate.score) >= float(self.config.top_apex_candidate_min_score)
                    and (point := _point(candidate)) is not None
                    and point[1] <= top_limit
                ]
                if samples[end_index].action in MEASURED_ACTIONS:
                    endpoint = samples[end_index].repaired
                    point = _point(endpoint)
                    if point is not None and point[1] <= top_limit:
                        endpoints.append(endpoint)
                for endpoint in endpoints:
                    chain = self._trace_top_candidate_chain(
                        samples,
                        left_index,
                        end_index,
                        endpoint,
                        top_limit,
                    )
                    if len(chain) < max(1, int(self.config.top_apex_min_chain_candidates)):
                        continue
                    span = chain[-1][0] - chain[0][0]
                    if (
                        span < max(1, int(self.config.top_apex_min_span_frames))
                        or max(float(track.score) for _, track in chain)
                        < float(self.config.top_apex_strong_score)
                    ):
                        continue
                    score_sum = sum(float(track.score) for _, track in chain)
                    path_error = self._chain_linear_error(left_index, left_point, chain)
                    rank = (len(chain), span, score_sum, -path_error)
                    if best is None or rank > best[0]:
                        best = (rank, chain)
            if best is not None:
                self._apply_candidate_chain(
                    samples,
                    left_index,
                    best[1],
                    candidate_source="fixed_lag_top_apex_candidate",
                    interpolation_source="fixed_lag_top_apex_interpolation",
                )

    def _trace_top_candidate_chain(
        self,
        samples: list[_Sample],
        left_index: int,
        end_index: int,
        endpoint: TrackResult,
        top_limit: float,
    ) -> list[tuple[int, TrackResult]]:
        left_point = _point(samples[left_index].repaired)
        endpoint_point = _point(endpoint)
        if left_point is None or endpoint_point is None or end_index <= left_index:
            return []
        max_step = max(
            float(self.config.top_apex_min_step_px),
            min(
                float(self.config.top_apex_max_step_px),
                float(self.config.top_apex_step_px_per_sec) / max(self.config.fps, 1.0),
            ),
        )
        chain: list[tuple[int, TrackResult]] = [(end_index, endpoint)]
        next_index = end_index
        next_point = endpoint_point
        for pos in range(end_index - 1, left_index, -1):
            frame_gap = next_index - pos
            alpha = (pos - left_index) / (end_index - left_index)
            expected = (
                left_point[0] * (1.0 - alpha) + endpoint_point[0] * alpha,
                left_point[1] * (1.0 - alpha) + endpoint_point[1] * alpha,
            )
            choices: list[tuple[float, TrackResult, Point]] = []
            for candidate in samples[pos].candidates:
                point = _point(candidate)
                if (
                    point is None
                    or float(candidate.score) < float(self.config.top_apex_candidate_min_score)
                    or point[1] > top_limit
                ):
                    continue
                transition = _distance(point, next_point)
                prediction_error = _distance(point, expected)
                if (
                    transition > max_step * frame_gap
                    or prediction_error > float(self.config.top_apex_max_prediction_error_px)
                ):
                    continue
                cost = (
                    prediction_error
                    + float(self.config.top_apex_transition_cost_weight) * transition
                    - float(self.config.top_apex_score_cost_weight) * float(candidate.score)
                )
                choices.append((cost, candidate, point))
            if not choices:
                continue
            _, chosen, chosen_point = min(choices, key=lambda item: item[0])
            chain.append((pos, chosen))
            next_index = pos
            next_point = chosen_point
        chain.reverse()
        if not chain:
            return []
        first_index, first_track = chain[0]
        first_point = _point(first_track)
        if first_point is None or _distance(left_point, first_point) > max_step * (first_index - left_index):
            return []
        return chain

    @staticmethod
    def _chain_linear_error(
        left_index: int,
        left_point: Point,
        chain: list[tuple[int, TrackResult]],
    ) -> float:
        end_index, end_track = chain[-1]
        end_point = _point(end_track)
        if end_point is None or end_index <= left_index:
            return float("inf")
        error = 0.0
        for pos, track in chain:
            point = _point(track)
            if point is None:
                continue
            alpha = (pos - left_index) / (end_index - left_index)
            expected = (
                left_point[0] * (1.0 - alpha) + end_point[0] * alpha,
                left_point[1] * (1.0 - alpha) + end_point[1] * alpha,
            )
            error += _distance(point, expected)
        return error

    def _apply_confirmed_relock_candidate_chain(self, samples: list[_Sample]) -> None:
        for relock_index, sample in enumerate(samples):
            if (
                sample.reason != "impact_direction_change"
                or relock_index + 1 >= len(samples)
                or not _is_reliable(sample, float(self.config.relock_anchor_min_score))
                or not _is_reliable(
                    samples[relock_index + 1], float(self.config.relock_following_min_score)
                )
            ):
                continue
            anchor = _point(sample.repaired)
            following = _point(samples[relock_index + 1].repaired)
            if anchor is None or following is None:
                continue
            left_index = relock_index - 1
            while left_index >= 0 and samples[left_index].action not in MEASURED_ACTIONS:
                left_index -= 1
            start_index = max(left_index + 1, relock_index - self.delay_frames)
            max_step = max(
                float(self.config.relock_min_step_px),
                min(
                    float(self.config.relock_max_step_px),
                    float(self.config.relock_step_px_per_sec) / max(self.config.fps, 1.0),
                ),
            )
            velocity = (following[0] - anchor[0], following[1] - anchor[1])
            next_index = relock_index
            next_point = anchor
            chain: list[tuple[int, TrackResult]] = []
            for pos in range(relock_index - 1, start_index - 1, -1):
                frame_gap = next_index - pos
                expected = (
                    next_point[0] - velocity[0] * frame_gap,
                    next_point[1] - velocity[1] * frame_gap,
                )
                choices: list[tuple[float, TrackResult, Point]] = []
                for candidate in samples[pos].candidates:
                    point = _point(candidate)
                    if point is None or float(candidate.score) < float(self.config.relock_candidate_min_score):
                        continue
                    transition = _distance(point, next_point)
                    prediction_error = _distance(point, expected)
                    candidate_velocity = (
                        (next_point[0] - point[0]) / frame_gap,
                        (next_point[1] - point[1]) / frame_gap,
                    )
                    if (
                        transition > max_step * frame_gap
                        or prediction_error
                        > max_step * float(self.config.relock_prediction_error_ratio) * frame_gap
                    ):
                        continue
                    if _distance(candidate_velocity, velocity) > max(
                        float(self.config.relock_velocity_change_min_px),
                        max_step * float(self.config.relock_velocity_change_ratio),
                    ):
                        continue
                    cost = (
                        prediction_error
                        + float(self.config.relock_transition_cost_weight) * transition
                        - float(self.config.relock_score_cost_weight) * float(candidate.score)
                    )
                    choices.append((cost, candidate, point))
                if not choices:
                    continue
                _, chosen, chosen_point = min(choices, key=lambda item: item[0])
                chain.append((pos, chosen))
                velocity = (
                    (next_point[0] - chosen_point[0]) / frame_gap,
                    (next_point[1] - chosen_point[1]) / frame_gap,
                )
                next_index = pos
                next_point = chosen_point
            chain.reverse()
            if (
                len(chain) < max(1, int(self.config.relock_min_chain_candidates))
                or statistics.mean(float(track.score) for _, track in chain)
                < float(self.config.relock_min_mean_score)
            ):
                continue
            self._apply_candidate_chain(
                samples,
                chain[0][0] - 1,
                chain,
                candidate_source="fixed_lag_relock_candidate",
                interpolation_source="fixed_lag_relock_interpolation",
                interpolate_from_left=False,
            )

    def _apply_future_confirmed_rejected_branch(self, samples: list[_Sample]) -> None:
        max_run = min(
            max(1, int(self.config.rejected_branch_max_frames)),
            max(max(1, int(self.config.rejected_branch_min_frames)), self.delay_frames),
        )
        max_step = max(
            float(self.config.rejected_branch_min_step_px),
            min(
                float(self.config.rejected_branch_max_step_px),
                float(self.config.rejected_branch_step_px_per_sec) / max(self.config.fps, 1.0),
            ),
        )
        for future_index in range(2, len(samples) - 1):
            future = samples[future_index]
            following = samples[future_index + 1]
            if not _is_reliable(future, float(self.config.rejected_branch_future_min_score)) or not _is_reliable(
                following, float(self.config.rejected_branch_future_min_score)
            ):
                continue
            future_point = _point(future.repaired)
            following_point = _point(following.repaired)
            if future_point is None or following_point is None:
                continue

            positions: list[int] = []
            pos = future_index - 1
            while pos >= 0 and len(positions) < max_run:
                sample = samples[pos]
                if (
                    sample.emitted
                    or sample.source != "causal"
                    or sample.action in MEASURED_ACTIONS
                    or sample.reason not in BRANCH_REJECTION_REASONS
                    or _point(sample.repaired) is not None
                ):
                    break
                positions.append(pos)
                pos -= 1
            positions.reverse()
            if len(positions) < max(1, int(self.config.rejected_branch_min_frames)):
                continue
            previous_index = positions[0] - 1
            if previous_index < 0 or _point(samples[previous_index].repaired) is None:
                continue

            velocity = (
                following_point[0] - future_point[0],
                following_point[1] - future_point[1],
            )
            next_index = future_index
            next_point = future_point
            chain: list[tuple[int, TrackResult]] = []
            for pos in reversed(positions):
                frame_gap = next_index - pos
                expected = (
                    next_point[0] - velocity[0] * frame_gap,
                    next_point[1] - velocity[1] * frame_gap,
                )
                choices: list[tuple[float, TrackResult, Point, Point]] = []
                for candidate in samples[pos].candidates:
                    point = _point(candidate)
                    if (
                        point is None
                        or float(candidate.score) < float(self.config.rejected_branch_candidate_min_score)
                    ):
                        continue
                    transition = _distance(point, next_point)
                    candidate_velocity = (
                        (next_point[0] - point[0]) / frame_gap,
                        (next_point[1] - point[1]) / frame_gap,
                    )
                    prediction_error = _distance(point, expected)
                    velocity_change = _distance(candidate_velocity, velocity)
                    if transition > max_step * frame_gap:
                        continue
                    if (
                        prediction_error
                        > max_step * float(self.config.rejected_branch_prediction_error_ratio) * frame_gap
                    ):
                        continue
                    if velocity_change > max(
                        float(self.config.rejected_branch_velocity_change_min_px),
                        max_step * float(self.config.rejected_branch_velocity_change_ratio),
                    ):
                        continue
                    if _dot(candidate_velocity, velocity) <= 0.0:
                        continue
                    cost = (
                        prediction_error
                        + float(self.config.rejected_branch_transition_cost_weight) * transition
                        - float(self.config.rejected_branch_score_cost_weight) * float(candidate.score)
                    )
                    choices.append((cost, candidate, point, candidate_velocity))
                if not choices:
                    chain.clear()
                    break
                _, chosen, chosen_point, velocity = min(choices, key=lambda item: item[0])
                chain.append((pos, chosen))
                next_index = pos
                next_point = chosen_point
            if len(chain) != len(positions):
                continue
            chain.reverse()
            scores = [float(track.score) for _, track in chain]
            if (
                statistics.mean(scores) < float(self.config.rejected_branch_min_mean_score)
                or max(scores) < float(self.config.rejected_branch_strong_score)
            ):
                continue
            for pos, candidate in chain:
                samples[pos].repaired = _copy_track(candidate)
                samples[pos].source = "fixed_lag_rejected_branch_candidate"

    def _apply_two_sided_short_visibility_gap(self, samples: list[_Sample]) -> None:
        index = 1
        max_step = max(
            float(self.config.short_visibility_min_step_px),
            min(
                float(self.config.short_visibility_max_step_px),
                float(self.config.short_visibility_step_px_per_sec) / max(self.config.fps, 1.0),
            ),
        )
        while index < len(samples) - 1:
            if samples[index].action in MEASURED_ACTIONS:
                index += 1
                continue
            start = index
            while index < len(samples) and samples[index].action not in MEASURED_ACTIONS:
                index += 1
            end = index - 1
            right_index = end + 1
            if (
                end - start + 1 > max(1, int(self.config.short_visibility_max_gap_frames))
                or right_index >= len(samples)
                or any(samples[pos].emitted for pos in range(start, right_index))
                or any(samples[pos].reason not in SHORT_VISIBILITY_GAP_REASONS for pos in range(start, right_index))
                or any(
                    samples[pos].source
                    not in {"causal", "fixed_lag_hermite", "fixed_lag_single_smooth"}
                    for pos in range(start, right_index)
                )
                or not _is_reliable(
                    samples[start - 1], float(self.config.short_visibility_anchor_min_score)
                )
                or not _is_reliable(
                    samples[right_index], float(self.config.short_visibility_anchor_min_score)
                )
            ):
                continue
            before = self._outer_reliable(samples, start - 1, -1)
            after = self._outer_reliable(samples, right_index, 1)
            left_point = _point(samples[start - 1].repaired)
            right_point = _point(samples[right_index].repaired)
            if before is None or after is None or left_point is None or right_point is None:
                continue
            before_index, before_point = before
            after_index, after_point = after
            incoming = (
                (left_point[0] - before_point[0]) / (start - 1 - before_index),
                (left_point[1] - before_point[1]) / (start - 1 - before_index),
            )
            outgoing = (
                (after_point[0] - right_point[0]) / (after_index - right_index),
                (after_point[1] - right_point[1]) / (after_index - right_index),
            )
            bridge = (
                (right_point[0] - left_point[0]) / (right_index - start + 1),
                (right_point[1] - left_point[1]) / (right_index - start + 1),
            )
            if _distance((0.0, 0.0), bridge) > max_step:
                continue
            if not self._directions_compatible(incoming, bridge) or not self._directions_compatible(
                bridge, outgoing
            ):
                continue

            span = right_index - (start - 1)
            for pos in range(start, right_index):
                alpha = (pos - (start - 1)) / span
                expected = (
                    left_point[0] * (1.0 - alpha) + right_point[0] * alpha,
                    left_point[1] * (1.0 - alpha) + right_point[1] * alpha,
                )
                tolerance = max(
                    float(self.config.short_visibility_tolerance_min_px),
                    min(
                        float(self.config.short_visibility_tolerance_max_px),
                        _distance((0.0, 0.0), bridge)
                        * float(self.config.short_visibility_tolerance_speed_ratio)
                        + float(self.config.short_visibility_tolerance_base_px),
                    ),
                )
                choices = [
                    candidate
                    for candidate in samples[pos].candidates
                    if float(candidate.score) >= float(self.config.short_visibility_candidate_min_score)
                    and (point := _point(candidate)) is not None
                    and _distance(point, expected) <= tolerance
                ]
                if choices:
                    chosen = min(choices, key=lambda candidate: _distance(_point(candidate) or expected, expected))
                    samples[pos].repaired = _copy_track(chosen)
                    samples[pos].source = "fixed_lag_weak_visibility_candidate"
                elif samples[pos].source == "causal":
                    score = (
                        min(samples[start - 1].decision_score, samples[right_index].decision_score)
                        * float(self.config.short_visibility_score_scale)
                    )
                    samples[pos].repaired = _track_from_point(expected, score, samples[pos].baseline)
                    samples[pos].source = "fixed_lag_short_visibility_gap"

    def _apply_candidate_chain(
        self,
        samples: list[_Sample],
        left_index: int,
        chain: list[tuple[int, TrackResult]],
        *,
        candidate_source: str,
        interpolation_source: str,
        interpolate_from_left: bool = True,
    ) -> None:
        anchors: list[tuple[int, Point, float]] = []
        if interpolate_from_left and left_index >= 0:
            left_point = _point(samples[left_index].repaired)
            if left_point is not None:
                anchors.append((left_index, left_point, float(samples[left_index].decision_score)))
        for pos, track in chain:
            if samples[pos].emitted:
                continue
            point = _point(track)
            if point is None:
                continue
            samples[pos].repaired = _copy_track(track)
            samples[pos].source = candidate_source
            anchors.append((pos, point, float(track.score)))
        for (left_pos, left_point, left_score), (right_pos, right_point, right_score) in zip(anchors, anchors[1:]):
            missing = right_pos - left_pos - 1
            if missing < 1 or missing > max(
                0, int(self.config.candidate_chain_max_interpolation_frames)
            ):
                continue
            for offset, pos in enumerate(range(left_pos + 1, right_pos), start=1):
                if samples[pos].emitted or _point(samples[pos].repaired) is not None:
                    continue
                alpha = offset / (missing + 1)
                point = (
                    left_point[0] * (1.0 - alpha) + right_point[0] * alpha,
                    left_point[1] * (1.0 - alpha) + right_point[1] * alpha,
                )
                samples[pos].repaired = _track_from_point(
                    point,
                    min(left_score, right_score)
                    * float(self.config.candidate_chain_interpolation_score_scale),
                    samples[pos].baseline,
                )
                samples[pos].source = interpolation_source

    def _apply_future_confirmed_candidate(self, samples: list[_Sample]) -> None:
        max_step = max(
            float(self.config.future_candidate_min_step_px),
            min(
                float(self.config.future_candidate_max_step_px),
                float(self.config.future_candidate_step_px_per_sec)
                / max(self.config.fps, 1.0),
            ),
        )
        max_backcast_error = max(
            float(self.config.future_candidate_backcast_min_error_px),
            max_step * float(self.config.future_candidate_backcast_error_ratio),
        )
        for index in range(1, len(samples) - 2):
            target = samples[index]
            if target.emitted or _point(target.repaired) is not None:
                continue
            following = samples[index + 1]
            after = samples[index + 2]
            if not _is_reliable(
                following, float(self.config.future_candidate_anchor_min_score)
            ) or not _is_reliable(after, float(self.config.future_candidate_anchor_min_score)):
                continue
            following_point = _point(following.repaired)
            after_point = _point(after.repaired)
            if following_point is None or after_point is None:
                continue
            predicted = (
                following_point[0] - (after_point[0] - following_point[0]),
                following_point[1] - (after_point[1] - following_point[1]),
            )
            choices = [
                candidate
                for candidate in target.candidates
                if float(candidate.score) >= float(self.config.future_candidate_min_score)
                and (point := _point(candidate)) is not None
                and _distance(point, predicted) <= max_backcast_error
                and _distance(point, following_point) <= max_step
            ]
            if not choices:
                continue
            chosen = min(
                choices,
                key=lambda candidate: _distance(_point(candidate) or predicted, predicted),
            )
            target.repaired = _copy_track(chosen)
            target.source = "fixed_lag_future_candidate"

    def _apply_confirmed_bootstrap(self, samples: list[_Sample]) -> None:
        confirm_frames = max(1, int(self.config.bootstrap_confirm_frames))
        velocity_points = max(
            2,
            min(confirm_frames, int(self.config.bootstrap_velocity_points)),
        )
        for index, sample in enumerate(samples):
            if sample.action != "bootstrap_accept" or index + confirm_frames > len(samples):
                continue
            chain = samples[index : index + confirm_frames]
            if not all(_is_reliable(item, 0.0) for item in chain):
                continue
            points = [_point(item.repaired) for item in chain[:velocity_points]]
            if any(point is None for point in points):
                continue
            resolved = [point for point in points if point is not None]
            velocity = (
                statistics.median(
                    resolved[pos][0] - resolved[pos - 1][0]
                    for pos in range(1, velocity_points)
                ),
                statistics.median(
                    resolved[pos][1] - resolved[pos - 1][1]
                    for pos in range(1, velocity_points)
                ),
            )
            anchor = resolved[0]
            for distance in range(1, max(0, int(self.config.bootstrap_backfill_frames)) + 1):
                target_index = index - distance
                if target_index < 0:
                    break
                target = samples[target_index]
                if target.emitted or _point(target.repaired) is not None:
                    continue
                predicted = (
                    anchor[0] - velocity[0] * distance,
                    anchor[1] - velocity[1] * distance,
                )
                candidates = [
                    candidate
                    for candidate in target.candidates
                    if candidate.visible
                    and float(candidate.score) >= float(self.config.bootstrap_candidate_min_score)
                    and (point := _point(candidate)) is not None
                    and _distance(point, predicted)
                    <= float(self.config.bootstrap_candidate_max_error_px)
                ]
                if not candidates:
                    continue
                chosen = min(
                    candidates,
                    key=lambda candidate: _distance(_point(candidate) or predicted, predicted),
                )
                target.repaired = _copy_track(chosen)
                target.source = "fixed_lag_bootstrap_candidate"

    def _apply_short_hermite_gaps(self, samples: list[_Sample]) -> None:
        history_frames = max(1, int(self.config.hermite_history_frames))
        index = 1
        while index < len(samples) - 1:
            if samples[index].action in MEASURED_ACTIONS:
                index += 1
                continue
            start = index
            while index < len(samples) and samples[index].action not in MEASURED_ACTIONS:
                index += 1
            end = index - 1
            gap_length = end - start + 1
            left_index = start - 1
            right_index = end + 1
            if (
                gap_length > max(0, int(self.config.hermite_max_gap_frames))
                or right_index >= len(samples)
                or any(samples[pos].emitted for pos in range(start, right_index))
                or not _is_reliable(samples[left_index], float(self.config.hermite_anchor_min_score))
                or not _is_reliable(samples[right_index], float(self.config.hermite_anchor_min_score))
                or start < history_frames
                or not all(
                    _point(samples[pos].repaired) is not None
                    for pos in range(start - history_frames, start)
                )
            ):
                continue
            left_point = _point(samples[left_index].repaired)
            right_point = _point(samples[right_index].repaired)
            if left_point is None or right_point is None:
                continue
            if (
                _distance(left_point, right_point) / (gap_length + 1)
                > float(self.config.hermite_max_average_step_px)
            ):
                continue
            before = self._outer_reliable(samples, left_index, -1)
            after = self._outer_reliable(samples, right_index, 1)
            if before is None or after is None:
                continue
            before_index, before_point = before
            after_index, after_point = after
            incoming = (
                (left_point[0] - before_point[0]) / (left_index - before_index),
                (left_point[1] - before_point[1]) / (left_index - before_index),
            )
            outgoing = (
                (after_point[0] - right_point[0]) / (after_index - right_index),
                (after_point[1] - right_point[1]) / (after_index - right_index),
            )
            span = right_index - left_index
            repaired_points: list[Point] = []
            for pos in range(start, right_index):
                t = (pos - left_index) / span
                h00 = 2 * t**3 - 3 * t**2 + 1
                h10 = t**3 - 2 * t**2 + t
                h01 = -2 * t**3 + 3 * t**2
                h11 = t**3 - t**2
                repaired_points.append(
                    (
                        h00 * left_point[0]
                        + h10 * span * incoming[0]
                        + h01 * right_point[0]
                        + h11 * span * outgoing[0],
                        h00 * left_point[1]
                        + h10 * span * incoming[1]
                        + h01 * right_point[1]
                        + h11 * span * outgoing[1],
                    )
                )
            if any(
                _distance(point, left_point if offset == 0 else repaired_points[offset - 1])
                > float(self.config.hermite_max_repaired_step_px)
                for offset, point in enumerate(repaired_points)
            ):
                continue
            score = (
                min(samples[left_index].decision_score, samples[right_index].decision_score)
                * float(self.config.hermite_score_scale)
            )
            for pos, point in zip(range(start, right_index), repaired_points):
                samples[pos].repaired = _track_from_point(point, score, samples[pos].baseline)
                samples[pos].source = "fixed_lag_hermite"

    def _outer_reliable(
        self,
        samples: list[_Sample],
        start: int,
        direction: int,
    ) -> tuple[int, Point] | None:
        for distance in range(1, max(0, int(self.config.outer_reliable_search_frames)) + 1):
            index = start + direction * distance
            if index < 0 or index >= len(samples):
                break
            sample = samples[index]
            point = _point(sample.repaired)
            if _is_reliable(sample, float(self.config.outer_reliable_min_score)) and point is not None:
                return index, point
        return None

    def _apply_single_frame_smoothing(self, samples: list[_Sample]) -> None:
        context_frames = max(1, int(self.config.single_smooth_context_frames))
        for index in range(context_frames, len(samples) - 1):
            previous, current, following = samples[index - 1 : index + 2]
            if (
                current.emitted
                or not _is_reliable(previous, float(self.config.single_smooth_anchor_min_score))
                or not _is_reliable(following, float(self.config.single_smooth_anchor_min_score))
            ):
                continue
            if "fixed_lag_future_candidate" in {previous.source, following.source}:
                continue
            previous_point = _point(previous.repaired)
            current_point = _point(current.repaired)
            following_point = _point(following.repaired)
            if previous_point is None or following_point is None:
                continue
            if "ground_bounce" in f"{previous.reason} {current.reason} {following.reason}":
                continue
            if (
                _distance(previous_point, following_point)
                > float(self.config.single_smooth_max_span_px)
            ):
                continue
            midpoint = (
                (previous_point[0] + following_point[0]) * 0.5,
                (previous_point[1] + following_point[1]) * 0.5,
            )
            deviation = float("inf") if current_point is None else _distance(current_point, midpoint)
            uncertain = current.action not in MEASURED_ACTIONS
            if uncertain and not all(
                _point(samples[pos].repaired) is not None
                for pos in range(index - context_frames, index)
            ):
                continue
            should_replace = (
                uncertain
                and (
                    current_point is None
                    or deviation >= float(self.config.single_smooth_missing_deviation_px)
                )
            ) or (
                current.action in MEASURED_ACTIONS
                and current.decision_score
                <= float(self.config.single_smooth_measured_max_score)
                and deviation >= float(self.config.single_smooth_measured_deviation_px)
            )
            if not should_replace:
                continue
            score = (
                current.decision_score
                if current_point is not None and current.action in MEASURED_ACTIONS
                else min(previous.decision_score, following.decision_score)
                * float(self.config.single_smooth_score_scale)
            )
            current.repaired = _track_from_point(midpoint, score, current.baseline)
            current.source = "fixed_lag_single_smooth"

    def _apply_relock_backtracking(self, samples: list[_Sample]) -> None:
        impact_frames = max(0, int(self.config.impact_backcast_frames))
        reset_history = max(0, int(self.config.occlusion_reset_history_frames))
        reset_future = max(0, int(self.config.occlusion_reset_future_frames))
        for index, sample in enumerate(samples):
            if (
                sample.reason == "impact_direction_change"
                and impact_frames > 0
                and index >= impact_frames
                and index + 1 < len(samples)
                and _is_reliable(sample, float(self.config.impact_backcast_anchor_min_score))
                and _is_reliable(
                    samples[index + 1], float(self.config.impact_backcast_anchor_min_score)
                )
            ):
                prior = samples[index - impact_frames : index]
                if not any(item.emitted for item in prior) and all(
                    item.action == "coast"
                    and item.decision_score <= float(self.config.impact_backcast_max_coast_score)
                    for item in prior
                ):
                    anchor = _point(sample.repaired)
                    following = _point(samples[index + 1].repaired)
                    if anchor is not None and following is not None:
                        velocity = (following[0] - anchor[0], following[1] - anchor[1])
                        score = (
                            min(sample.decision_score, samples[index + 1].decision_score)
                            * float(self.config.impact_backcast_score_scale)
                        )
                        for distance in range(1, impact_frames + 1):
                            point = (
                                anchor[0] - velocity[0] * distance,
                                anchor[1] - velocity[1] * distance,
                            )
                            target = samples[index - distance]
                            target.repaired = _track_from_point(point, score, target.baseline)
                            target.source = "fixed_lag_impact_backcast"
            if (
                sample.reason == "literature_occlusion_model_reset"
                and reset_history > 0
                and reset_future > 0
                and index >= reset_history
                and index + reset_future < len(samples)
                and all(
                    _is_reliable(
                        samples[index + distance],
                        float(self.config.occlusion_reset_future_min_score),
                    )
                    for distance in range(1, reset_future + 1)
                )
            ):
                prior = samples[index - reset_history : index]
                if (
                    not any(item.emitted for item in prior)
                    and sum(item.action == "coast" for item in prior)
                    >= max(0, int(self.config.occlusion_reset_min_coast_frames))
                ):
                    for target in prior:
                        target.repaired = _invisible_track(target.baseline)
                        target.source = "fixed_lag_stale_branch_removed"

    def _emit_oldest(self) -> FixedLagTrackFrame | None:
        for sample in self._samples:
            if sample.emitted:
                continue
            sample.emitted = True
            debug = self._emitted_debug(sample)
            payload = sample.payload
            sample.payload = None
            sample.candidates = ()
            sample.debug_record = None
            frame = FixedLagTrackFrame(
                frame_index=sample.frame_index,
                track=_copy_track(sample.repaired),
                payload=payload,
                debug_record=debug,
                source=sample.source,
            )
            self._trim_history()
            return frame
        return None

    def _emitted_debug(self, sample: _Sample) -> dict[str, object] | None:
        if sample.debug_record is None:
            return None
        record = dict(sample.debug_record)
        record.update(
            {
                "output_visible": int(bool(sample.repaired.visible)),
                "output_x": sample.repaired.ball_xy[0] if sample.repaired.visible else -1.0,
                "output_y": sample.repaired.ball_xy[1] if sample.repaired.visible else -1.0,
                "output_score": float(sample.repaired.score),
                "fixed_lag_frames": self.delay_frames,
                "fixed_lag_source": sample.source,
            }
        )
        if sample.source != "causal":
            record["action"] = "fixed_lag_repair" if sample.repaired.visible else "reject"
            record["reason"] = sample.source
        return record

    def _trim_history(self) -> None:
        keep = max(
            max(0, int(self.config.min_retained_history_frames)),
            int(self.config.history_frames),
        )
        emitted = sum(sample.emitted for sample in self._samples)
        while self._samples and self._samples[0].emitted and emitted > keep:
            self._samples.popleft()
            emitted -= 1

    def _directions_compatible(self, a: Point, b: Point) -> bool:
        a_length = _distance((0.0, 0.0), a)
        b_length = _distance((0.0, 0.0), b)
        if max(a_length, b_length) <= float(self.config.short_visibility_stationary_max_px):
            return True
        if min(a_length, b_length) <= float(self.config.short_visibility_min_direction_px):
            return False
        return (
            _dot(a, b) / (a_length * b_length)
            >= float(self.config.short_visibility_min_direction_cosine)
        )


def _is_reliable(sample: _Sample, min_score: float) -> bool:
    return (
        sample.action in MEASURED_ACTIONS
        and sample.decision_score >= min_score
        and _point(sample.repaired) is not None
    )


def _point(track: TrackResult) -> Point | None:
    if not track.visible or len(track.ball_xy) < 2:
        return None
    x, y = float(track.ball_xy[0]), float(track.ball_xy[1])
    if x < 0 or y < 0 or not isfinite(x) or not isfinite(y):
        return None
    return x, y


def _distance(a: Point, b: Point) -> float:
    return hypot(a[0] - b[0], a[1] - b[1])


def _dot(a: Point, b: Point) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _copy_track(track: TrackResult) -> TrackResult:
    return TrackResult(
        ball_xy=[float(value) for value in track.ball_xy],
        visible=int(bool(track.visible)),
        score=float(track.score),
        heatmap_shape=[int(value) for value in track.heatmap_shape],
    )


def _track_from_point(point: Point, score: float, metadata: TrackResult) -> TrackResult:
    return TrackResult(
        ball_xy=[float(point[0]), float(point[1])],
        visible=1,
        score=max(0.0, min(float(score), 1.0)),
        heatmap_shape=list(metadata.heatmap_shape),
    )


def _invisible_track(metadata: TrackResult) -> TrackResult:
    return TrackResult(
        ball_xy=[-1.0, -1.0],
        visible=0,
        score=0.0,
        heatmap_shape=list(metadata.heatmap_shape),
    )
