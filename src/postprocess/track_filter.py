from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import cos, hypot, isfinite, radians
from typing import Any, Protocol, Sequence

from src.utils.structures import TrackResult


Point = tuple[float, float]
FrameSize = tuple[float, float]
FrameShape = tuple[int, ...] | list[int] | None
PersonBBox = tuple[float, float, float, float]
PersonBBoxes = Sequence[Sequence[float]] | None
CourtMatrix = tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
COURT_WIDTH = 610.0
COURT_LENGTH = 1340.0


class TrackFilterAlgorithm(Protocol):
    """Interface for pluggable ball trajectory filters."""

    debug_records: list[dict[str, object]]

    def reset(self) -> None:
        ...

    def update(
        self,
        track: TrackResult,
        *,
        dt: float | None = None,
        frame_shape: FrameShape = None,
        court_prediction: Any | None = None,
        person_bboxes: PersonBBoxes = None,
    ) -> TrackResult:
        ...

    def update_candidates(
        self,
        tracks: Sequence[TrackResult],
        *,
        dt: float | None = None,
        frame_shape: FrameShape = None,
        court_prediction: Any | None = None,
        person_bboxes: PersonBBoxes = None,
    ) -> TrackResult:
        ...

    def last_debug_record(self) -> dict[str, object] | None:
        ...


@dataclass(slots=True)
class BallTrackFilterConfig:
    fps: float = 25.0
    min_confidence: float = 0.35
    soft_min_confidence: float = 0.32
    relock_confidence: float = 0.50
    strong_relock_confidence: float = 0.72
    bootstrap_confirm_min_confidence: float = 0.55
    impact_relock_confidence: float = 0.60
    impact_relock_confirm_frames: int = 1
    impact_relock_min_missed_frames: int = 1
    impact_relock_min_angle_deg: float = 94.0
    impact_relock_max_prediction_error_px: float = 260.0
    impact_relock_max_prediction_error_per_missed_px: float = 28.0
    close_gate_confidence: float = 0.50
    close_gate_px: float = 72.0
    inertia_relax_confidence: float = 0.65
    inertia_relax_max_confidence: float = 0.75
    inertia_relax_prediction_gate_ratio: float = 1.0
    court_filter_enabled: bool = True
    court_filter_margin_cm: float = 140.0
    court_filter_margin_px: float = 96.0
    court_air_extension_ratio: float = 1.0
    court_air_lateral_expansion_ratio: float = 1.25
    base_gate_px: float = 80.0
    max_gate_px: float = 360.0
    missed_gate_growth_px: float = 55.0
    max_speed_px_per_sec: float = 12000.0
    velocity_blend: float = 0.66
    inertia_min_speed_px_per_sec: float = 250.0
    max_accel_px_per_sec2: float = 120000.0
    max_lateral_error_px: float = 82.0
    max_reverse_px: float = 36.0
    max_coast_frames: int = 10
    min_coast_speed_px_per_sec: float = 450.0
    coast_velocity_decay: float = 0.82
    coast_score_decay: float = 0.55
    coast_on_outlier: bool = False
    person_occlusion_enabled: bool = True
    person_occlusion_padding_px: float = 18.0
    person_occlusion_coast_frames: int = 7
    person_occlusion_min_speed_px_per_sec: float = 250.0
    person_occlusion_candidate_penalty: float = 96.0
    person_occlusion_suppress_coast_candidate_score: float = 0.35
    person_occlusion_accept_confidence: float = 0.50
    frame_measurement_margin_px: float = 6.0
    out_of_frame_prediction_margin_px: float = 12.0
    parabola_enabled: bool = True
    parabola_min_points: int = 4
    parabola_min_real_detections: int = 3
    parabola_history_frames: int = 12
    parabola_max_gap_frames: int = 8
    parabola_min_motion_px: float = 42.0
    parabola_max_fit_rmse_px: float = 48.0
    parabola_gate_px: float = 90.0
    parabola_max_gate_px: float = 160.0
    parabola_gate_growth_px: float = 24.0
    parabola_fit_error_scale: float = 1.8
    parabola_score_bonus_px: float = 20.0
    parabola_score_decay: float = 0.58
    parabola_fill_on_outlier: bool = True
    parabola_fill_max_outlier_distance_px: float = 180.0
    parabola_direction_conflict_min_delta_px: float = 6.0
    relock_distance_px: float = 220.0
    relock_max_speed_px_per_sec: float = 9000.0
    relock_confirm_frames: int = 3
    high_score_fast_relock_confidence: float = 0.72
    relock_after_missed_frames: int = 2
    max_missed_frames: int = 12
    render_smoothing: float = 0.0
    top_exit_enabled: bool = True
    top_exit_margin_px: float = 24.0
    top_exit_margin_ratio: float = 0.04
    top_exit_min_up_speed_px_per_sec: float = 650.0
    top_exit_min_up_motion_px: float = 36.0
    top_exit_history_frames: int = 4
    top_exit_suppression_frames: int = 6
    top_exit_reversal_min_delta_px: float = 8.0
    top_edge_hallucination_min_gap_px: float = 120.0
    top_edge_hallucination_min_gap_ratio: float = 0.10
    ground_bounce_guard_enabled: bool = True
    ground_bounce_guard_max_score: float = 0.35
    ground_bounce_guard_min_downward_px: float = 8.0
    ground_bounce_guard_min_upward_px: float = 4.0
    ground_bounce_guard_min_upward_points: int = 3
    ground_bounce_guard_min_low_score_points: int = 2
    ground_bounce_guard_max_backtrack_px: float = 2.0
    ground_bounce_guard_max_lateral_px: float = 80.0
    ground_bounce_guard_compact_max_upward_px: float = 24.0
    ground_bounce_guard_compact_max_lateral_px: float = 16.0
    ground_bounce_guard_compact_max_speed_ratio: float = 0.45
    ground_bounce_guard_person_clearance_px: float = 96.0
    ground_bounce_guard_min_height_ratio: float = 0.35
    ground_bounce_guard_suppression_seconds: float = 0.75
    static_hotspot_enabled: bool = True
    static_hotspot_radius_px: float = 18.0
    static_hotspot_min_frames: int = 5
    static_hotspot_max_motion_px: float = 12.0
    static_hotspot_memory_frames: int = 90
    static_hotspot_suppression_frames: int = 45
    static_hotspot_tracking_speed_px_per_sec: float = 50.0
    static_hotspot_edge_band_ratio: float = 0.07
    static_hotspot_edge_max_motion_px: float = 32.0
    bootstrap_static_max_x_span_px: float = 3.0
    bootstrap_static_max_y_std_px: float = 15.0
    literature_short_gap_enabled: bool = True
    literature_short_gap_frames: int = 1
    literature_short_gap_min_real_points: int = 3
    literature_short_gap_top_band_ratio: float = 0.06
    literature_branch_guard_enabled: bool = True
    literature_branch_guard_min_confidence: float = 0.80
    literature_branch_guard_max_speed_px_per_sec: float = 260.0
    literature_branch_guard_min_prediction_error_px: float = 32.0
    literature_branch_guard_min_downward_jump_px: float = 28.0
    literature_branch_recovery_frames: int = 3
    literature_branch_recovery_distance_scale_px: float = 4.0
    literature_occlusion_reset_enabled: bool = True
    literature_occlusion_reset_confidence: float = 0.85
    literature_occlusion_reset_max_prediction_error_px: float = 480.0
    literature_occlusion_reset_min_angle_deg: float = 94.0


@dataclass(slots=True)
class _RelockCandidate:
    point: Point
    score: float
    count: int = 1
    last_measurement: Point | None = None
    direction_consistent: bool = True
    measurements: tuple[Point, ...] = ()


@dataclass(slots=True)
class _StaticHotspot:
    anchor: Point
    point: Point
    score: float
    first_frame: int
    last_frame: int
    count: int = 1
    max_distance: float = 0.0
    suppressed_until: int = -1


@dataclass(slots=True)
class _TrajectoryPoint:
    frame_index: int
    time_seconds: float
    point: Point
    score: float


@dataclass(slots=True)
class _ParabolaPrediction:
    point: Point
    fit_rmse: float
    gap_frames: int


@dataclass(slots=True)
class _CourtFilter:
    image_to_court_h: CourtMatrix | None
    corners: tuple[Point, Point, Point, Point] | None


class BallTrackFilter:
    """Low-latency robust gate for shuttle detections.

    The predicted position is used for gating and short coasting. Recent real
    detections are also fit with a lightweight quadratic motion model, so small
    gaps can be filled along the arc and points far outside that arc are ignored
    until they form a stable new trajectory.
    """

    def __init__(
        self,
        config: BallTrackFilterConfig | None = None,
        *,
        fps: float | None = None,
        debug_enabled: bool = False,
        algorithm: TrackFilterAlgorithm | None = None,
    ) -> None:
        self._algorithm = algorithm
        self.config = config or BallTrackFilterConfig()
        if fps is not None and fps > 0:
            self.config.fps = float(fps)
        self._last_point: Point | None = None
        self._render_point: Point | None = None
        self._velocity: Point = (0.0, 0.0)
        self._missed_frames = 0
        self._coast_frames = 0
        self._locked = False
        self._candidate: _RelockCandidate | None = None
        self._real_detections_since_relock = 0
        self._history: deque[_TrajectoryPoint] = deque(maxlen=max(1, self.config.parabola_history_frames))
        self._static_hotspots: list[_StaticHotspot] = []
        self._frame_index = -1
        self._time_seconds = 0.0
        self._last_frame_size: FrameSize | None = None
        self._top_exit_frames_remaining = 0
        self._ground_bounce_suppression_remaining_s = 0.0
        self.debug_enabled = debug_enabled
        self._debug_records: list[dict[str, object]] = []
        self._last_debug_record: dict[str, object] | None = None
        self._pending_candidate_debug: dict[str, object] | None = None
        self._raw_candidate_count = 0
        self._court_filter_active = False
        self._court_filtered_count = 0
        self._static_filtered_count = 0
        self._static_hotspot_count = 0
        self._decision_action = "unknown"
        self._decision_reason = ""
        self._literature_branch_recovery_remaining = 0

    @property
    def debug_records(self) -> list[dict[str, object]]:
        if self._algorithm is not None:
            return self._algorithm.debug_records
        return self._debug_records

    def reset(self) -> None:
        if self._algorithm is not None:
            self._algorithm.reset()
            return

        self._last_point = None
        self._render_point = None
        self._velocity = (0.0, 0.0)
        self._missed_frames = 0
        self._coast_frames = 0
        self._locked = False
        self._candidate = None
        self._real_detections_since_relock = 0
        self._history.clear()
        self._static_hotspots.clear()
        self._frame_index = -1
        self._time_seconds = 0.0
        self._last_frame_size = None
        self._top_exit_frames_remaining = 0
        self._ground_bounce_suppression_remaining_s = 0.0
        self.debug_records.clear()
        self._last_debug_record = None
        self._pending_candidate_debug = None
        self._raw_candidate_count = 0
        self._court_filter_active = False
        self._court_filtered_count = 0
        self._static_filtered_count = 0
        self._static_hotspot_count = 0
        self._decision_action = "unknown"
        self._decision_reason = ""
        self._literature_branch_recovery_remaining = 0

    def update(
        self,
        track: TrackResult,
        *,
        dt: float | None = None,
        frame_shape: FrameShape = None,
        court_prediction: Any | None = None,
        person_bboxes: PersonBBoxes = None,
    ) -> TrackResult:
        if self._algorithm is not None:
            return self._algorithm.update(
                track,
                dt=dt,
                frame_shape=frame_shape,
                court_prediction=court_prediction,
                person_bboxes=person_bboxes,
            )

        step_dt = self._resolve_dt(dt)
        self._frame_index += 1
        self._time_seconds += step_dt
        frame_size = self._resolve_frame_size(frame_shape)
        court_filter = self._court_filter(court_prediction)
        normalized_person_bboxes = self._normalize_person_bboxes(person_bboxes, frame_size)
        debug_before = self._debug_before_state(step_dt, frame_size)
        self._start_decision()

        if self._ground_bounce_suppression_remaining_s > 0.0:
            self._ground_bounce_suppression_remaining_s = max(
                0.0,
                self._ground_bounce_suppression_remaining_s - step_dt,
            )
            self._mark_decision("ground_bounce_suppress", "active_ground_bounce_suppression", force=True)
            result = self._invisible(track)
            return self._finish_debug(track, result, step_dt, frame_size, debug_before)

        if self._top_exit_frames_remaining > 0:
            self._top_exit_frames_remaining -= 1
            self._mark_decision("top_exit_suppress", "active_top_exit_suppression", force=True)
            result = self._invisible(track)
            return self._finish_debug(track, result, step_dt, frame_size, debug_before)

        raw_measurement = self._raw_measurement(track, frame_size)
        measurement = self._measurement(track, frame_size, court_filter=court_filter)
        soft_measurement = False
        if measurement is None and self._can_use_soft_measurement(track, frame_size, court_filter):
            measurement = self._raw_measurement(track, frame_size)
            soft_measurement = measurement is not None

        person_occlusion_active = self._person_occlusion_likely(step_dt, frame_size, normalized_person_bboxes)
        ground_bounce_reason = self._ground_bounce_reason(
            measurement,
            float(track.score),
            frame_size,
            court_filter,
            normalized_person_bboxes,
        )
        if ground_bounce_reason is not None:
            self._enter_ground_bounce_suppression()
            self._mark_decision("ground_bounce_enter", ground_bounce_reason, force=True)
            result = self._invisible(track)
        elif measurement is None:
            candidate_conflict = self._low_confidence_candidate_conflicts_with_prediction(
                raw_measurement,
                step_dt,
                frame_size,
                court_filter,
            )
            reason = "low_confidence_candidate_conflict" if candidate_conflict else "missing_or_low_confidence"
            self._mark_decision("reject", reason)
            result = self._reject(
                track,
                step_dt,
                allow_coast=not candidate_conflict,
                allow_short_gap_bridge=not candidate_conflict,
                frame_size=frame_size,
                court_filter=court_filter,
                occlusion_active=person_occlusion_active,
                coast_reason="person_occlusion_prediction" if person_occlusion_active else None,
            )
        elif not self._locked or self._last_point is None:
            result = self._bootstrap(track, measurement, step_dt, frame_size)
        elif self._prediction_is_out_of_frame(step_dt, frame_size):
            self._mark_decision("reject", "prediction_out_of_frame")
            result = self._reject(track, step_dt, allow_coast=True, frame_size=frame_size, court_filter=court_filter)
        elif self._measurement_reverses_after_top_exit(measurement, step_dt, frame_size):
            self._enter_top_exit()
            self._mark_decision("top_exit_enter", "measurement_reverses_after_top_exit", force=True)
            result = self._invisible(track)
        elif self._measurement_is_top_edge_hallucination(measurement, step_dt, frame_size):
            self._mark_decision("reject", "top_edge_hallucination")
            result = self._reject(track, step_dt, allow_coast=True, frame_size=frame_size, court_filter=court_filter)
        elif self._literature_prediction_outlier(measurement, float(track.score), step_dt, frame_size):
            self._literature_branch_recovery_remaining = max(
                1,
                int(self.config.literature_branch_recovery_frames),
            )
            self._mark_decision("reject", "literature_prediction_outlier", force=True)
            result = self._reject(
                track,
                step_dt,
                allow_coast=False,
                frame_size=frame_size,
                court_filter=court_filter,
            )
        elif (
            person_occlusion_active
            and self._point_inside_person_bbox(measurement, normalized_person_bboxes)
        ):
            score = float(track.score)
            if self._literature_occlusion_model_reset(measurement, score, step_dt):
                self._drop_lock()
                self._mark_decision("relock_accept", "literature_occlusion_model_reset", force=True)
                result = self._accept(track, measurement, step_dt, frame_size)
            elif (
                score >= self.config.person_occlusion_accept_confidence
                and self._passes_gate(measurement, score, step_dt, frame_size)
            ):
                predicted = self._predict(step_dt)
                if self._can_relax_inertia(score, _distance(measurement, predicted)):
                    self._invalidate_parabola_on_y_direction_conflict(measurement, step_dt)
                self._mark_decision("accept", "person_occlusion_motion_gate", force=True)
                result = self._accept(track, measurement, step_dt, frame_size)
            else:
                suppress_coast = (
                    score > self.config.person_occlusion_suppress_coast_candidate_score
                )
                prediction_recovery = (
                    suppress_coast
                    and self._literature_short_gap_can_bridge(step_dt, frame_size, court_filter)
                    and self._last_point is not None
                    and _distance(measurement, self._predict(step_dt))
                    >= float(self.config.literature_branch_guard_min_prediction_error_px)
                )
                if suppress_coast:
                    self._mark_decision(
                        "reject",
                        "literature_prediction_bridge" if prediction_recovery else "person_occlusion_candidate_high_score",
                        force=True,
                    )
                else:
                    self._mark_decision("reject", "person_occlusion_candidate")
                result = self._reject(
                    track,
                    step_dt,
                    allow_coast=prediction_recovery or not suppress_coast,
                    allow_short_gap_bridge=prediction_recovery,
                    frame_size=frame_size,
                    court_filter=court_filter,
                    occlusion_active=True,
                    coast_reason=(
                        "literature_prediction_bridge"
                        if prediction_recovery
                        else "person_occlusion_prediction"
                    ),
                )
        elif self._passes_gate(measurement, float(track.score), step_dt, frame_size):
            predicted = self._predict(step_dt)
            if self._can_relax_inertia(float(track.score), _distance(measurement, predicted)):
                self._invalidate_parabola_on_y_direction_conflict(measurement, step_dt)
            if soft_measurement:
                self._mark_decision("accept", "soft_confidence_motion_gate")
            result = self._accept(track, measurement, step_dt, frame_size)
        elif self._passes_close_gate(measurement, float(track.score), step_dt, frame_size):
            self._mark_decision("accept", "close_prediction_motion_break")
            self._invalidate_parabola_on_y_direction_conflict(measurement, step_dt)
            result = self._accept(track, measurement, step_dt, frame_size)
        else:
            relock = self._update_candidate(measurement, float(track.score), step_dt)
            impact_relock = self._should_impact_relock(measurement, float(track.score), step_dt)
            fast_relock = self._should_fast_relock(float(track.score))
            if (relock and self._should_relock()) or impact_relock or fast_relock:
                self._drop_lock()
                if impact_relock:
                    reason = "impact_direction_change"
                elif fast_relock:
                    reason = "high_score_fast_relock"
                else:
                    reason = "stable_new_candidate"
                self._mark_decision("relock_accept", reason, force=True)
                result = self._accept(track, measurement, step_dt, frame_size)
            else:
                self._mark_decision("reject", "candidate_failed_motion_gate")
                result = self._reject(
                    track,
                    step_dt,
                    allow_coast=self.config.coast_on_outlier,
                    frame_size=frame_size,
                    court_filter=court_filter,
                    occlusion_active=person_occlusion_active,
                    coast_reason="person_occlusion_prediction" if person_occlusion_active else None,
                )

        return self._finish_debug(track, result, step_dt, frame_size, debug_before)

    def update_candidates(
        self,
        tracks: Sequence[TrackResult],
        *,
        dt: float | None = None,
        frame_shape: FrameShape = None,
        court_prediction: Any | None = None,
        person_bboxes: PersonBBoxes = None,
    ) -> TrackResult:
        if self._algorithm is not None:
            return self._algorithm.update_candidates(
                tracks,
                dt=dt,
                frame_shape=frame_shape,
                court_prediction=court_prediction,
                person_bboxes=person_bboxes,
            )

        step_dt = self._resolve_dt(dt)
        target_time_seconds = self._time_seconds + step_dt
        frame_size = self._resolve_frame_size(frame_shape)
        court_filter = self._court_filter(court_prediction)
        normalized_person_bboxes = self._normalize_person_bboxes(person_bboxes, frame_size)
        self._raw_candidate_count = sum(
            self._raw_measurement(track, frame_size) is not None
            for track in tracks
        )
        self._court_filter_active = court_filter is not None
        filtered_tracks = self._filter_candidates_by_court(tracks, frame_size, court_filter)
        filtered_candidate_count = sum(
            self._raw_measurement(track, frame_size) is not None
            for track in filtered_tracks
        )
        self._court_filtered_count = max(0, self._raw_candidate_count - filtered_candidate_count)
        tracks = filtered_tracks
        candidate_frame_index = self._frame_index + 1
        self._observe_static_hotspots(tracks, frame_size, court_filter, candidate_frame_index)
        tracks = self._filter_static_hotspots(
            tracks,
            frame_size,
            court_filter,
            step_dt,
            candidate_frame_index,
            target_time_seconds,
        )
        recovery_before_update = self._literature_branch_recovery_remaining
        track = self._select_candidate(
            tracks,
            step_dt,
            frame_size,
            court_filter,
            normalized_person_bboxes,
            target_time_seconds,
        )
        result = self.update(
            track,
            dt=step_dt,
            frame_shape=frame_shape,
            court_prediction=court_prediction,
            person_bboxes=normalized_person_bboxes,
        )
        if recovery_before_update > 0:
            self._literature_branch_recovery_remaining = max(
                0,
                self._literature_branch_recovery_remaining - 1,
            )
        return result

    def _select_candidate(
        self,
        tracks: Sequence[TrackResult],
        dt: float,
        frame_size: FrameSize | None,
        court_filter: _CourtFilter | None,
        person_bboxes: Sequence[PersonBBox],
        target_time_seconds: float,
    ) -> TrackResult:
        if not tracks:
            self._pending_candidate_debug = {
                "candidate_count": 0,
                "selected_candidate_index": -1,
                "selected_candidate_rank": "",
                "candidates": "",
            }
            return TrackResult(ball_xy=[-1.0, -1.0], visible=0, score=0.0)

        primary = max(tracks, key=lambda item: float(item.score))
        if not self._locked or self._last_point is None or self._top_exit_frames_remaining > 0:
            selected_index = list(tracks).index(primary)
            self._pending_candidate_debug = self._candidate_debug(
                tracks,
                dt,
                frame_size,
                court_filter,
                person_bboxes,
                selected_index=selected_index,
                selected_rank="primary",
                target_time_seconds=target_time_seconds,
            )
            return primary

        predicted = self._predict(dt, target_time_seconds=target_time_seconds)
        best_track = primary
        best_rank = float("-inf")
        best_index = -1
        person_occlusion_active = self._person_occlusion_likely(
            dt,
            frame_size,
            person_bboxes,
            target_time_seconds=target_time_seconds,
        )
        for index, candidate in enumerate(tracks):
            rank = self._candidate_rank(
                candidate,
                predicted,
                dt,
                frame_size,
                court_filter,
                person_bboxes,
                person_occlusion_active,
                index,
                target_time_seconds,
            )
            if rank > best_rank:
                best_rank = rank
                best_track = candidate
                best_index = index

        self._pending_candidate_debug = self._candidate_debug(
            tracks,
            dt,
            frame_size,
            court_filter,
            person_bboxes,
            selected_index=best_index,
            selected_rank=f"{best_rank:.4f}",
            target_time_seconds=target_time_seconds,
        )
        return best_track

    def _candidate_rank(
        self,
        track: TrackResult,
        predicted: Point,
        dt: float,
        frame_size: FrameSize | None,
        court_filter: _CourtFilter | None,
        person_bboxes: Sequence[PersonBBox],
        person_occlusion_active: bool,
        index: int,
        target_time_seconds: float,
    ) -> float:
        measurement = self._measurement(track, frame_size, court_filter=court_filter)
        soft_measurement = False
        if measurement is None and self._can_use_soft_measurement(track, frame_size, court_filter):
            measurement = self._raw_measurement(track, frame_size)
            soft_measurement = measurement is not None
        if measurement is None:
            return -1000.0 + float(track.score) - index * 0.05

        score = float(track.score)
        distance_to_prediction = _distance(measurement, predicted)
        passes_gate = self._passes_gate(
            measurement,
            score,
            dt,
            frame_size,
            target_time_seconds=target_time_seconds,
        )
        if soft_measurement and not passes_gate:
            return -500.0 + score - distance_to_prediction / 56.0 - index * 0.05
        pending_relock_continuity = self._continues_pending_relock_candidate(
            measurement,
            score,
            dt,
        )
        gate_bonus = 100.0 if passes_gate or pending_relock_continuity else 0.0
        distance_penalty = distance_to_prediction / 56.0
        heatmap_rank_penalty = index * 0.08
        occlusion_penalty = (
            self.config.person_occlusion_candidate_penalty
            if person_occlusion_active and self._point_inside_person_bbox(measurement, person_bboxes)
            else 0.0
        )
        recovery_penalty = 0.0
        if self._literature_branch_recovery_remaining > 0:
            recovery_penalty = distance_to_prediction / max(
                1.0,
                float(self.config.literature_branch_recovery_distance_scale_px),
            )
        return (
            gate_bonus
            + score * 12.0
            - distance_penalty
            - recovery_penalty
            - heatmap_rank_penalty
            - occlusion_penalty
        )

    def _resolve_frame_size(self, frame_shape: tuple[int, ...] | list[int] | None) -> FrameSize | None:
        if frame_shape is None:
            return self._last_frame_size
        if len(frame_shape) < 2:
            return self._last_frame_size

        height = float(frame_shape[0])
        width = float(frame_shape[1])
        if width <= 0.0 or height <= 0.0:
            return self._last_frame_size

        self._last_frame_size = (width, height)
        return self._last_frame_size

    def _resolve_dt(self, dt: float | None) -> float:
        if dt is not None and dt > 0:
            return float(dt)
        fps = self.config.fps if self.config.fps > 0 else 25.0
        return 1.0 / fps

    def _court_filter(self, court_prediction: Any | None) -> _CourtFilter | None:
        if not self.config.court_filter_enabled:
            return None
        return _extract_court_filter(court_prediction)

    def _measurement(
        self,
        track: TrackResult,
        frame_size: FrameSize | None,
        *,
        court_filter: _CourtFilter | None = None,
    ) -> Point | None:
        if not track.visible or float(track.score) < self.config.min_confidence or len(track.ball_xy) < 2:
            return None

        measurement = self._raw_measurement(track, frame_size)
        if measurement is None:
            return None
        if court_filter is not None and not self._point_inside_court_region(measurement, court_filter, frame_size):
            return None
        return measurement

    def _raw_measurement(self, track: TrackResult, frame_size: FrameSize | None) -> Point | None:
        if not track.visible or len(track.ball_xy) < 2:
            return None

        x, y = float(track.ball_xy[0]), float(track.ball_xy[1])
        if x < 0 or y < 0 or not isfinite(x) or not isfinite(y):
            return None
        if frame_size is not None and not _point_inside_frame(
            (x, y),
            frame_size,
            margin=self.config.frame_measurement_margin_px,
        ):
            return None
        return x, y

    def _can_use_soft_measurement(
        self,
        track: TrackResult,
        frame_size: FrameSize | None,
        court_filter: _CourtFilter | None,
    ) -> bool:
        if not self._locked or self._last_point is None:
            return False
        score = float(track.score)
        if score < self.config.soft_min_confidence or score >= self.config.min_confidence:
            return False
        measurement = self._raw_measurement(track, frame_size)
        if measurement is None:
            return False
        if court_filter is not None and not self._point_inside_court_region(measurement, court_filter, frame_size):
            return False
        return True

    def _normalize_person_bboxes(
        self,
        person_bboxes: Sequence[Sequence[float]] | None,
        frame_size: FrameSize | None,
    ) -> list[PersonBBox]:
        if not self.config.person_occlusion_enabled or not person_bboxes:
            return []

        normalized: list[PersonBBox] = []
        for raw_bbox in person_bboxes:
            try:
                x1, y1, x2, y2 = [float(value) for value in raw_bbox[:4]]
            except (TypeError, ValueError, IndexError):
                continue
            if not all(isfinite(value) for value in (x1, y1, x2, y2)):
                continue
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1
            if frame_size is not None:
                width, height = frame_size
                x1 = min(max(x1, 0.0), width)
                x2 = min(max(x2, 0.0), width)
                y1 = min(max(y1, 0.0), height)
                y2 = min(max(y2, 0.0), height)
            if x2 - x1 < 2.0 or y2 - y1 < 2.0:
                continue
            normalized.append((x1, y1, x2, y2))
        return normalized

    def _person_occlusion_likely(
        self,
        dt: float,
        frame_size: FrameSize | None,
        person_bboxes: Sequence[PersonBBox],
        *,
        target_time_seconds: float | None = None,
    ) -> bool:
        if not self.config.person_occlusion_enabled or not person_bboxes:
            return False
        if not self._locked or self._last_point is None:
            return False
        if _length(self._velocity) < max(0.0, float(self.config.person_occlusion_min_speed_px_per_sec)):
            return False

        predicted = self._predict(dt, target_time_seconds=target_time_seconds)
        if self._point_is_out_of_frame_prediction(predicted, frame_size):
            return False

        for bbox in person_bboxes:
            if _point_inside_rect(self._last_point, bbox, padding=self.config.person_occlusion_padding_px):
                return True
            if _point_inside_rect(predicted, bbox, padding=self.config.person_occlusion_padding_px):
                return True
            if _segment_intersects_rect(
                self._last_point,
                predicted,
                bbox,
                padding=self.config.person_occlusion_padding_px,
            ):
                return True
        return False

    def _point_inside_person_bbox(self, point: Point, person_bboxes: Sequence[PersonBBox]) -> bool:
        for bbox in person_bboxes:
            if _point_inside_rect(point, bbox, padding=self.config.person_occlusion_padding_px):
                return True
        return False

    def _bootstrap(
        self,
        track: TrackResult,
        measurement: Point,
        dt: float,
        frame_size: FrameSize | None,
    ) -> TrackResult:
        if (
            float(track.score) >= self.config.strong_relock_confidence
            and not self._measurement_is_near_top_edge(measurement, frame_size)
        ):
            self._mark_decision("bootstrap_accept", "strong_confidence")
            return self._accept(track, measurement, dt, frame_size)

        candidate_confirmed = self._update_candidate(measurement, float(track.score), dt)
        if (
            candidate_confirmed
            and self._candidate is not None
            and self._candidate.score >= self.config.bootstrap_confirm_min_confidence
        ):
            if self._candidate_is_static_bootstrap_cluster():
                self._mark_decision("bootstrap_wait", "static_bootstrap_candidate", force=True)
                return self._invisible(track)
            self._mark_decision("bootstrap_accept", "candidate_confirmed")
            return self._accept(track, measurement, dt, frame_size)

        self._mark_decision("bootstrap_wait", "waiting_for_candidate_confirmation")
        return self._invisible(track)

    def _passes_gate(
        self,
        measurement: Point,
        score: float,
        dt: float,
        frame_size: FrameSize | None,
        *,
        target_time_seconds: float | None = None,
    ) -> bool:
        assert self._last_point is not None

        predicted = self._predict(dt, target_time_seconds=target_time_seconds)
        if self._point_is_out_of_frame_prediction(predicted, frame_size):
            return False

        distance_to_prediction = _distance(measurement, predicted)
        distance_to_last = _distance(measurement, self._last_point)
        observed_speed = distance_to_last / max(dt * max(self._missed_frames + 1, 1), 1e-6)

        if observed_speed > self.config.max_speed_px_per_sec:
            return False

        velocity_px_per_frame = _length(self._velocity) * dt
        score_bonus = max(0.0, score - self.config.min_confidence) * 160.0
        allowed_distance = (
            self.config.base_gate_px
            + min(velocity_px_per_frame * 1.8, self.config.max_gate_px * 0.55)
            + self._missed_frames * self.config.missed_gate_growth_px
            + score_bonus
        )
        allowed_distance = min(max(allowed_distance, self.config.base_gate_px), self.config.max_gate_px)
        if distance_to_prediction > allowed_distance:
            return False

        if not self._passes_parabola_gate(
            measurement,
            score,
            target_time_seconds=target_time_seconds,
        ):
            return False

        if self._can_relax_inertia(score, distance_to_prediction):
            return True

        return self._passes_inertia(measurement, score, dt)

    def _passes_close_gate(
        self,
        measurement: Point,
        score: float,
        dt: float,
        frame_size: FrameSize | None,
    ) -> bool:
        assert self._last_point is not None

        if score < self.config.close_gate_confidence:
            return False

        predicted = self._predict(dt)
        if self._point_is_out_of_frame_prediction(predicted, frame_size):
            return False

        elapsed = max(dt * max(self._missed_frames + 1, 1), 1e-6)
        distance_to_last = _distance(measurement, self._last_point)
        if distance_to_last / elapsed > self.config.max_speed_px_per_sec:
            return False

        return _distance(measurement, predicted) <= self.config.close_gate_px

    def _invalidate_parabola_on_y_direction_conflict(self, measurement: Point, dt: float) -> None:
        if self._last_point is None:
            return

        parabola_prediction = self._parabola_prediction()
        if parabola_prediction is None:
            return

        actual_delta_y = measurement[1] - self._last_point[1]
        if not self._parabola_y_direction_conflicts(measurement, parabola_prediction):
            return

        elapsed = max(dt * max(self._missed_frames + 1, 1), 1e-6)
        actual_velocity_y = actual_delta_y / elapsed
        self._history.clear()
        self._real_detections_since_relock = 0
        self._velocity = (self._velocity[0], actual_velocity_y)

    def _parabola_y_direction_conflicts(
        self,
        measurement: Point,
        prediction: _ParabolaPrediction,
    ) -> bool:
        if self._last_point is None:
            return False
        min_delta = max(0.0, float(self.config.parabola_direction_conflict_min_delta_px))
        parabola_delta_y = prediction.point[1] - self._last_point[1]
        actual_delta_y = measurement[1] - self._last_point[1]
        return (
            abs(parabola_delta_y) >= min_delta
            and abs(actual_delta_y) >= min_delta
            and parabola_delta_y * actual_delta_y < 0.0
        )

    def _passes_parabola_gate(
        self,
        measurement: Point,
        score: float,
        *,
        target_time_seconds: float | None = None,
    ) -> bool:
        prediction = self._parabola_prediction(target_time_seconds)
        if prediction is None:
            return True

        score_bonus = max(0.0, score - self.config.min_confidence) * self.config.parabola_score_bonus_px
        allowed_distance = (
            self.config.parabola_gate_px
            + prediction.fit_rmse * self.config.parabola_fit_error_scale
            + prediction.gap_frames * self.config.parabola_gate_growth_px
            + score_bonus
        )
        allowed_distance = min(max(allowed_distance, self.config.parabola_gate_px), self.config.parabola_max_gate_px)
        return _distance(measurement, prediction.point) <= allowed_distance

    def _should_fill_outlier_with_parabola(self, measurement: Point) -> bool:
        if not self.config.parabola_fill_on_outlier:
            return False
        prediction = self._parabola_prediction()
        if prediction is None:
            return False
        if self._parabola_y_direction_conflicts(measurement, prediction):
            return False
        max_distance = max(0.0, float(self.config.parabola_fill_max_outlier_distance_px))
        return _distance(measurement, prediction.point) <= max_distance

    def _passes_inertia(self, measurement: Point, score: float, dt: float) -> bool:
        assert self._last_point is not None

        speed = _length(self._velocity)
        if speed < self.config.inertia_min_speed_px_per_sec:
            return True

        elapsed = max(dt * max(self._missed_frames + 1, 1), 1e-6)
        displacement = (
            measurement[0] - self._last_point[0],
            measurement[1] - self._last_point[1],
        )
        candidate_velocity = (displacement[0] / elapsed, displacement[1] / elapsed)
        acceleration = _distance(candidate_velocity, self._velocity) / elapsed
        if acceleration > self.config.max_accel_px_per_sec2:
            return False

        forward_px = _dot(displacement, self._velocity) / max(speed, 1e-6)
        if forward_px < -self.config.max_reverse_px:
            return False

        lateral_px = abs(displacement[0] * self._velocity[1] - displacement[1] * self._velocity[0]) / max(speed, 1e-6)
        expected_step_px = speed * elapsed
        score_bonus = max(0.0, score - self.config.min_confidence) * 35.0
        allowed_lateral = min(
            self.config.max_lateral_error_px,
            34.0 + expected_step_px * 0.45 + score_bonus,
        )
        return lateral_px <= allowed_lateral

    def _measurement_is_near_top_edge(self, measurement: Point, frame_size: FrameSize | None) -> bool:
        if frame_size is None:
            return False
        return measurement[1] <= self._top_exit_band(frame_size)

    def _can_relax_inertia(self, score: float, distance_to_prediction: float) -> bool:
        if score < self.config.inertia_relax_confidence:
            return False
        if score > self.config.inertia_relax_max_confidence:
            return False
        max_distance = self.config.base_gate_px * max(0.0, float(self.config.inertia_relax_prediction_gate_ratio))
        return distance_to_prediction <= max_distance

    def _predict(self, dt: float, *, target_time_seconds: float | None = None) -> Point:
        assert self._last_point is not None
        parabola_prediction = self._parabola_prediction(target_time_seconds)
        if parabola_prediction is not None:
            return parabola_prediction.point

        frames = 1 if self._coast_frames > 0 else max(self._missed_frames + 1, 1)
        return (
            self._last_point[0] + self._velocity[0] * dt * frames,
            self._last_point[1] + self._velocity[1] * dt * frames,
        )

    def _prediction_is_out_of_frame(self, dt: float, frame_size: FrameSize | None) -> bool:
        if self._last_point is None:
            return False
        return self._point_is_out_of_frame_prediction(self._predict(dt), frame_size)

    def _point_is_out_of_frame_prediction(self, point: Point, frame_size: FrameSize | None) -> bool:
        if frame_size is None:
            return False
        return not _point_inside_frame(
            point,
            frame_size,
            margin=self.config.out_of_frame_prediction_margin_px,
        )

    def _accept(
        self,
        track: TrackResult,
        measurement: Point,
        dt: float,
        frame_size: FrameSize | None,
    ) -> TrackResult:
        continues_real_detection = self._locked and self._missed_frames == 0 and self._coast_frames == 0
        if self._last_point is not None:
            raw_velocity = (
                (measurement[0] - self._last_point[0]) / max(dt * max(self._missed_frames + 1, 1), 1e-6),
                (measurement[1] - self._last_point[1]) / max(dt * max(self._missed_frames + 1, 1), 1e-6),
            )
            blend = min(max(self.config.velocity_blend, 0.0), 1.0)
            self._velocity = (
                blend * raw_velocity[0] + (1.0 - blend) * self._velocity[0],
                blend * raw_velocity[1] + (1.0 - blend) * self._velocity[1],
            )
        else:
            self._velocity = (0.0, 0.0)

        self._last_point = measurement
        self._render_point = self._smooth_render_point(measurement)
        self._missed_frames = 0
        self._coast_frames = 0
        self._locked = True
        self._candidate = None
        self._record_history(measurement, float(track.score))
        if continues_real_detection:
            self._real_detections_since_relock += 1
        else:
            self._real_detections_since_relock = 1
        self._mark_decision("accept", "passes_motion_gate")
        return self._visible(track, self._render_point, frame_size)

    def _reject(
        self,
        track: TrackResult,
        dt: float,
        *,
        allow_coast: bool,
        allow_short_gap_bridge: bool = False,
        frame_size: FrameSize | None,
        court_filter: _CourtFilter | None,
        occlusion_active: bool = False,
        coast_reason: str | None = None,
    ) -> TrackResult:
        if self._top_exit_is_likely(dt, frame_size):
            self._enter_top_exit()
            self._mark_decision("top_exit_enter", "likely_top_exit", force=True)
            return self._invisible(track)

        if allow_coast and (
            self._can_coast(dt, frame_size, court_filter, occlusion_active=occlusion_active)
            or (
                allow_short_gap_bridge
                and self._literature_short_gap_can_bridge(dt, frame_size, court_filter)
            )
        ):
            return self._coast(track, dt, frame_size, court_filter, reason=coast_reason)

        self._missed_frames += 1
        if self._missed_frames > self.config.max_missed_frames:
            self._drop_lock()
            self._mark_decision("drop_lock", "max_missed_frames", force=True)
        else:
            self._mark_decision("reject", "hidden_after_reject")
        return self._invisible(track)

    def _literature_short_gap_can_bridge(
        self,
        dt: float,
        frame_size: FrameSize | None,
        court_filter: _CourtFilter | None,
    ) -> bool:
        if not self.config.literature_short_gap_enabled:
            return False
        if not self._locked or self._last_point is None:
            return False
        if self._missed_frames >= max(0, int(self.config.literature_short_gap_frames)):
            return False
        min_points = max(2, int(self.config.literature_short_gap_min_real_points))
        if self._real_detections_since_relock < min_points or len(self._history) < min_points:
            return False
        recent = list(self._history)[-min_points:]
        if recent[-1].frame_index != self._frame_index - 1:
            return False
        if any(
            recent[index].frame_index != recent[index - 1].frame_index + 1
            for index in range(1, len(recent))
        ):
            return False
        prediction = self._predict(dt)
        if frame_size is not None:
            _, height = frame_size
            top_band = height * max(0.0, float(self.config.literature_short_gap_top_band_ratio))
            if self._last_point[1] <= top_band or prediction[1] <= top_band:
                return False
        if self._top_exit_is_likely(dt, frame_size):
            return False
        return self._point_can_be_predicted(prediction, frame_size, court_filter)

    def _literature_prediction_outlier(
        self,
        measurement: Point,
        score: float,
        dt: float,
        frame_size: FrameSize | None,
    ) -> bool:
        if not self.config.literature_branch_guard_enabled:
            return False
        if not self._locked or self._last_point is None:
            return False
        if self._missed_frames or self._coast_frames:
            return False
        min_points = max(3, int(self.config.literature_short_gap_min_real_points))
        if len(self._history) < min_points:
            return False
        recent = list(self._history)[-min_points:]
        if recent[-1].frame_index != self._frame_index - 1:
            return False
        if recent[-1].frame_index - recent[0].frame_index > min_points:
            return False
        if score < float(self.config.literature_branch_guard_min_confidence):
            return False
        speed = _length(self._velocity)
        if speed > max(0.0, float(self.config.literature_branch_guard_max_speed_px_per_sec)):
            return False
        displacement_y = measurement[1] - self._last_point[1]
        if displacement_y < max(0.0, float(self.config.literature_branch_guard_min_downward_jump_px)):
            return False
        prediction = self._predict(dt)
        if _distance(measurement, prediction) < max(
            0.0,
            float(self.config.literature_branch_guard_min_prediction_error_px),
        ):
            return False
        if self._point_is_out_of_frame_prediction(prediction, frame_size):
            return False
        return True

    def _literature_occlusion_model_reset(
        self,
        measurement: Point,
        score: float,
        dt: float,
    ) -> bool:
        if not self.config.literature_occlusion_reset_enabled:
            return False
        if self._last_point is None or self._missed_frames < 1:
            return False
        if score < float(self.config.literature_occlusion_reset_confidence):
            return False
        speed = _length(self._velocity)
        if speed < max(1.0, float(self.config.inertia_min_speed_px_per_sec)):
            return False
        displacement = (
            measurement[0] - self._last_point[0],
            measurement[1] - self._last_point[1],
        )
        displacement_length = _length(displacement)
        if displacement_length <= 1e-6:
            return False
        prediction = self._predict(dt)
        if _distance(measurement, prediction) > float(
            self.config.literature_occlusion_reset_max_prediction_error_px
        ):
            return False
        direction_cos = _dot(displacement, self._velocity) / max(displacement_length * speed, 1e-6)
        angle_threshold = cos(radians(float(self.config.literature_occlusion_reset_min_angle_deg)))
        return direction_cos <= angle_threshold

    def _can_coast(
        self,
        dt: float,
        frame_size: FrameSize | None,
        court_filter: _CourtFilter | None,
        *,
        occlusion_active: bool = False,
    ) -> bool:
        if not self._locked or self._last_point is None:
            return False

        parabola_prediction = self._parabola_prediction()
        if (
            self._coast_frames < self.config.parabola_max_gap_frames
            and parabola_prediction is not None
        ):
            return self._point_can_be_predicted(parabola_prediction.point, frame_size, court_filter)

        predicted = (
            self._last_point[0] + self._velocity[0] * dt,
            self._last_point[1] + self._velocity[1] * dt,
        )

        max_coast_frames = self.config.max_coast_frames
        min_speed = self.config.min_coast_speed_px_per_sec
        if occlusion_active:
            max_coast_frames = max(max_coast_frames, int(self.config.person_occlusion_coast_frames))
            min_speed = min(min_speed, float(self.config.person_occlusion_min_speed_px_per_sec))

        return (
            self._coast_frames < max_coast_frames
            and _length(self._velocity) >= min_speed
            and self._point_can_be_predicted(predicted, frame_size, court_filter)
        )

    def _point_can_be_predicted(
        self,
        point: Point,
        frame_size: FrameSize | None,
        court_filter: _CourtFilter | None,
    ) -> bool:
        if self._point_is_out_of_frame_prediction(point, frame_size):
            return False
        if court_filter is not None and not self._point_inside_court_region(point, court_filter, frame_size):
            return False
        return True

    def _point_inside_court_region(
        self,
        point: Point,
        court_filter: _CourtFilter,
        frame_size: FrameSize | None,
    ) -> bool:
        if court_filter.corners is not None:
            return _point_inside_projected_court_air(
                point,
                court_filter.corners,
                margin_px=self.config.court_filter_margin_px,
                air_extension_ratio=self.config.court_air_extension_ratio,
                lateral_expansion_ratio=self.config.court_air_lateral_expansion_ratio,
                frame_size=frame_size,
            )

        if court_filter.image_to_court_h is None:
            return False
        return _point_inside_court_plane(
            point,
            court_filter.image_to_court_h,
            self.config.court_filter_margin_cm,
        )

    def _filter_candidates_by_court(
        self,
        tracks: Sequence[TrackResult],
        frame_size: FrameSize | None,
        court_filter: _CourtFilter | None,
    ) -> list[TrackResult]:
        if court_filter is None:
            return list(tracks)

        selected: list[TrackResult] = []
        best_score = 0.0
        heatmap_shape: list[int] = []
        for track in tracks:
            best_score = max(best_score, float(track.score))
            if not heatmap_shape:
                heatmap_shape = list(track.heatmap_shape)

            measurement = self._raw_measurement(track, frame_size)
            if measurement is None:
                continue
            if self._point_inside_court_region(measurement, court_filter, frame_size):
                selected.append(track)

        if selected:
            return selected

        return [
            TrackResult(
                ball_xy=[-1.0, -1.0],
                visible=0,
                score=best_score,
                heatmap_shape=heatmap_shape,
            )
        ]

    def _observe_static_hotspots(
        self,
        tracks: Sequence[TrackResult],
        frame_size: FrameSize | None,
        court_filter: _CourtFilter | None,
        frame_index: int,
    ) -> None:
        if not self.config.static_hotspot_enabled:
            self._static_hotspot_count = 0
            return

        self._prune_static_hotspots(frame_index)
        updated_indices: set[int] = set()
        for track in tracks:
            if float(track.score) < self.config.min_confidence:
                continue
            measurement = self._measurement(track, frame_size, court_filter=court_filter)
            if measurement is None:
                continue

            index = self._nearest_static_hotspot_index(measurement)
            if index is None:
                self._static_hotspots.append(
                    _StaticHotspot(
                        anchor=measurement,
                        point=measurement,
                        score=float(track.score),
                        first_frame=frame_index,
                        last_frame=frame_index,
                    )
                )
                updated_indices.add(len(self._static_hotspots) - 1)
                continue

            if index in updated_indices:
                continue
            hotspot = self._static_hotspots[index]
            if frame_index > hotspot.last_frame:
                hotspot.count += 1
            hotspot.last_frame = frame_index
            hotspot.score = max(hotspot.score, float(track.score))
            hotspot.max_distance = max(hotspot.max_distance, _distance(measurement, hotspot.anchor))
            hotspot.point = (
                0.78 * hotspot.point[0] + 0.22 * measurement[0],
                0.78 * hotspot.point[1] + 0.22 * measurement[1],
            )
            if self._hotspot_is_static(hotspot):
                hotspot.suppressed_until = max(
                    hotspot.suppressed_until,
                    frame_index + max(1, int(self.config.static_hotspot_suppression_frames)),
                )
            updated_indices.add(index)

        self._static_hotspot_count = sum(
            1 for hotspot in self._static_hotspots if self._hotspot_is_active(hotspot, frame_index)
        )

    def _filter_static_hotspots(
        self,
        tracks: Sequence[TrackResult],
        frame_size: FrameSize | None,
        court_filter: _CourtFilter | None,
        dt: float,
        frame_index: int,
        target_time_seconds: float,
    ) -> list[TrackResult]:
        self._static_filtered_count = 0
        if not self.config.static_hotspot_enabled:
            return list(tracks)

        selected: list[TrackResult] = []
        best_score = 0.0
        heatmap_shape: list[int] = []
        for track in tracks:
            best_score = max(best_score, float(track.score))
            if not heatmap_shape:
                heatmap_shape = list(track.heatmap_shape)

            measurement = self._measurement(track, frame_size, court_filter=court_filter)
            if measurement is not None and self._point_matches_static_hotspot(
                measurement,
                dt,
                frame_index,
                frame_size,
                target_time_seconds,
            ):
                self._static_filtered_count += 1
                continue
            selected.append(track)

        if selected:
            return selected
        if self._static_filtered_count <= 0:
            return list(tracks)
        return [
            TrackResult(
                ball_xy=[-1.0, -1.0],
                visible=0,
                score=best_score,
                heatmap_shape=heatmap_shape,
            )
        ]

    def _nearest_static_hotspot_index(self, point: Point) -> int | None:
        radius = max(1.0, float(self.config.static_hotspot_radius_px))
        best_index: int | None = None
        best_distance = radius
        for index, hotspot in enumerate(self._static_hotspots):
            distance = _distance(point, hotspot.point)
            if distance <= best_distance:
                best_index = index
                best_distance = distance
        return best_index

    def _prune_static_hotspots(self, frame_index: int) -> None:
        memory_frames = max(1, int(self.config.static_hotspot_memory_frames))
        self._static_hotspots = [
            hotspot
            for hotspot in self._static_hotspots
            if frame_index - hotspot.last_frame <= memory_frames or frame_index <= hotspot.suppressed_until
        ]

    def _hotspot_is_static(self, hotspot: _StaticHotspot) -> bool:
        return (
            hotspot.count >= max(1, int(self.config.static_hotspot_min_frames))
            and hotspot.max_distance <= max(0.0, float(self.config.static_hotspot_max_motion_px))
        )

    def _hotspot_is_active(self, hotspot: _StaticHotspot, frame_index: int) -> bool:
        return frame_index <= hotspot.suppressed_until or self._hotspot_is_static(hotspot)

    def _point_matches_static_hotspot(
        self,
        point: Point,
        dt: float,
        frame_index: int,
        frame_size: FrameSize | None,
        target_time_seconds: float,
    ) -> bool:
        radius = max(1.0, float(self.config.static_hotspot_radius_px))
        for hotspot in self._static_hotspots:
            if not self._hotspot_is_active(hotspot, frame_index) and not self._hotspot_is_slow_edge_drift(
                hotspot,
                point,
                frame_size,
            ):
                continue
            if _distance(point, hotspot.point) > radius:
                continue
            if self._moving_track_can_claim_static_point(point, dt, target_time_seconds):
                return False
            return True
        return False

    def _hotspot_is_slow_edge_drift(
        self,
        hotspot: _StaticHotspot,
        point: Point,
        frame_size: FrameSize | None,
    ) -> bool:
        if frame_size is None:
            return False
        if hotspot.count < max(1, int(self.config.static_hotspot_min_frames)):
            return False
        if hotspot.max_distance > max(0.0, float(self.config.static_hotspot_edge_max_motion_px)):
            return False

        width, height = frame_size
        x_margin = width * max(0.0, float(self.config.static_hotspot_edge_band_ratio))
        y_margin = height * max(0.0, float(self.config.static_hotspot_edge_band_ratio))
        return (
            point[0] <= x_margin
            or point[0] >= width - x_margin
            or point[1] <= y_margin
            or point[1] >= height - y_margin
        )

    def _moving_track_can_claim_static_point(
        self,
        point: Point,
        dt: float,
        target_time_seconds: float,
    ) -> bool:
        if not self._locked or self._last_point is None:
            return False
        speed = _length(self._velocity)
        if speed < max(0.0, float(self.config.static_hotspot_tracking_speed_px_per_sec)):
            return False
        predicted = self._predict(dt, target_time_seconds=target_time_seconds)
        return _distance(point, predicted) <= max(float(self.config.close_gate_px), float(self.config.base_gate_px))

    def _coast(
        self,
        track: TrackResult,
        dt: float,
        frame_size: FrameSize | None,
        court_filter: _CourtFilter | None,
        *,
        reason: str | None = None,
    ) -> TrackResult:
        assert self._last_point is not None

        parabola_prediction = self._parabola_prediction()
        if parabola_prediction is not None:
            predicted = parabola_prediction.point
            self._velocity = (
                (predicted[0] - self._last_point[0]) / max(dt, 1e-6),
                (predicted[1] - self._last_point[1]) / max(dt, 1e-6),
            )
            score_decay = self.config.parabola_score_decay
            self._mark_decision("coast", reason or "parabola_prediction", force=True)
        else:
            predicted = (
                self._last_point[0] + self._velocity[0] * dt,
                self._last_point[1] + self._velocity[1] * dt,
            )
            self._velocity = (
                self._velocity[0] * self.config.coast_velocity_decay,
                self._velocity[1] * self.config.coast_velocity_decay,
            )
            score_decay = self.config.coast_score_decay
            self._mark_decision("coast", reason or "velocity_prediction", force=True)

        if not self._point_can_be_predicted(predicted, frame_size, court_filter):
            self._missed_frames += 1
            self._mark_decision("reject", "prediction_out_of_court", force=True)
            return self._invisible(track)

        self._last_point = predicted
        self._render_point = self._smooth_render_point(predicted)
        self._missed_frames += 1
        self._coast_frames += 1
        source_score = min(max(float(track.score), 0.0), self.config.min_confidence)
        score = source_score * (score_decay ** self._coast_frames)
        if frame_size is not None and not _point_inside_frame(self._render_point, frame_size, margin=0.0):
            return TrackResult(
                ball_xy=[-1.0, -1.0],
                visible=0,
                score=max(0.0, score),
                heatmap_shape=list(track.heatmap_shape),
            )
        return TrackResult(
            ball_xy=[float(self._render_point[0]), float(self._render_point[1])],
            visible=1,
            score=max(0.0, score),
            heatmap_shape=list(track.heatmap_shape),
        )

    def _update_candidate(self, measurement: Point, score: float, dt: float) -> bool:
        relock_distance = max(
            self.config.relock_distance_px,
            self.config.relock_max_speed_px_per_sec * max(dt, 1e-6),
        )
        if self._candidate is None or _distance(measurement, self._candidate.point) > relock_distance:
            self._candidate = _RelockCandidate(
                point=measurement,
                score=score,
                last_measurement=measurement,
                measurements=(measurement,),
            )
        else:
            if self._last_point is not None and self._candidate.last_measurement is not None:
                previous_direction = (
                    self._candidate.last_measurement[0] - self._last_point[0],
                    self._candidate.last_measurement[1] - self._last_point[1],
                )
                current_direction = (
                    measurement[0] - self._last_point[0],
                    measurement[1] - self._last_point[1],
                )
                self._candidate.direction_consistent = (
                    self._candidate.direction_consistent
                    and _dot(previous_direction, current_direction) > 0.0
                )
            self._candidate.point = (
                0.35 * self._candidate.point[0] + 0.65 * measurement[0],
                0.35 * self._candidate.point[1] + 0.65 * measurement[1],
            )
            self._candidate.score = max(self._candidate.score, score)
            self._candidate.count += 1
            self._candidate.last_measurement = measurement
            measurement_limit = max(1, int(self.config.relock_confirm_frames))
            self._candidate.measurements = (*self._candidate.measurements, measurement)[-measurement_limit:]

        return (
            self._candidate.count >= self.config.relock_confirm_frames
            and self._candidate.score >= self.config.relock_confidence
        )

    def _continues_pending_relock_candidate(
        self,
        measurement: Point,
        score: float,
        dt: float,
    ) -> bool:
        if self._candidate is None or self._candidate.last_measurement is None or self._last_point is None:
            return False
        if self._missed_frames < self.config.impact_relock_min_missed_frames:
            return False
        if score < self.config.impact_relock_confidence:
            return False
        if self._candidate.score < self.config.impact_relock_confidence:
            return False

        relock_distance = max(
            self.config.relock_distance_px,
            self.config.relock_max_speed_px_per_sec * max(dt, 1e-6),
        )
        if _distance(measurement, self._candidate.last_measurement) > relock_distance:
            return False

        previous_direction = (
            self._candidate.last_measurement[0] - self._last_point[0],
            self._candidate.last_measurement[1] - self._last_point[1],
        )
        current_direction = (
            measurement[0] - self._last_point[0],
            measurement[1] - self._last_point[1],
        )
        return _dot(previous_direction, current_direction) > 0.0

    def _candidate_is_static_bootstrap_cluster(self) -> bool:
        if self._candidate is None:
            return False

        min_count = max(2, int(self.config.relock_confirm_frames))
        measurements = self._candidate.measurements[-min_count:]
        if len(measurements) < min_count:
            return False

        xs = [point[0] for point in measurements]
        ys = [point[1] for point in measurements]
        x_span = max(xs) - min(xs)
        mean_y = sum(ys) / len(ys)
        y_std = (sum((y - mean_y) ** 2 for y in ys) / len(ys)) ** 0.5
        return (
            x_span < max(0.0, float(self.config.bootstrap_static_max_x_span_px))
            and y_std < max(0.0, float(self.config.bootstrap_static_max_y_std_px))
        )

    def _should_fast_relock(self, current_score: float) -> bool:
        if self._candidate is None:
            return False
        return (
            self._candidate.count >= 2
            and current_score > self.config.high_score_fast_relock_confidence
            and self._candidate.score > self.config.high_score_fast_relock_confidence
            and self._candidate.direction_consistent
        )

    def _should_relock(self) -> bool:
        if self._candidate is None:
            return False
        if self._candidate.score >= self.config.strong_relock_confidence:
            return True
        return self._missed_frames >= self.config.relock_after_missed_frames

    def _should_impact_relock(self, measurement: Point, score: float, dt: float) -> bool:
        if self._candidate is None or self._last_point is None:
            return False
        if score < self.config.impact_relock_confidence:
            return False
        if self._missed_frames < self.config.impact_relock_min_missed_frames:
            return False
        if self._candidate.count < self.config.impact_relock_confirm_frames:
            return False
        if self._candidate.score < self.config.impact_relock_confidence:
            return False

        speed = _length(self._velocity)
        if speed < self.config.inertia_min_speed_px_per_sec:
            return False

        elapsed = max(dt * max(self._missed_frames + 1, 1), 1e-6)
        displacement = (
            measurement[0] - self._last_point[0],
            measurement[1] - self._last_point[1],
        )
        displacement_length = _length(displacement)
        if displacement_length <= 1e-6:
            return False

        predicted = self._predict(dt)
        max_prediction_error = (
            float(self.config.impact_relock_max_prediction_error_px)
            + max(0, self._missed_frames - 1) * float(self.config.impact_relock_max_prediction_error_per_missed_px)
        )
        if _distance(measurement, predicted) > max_prediction_error:
            return False

        direction_cos = _dot(displacement, self._velocity) / max(displacement_length * speed, 1e-6)
        angle_threshold_cos = cos(radians(self.config.impact_relock_min_angle_deg))
        return direction_cos <= angle_threshold_cos

    def _drop_lock(self) -> None:
        self._locked = False
        self._last_point = None
        self._render_point = None
        self._velocity = (0.0, 0.0)
        self._missed_frames = 0
        self._coast_frames = 0
        self._candidate = None
        self._real_detections_since_relock = 0
        self._history.clear()
        self._top_exit_frames_remaining = 0
        self._literature_branch_recovery_remaining = 0

    def _start_decision(self) -> None:
        self._decision_action = "unknown"
        self._decision_reason = ""

    def _mark_decision(self, action: str, reason: str, *, force: bool = False) -> None:
        if force or self._decision_action == "unknown":
            self._decision_action = action
            self._decision_reason = reason

    def _enter_top_exit(self) -> None:
        suppress_frames = max(0, int(self.config.top_exit_suppression_frames))
        self._drop_lock()
        self._top_exit_frames_remaining = suppress_frames

    def _enter_ground_bounce_suppression(self) -> None:
        self._drop_lock()
        self._ground_bounce_suppression_remaining_s = max(
            0.0,
            float(self.config.ground_bounce_guard_suppression_seconds),
        )

    def _ground_bounce_reason(
        self,
        measurement: Point | None,
        score: float,
        frame_size: FrameSize | None,
        court_filter: _CourtFilter | None,
        person_bboxes: Sequence[PersonBBox],
    ) -> str | None:
        if not self.config.ground_bounce_guard_enabled or measurement is None:
            return None
        if not self._locked:
            return None
        history = list(self._history)
        if len(history) < 4:
            return None
        items = history + [
            _TrajectoryPoint(
                frame_index=self._frame_index,
                time_seconds=self._time_seconds,
                point=measurement,
                score=float(score),
            )
        ]
        peak_index = max(range(len(items)), key=lambda index: items[index].point[1])
        before = items[:peak_index]
        after = items[peak_index + 1 :]
        if not before or not after:
            return None
        peak = items[peak_index].point
        downward_px = peak[1] - min(item.point[1] for item in before)
        upward_px = peak[1] - measurement[1]
        if downward_px < float(self.config.ground_bounce_guard_min_downward_px):
            return None
        if upward_px < float(self.config.ground_bounce_guard_min_upward_px):
            return None
        previous_y = peak[1]
        max_backtrack = float(self.config.ground_bounce_guard_max_backtrack_px)
        for item in after:
            if item.point[1] > previous_y + max_backtrack:
                return None
            previous_y = item.point[1]
        if not self._point_is_plausible_ground_contact(peak, frame_size, court_filter):
            return None

        contact_clear_of_players = bool(person_bboxes) and not any(
            _point_inside_rect(
                peak,
                bbox,
                padding=float(self.config.ground_bounce_guard_person_clearance_px),
            )
            for bbox in person_bboxes
        )
        if contact_clear_of_players and len(after) >= 2:
            approach_reference = items[max(0, peak_index - len(after))]
            approach_duration = items[peak_index].time_seconds - approach_reference.time_seconds
            rebound_duration = items[-1].time_seconds - items[peak_index].time_seconds
            approach_speed = (
                peak[1] - approach_reference.point[1]
            ) / max(approach_duration, 1e-6)
            rebound_speed = upward_px / max(rebound_duration, 1e-6)
            compact_lateral_px = abs(measurement[0] - peak[0])
            if (
                approach_speed > 0.0
                and upward_px <= float(self.config.ground_bounce_guard_compact_max_upward_px)
                and compact_lateral_px <= float(self.config.ground_bounce_guard_compact_max_lateral_px)
                and rebound_speed / approach_speed
                <= float(self.config.ground_bounce_guard_compact_max_speed_ratio)
            ):
                return "compact_ground_bounce"

        if score > float(self.config.ground_bounce_guard_max_score):
            return None
        if len(after) < max(1, int(self.config.ground_bounce_guard_min_upward_points)):
            return None
        low_score_points = sum(
            item.score <= float(self.config.ground_bounce_guard_max_score)
            for item in after
        )
        if low_score_points < max(1, int(self.config.ground_bounce_guard_min_low_score_points)):
            return None
        if abs(measurement[0] - peak[0]) > float(self.config.ground_bounce_guard_max_lateral_px):
            return None
        return "low_confidence_ground_bounce_tail"

    def _point_is_plausible_ground_contact(
        self,
        point: Point,
        frame_size: FrameSize | None,
        court_filter: _CourtFilter | None,
    ) -> bool:
        if court_filter is not None and court_filter.image_to_court_h is not None:
            return _point_inside_court_plane(
                point,
                court_filter.image_to_court_h,
                self.config.court_filter_margin_cm,
            )
        if court_filter is not None and court_filter.corners is not None:
            return _point_inside_projected_court_air(
                point,
                court_filter.corners,
                margin_px=self.config.court_filter_margin_px,
                air_extension_ratio=0.0,
                frame_size=frame_size,
            )
        if frame_size is None:
            return False
        _, height = frame_size
        return point[1] >= height * float(self.config.ground_bounce_guard_min_height_ratio)

    def _top_exit_is_likely(self, dt: float, frame_size: FrameSize | None) -> bool:
        if not self.config.top_exit_enabled or frame_size is None:
            return False
        if not self._locked or self._last_point is None:
            return False

        moving_up_fast = self._velocity[1] <= -abs(self.config.top_exit_min_up_speed_px_per_sec)
        moving_up_from_history = self._history_moves_up_toward_top(dt)
        if not moving_up_fast and not moving_up_from_history:
            return False

        top_band = self._top_exit_band(frame_size)
        linear_prediction = self._linear_prediction(dt)
        return min(self._last_point[1], linear_prediction[1]) <= top_band

    def _history_moves_up_toward_top(self, dt: float) -> bool:
        if len(self._history) < 2:
            return False

        window_size = max(2, int(self.config.top_exit_history_frames))
        history = list(self._history)[-window_size:]
        first = history[0]
        last = history[-1]
        frame_span = max(last.frame_index - first.frame_index, 1)
        elapsed_seconds = last.time_seconds - first.time_seconds
        if elapsed_seconds <= 0.0:
            elapsed_seconds = dt * frame_span
        upward_motion = first.point[1] - last.point[1]
        upward_speed = upward_motion / max(elapsed_seconds, 1e-6)
        return (
            upward_motion >= self.config.top_exit_min_up_motion_px
            and upward_speed >= abs(self.config.top_exit_min_up_speed_px_per_sec) * 0.45
        )

    def _measurement_reverses_after_top_exit(
        self,
        measurement: Point,
        dt: float,
        frame_size: FrameSize | None,
    ) -> bool:
        if not self._top_exit_is_likely(dt, frame_size) or self._last_point is None:
            return False

        linear_prediction = self._linear_prediction(dt)
        top_band = self._top_exit_band(frame_size)
        if linear_prediction[1] > top_band:
            return False

        return measurement[1] > self._last_point[1] + self.config.top_exit_reversal_min_delta_px

    def _measurement_is_top_edge_hallucination(
        self,
        measurement: Point,
        dt: float,
        frame_size: FrameSize | None,
    ) -> bool:
        if not self.config.top_exit_enabled or frame_size is None:
            return False
        if not self._locked or self._last_point is None:
            return False
        if self._top_exit_is_likely(dt, frame_size):
            return False

        _, height = frame_size
        top_band = self._top_exit_band(frame_size)
        if measurement[1] > top_band:
            return False

        predicted = self._predict(dt)
        nearest_track_y = min(self._last_point[1], predicted[1])
        required_gap = max(
            float(self.config.top_edge_hallucination_min_gap_px),
            float(height) * max(0.0, float(self.config.top_edge_hallucination_min_gap_ratio)),
        )
        return nearest_track_y - measurement[1] >= required_gap

    def _top_exit_band(self, frame_size: FrameSize | None) -> float:
        if frame_size is None:
            return float(self.config.top_exit_margin_px)
        _, height = frame_size
        return max(
            float(self.config.top_exit_margin_px),
            float(height) * max(0.0, float(self.config.top_exit_margin_ratio)),
        )

    def _linear_prediction(self, dt: float) -> Point:
        assert self._last_point is not None
        frames = max(self._missed_frames + 1, 1)
        return (
            self._last_point[0] + self._velocity[0] * dt * frames,
            self._last_point[1] + self._velocity[1] * dt * frames,
        )

    def _record_history(self, point: Point, score: float) -> None:
        self._history.append(
            _TrajectoryPoint(
                frame_index=self._frame_index,
                time_seconds=self._time_seconds,
                point=point,
                score=float(score),
            )
        )

    def _parabola_prediction(self, target_time_seconds: float | None = None) -> _ParabolaPrediction | None:
        if not self.config.parabola_enabled:
            return None

        min_points = max(3, int(self.config.parabola_min_points))
        min_real_detections = max(1, int(self.config.parabola_min_real_detections))
        usable_history_count = min(len(self._history), self._real_detections_since_relock)
        if usable_history_count < max(min_points, min_real_detections):
            return None

        usable_history = list(self._history)[-usable_history_count:]
        last_time_seconds = usable_history[-1].time_seconds
        target_time = self._time_seconds if target_time_seconds is None else float(target_time_seconds)
        gap_seconds = target_time - last_time_seconds
        fps = self.config.fps if self.config.fps > 0 else 25.0
        gap_frames_float = gap_seconds * fps
        if gap_frames_float < -1e-6 or gap_frames_float > float(self.config.parabola_max_gap_frames):
            return None

        history = usable_history[-max(min_points, int(self.config.parabola_history_frames)) :]
        if history[-1].time_seconds - history[0].time_seconds <= 1e-9:
            return None

        first_point = history[0].point
        motion_span = max(_distance(first_point, item.point) for item in history[1:])
        if motion_span < self.config.parabola_min_motion_px:
            return None

        times = [float(item.time_seconds - last_time_seconds) for item in history]
        xs = [float(item.point[0]) for item in history]
        ys = [float(item.point[1]) for item in history]
        coeff_x = _fit_quadratic(times, xs)
        coeff_y = _fit_quadratic(times, ys)
        if coeff_x is None or coeff_y is None:
            return None

        residual_sum = 0.0
        for t, x, y in zip(times, xs, ys):
            fitted_x = _eval_quadratic(coeff_x, t)
            fitted_y = _eval_quadratic(coeff_y, t)
            residual_sum += _distance((x, y), (fitted_x, fitted_y)) ** 2
        fit_rmse = (residual_sum / max(len(history), 1)) ** 0.5
        if not isfinite(fit_rmse) or fit_rmse > self.config.parabola_max_fit_rmse_px:
            return None

        target_t = float(target_time - last_time_seconds)
        predicted = (_eval_quadratic(coeff_x, target_t), _eval_quadratic(coeff_y, target_t))
        if not isfinite(predicted[0]) or not isfinite(predicted[1]):
            return None

        return _ParabolaPrediction(
            point=predicted,
            fit_rmse=fit_rmse,
            gap_frames=max(0, int(round(gap_frames_float))),
        )

    def _smooth_render_point(self, measurement: Point) -> Point:
        smoothing = min(max(self.config.render_smoothing, 0.0), 0.85)
        if self._render_point is None or smoothing <= 0.0:
            return measurement
        return (
            smoothing * self._render_point[0] + (1.0 - smoothing) * measurement[0],
            smoothing * self._render_point[1] + (1.0 - smoothing) * measurement[1],
        )

    def _low_confidence_candidate_conflicts_with_prediction(
        self,
        measurement: Point | None,
        dt: float,
        frame_size: FrameSize | None,
        court_filter: _CourtFilter | None,
    ) -> bool:
        if measurement is None or not self._locked or self._last_point is None:
            return False
        if court_filter is not None and not self._point_inside_court_region(measurement, court_filter, frame_size):
            return False
        predicted = self._predict(dt)
        conflict_distance = max(float(self.config.base_gate_px), float(self.config.close_gate_px))
        return _distance(measurement, predicted) > conflict_distance

    def last_debug_record(self) -> dict[str, object] | None:
        if self._algorithm is not None:
            return self._algorithm.last_debug_record()
        if self._last_debug_record is None:
            return None
        return dict(self._last_debug_record)

    def _debug_before_state(self, dt: float, frame_size: FrameSize | None) -> dict[str, object]:
        predicted: Point | None = None
        if self._last_point is not None:
            predicted = self._predict(dt)

        width, height = frame_size if frame_size is not None else (0.0, 0.0)
        return {
            "locked_before": self._locked,
            "missed_before": self._missed_frames,
            "coast_before": self._coast_frames,
            "last_x_before": self._last_point[0] if self._last_point is not None else -1.0,
            "last_y_before": self._last_point[1] if self._last_point is not None else -1.0,
            "pred_x": predicted[0] if predicted is not None else -1.0,
            "pred_y": predicted[1] if predicted is not None else -1.0,
            "velocity_x_before": self._velocity[0],
            "velocity_y_before": self._velocity[1],
            "frame_w": width,
            "frame_h": height,
        }

    def _finish_debug(
        self,
        input_track: TrackResult,
        output_track: TrackResult,
        dt: float,
        frame_size: FrameSize | None,
        before: dict[str, object],
    ) -> TrackResult:
        if not self.debug_enabled:
            self._pending_candidate_debug = None
            return output_track

        candidate_debug = self._pending_candidate_debug or {
            "candidate_count": 1,
            "selected_candidate_index": 0 if input_track.visible else -1,
            "selected_candidate_rank": "single",
            "candidates": self._format_track(input_track),
        }
        self._pending_candidate_debug = None

        record: dict[str, object] = {
            **before,
            **candidate_debug,
            "frame_index": self._frame_index,
            "dt": dt,
            "raw_candidate_count": self._raw_candidate_count,
            "court_filter_active": int(self._court_filter_active),
            "court_filtered_count": self._court_filtered_count,
            "static_filtered_count": self._static_filtered_count,
            "static_hotspot_count": self._static_hotspot_count,
            "action": self._decision_action,
            "reason": self._decision_reason,
            "input_visible": int(bool(input_track.visible)),
            "input_x": input_track.ball_xy[0] if len(input_track.ball_xy) > 0 else -1.0,
            "input_y": input_track.ball_xy[1] if len(input_track.ball_xy) > 1 else -1.0,
            "input_score": float(input_track.score),
            "output_visible": int(bool(output_track.visible)),
            "output_x": output_track.ball_xy[0] if len(output_track.ball_xy) > 0 else -1.0,
            "output_y": output_track.ball_xy[1] if len(output_track.ball_xy) > 1 else -1.0,
            "output_score": float(output_track.score),
            "locked_after": self._locked,
            "missed_after": self._missed_frames,
            "coast_after": self._coast_frames,
            "velocity_x_after": self._velocity[0],
            "velocity_y_after": self._velocity[1],
            "top_exit_remaining": self._top_exit_frames_remaining,
        }
        self._last_debug_record = record
        self.debug_records.append(record)
        return output_track

    def _candidate_debug(
        self,
        tracks: Sequence[TrackResult],
        dt: float,
        frame_size: FrameSize | None,
        court_filter: _CourtFilter | None,
        person_bboxes: Sequence[PersonBBox],
        *,
        selected_index: int,
        selected_rank: str,
        target_time_seconds: float,
    ) -> dict[str, object]:
        items = []
        predicted: Point | None = (
            self._predict(dt, target_time_seconds=target_time_seconds)
            if self._last_point is not None
            else None
        )
        person_occlusion_active = self._person_occlusion_likely(
            dt,
            frame_size,
            person_bboxes,
            target_time_seconds=target_time_seconds,
        )
        for index, track in enumerate(tracks):
            measurement = self._measurement(track, frame_size, court_filter=court_filter)
            if measurement is None and self._can_use_soft_measurement(track, frame_size, court_filter):
                measurement = self._raw_measurement(track, frame_size)
            distance = _distance(measurement, predicted) if measurement is not None and predicted is not None else -1.0
            gate = (
                int(
                    self._passes_gate(
                        measurement,
                        float(track.score),
                        dt,
                        frame_size,
                        target_time_seconds=target_time_seconds,
                    )
                )
                if measurement is not None and self._locked and self._last_point is not None
                else -1
            )
            rank = (
                self._candidate_rank(
                    track,
                    predicted,
                    dt,
                    frame_size,
                    court_filter,
                    person_bboxes,
                    person_occlusion_active,
                    index,
                    target_time_seconds,
                )
                if predicted is not None and self._locked and self._last_point is not None
                else float("nan")
            )
            person_occ = int(
                measurement is not None
                and person_occlusion_active
                and self._point_inside_person_bbox(measurement, person_bboxes)
            )
            x = track.ball_xy[0] if len(track.ball_xy) > 0 else -1.0
            y = track.ball_xy[1] if len(track.ball_xy) > 1 else -1.0
            items.append(
                f"{index}:x={float(x):.1f},y={float(y):.1f},s={float(track.score):.3f},"
                f"v={int(bool(track.visible))},d={distance:.1f},gate={gate},occ={person_occ},rank={rank:.3f}"
            )
        return {
            "candidate_count": len(tracks),
            "selected_candidate_index": selected_index,
            "selected_candidate_rank": selected_rank,
            "static_filtered_count": self._static_filtered_count,
            "static_hotspot_count": self._static_hotspot_count,
            "candidates": " | ".join(items),
        }

    def _format_track(self, track: TrackResult) -> str:
        x = track.ball_xy[0] if len(track.ball_xy) > 0 else -1.0
        y = track.ball_xy[1] if len(track.ball_xy) > 1 else -1.0
        return f"0:x={float(x):.1f},y={float(y):.1f},s={float(track.score):.3f},v={int(bool(track.visible))}"

    def _visible(self, original: TrackResult, point: Point, frame_size: FrameSize | None = None) -> TrackResult:
        if frame_size is not None and not _point_inside_frame(point, frame_size, margin=0.0):
            return self._invisible(original)
        return TrackResult(
            ball_xy=[float(point[0]), float(point[1])],
            visible=1,
            score=float(original.score),
            heatmap_shape=list(original.heatmap_shape),
        )

    def _invisible(self, original: TrackResult) -> TrackResult:
        return TrackResult(
            ball_xy=[-1.0, -1.0],
            visible=0,
            score=float(original.score),
            heatmap_shape=list(original.heatmap_shape),
        )


class LegacyBallTrackFilterAlgorithm(BallTrackFilter):
    """Named original trajectory filter implementation for explicit algorithm selection."""


def _distance(a: Point, b: Point) -> float:
    return hypot(a[0] - b[0], a[1] - b[1])


def _length(v: Point) -> float:
    return hypot(v[0], v[1])


def _dot(a: Point, b: Point) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _point_inside_frame(point: Point, frame_size: FrameSize, *, margin: float) -> bool:
    width, height = frame_size
    allowed_margin = max(0.0, float(margin))
    return (
        -allowed_margin <= point[0] < width + allowed_margin
        and -allowed_margin <= point[1] < height + allowed_margin
    )


def _point_inside_rect(point: Point, rect: PersonBBox, *, padding: float) -> bool:
    x1, y1, x2, y2 = rect
    pad = max(0.0, float(padding))
    return x1 - pad <= point[0] <= x2 + pad and y1 - pad <= point[1] <= y2 + pad


def _segment_intersects_rect(a: Point, b: Point, rect: PersonBBox, *, padding: float) -> bool:
    if _point_inside_rect(a, rect, padding=padding) or _point_inside_rect(b, rect, padding=padding):
        return True

    x1, y1, x2, y2 = rect
    pad = max(0.0, float(padding))
    min_x, max_x = x1 - pad, x2 + pad
    min_y, max_y = y1 - pad, y2 + pad
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    t0 = 0.0
    t1 = 1.0

    for p, q in (
        (-dx, a[0] - min_x),
        (dx, max_x - a[0]),
        (-dy, a[1] - min_y),
        (dy, max_y - a[1]),
    ):
        if abs(p) < 1e-9:
            if q < 0.0:
                return False
            continue
        r = q / p
        if p < 0.0:
            if r > t1:
                return False
            if r > t0:
                t0 = r
        else:
            if r < t0:
                return False
            if r < t1:
                t1 = r
    return True


def _extract_court_filter(court_prediction: Any | None) -> _CourtFilter | None:
    if court_prediction is None:
        return None
    if not _prediction_value(court_prediction, "valid", False):
        return None

    image_to_court_h = _extract_image_to_court_h(court_prediction)
    corners = _extract_projected_court_corners(court_prediction)
    if image_to_court_h is None and corners is None:
        return None
    return _CourtFilter(image_to_court_h=image_to_court_h, corners=corners)


def _extract_image_to_court_h(court_prediction: Any | None) -> CourtMatrix | None:
    if court_prediction is None:
        return None

    raw_h = _prediction_value(court_prediction, "image_to_court_h", None)
    if raw_h is None:
        return None

    try:
        rows = tuple(tuple(float(value) for value in row) for row in raw_h)
    except (TypeError, ValueError):
        return None
    if len(rows) != 3 or any(len(row) != 3 for row in rows):
        return None
    if any(not isfinite(value) for row in rows for value in row):
        return None
    return rows  # type: ignore[return-value]


def _extract_projected_court_corners(court_prediction: Any | None) -> tuple[Point, Point, Point, Point] | None:
    if court_prediction is None:
        return None

    raw_corners = _prediction_value(court_prediction, "corners", None)
    if raw_corners is None:
        projected_lines = _prediction_value(court_prediction, "projected_lines", {})
        if isinstance(projected_lines, dict):
            raw_corners = projected_lines.get("doubles_outer")

    if raw_corners is None:
        return None

    try:
        points = tuple((float(point[0]), float(point[1])) for point in raw_corners)
    except (TypeError, ValueError, IndexError):
        return None
    if len(points) != 4:
        return None
    if any(not isfinite(value) for point in points for value in point):
        return None
    return points  # type: ignore[return-value]


def _prediction_value(prediction: Any, key: str, default: Any) -> Any:
    if isinstance(prediction, dict):
        return prediction.get(key, default)
    return getattr(prediction, key, default)


def _project_image_point(h: CourtMatrix, point: Point) -> Point | None:
    x, y = point
    u = h[0][0] * x + h[0][1] * y + h[0][2]
    v = h[1][0] * x + h[1][1] * y + h[1][2]
    w = h[2][0] * x + h[2][1] * y + h[2][2]
    if abs(w) < 1e-9:
        return None
    court_x = u / w
    court_y = v / w
    if not isfinite(court_x) or not isfinite(court_y):
        return None
    return court_x, court_y


def _point_inside_court_plane(point: Point, image_to_court_h: CourtMatrix, margin_cm: float) -> bool:
    court_point = _project_image_point(image_to_court_h, point)
    if court_point is None:
        return False
    x, y = court_point
    margin = max(0.0, float(margin_cm))
    return -margin <= x <= COURT_WIDTH + margin and -margin <= y <= COURT_LENGTH + margin


def _point_inside_projected_court_air(
    point: Point,
    corners: tuple[Point, Point, Point, Point],
    *,
    margin_px: float,
    air_extension_ratio: float,
    lateral_expansion_ratio: float = 0.0,
    frame_size: FrameSize | None,
) -> bool:
    tl, tr, br, bl = corners
    x, y = point
    margin = max(0.0, float(margin_px))

    top_y = min(tl[1], tr[1])
    bottom_y = max(bl[1], br[1])
    court_height = max(1.0, bottom_y - top_y)
    extension = court_height * max(0.0, float(air_extension_ratio))
    min_y = top_y - extension - margin
    if frame_size is not None:
        min_y = max(-margin, min_y)
    if y < min_y or y > bottom_y + margin:
        return False

    left_x = _line_x_at_y(tl, bl, y)
    right_x = _line_x_at_y(tr, br, y)
    if left_x is None or right_x is None:
        xs = [corner[0] for corner in corners]
        left_x = min(xs)
        right_x = max(xs)

    above_court = min(max(0.0, top_y - y), extension)
    lateral_margin = margin + above_court * max(0.0, float(lateral_expansion_ratio))
    min_x = min(left_x, right_x) - lateral_margin
    max_x = max(left_x, right_x) + lateral_margin
    return min_x <= x <= max_x


def _line_x_at_y(a: Point, b: Point, y: float) -> float | None:
    dy = b[1] - a[1]
    if abs(dy) < 1e-6:
        return (a[0] + b[0]) * 0.5
    t = (y - a[1]) / dy
    return a[0] + (b[0] - a[0]) * t


def _fit_quadratic(times: list[float], values: list[float]) -> tuple[float, float, float] | None:
    if len(times) != len(values) or len(times) < 3:
        return None

    s0 = float(len(times))
    s1 = sum(times)
    s2 = sum(t * t for t in times)
    s3 = sum(t * t * t for t in times)
    s4 = sum(t * t * t * t for t in times)
    rhs0 = sum(v for v in values)
    rhs1 = sum(t * v for t, v in zip(times, values))
    rhs2 = sum(t * t * v for t, v in zip(times, values))
    return _solve_3x3(
        [
            [s4, s3, s2],
            [s3, s2, s1],
            [s2, s1, s0],
        ],
        [rhs2, rhs1, rhs0],
    )


def _solve_3x3(matrix: list[list[float]], rhs: list[float]) -> tuple[float, float, float] | None:
    rows = [list(matrix[index]) + [float(rhs[index])] for index in range(3)]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda row: abs(rows[row][col]))
        if abs(rows[pivot][col]) < 1e-9:
            return None
        if pivot != col:
            rows[col], rows[pivot] = rows[pivot], rows[col]

        pivot_value = rows[col][col]
        for item in range(col, 4):
            rows[col][item] /= pivot_value

        for row in range(3):
            if row == col:
                continue
            factor = rows[row][col]
            if abs(factor) < 1e-12:
                continue
            for item in range(col, 4):
                rows[row][item] -= factor * rows[col][item]

    return rows[0][3], rows[1][3], rows[2][3]


def _eval_quadratic(coefficients: tuple[float, float, float], t: float) -> float:
    return coefficients[0] * t * t + coefficients[1] * t + coefficients[2]


def filter_track_results(tracks: list[TrackResult], *, fps: float = 25.0) -> list[TrackResult]:
    from src.postprocess.tracknet_v3_filter import create_tracknet_v3_ball_track_filter

    tracker = create_tracknet_v3_ball_track_filter(fps=fps)
    return [tracker.update(track) for track in tracks]
