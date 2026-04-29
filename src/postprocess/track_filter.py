from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import cos, hypot, isfinite, radians
from typing import Any, Sequence

from src.utils.structures import TrackResult


Point = tuple[float, float]
FrameSize = tuple[float, float]
CourtMatrix = tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
COURT_WIDTH = 610.0
COURT_LENGTH = 1340.0


@dataclass(slots=True)
class BallTrackFilterConfig:
    fps: float = 25.0
    min_confidence: float = 0.35
    soft_min_confidence: float = 0.25
    relock_confidence: float = 0.50
    strong_relock_confidence: float = 0.85
    impact_relock_confidence: float = 0.48
    impact_relock_confirm_frames: int = 2
    impact_relock_min_missed_frames: int = 1
    impact_relock_min_angle_deg: float = 100.0
    close_gate_confidence: float = 0.50
    close_gate_px: float = 72.0
    court_filter_enabled: bool = True
    court_filter_margin_cm: float = 140.0
    court_filter_margin_px: float = 96.0
    court_air_extension_ratio: float = 1.0
    base_gate_px: float = 80.0
    max_gate_px: float = 360.0
    missed_gate_growth_px: float = 55.0
    max_speed_px_per_sec: float = 12000.0
    velocity_blend: float = 0.66
    inertia_min_speed_px_per_sec: float = 250.0
    max_accel_px_per_sec2: float = 120000.0
    max_lateral_error_px: float = 82.0
    max_reverse_px: float = 36.0
    max_coast_frames: int = 3
    min_coast_speed_px_per_sec: float = 450.0
    coast_velocity_decay: float = 0.82
    coast_score_decay: float = 0.55
    coast_on_outlier: bool = False
    frame_measurement_margin_px: float = 6.0
    out_of_frame_prediction_margin_px: float = 12.0
    parabola_enabled: bool = True
    parabola_min_points: int = 4
    parabola_history_frames: int = 12
    parabola_max_gap_frames: int = 6
    parabola_min_motion_px: float = 42.0
    parabola_max_fit_rmse_px: float = 48.0
    parabola_gate_px: float = 62.0
    parabola_max_gate_px: float = 160.0
    parabola_gate_growth_px: float = 24.0
    parabola_fit_error_scale: float = 1.8
    parabola_score_bonus_px: float = 20.0
    parabola_score_decay: float = 0.58
    parabola_fill_on_outlier: bool = True
    relock_distance_px: float = 220.0
    relock_max_speed_px_per_sec: float = 9000.0
    relock_confirm_frames: int = 3
    relock_after_missed_frames: int = 2
    max_missed_frames: int = 8
    render_smoothing: float = 0.0
    top_exit_enabled: bool = True
    top_exit_margin_px: float = 24.0
    top_exit_margin_ratio: float = 0.04
    top_exit_min_up_speed_px_per_sec: float = 650.0
    top_exit_min_up_motion_px: float = 36.0
    top_exit_history_frames: int = 4
    top_exit_suppression_frames: int = 6
    top_exit_reversal_min_delta_px: float = 8.0


@dataclass(slots=True)
class _RelockCandidate:
    point: Point
    score: float
    count: int = 1


@dataclass(slots=True)
class _TrajectoryPoint:
    frame_index: int
    point: Point


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
    ) -> None:
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
        self._history: deque[_TrajectoryPoint] = deque(maxlen=max(1, self.config.parabola_history_frames))
        self._frame_index = -1
        self._last_frame_size: FrameSize | None = None
        self._top_exit_frames_remaining = 0
        self.debug_enabled = debug_enabled
        self.debug_records: list[dict[str, object]] = []
        self._last_debug_record: dict[str, object] | None = None
        self._pending_candidate_debug: dict[str, object] | None = None
        self._decision_action = "unknown"
        self._decision_reason = ""

    def reset(self) -> None:
        self._last_point = None
        self._render_point = None
        self._velocity = (0.0, 0.0)
        self._missed_frames = 0
        self._coast_frames = 0
        self._locked = False
        self._candidate = None
        self._history.clear()
        self._frame_index = -1
        self._last_frame_size = None
        self._top_exit_frames_remaining = 0
        self.debug_records.clear()
        self._last_debug_record = None
        self._pending_candidate_debug = None
        self._decision_action = "unknown"
        self._decision_reason = ""

    def update(
        self,
        track: TrackResult,
        *,
        dt: float | None = None,
        frame_shape: tuple[int, ...] | list[int] | None = None,
        court_prediction: Any | None = None,
    ) -> TrackResult:
        self._frame_index += 1
        step_dt = self._resolve_dt(dt)
        frame_size = self._resolve_frame_size(frame_shape)
        court_filter = self._court_filter(court_prediction)
        debug_before = self._debug_before_state(step_dt, frame_size)
        self._start_decision()

        if self._top_exit_frames_remaining > 0:
            self._top_exit_frames_remaining -= 1
            self._mark_decision("top_exit_suppress", "active_top_exit_suppression", force=True)
            result = self._invisible(track)
            return self._finish_debug(track, result, step_dt, frame_size, debug_before)

        measurement = self._measurement(track, frame_size, court_filter=court_filter)
        soft_measurement = False
        if measurement is None and self._can_use_soft_measurement(track, frame_size, court_filter):
            measurement = self._raw_measurement(track, frame_size)
            soft_measurement = measurement is not None

        if measurement is None:
            self._mark_decision("reject", "missing_or_low_confidence")
            result = self._reject(track, step_dt, allow_coast=True, frame_size=frame_size, court_filter=court_filter)
        elif not self._locked or self._last_point is None:
            result = self._bootstrap(track, measurement, step_dt, frame_size)
        elif self._prediction_is_out_of_frame(step_dt, frame_size):
            self._mark_decision("reject", "prediction_out_of_frame")
            result = self._reject(track, step_dt, allow_coast=True, frame_size=frame_size, court_filter=court_filter)
        elif self._measurement_reverses_after_top_exit(measurement, step_dt, frame_size):
            self._enter_top_exit()
            self._mark_decision("top_exit_enter", "measurement_reverses_after_top_exit", force=True)
            result = self._invisible(track)
        elif self._passes_gate(measurement, float(track.score), step_dt, frame_size):
            if soft_measurement:
                self._mark_decision("accept", "soft_confidence_motion_gate")
            result = self._accept(track, measurement, step_dt, frame_size)
        elif self._passes_close_gate(measurement, float(track.score), step_dt, frame_size):
            self._mark_decision("accept", "close_prediction_motion_break")
            result = self._accept(track, measurement, step_dt, frame_size)
        else:
            relock = self._update_candidate(measurement, float(track.score), step_dt)
            impact_relock = self._should_impact_relock(measurement, step_dt)
            if (relock and self._should_relock()) or impact_relock:
                self._drop_lock()
                reason = "impact_direction_change" if impact_relock else "stable_new_candidate"
                self._mark_decision("relock_accept", reason, force=True)
                result = self._accept(track, measurement, step_dt, frame_size)
            else:
                allow_parabola_fill = (
                    self.config.parabola_fill_on_outlier
                    and self._parabola_prediction(self._frame_index) is not None
                )
                self._mark_decision("reject", "candidate_failed_motion_gate")
                result = self._reject(
                    track,
                    step_dt,
                    allow_coast=self.config.coast_on_outlier or allow_parabola_fill,
                    frame_size=frame_size,
                    court_filter=court_filter,
                )

        return self._finish_debug(track, result, step_dt, frame_size, debug_before)

    def update_candidates(
        self,
        tracks: Sequence[TrackResult],
        *,
        dt: float | None = None,
        frame_shape: tuple[int, ...] | list[int] | None = None,
        court_prediction: Any | None = None,
    ) -> TrackResult:
        step_dt = self._resolve_dt(dt)
        frame_size = self._resolve_frame_size(frame_shape)
        court_filter = self._court_filter(court_prediction)
        tracks = self._filter_candidates_by_court(tracks, frame_size, court_filter)
        track = self._select_candidate(tracks, step_dt, frame_size, court_filter)
        return self.update(track, dt=step_dt, frame_shape=frame_shape, court_prediction=court_prediction)

    def _select_candidate(
        self,
        tracks: Sequence[TrackResult],
        dt: float,
        frame_size: FrameSize | None,
        court_filter: _CourtFilter | None,
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
                selected_index=selected_index,
                selected_rank="primary",
            )
            return primary

        predicted = self._predict(dt)
        best_track = primary
        best_rank = float("-inf")
        best_index = -1
        for index, candidate in enumerate(tracks):
            rank = self._candidate_rank(candidate, predicted, dt, frame_size, court_filter, index)
            if rank > best_rank:
                best_rank = rank
                best_track = candidate
                best_index = index

        self._pending_candidate_debug = self._candidate_debug(
            tracks,
            dt,
            frame_size,
            court_filter,
            selected_index=best_index,
            selected_rank=f"{best_rank:.4f}",
        )
        return best_track

    def _candidate_rank(
        self,
        track: TrackResult,
        predicted: Point,
        dt: float,
        frame_size: FrameSize | None,
        court_filter: _CourtFilter | None,
        index: int,
    ) -> float:
        measurement = self._measurement(track, frame_size, court_filter=court_filter)
        if measurement is None:
            return -1000.0 + float(track.score) - index * 0.05

        score = float(track.score)
        distance_to_prediction = _distance(measurement, predicted)
        gate_bonus = 100.0 if self._passes_gate(measurement, score, dt, frame_size) else 0.0
        distance_penalty = distance_to_prediction / 56.0
        heatmap_rank_penalty = index * 0.08
        return gate_bonus + score * 12.0 - distance_penalty - heatmap_rank_penalty

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

    def _bootstrap(
        self,
        track: TrackResult,
        measurement: Point,
        dt: float,
        frame_size: FrameSize | None,
    ) -> TrackResult:
        if float(track.score) >= self.config.strong_relock_confidence:
            self._mark_decision("bootstrap_accept", "strong_confidence")
            return self._accept(track, measurement, dt, frame_size)

        if self._update_candidate(measurement, float(track.score), dt):
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
    ) -> bool:
        assert self._last_point is not None

        predicted = self._predict(dt)
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

        if not self._passes_parabola_gate(measurement, score):
            return False

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

    def _passes_parabola_gate(self, measurement: Point, score: float) -> bool:
        prediction = self._parabola_prediction(self._frame_index)
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

    def _predict(self, dt: float) -> Point:
        assert self._last_point is not None
        parabola_prediction = self._parabola_prediction(self._frame_index)
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
        self._record_history(measurement)
        self._mark_decision("accept", "passes_motion_gate")
        return self._visible(track, self._render_point, frame_size)

    def _reject(
        self,
        track: TrackResult,
        dt: float,
        *,
        allow_coast: bool,
        frame_size: FrameSize | None,
        court_filter: _CourtFilter | None,
    ) -> TrackResult:
        if self._top_exit_is_likely(dt, frame_size):
            self._enter_top_exit()
            self._mark_decision("top_exit_enter", "likely_top_exit", force=True)
            return self._invisible(track)

        if allow_coast and self._can_coast(dt, frame_size, court_filter):
            return self._coast(track, dt, frame_size, court_filter)

        self._missed_frames += 1
        if self._missed_frames > self.config.max_missed_frames:
            self._drop_lock()
            self._mark_decision("drop_lock", "max_missed_frames", force=True)
        else:
            self._mark_decision("reject", "hidden_after_reject")
        return self._invisible(track)

    def _can_coast(self, dt: float, frame_size: FrameSize | None, court_filter: _CourtFilter | None) -> bool:
        if not self._locked or self._last_point is None:
            return False

        parabola_prediction = self._parabola_prediction(self._frame_index)
        if (
            self._coast_frames < self.config.parabola_max_gap_frames
            and parabola_prediction is not None
        ):
            return self._point_can_be_predicted(parabola_prediction.point, frame_size, court_filter)

        predicted = (
            self._last_point[0] + self._velocity[0] * dt,
            self._last_point[1] + self._velocity[1] * dt,
        )

        return (
            self._coast_frames < self.config.max_coast_frames
            and _length(self._velocity) >= self.config.min_coast_speed_px_per_sec
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

    def _coast(
        self,
        track: TrackResult,
        dt: float,
        frame_size: FrameSize | None,
        court_filter: _CourtFilter | None,
    ) -> TrackResult:
        assert self._last_point is not None

        parabola_prediction = self._parabola_prediction(self._frame_index)
        if parabola_prediction is not None:
            predicted = parabola_prediction.point
            self._velocity = (
                (predicted[0] - self._last_point[0]) / max(dt, 1e-6),
                (predicted[1] - self._last_point[1]) / max(dt, 1e-6),
            )
            score_decay = self.config.parabola_score_decay
            self._mark_decision("coast", "parabola_prediction", force=True)
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
            self._mark_decision("coast", "velocity_prediction", force=True)

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
            self._candidate = _RelockCandidate(point=measurement, score=score)
        else:
            self._candidate.point = (
                0.35 * self._candidate.point[0] + 0.65 * measurement[0],
                0.35 * self._candidate.point[1] + 0.65 * measurement[1],
            )
            self._candidate.score = max(self._candidate.score, score)
            self._candidate.count += 1

        return (
            self._candidate.count >= self.config.relock_confirm_frames
            and self._candidate.score >= self.config.relock_confidence
        )

    def _should_relock(self) -> bool:
        if self._candidate is None:
            return False
        if self._candidate.score >= self.config.strong_relock_confidence:
            return True
        return self._missed_frames >= self.config.relock_after_missed_frames

    def _should_impact_relock(self, measurement: Point, dt: float) -> bool:
        if self._candidate is None or self._last_point is None:
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
        self._history.clear()
        self._top_exit_frames_remaining = 0

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
        upward_motion = first.point[1] - last.point[1]
        upward_speed = upward_motion / max(dt * frame_span, 1e-6)
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

    def _record_history(self, point: Point) -> None:
        self._history.append(_TrajectoryPoint(frame_index=self._frame_index, point=point))

    def _parabola_prediction(self, target_frame: int) -> _ParabolaPrediction | None:
        if not self.config.parabola_enabled:
            return None

        min_points = max(3, int(self.config.parabola_min_points))
        if len(self._history) < min_points:
            return None

        last_frame = self._history[-1].frame_index
        gap_frames = target_frame - last_frame
        if gap_frames < 0 or gap_frames > self.config.parabola_max_gap_frames:
            return None

        history = list(self._history)[-max(min_points, int(self.config.parabola_history_frames)) :]
        if history[-1].frame_index - history[0].frame_index < min_points - 1:
            return None

        first_point = history[0].point
        motion_span = max(_distance(first_point, item.point) for item in history[1:])
        if motion_span < self.config.parabola_min_motion_px:
            return None

        times = [float(item.frame_index - last_frame) for item in history]
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

        target_t = float(target_frame - last_frame)
        predicted = (_eval_quadratic(coeff_x, target_t), _eval_quadratic(coeff_y, target_t))
        if not isfinite(predicted[0]) or not isfinite(predicted[1]):
            return None

        return _ParabolaPrediction(point=predicted, fit_rmse=fit_rmse, gap_frames=max(0, gap_frames))

    def _smooth_render_point(self, measurement: Point) -> Point:
        smoothing = min(max(self.config.render_smoothing, 0.0), 0.85)
        if self._render_point is None or smoothing <= 0.0:
            return measurement
        return (
            smoothing * self._render_point[0] + (1.0 - smoothing) * measurement[0],
            smoothing * self._render_point[1] + (1.0 - smoothing) * measurement[1],
        )

    def last_debug_record(self) -> dict[str, object] | None:
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
        *,
        selected_index: int,
        selected_rank: str,
    ) -> dict[str, object]:
        items = []
        predicted: Point | None = self._predict(dt) if self._last_point is not None else None
        for index, track in enumerate(tracks):
            measurement = self._measurement(track, frame_size, court_filter=court_filter)
            distance = _distance(measurement, predicted) if measurement is not None and predicted is not None else -1.0
            gate = (
                int(self._passes_gate(measurement, float(track.score), dt, frame_size))
                if measurement is not None and self._locked and self._last_point is not None
                else -1
            )
            rank = (
                self._candidate_rank(track, predicted, dt, frame_size, court_filter, index)
                if predicted is not None and self._locked and self._last_point is not None
                else float("nan")
            )
            x = track.ball_xy[0] if len(track.ball_xy) > 0 else -1.0
            y = track.ball_xy[1] if len(track.ball_xy) > 1 else -1.0
            items.append(
                f"{index}:x={float(x):.1f},y={float(y):.1f},s={float(track.score):.3f},"
                f"v={int(bool(track.visible))},d={distance:.1f},gate={gate},rank={rank:.3f}"
            )
        return {
            "candidate_count": len(tracks),
            "selected_candidate_index": selected_index,
            "selected_candidate_rank": selected_rank,
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

    min_x = min(left_x, right_x) - margin
    max_x = max(left_x, right_x) + margin
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
    tracker = BallTrackFilter(fps=fps)
    return [tracker.update(track) for track in tracks]
