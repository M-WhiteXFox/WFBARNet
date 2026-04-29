from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import hypot, isfinite

from src.utils.structures import TrackResult


Point = tuple[float, float]
FrameSize = tuple[float, float]


@dataclass(slots=True)
class BallTrackFilterConfig:
    fps: float = 25.0
    min_confidence: float = 0.35
    relock_confidence: float = 0.50
    strong_relock_confidence: float = 0.85
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


class BallTrackFilter:
    """Low-latency robust gate for shuttle detections.

    The predicted position is used for gating and short coasting. Recent real
    detections are also fit with a lightweight quadratic motion model, so small
    gaps can be filled along the arc and points far outside that arc are ignored
    until they form a stable new trajectory.
    """

    def __init__(self, config: BallTrackFilterConfig | None = None, *, fps: float | None = None) -> None:
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

    def update(
        self,
        track: TrackResult,
        *,
        dt: float | None = None,
        frame_shape: tuple[int, ...] | list[int] | None = None,
    ) -> TrackResult:
        self._frame_index += 1
        step_dt = self._resolve_dt(dt)
        frame_size = self._resolve_frame_size(frame_shape)
        measurement = self._measurement(track, frame_size)

        if measurement is None:
            return self._reject(track, step_dt, allow_coast=True, frame_size=frame_size)

        if not self._locked or self._last_point is None:
            return self._bootstrap(track, measurement, step_dt, frame_size)

        if self._prediction_is_out_of_frame(step_dt, frame_size):
            return self._reject(track, step_dt, allow_coast=True, frame_size=frame_size)

        if self._passes_gate(measurement, float(track.score), step_dt, frame_size):
            return self._accept(track, measurement, step_dt, frame_size)

        relock = self._update_candidate(measurement, float(track.score), step_dt)
        if relock and self._should_relock():
            self._drop_lock()
            return self._accept(track, measurement, step_dt, frame_size)

        allow_parabola_fill = (
            self.config.parabola_fill_on_outlier
            and self._parabola_prediction(self._frame_index) is not None
        )
        return self._reject(
            track,
            step_dt,
            allow_coast=self.config.coast_on_outlier or allow_parabola_fill,
            frame_size=frame_size,
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

    def _measurement(self, track: TrackResult, frame_size: FrameSize | None) -> Point | None:
        if not track.visible or float(track.score) < self.config.min_confidence or len(track.ball_xy) < 2:
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

    def _bootstrap(
        self,
        track: TrackResult,
        measurement: Point,
        dt: float,
        frame_size: FrameSize | None,
    ) -> TrackResult:
        if float(track.score) >= self.config.strong_relock_confidence:
            return self._accept(track, measurement, dt, frame_size)

        if self._update_candidate(measurement, float(track.score), dt):
            return self._accept(track, measurement, dt, frame_size)

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
        return self._visible(track, self._render_point, frame_size)

    def _reject(
        self,
        track: TrackResult,
        dt: float,
        *,
        allow_coast: bool,
        frame_size: FrameSize | None,
    ) -> TrackResult:
        if allow_coast and self._can_coast(frame_size):
            return self._coast(track, dt, frame_size)

        self._missed_frames += 1
        if self._missed_frames > self.config.max_missed_frames:
            self._drop_lock()
        return self._invisible(track)

    def _can_coast(self, frame_size: FrameSize | None) -> bool:
        if not self._locked or self._last_point is None:
            return False

        if (
            self._coast_frames < self.config.parabola_max_gap_frames
            and self._parabola_prediction(self._frame_index) is not None
        ):
            return True

        return (
            self._coast_frames < self.config.max_coast_frames
            and _length(self._velocity) >= self.config.min_coast_speed_px_per_sec
        )

    def _coast(self, track: TrackResult, dt: float, frame_size: FrameSize | None) -> TrackResult:
        assert self._last_point is not None

        parabola_prediction = self._parabola_prediction(self._frame_index)
        if parabola_prediction is not None:
            predicted = parabola_prediction.point
            self._velocity = (
                (predicted[0] - self._last_point[0]) / max(dt, 1e-6),
                (predicted[1] - self._last_point[1]) / max(dt, 1e-6),
            )
            score_decay = self.config.parabola_score_decay
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

    def _drop_lock(self) -> None:
        self._locked = False
        self._last_point = None
        self._render_point = None
        self._velocity = (0.0, 0.0)
        self._missed_frames = 0
        self._coast_frames = 0
        self._candidate = None
        self._history.clear()

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
