from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite

from src.utils.structures import TrackResult


Point = tuple[float, float]


@dataclass(slots=True)
class BallTrackFilterConfig:
    fps: float = 25.0
    min_confidence: float = 0.35
    relock_confidence: float = 0.45
    strong_relock_confidence: float = 0.70
    base_gate_px: float = 95.0
    max_gate_px: float = 420.0
    missed_gate_growth_px: float = 60.0
    max_speed_px_per_sec: float = 15000.0
    velocity_blend: float = 0.72
    relock_distance_px: float = 180.0
    relock_confirm_frames: int = 2
    max_missed_frames: int = 6
    render_smoothing: float = 0.0


@dataclass(slots=True)
class _RelockCandidate:
    point: Point
    score: float
    count: int = 1


class BallTrackFilter:
    """Low-latency robust gate for shuttle detections.

    The predicted position is used only to decide whether a detection belongs to
    the current trajectory. Rejected or missing detections return visible=0, so
    callers do not render drifting prediction points.
    """

    def __init__(self, config: BallTrackFilterConfig | None = None, *, fps: float | None = None) -> None:
        self.config = config or BallTrackFilterConfig()
        if fps is not None and fps > 0:
            self.config.fps = float(fps)
        self._last_point: Point | None = None
        self._render_point: Point | None = None
        self._velocity: Point = (0.0, 0.0)
        self._missed_frames = 0
        self._locked = False
        self._candidate: _RelockCandidate | None = None

    def reset(self) -> None:
        self._last_point = None
        self._render_point = None
        self._velocity = (0.0, 0.0)
        self._missed_frames = 0
        self._locked = False
        self._candidate = None

    def update(self, track: TrackResult, *, dt: float | None = None) -> TrackResult:
        step_dt = self._resolve_dt(dt)
        measurement = self._measurement(track)

        if measurement is None:
            return self._reject(track)

        if not self._locked or self._last_point is None:
            return self._bootstrap(track, measurement, step_dt)

        if self._passes_gate(measurement, float(track.score), step_dt):
            return self._accept(track, measurement, step_dt)

        relock = self._update_candidate(measurement, float(track.score))
        if relock:
            self._locked = False
            return self._accept(track, measurement, step_dt)

        return self._reject(track)

    def _resolve_dt(self, dt: float | None) -> float:
        if dt is not None and dt > 0:
            return float(dt)
        fps = self.config.fps if self.config.fps > 0 else 25.0
        return 1.0 / fps

    def _measurement(self, track: TrackResult) -> Point | None:
        if not track.visible or float(track.score) < self.config.min_confidence or len(track.ball_xy) < 2:
            return None

        x, y = float(track.ball_xy[0]), float(track.ball_xy[1])
        if x < 0 or y < 0 or not isfinite(x) or not isfinite(y):
            return None
        return x, y

    def _bootstrap(self, track: TrackResult, measurement: Point, dt: float) -> TrackResult:
        if float(track.score) >= self.config.strong_relock_confidence:
            return self._accept(track, measurement, dt)

        if self._update_candidate(measurement, float(track.score)):
            return self._accept(track, measurement, dt)

        return self._invisible(track)

    def _passes_gate(self, measurement: Point, score: float, dt: float) -> bool:
        assert self._last_point is not None

        predicted = self._predict(dt)
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
        return distance_to_prediction <= allowed_distance

    def _predict(self, dt: float) -> Point:
        assert self._last_point is not None
        frames = max(self._missed_frames + 1, 1)
        return (
            self._last_point[0] + self._velocity[0] * dt * frames,
            self._last_point[1] + self._velocity[1] * dt * frames,
        )

    def _accept(self, track: TrackResult, measurement: Point, dt: float) -> TrackResult:
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
        self._locked = True
        self._candidate = None
        return self._visible(track, self._render_point)

    def _reject(self, track: TrackResult) -> TrackResult:
        self._missed_frames += 1
        if self._missed_frames > self.config.max_missed_frames:
            self._locked = False
            self._last_point = None
            self._render_point = None
            self._velocity = (0.0, 0.0)
        return self._invisible(track)

    def _update_candidate(self, measurement: Point, score: float) -> bool:
        if self._candidate is None or _distance(measurement, self._candidate.point) > self.config.relock_distance_px:
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

    def _smooth_render_point(self, measurement: Point) -> Point:
        smoothing = min(max(self.config.render_smoothing, 0.0), 0.85)
        if self._render_point is None or smoothing <= 0.0:
            return measurement
        return (
            smoothing * self._render_point[0] + (1.0 - smoothing) * measurement[0],
            smoothing * self._render_point[1] + (1.0 - smoothing) * measurement[1],
        )

    def _visible(self, original: TrackResult, point: Point) -> TrackResult:
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


def filter_track_results(tracks: list[TrackResult], *, fps: float = 25.0) -> list[TrackResult]:
    tracker = BallTrackFilter(fps=fps)
    return [tracker.update(track) for track in tracks]
