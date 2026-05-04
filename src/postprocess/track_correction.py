from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import hypot, isfinite
from typing import Any, Sequence

from src.postprocess.track_filter import FrameShape, PersonBBoxes
from src.utils.structures import TrackResult


Point = tuple[float, float]
FrameSize = tuple[float, float]
PersonBBox = tuple[float, float, float, float]


@dataclass(slots=True)
class RealtimeKalmanTrackCorrectorConfig:
    fps: float = 25.0
    min_confidence: float = 0.35
    candidate_min_confidence: float = 0.28
    bootstrap_confidence: float = 0.55
    relock_confidence: float = 0.65
    strong_relock_confidence: float = 0.78
    gate_chi2: float = 10.0
    gate_missed_growth: float = 3.0
    score_weight: float = 7.5
    heatmap_rank_penalty: float = 0.22
    high_confidence_measurement_std_px: float = 10.0
    low_confidence_measurement_std_px: float = 80.0
    stable_accel_noise_px_per_sec2: float = 3600.0
    maneuver_accel_noise_px_per_sec2: float = 22000.0
    initial_position_std_px: float = 28.0
    initial_velocity_std_px_per_sec: float = 2800.0
    maneuver_innovation_px: float = 80.0
    maneuver_frames: int = 3
    maneuver_snap_innovation_px: float = 45.0
    maneuver_snap_confidence: float = 0.55
    maneuver_snap_weight: float = 0.85
    max_coast_frames: int = 8
    occlusion_coast_frames: int = 10
    min_coast_speed_px_per_sec: float = 180.0
    coast_score_decay: float = 0.55
    person_occlusion_enabled: bool = True
    person_occlusion_padding_px: float = 18.0
    person_occlusion_candidate_penalty: float = 40.0
    person_occlusion_reject_below_score: float = 0.75
    out_of_frame_margin_px: float = 10.0
    out_of_frame_suppression_frames: int = 10
    edge_suppression_band_px: float = 36.0
    relock_after_missed_frames: int = 2
    max_missed_frames: int = 14
    fixed_lag_frames: int = 0
    fixed_lag_future_decay: float = 0.38


@dataclass(slots=True)
class _CandidateMatch:
    track: TrackResult
    index: int
    point: Point
    score: float
    mahalanobis: float
    gate: float
    cost: float
    inside_person: bool


class RealtimeKalmanTrackCorrector:
    """Realtime multi-candidate adaptive Kalman correction algorithm.

    The module consumes TrackNet Top-K candidates, associates the best candidate
    against the Kalman prediction, adapts process and measurement noise, handles
    short occlusion/out-of-frame states, and can emit a tiny fixed-lag smoothed
    track when downstream consumers explicitly allow a display delay.
    """

    def __init__(
        self,
        config: RealtimeKalmanTrackCorrectorConfig | None = None,
        *,
        fps: float | None = None,
        debug_enabled: bool = False,
        fixed_lag_frames: int | None = None,
    ) -> None:
        self.config = config or RealtimeKalmanTrackCorrectorConfig()
        if fps is not None and fps > 0:
            self.config.fps = float(fps)
        if fixed_lag_frames is not None:
            self.config.fixed_lag_frames = max(0, int(fixed_lag_frames))
        self.debug_enabled = debug_enabled
        self.debug_records: list[dict[str, object]] = []
        self._state: list[float] | None = None
        self._covariance: list[list[float]] | None = None
        self._frame_index = -1
        self._missed_frames = 0
        self._coast_frames = 0
        self._maneuver_frames_remaining = 0
        self._out_of_frame_suppression_remaining = 0
        self._last_output_score = 0.0
        self._last_heatmap_shape: list[int] = []
        self._output_buffer: deque[TrackResult] = deque()
        self._last_debug_record: dict[str, object] | None = None

    def reset(self) -> None:
        self.debug_records.clear()
        self._state = None
        self._covariance = None
        self._frame_index = -1
        self._missed_frames = 0
        self._coast_frames = 0
        self._maneuver_frames_remaining = 0
        self._out_of_frame_suppression_remaining = 0
        self._last_output_score = 0.0
        self._last_heatmap_shape = []
        self._output_buffer.clear()
        self._last_debug_record = None

    def update(
        self,
        track: TrackResult,
        *,
        dt: float | None = None,
        frame_shape: FrameShape = None,
        court_prediction: Any | None = None,
        person_bboxes: PersonBBoxes = None,
    ) -> TrackResult:
        return self.update_candidates(
            [track],
            dt=dt,
            frame_shape=frame_shape,
            court_prediction=court_prediction,
            person_bboxes=person_bboxes,
        )

    def update_candidates(
        self,
        tracks: Sequence[TrackResult],
        *,
        dt: float | None = None,
        frame_shape: FrameShape = None,
        court_prediction: Any | None = None,
        person_bboxes: PersonBBoxes = None,
    ) -> TrackResult:
        del court_prediction
        self._frame_index += 1
        step_dt = self._resolve_dt(dt)
        frame_size = _frame_size(frame_shape)
        bboxes = _normalize_person_bboxes(person_bboxes)
        candidates = self._valid_candidates(tracks, frame_size)
        raw_candidate_count = len(tracks)

        if self._state is None or self._covariance is None:
            result, selected = self._bootstrap(candidates, frame_size)
            emitted = self._emit(result, flush=not result.visible)
            self._record_debug(
                action="bootstrap_accept" if result.visible else "bootstrap_wait",
                reason="candidate_initialized" if result.visible else "no_reliable_candidate",
                input_track=selected.track if selected is not None else None,
                output_track=emitted,
                dt=step_dt,
                frame_size=frame_size,
                raw_candidate_count=raw_candidate_count,
                    candidate_count=len(candidates),
                    candidates=candidates,
                    selected=selected,
                    predicted=None,
                    occlusion_active=False,
                )
            return emitted

        predicted_state, predicted_covariance = self._predict(step_dt)
        predicted_point = (float(predicted_state[0]), float(predicted_state[1]))
        occlusion_active = self._occlusion_likely(predicted_point, bboxes)

        if self._should_enter_out_of_frame(predicted_state, frame_size):
            self._state = predicted_state
            self._covariance = predicted_covariance
            self._out_of_frame_suppression_remaining = max(
                self._out_of_frame_suppression_remaining,
                int(self.config.out_of_frame_suppression_frames),
            )

        if self._out_of_frame_suppression_remaining > 0:
            relock = self._out_of_frame_relock_candidate(candidates, frame_size)
            if relock is None:
                self._missed_frames += 1
                self._out_of_frame_suppression_remaining -= 1
                result = self._invisible(score=self._last_output_score * self.config.coast_score_decay)
                reason = "suppress_edge_noise"
                if self._missed_frames > int(self.config.max_missed_frames):
                    self._state = None
                    self._covariance = None
                    self._maneuver_frames_remaining = 0
                    self._out_of_frame_suppression_remaining = 0
                    reason = "unlock_after_long_exit"
                emitted = self._emit(result, flush=True)
                self._record_debug(
                    action="out_of_frame",
                    reason=reason,
                    input_track=None,
                    output_track=emitted,
                    dt=step_dt,
                    frame_size=frame_size,
                    raw_candidate_count=raw_candidate_count,
                    candidate_count=len(candidates),
                    candidates=candidates,
                    selected=None,
                    predicted=predicted_point,
                    occlusion_active=occlusion_active,
                )
                return emitted
            self._initialize(relock.point, relock.score, relock.track.heatmap_shape)
            self._out_of_frame_suppression_remaining = 0
            result = self._visible_from_state(relock.score, relock.track.heatmap_shape)
            emitted = self._emit(result)
            self._record_debug(
                action="relock_accept",
                reason="out_of_frame_reentry",
                input_track=relock.track,
                output_track=emitted,
                dt=step_dt,
                frame_size=frame_size,
                raw_candidate_count=raw_candidate_count,
                candidate_count=len(candidates),
                candidates=candidates,
                selected=relock,
                predicted=predicted_point,
                occlusion_active=occlusion_active,
            )
            return emitted

        selected = self._select_candidate(
            candidates,
            predicted_state,
            predicted_covariance,
            bboxes,
            occlusion_active,
        )

        if selected is not None:
            self._state = predicted_state
            self._covariance = predicted_covariance
            self._update_with_measurement(selected)
            self._missed_frames = 0
            self._coast_frames = 0
            self._last_output_score = selected.score
            self._last_heatmap_shape = list(selected.track.heatmap_shape)
            result = self._visible_from_state(selected.score, selected.track.heatmap_shape)
            emitted = self._emit(result)
            self._record_debug(
                action="accept",
                reason="adaptive_kalman_gate",
                input_track=selected.track,
                output_track=emitted,
                dt=step_dt,
                frame_size=frame_size,
                raw_candidate_count=raw_candidate_count,
                candidate_count=len(candidates),
                candidates=candidates,
                selected=selected,
                predicted=predicted_point,
                occlusion_active=occlusion_active,
            )
            return emitted

        self._state = predicted_state
        self._covariance = predicted_covariance
        self._missed_frames += 1
        reason = "occlusion_prediction" if occlusion_active else "velocity_prediction"
        allow_coast = self._can_coast(occlusion_active) and (occlusion_active or not candidates)
        if allow_coast:
            self._coast_frames += 1
            score = self._last_output_score * (self.config.coast_score_decay ** max(1, self._coast_frames))
            result = self._visible_from_state(score, self._last_heatmap_shape, frame_size=frame_size)
            if not result.visible:
                emitted = self._emit(result, flush=True)
            else:
                emitted = self._emit(result)
            self._record_debug(
                action="coast" if result.visible else "reject",
                reason=reason if result.visible else "coast_out_of_frame",
                input_track=None,
                output_track=emitted,
                dt=step_dt,
                frame_size=frame_size,
                raw_candidate_count=raw_candidate_count,
                candidate_count=len(candidates),
                candidates=candidates,
                selected=None,
                predicted=predicted_point,
                occlusion_active=occlusion_active,
            )
            return emitted

        if self._missed_frames > self.config.max_missed_frames:
            self._state = None
            self._covariance = None
        result = self._invisible(score=self._last_output_score * self.config.coast_score_decay)
        emitted = self._emit(result, flush=True)
        self._record_debug(
            action="reject",
            reason="no_candidate_inside_gate",
            input_track=None,
            output_track=emitted,
            dt=step_dt,
            frame_size=frame_size,
            raw_candidate_count=raw_candidate_count,
            candidate_count=len(candidates),
            candidates=candidates,
            selected=None,
            predicted=predicted_point,
            occlusion_active=occlusion_active,
        )
        return emitted

    def last_debug_record(self) -> dict[str, object] | None:
        if self._last_debug_record is None:
            return None
        return dict(self._last_debug_record)

    def _resolve_dt(self, dt: float | None) -> float:
        if dt is not None and dt > 0:
            return float(dt)
        fps = self.config.fps if self.config.fps > 0 else 25.0
        return 1.0 / fps

    def _valid_candidates(
        self,
        tracks: Sequence[TrackResult],
        frame_size: FrameSize | None,
    ) -> list[_CandidateMatch]:
        candidates: list[_CandidateMatch] = []
        for index, track in enumerate(tracks):
            point = _track_point(track)
            score = float(track.score)
            if point is None or not track.visible or score < self.config.candidate_min_confidence:
                continue
            if frame_size is not None and not _point_inside_frame(point, frame_size, margin=0.0):
                continue
            candidates.append(
                _CandidateMatch(
                    track=track,
                    index=index,
                    point=point,
                    score=score,
                    mahalanobis=float("inf"),
                    gate=0.0,
                    cost=float("inf"),
                    inside_person=False,
                )
            )
        return candidates

    def _bootstrap(
        self,
        candidates: Sequence[_CandidateMatch],
        frame_size: FrameSize | None,
    ) -> tuple[TrackResult, _CandidateMatch | None]:
        if not candidates:
            return self._invisible(score=0.0), None
        selected = max(candidates, key=lambda item: (item.score, -item.index))
        if selected.score < self.config.bootstrap_confidence:
            return self._invisible(score=selected.score), selected
        if self._near_edge(selected.point, frame_size) and selected.score < self.config.strong_relock_confidence:
            return self._invisible(score=selected.score), selected

        self._initialize(selected.point, selected.score, selected.track.heatmap_shape)
        return self._visible_from_state(selected.score, selected.track.heatmap_shape), selected

    def _initialize(self, point: Point, score: float, heatmap_shape: Sequence[int]) -> None:
        self._state = [point[0], point[1], 0.0, 0.0]
        self._covariance = [
            [self.config.initial_position_std_px**2, 0.0, 0.0, 0.0],
            [0.0, self.config.initial_position_std_px**2, 0.0, 0.0],
            [0.0, 0.0, self.config.initial_velocity_std_px_per_sec**2, 0.0],
            [0.0, 0.0, 0.0, self.config.initial_velocity_std_px_per_sec**2],
        ]
        self._missed_frames = 0
        self._coast_frames = 0
        self._maneuver_frames_remaining = 0
        self._last_output_score = float(score)
        self._last_heatmap_shape = list(heatmap_shape)
        self._output_buffer.clear()

    def _predict(self, dt: float) -> tuple[list[float], list[list[float]]]:
        assert self._state is not None
        assert self._covariance is not None
        transition = [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        accel_noise = (
            self.config.maneuver_accel_noise_px_per_sec2
            if self._maneuver_frames_remaining > 0
            else self.config.stable_accel_noise_px_per_sec2
        )
        if self._maneuver_frames_remaining > 0:
            self._maneuver_frames_remaining -= 1
        q = self._process_noise(dt, accel_noise)
        state = _matvec(transition, self._state)
        covariance = _matadd(_matmul(_matmul(transition, self._covariance), _transpose(transition)), q)
        return state, covariance

    def _process_noise(self, dt: float, accel_noise: float) -> list[list[float]]:
        variance = float(accel_noise) ** 2
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        return [
            [dt4 * 0.25 * variance, 0.0, dt3 * 0.5 * variance, 0.0],
            [0.0, dt4 * 0.25 * variance, 0.0, dt3 * 0.5 * variance],
            [dt3 * 0.5 * variance, 0.0, dt2 * variance, 0.0],
            [0.0, dt3 * 0.5 * variance, 0.0, dt2 * variance],
        ]

    def _select_candidate(
        self,
        candidates: Sequence[_CandidateMatch],
        predicted_state: list[float],
        predicted_covariance: list[list[float]],
        person_bboxes: Sequence[PersonBBox],
        occlusion_active: bool,
    ) -> _CandidateMatch | None:
        best: _CandidateMatch | None = None
        for candidate in candidates:
            inside_person = (
                occlusion_active
                and self.config.person_occlusion_enabled
                and _point_inside_any_bbox(
                    candidate.point,
                    person_bboxes,
                    padding=self.config.person_occlusion_padding_px,
                )
            )
            if inside_person and candidate.score < self.config.person_occlusion_reject_below_score:
                continue

            mahalanobis = self._mahalanobis(candidate.point, predicted_state, predicted_covariance, candidate.score)
            if self._missed_frames > 0 and candidate.score < self.config.bootstrap_confidence:
                continue
            gate = self.config.gate_chi2 + self._missed_frames * self.config.gate_missed_growth
            if mahalanobis > gate:
                if self._can_relock(candidate):
                    return candidate
                continue

            occlusion_penalty = self.config.person_occlusion_candidate_penalty if inside_person else 0.0
            cost = (
                mahalanobis
                - self.config.score_weight * candidate.score
                + self.config.heatmap_rank_penalty * candidate.index
                + occlusion_penalty
            )
            candidate.mahalanobis = mahalanobis
            candidate.gate = gate
            candidate.cost = cost
            candidate.inside_person = inside_person
            if best is None or candidate.cost < best.cost:
                best = candidate
        return best

    def _mahalanobis(
        self,
        point: Point,
        predicted_state: list[float],
        predicted_covariance: list[list[float]],
        score: float,
    ) -> float:
        innovation_x = point[0] - predicted_state[0]
        innovation_y = point[1] - predicted_state[1]
        measurement_variance = self._measurement_noise(score)
        s00 = predicted_covariance[0][0] + measurement_variance
        s01 = predicted_covariance[0][1]
        s10 = predicted_covariance[1][0]
        s11 = predicted_covariance[1][1] + measurement_variance
        det = s00 * s11 - s01 * s10
        if abs(det) < 1e-9:
            value = innovation_x * innovation_x + innovation_y * innovation_y
        else:
            inv00 = s11 / det
            inv01 = -s01 / det
            inv10 = -s10 / det
            inv11 = s00 / det
            value = (
                innovation_x * (inv00 * innovation_x + inv01 * innovation_y)
                + innovation_y * (inv10 * innovation_x + inv11 * innovation_y)
            )
        return value if isfinite(value) else float("inf")

    def _measurement_noise(self, score: float) -> float:
        confidence = _clamp(
            (float(score) - self.config.candidate_min_confidence)
            / max(1e-6, 1.0 - self.config.candidate_min_confidence),
            0.0,
            1.0,
        )
        std = (
            self.config.low_confidence_measurement_std_px
            + confidence
            * (self.config.high_confidence_measurement_std_px - self.config.low_confidence_measurement_std_px)
        )
        return std * std

    def _update_with_measurement(self, selected: _CandidateMatch) -> None:
        assert self._state is not None
        assert self._covariance is not None
        innovation_x = selected.point[0] - self._state[0]
        innovation_y = selected.point[1] - self._state[1]
        measurement_variance = self._measurement_noise(selected.score)
        s00 = self._covariance[0][0] + measurement_variance
        s01 = self._covariance[0][1]
        s10 = self._covariance[1][0]
        s11 = self._covariance[1][1] + measurement_variance
        det = s00 * s11 - s01 * s10
        if abs(det) < 1e-9:
            return
        inv_s = [[s11 / det, -s01 / det], [-s10 / det, s00 / det]]
        gain = [
            [
                self._covariance[row][0] * inv_s[0][0] + self._covariance[row][1] * inv_s[1][0],
                self._covariance[row][0] * inv_s[0][1] + self._covariance[row][1] * inv_s[1][1],
            ]
            for row in range(4)
        ]

        self._state = [
            self._state[row] + gain[row][0] * innovation_x + gain[row][1] * innovation_y
            for row in range(4)
        ]
        kh = [[gain[row][0], gain[row][1], 0.0, 0.0] for row in range(4)]
        identity_minus_kh = [
            [(1.0 if row == col else 0.0) - kh[row][col] for col in range(4)]
            for row in range(4)
        ]
        self._covariance = _matmul(identity_minus_kh, self._covariance)

        if self._should_snap_to_measurement(selected, innovation_x, innovation_y):
            weight = _clamp(float(self.config.maneuver_snap_weight), 0.0, 1.0)
            self._state[0] = self._state[0] + weight * (selected.point[0] - self._state[0])
            self._state[1] = self._state[1] + weight * (selected.point[1] - self._state[1])

        if hypot(innovation_x, innovation_y) >= self.config.maneuver_innovation_px:
            self._maneuver_frames_remaining = max(
                self._maneuver_frames_remaining,
                int(self.config.maneuver_frames),
            )

    def _should_snap_to_measurement(
        self,
        selected: _CandidateMatch,
        innovation_x: float,
        innovation_y: float,
    ) -> bool:
        if selected.score < self.config.maneuver_snap_confidence:
            return False
        return hypot(innovation_x, innovation_y) >= self.config.maneuver_snap_innovation_px

    def _can_relock(self, candidate: _CandidateMatch) -> bool:
        if self._missed_frames >= self.config.relock_after_missed_frames:
            return candidate.score >= self.config.relock_confidence
        return False

    def _can_coast(self, occlusion_active: bool) -> bool:
        if self._state is None:
            return False
        vx = float(self._state[2])
        vy = float(self._state[3])
        if hypot(vx, vy) < self.config.min_coast_speed_px_per_sec:
            return False
        limit = self.config.occlusion_coast_frames if occlusion_active else self.config.max_coast_frames
        return self._coast_frames < int(limit)

    def _occlusion_likely(self, predicted_point: Point, person_bboxes: Sequence[PersonBBox]) -> bool:
        if not self.config.person_occlusion_enabled or not person_bboxes:
            return False
        if _point_inside_any_bbox(
            predicted_point,
            person_bboxes,
            padding=self.config.person_occlusion_padding_px,
        ):
            return True
        if self._state is None:
            return False
        previous = (float(self._state[0]), float(self._state[1]))
        return any(
            _segment_intersects_rect(
                previous,
                predicted_point,
                bbox,
                padding=self.config.person_occlusion_padding_px,
            )
            for bbox in person_bboxes
        )

    def _should_enter_out_of_frame(
        self,
        predicted_state: list[float],
        frame_size: FrameSize | None,
    ) -> bool:
        if frame_size is None:
            return False
        point = (float(predicted_state[0]), float(predicted_state[1]))
        if _point_inside_frame(point, frame_size, margin=self.config.out_of_frame_margin_px):
            return False
        vx = float(predicted_state[2])
        vy = float(predicted_state[3])
        width, height = frame_size
        margin = self.config.out_of_frame_margin_px
        return (
            (point[0] < -margin and vx <= 0.0)
            or (point[0] > width + margin and vx >= 0.0)
            or (point[1] < -margin and vy <= 0.0)
            or (point[1] > height + margin and vy >= 0.0)
        )

    def _out_of_frame_relock_candidate(
        self,
        candidates: Sequence[_CandidateMatch],
        frame_size: FrameSize | None,
    ) -> _CandidateMatch | None:
        if not candidates:
            return None
        reliable = [
            candidate
            for candidate in candidates
            if candidate.score >= self.config.relock_confidence and not self._near_edge(candidate.point, frame_size)
        ]
        if not reliable:
            return None
        return max(reliable, key=lambda item: (item.score, -item.index))

    def _near_edge(self, point: Point, frame_size: FrameSize | None) -> bool:
        if frame_size is None:
            return False
        width, height = frame_size
        band = max(0.0, float(self.config.edge_suppression_band_px))
        return point[0] <= band or point[0] >= width - band or point[1] <= band or point[1] >= height - band

    def _visible_from_state(
        self,
        score: float,
        heatmap_shape: Sequence[int],
        *,
        frame_size: FrameSize | None = None,
    ) -> TrackResult:
        if self._state is None:
            return self._invisible(score=score)
        point = (float(self._state[0]), float(self._state[1]))
        if frame_size is not None and not _point_inside_frame(point, frame_size, margin=0.0):
            return self._invisible(score=score)
        return TrackResult(
            ball_xy=[point[0], point[1]],
            visible=1,
            score=float(score),
            heatmap_shape=list(heatmap_shape),
        )

    def _invisible(self, *, score: float) -> TrackResult:
        return TrackResult(ball_xy=[-1.0, -1.0], visible=0, score=float(score), heatmap_shape=list(self._last_heatmap_shape))

    def _emit(self, track: TrackResult, *, flush: bool = False) -> TrackResult:
        lag = max(0, int(self.config.fixed_lag_frames))
        if flush or lag <= 0:
            if flush:
                self._output_buffer.clear()
            return track
        self._output_buffer.append(track)
        if len(self._output_buffer) <= lag:
            return track
        return self._pop_smoothed_output()

    def _pop_smoothed_output(self) -> TrackResult:
        oldest = self._output_buffer.popleft()
        if not oldest.visible:
            return oldest

        weighted_x = 0.0
        weighted_y = 0.0
        total_weight = 0.0
        for index, item in enumerate([oldest, *list(self._output_buffer)]):
            if not item.visible or len(item.ball_xy) < 2:
                continue
            weight = self.config.fixed_lag_future_decay**index
            weighted_x += float(item.ball_xy[0]) * weight
            weighted_y += float(item.ball_xy[1]) * weight
            total_weight += weight

        if total_weight <= 0.0:
            return oldest
        return TrackResult(
            ball_xy=[weighted_x / total_weight, weighted_y / total_weight],
            visible=1,
            score=float(oldest.score),
            heatmap_shape=list(oldest.heatmap_shape),
        )

    def _record_debug(
        self,
        *,
        action: str,
        reason: str,
        input_track: TrackResult | None,
        output_track: TrackResult,
        dt: float,
        frame_size: FrameSize | None,
        raw_candidate_count: int,
        candidate_count: int,
        candidates: Sequence[_CandidateMatch],
        selected: _CandidateMatch | None,
        predicted: Point | None,
        occlusion_active: bool,
    ) -> None:
        if not self.debug_enabled:
            return
        width, height = frame_size if frame_size is not None else (0.0, 0.0)
        input_x = input_track.ball_xy[0] if input_track is not None and len(input_track.ball_xy) > 0 else -1.0
        input_y = input_track.ball_xy[1] if input_track is not None and len(input_track.ball_xy) > 1 else -1.0
        input_score = float(input_track.score) if input_track is not None else 0.0
        output_x = output_track.ball_xy[0] if len(output_track.ball_xy) > 0 else -1.0
        output_y = output_track.ball_xy[1] if len(output_track.ball_xy) > 1 else -1.0
        velocity_x = float(self._state[2]) if self._state is not None else 0.0
        velocity_y = float(self._state[3]) if self._state is not None else 0.0
        record: dict[str, object] = {
            "frame_index": self._frame_index,
            "action": action,
            "reason": reason,
            "raw_candidate_count": raw_candidate_count,
            "candidate_count": candidate_count,
            "selected_candidate_index": selected.index if selected is not None else -1,
            "selected_candidate_rank": f"{selected.cost:.4f}" if selected is not None and isfinite(selected.cost) else "",
            "input_visible": int(bool(input_track.visible)) if input_track is not None else 0,
            "input_x": input_x,
            "input_y": input_y,
            "input_score": input_score,
            "output_visible": int(bool(output_track.visible)),
            "output_x": output_x,
            "output_y": output_y,
            "output_score": float(output_track.score),
            "locked_before": int(self._state is not None),
            "locked_after": int(self._state is not None),
            "missed_before": max(0, self._missed_frames - (1 if action in ("coast", "reject") else 0)),
            "missed_after": self._missed_frames,
            "coast_before": max(0, self._coast_frames - (1 if action == "coast" else 0)),
            "coast_after": self._coast_frames,
            "last_x_before": -1.0,
            "last_y_before": -1.0,
            "pred_x": predicted[0] if predicted is not None else -1.0,
            "pred_y": predicted[1] if predicted is not None else -1.0,
            "velocity_x_before": velocity_x,
            "velocity_y_before": velocity_y,
            "velocity_x_after": velocity_x,
            "velocity_y_after": velocity_y,
            "top_exit_remaining": self._out_of_frame_suppression_remaining,
            "frame_w": width,
            "frame_h": height,
            "dt": dt,
            "occlusion_active": int(occlusion_active),
            "candidate_mahalanobis": selected.mahalanobis if selected is not None else -1.0,
            "candidate_gate": selected.gate if selected is not None else -1.0,
            "candidates": self._format_candidates(candidates, frame_size),
        }
        self._last_debug_record = record
        self.debug_records.append(record)

    def _format_candidates(
        self,
        candidates: Sequence[_CandidateMatch],
        frame_size: FrameSize | None,
    ) -> str:
        items = []
        for candidate in candidates[:5]:
            edge = "edge" if self._near_edge(candidate.point, frame_size) else "mid"
            items.append(
                f"#{candidate.index}:({candidate.point[0]:.1f},{candidate.point[1]:.1f})"
                f"/s={candidate.score:.3f}/{edge}"
            )
        return " | ".join(items)


def _frame_size(frame_shape: FrameShape) -> FrameSize | None:
    if frame_shape is None or len(frame_shape) < 2:
        return None
    height = float(frame_shape[0])
    width = float(frame_shape[1])
    if width <= 0.0 or height <= 0.0:
        return None
    return width, height


def _track_point(track: TrackResult) -> Point | None:
    if len(track.ball_xy) < 2:
        return None
    try:
        x = float(track.ball_xy[0])
        y = float(track.ball_xy[1])
    except (TypeError, ValueError):
        return None
    if not isfinite(x) or not isfinite(y):
        return None
    return x, y


def _normalize_person_bboxes(person_bboxes: PersonBBoxes) -> list[PersonBBox]:
    if not person_bboxes:
        return []
    normalized: list[PersonBBox] = []
    for bbox in person_bboxes:
        if len(bbox) < 4:
            continue
        try:
            x1, y1, x2, y2 = (float(value) for value in bbox[:4])
        except (TypeError, ValueError):
            continue
        if not all(isfinite(value) for value in (x1, y1, x2, y2)):
            continue
        normalized.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
    return normalized


def _matvec(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> list[float]:
    return [sum(float(value) * float(vector[index]) for index, value in enumerate(row)) for row in matrix]


def _matmul(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> list[list[float]]:
    columns = list(zip(*right))
    return [
        [sum(float(a) * float(b) for a, b in zip(row, column)) for column in columns]
        for row in left
    ]


def _transpose(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    return [list(row) for row in zip(*matrix)]


def _matadd(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> list[list[float]]:
    return [
        [float(a) + float(b) for a, b in zip(left_row, right_row)]
        for left_row, right_row in zip(left, right)
    ]


def _point_inside_frame(point: Point, frame_size: FrameSize, *, margin: float) -> bool:
    width, height = frame_size
    return -margin <= point[0] < width + margin and -margin <= point[1] < height + margin


def _point_inside_any_bbox(point: Point, bboxes: Sequence[PersonBBox], *, padding: float) -> bool:
    return any(_point_inside_rect(point, bbox, padding=padding) for bbox in bboxes)


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


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))
