from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import hypot, isfinite
from typing import Any, Sequence

from src.postprocess.track_filter import (
    BallTrackFilter,
    FrameShape,
    PersonBBoxes,
    _extract_court_filter,
    _point_inside_court_plane,
    _point_inside_projected_court_air,
)
from src.utils.structures import TrackResult


Point = tuple[float, float]
FrameSize = tuple[float, float]


@dataclass(slots=True)
class TrackNetV3TrajectoryFilterConfig:
    fps: float = 30.0
    candidate_min_confidence: float = 0.35
    motion_consistency_enabled: bool = True
    motion_consistency_gate_px: float = 96.0
    motion_consistency_velocity_scale: float = 1.25
    motion_consistency_score_distance_px: float = 120.0
    inpaint_top_threshold_px: float = 30.0
    inpaint_top_threshold_ratio: float = 0.05
    inpaint_score: float = 0.35
    fixed_lag_frames: int = 0
    buffer_frames: int = 64
    court_filter_enabled: bool = True
    court_filter_margin_cm: float = 120.0
    court_filter_margin_px: float = 72.0
    court_air_extension_ratio: float = 1.0
    causal_bridge_enabled: bool = True
    causal_bridge_frames: int = 4
    causal_bridge_score: float = 0.30
    drift_guard_enabled: bool = True
    drift_guard_min_history_speed_px: float = 8.0
    drift_guard_min_jump_px: float = 48.0
    drift_guard_direction_gate_scale: float = 1.0
    drift_guard_hard_gate_scale: float = 2.5
    drift_guard_opposite_cosine: float = -0.15
    drift_guard_upward_impact_px: float = 40.0
    drift_guard_top_band_ratio: float = 0.08
    drift_guard_top_band_px: float = 58.0


@dataclass(slots=True)
class _BufferedTrack:
    frame_index: int
    raw: TrackResult
    repaired: TrackResult
    inpaint_mask: int = 0


@dataclass(slots=True)
class _Candidate:
    track: TrackResult
    index: int
    point: Point
    score: float


class TrackNetV3TrajectoryFilter:
    """TrackNetV3-style trajectory rectification.

    The original TrackNetV3 postprocess keeps predicted visible coordinates as
    they are, generates an inpaint mask for middle-of-frame disappearances, then
    fills masked spans with InpaintNet or linear interpolation. This adapter uses
    the same mask and linear interpolation rules while fitting the project's
    realtime TrackFilterAlgorithm interface.
    """

    def __init__(
        self,
        config: TrackNetV3TrajectoryFilterConfig | None = None,
        *,
        fps: float | None = None,
        debug_enabled: bool = False,
        fixed_lag_frames: int | None = None,
    ) -> None:
        self.config = config or TrackNetV3TrajectoryFilterConfig()
        if fps is not None and fps > 0:
            self.config.fps = float(fps)
        if fixed_lag_frames is not None:
            self.config.fixed_lag_frames = max(0, int(fixed_lag_frames))
        self.debug_enabled = debug_enabled
        self.debug_records: list[dict[str, object]] = []
        self._frame_index = -1
        self._buffer: deque[_BufferedTrack] = deque()
        self._last_debug_record: dict[str, object] | None = None
        self._last_visible_point: Point | None = None
        self._missing_frames = 0

    def reset(self) -> None:
        self.debug_records.clear()
        self._frame_index = -1
        self._buffer.clear()
        self._last_debug_record = None
        self._last_visible_point = None
        self._missing_frames = 0

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
        del person_bboxes
        self._frame_index += 1
        step_dt = self._resolve_dt(dt)
        frame_size = _frame_size(frame_shape)
        court_filter = _extract_court_filter(court_prediction) if self.config.court_filter_enabled else None
        candidates = self._valid_candidates(tracks, frame_size)
        candidate_count_before_court = len(candidates)
        candidates = self._filter_candidates_by_court(candidates, frame_size, court_filter)
        selected = self._select_candidate(candidates, frame_size)
        raw_track = selected.track if selected is not None else self._invisible(score=_max_score(tracks))
        last_visible_before = self._last_visible_point
        if selected is None:
            self._missing_frames += 1
        else:
            self._missing_frames = 0
            self._last_visible_point = selected.point

        sample = _BufferedTrack(
            frame_index=self._frame_index,
            raw=_copy_track(raw_track),
            repaired=_copy_track(raw_track),
        )
        self._buffer.append(sample)
        self._apply_tracknet_v3_repair(frame_size)
        self._apply_causal_bridge(sample, frame_size)
        emitted, source_offset, emitted_inpaint_mask = self._emit(sample)

        action, reason = self._action_reason(selected, emitted, source_offset, emitted_inpaint_mask)
        self._record_debug(
            action=action,
            reason=reason,
            input_track=selected.track if selected is not None else None,
            output_track=emitted,
            dt=step_dt,
            frame_size=frame_size,
            raw_candidate_count=len(tracks),
            candidate_count=len(candidates),
            court_filter_active=court_filter is not None,
            court_filtered_count=max(0, candidate_count_before_court - len(candidates)),
            candidates=candidates,
            selected=selected,
            source_offset=source_offset,
            emitted_inpaint_mask=emitted_inpaint_mask,
            last_visible_before=last_visible_before,
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
    ) -> list[_Candidate]:
        candidates: list[_Candidate] = []
        for index, track in enumerate(tracks):
            point = _track_point(track)
            score = float(track.score)
            if point is None or not track.visible or score < self.config.candidate_min_confidence:
                continue
            if frame_size is not None and not _point_inside_frame(point, frame_size):
                continue
            candidates.append(_Candidate(track=track, index=index, point=point, score=score))
        return candidates

    def _filter_candidates_by_court(
        self,
        candidates: Sequence[_Candidate],
        frame_size: FrameSize | None,
        court_filter: object | None,
    ) -> list[_Candidate]:
        if court_filter is None:
            return list(candidates)
        return [
            candidate
            for candidate in candidates
            if self._point_inside_court_region(candidate.point, court_filter, frame_size)
        ]

    def _point_inside_court_region(
        self,
        point: Point,
        court_filter: object,
        frame_size: FrameSize | None,
    ) -> bool:
        corners = getattr(court_filter, "corners", None)
        if corners is not None:
            return _point_inside_projected_court_air(
                point,
                corners,
                margin_px=self.config.court_filter_margin_px,
                air_extension_ratio=self.config.court_air_extension_ratio,
                frame_size=frame_size,
            )

        image_to_court_h = getattr(court_filter, "image_to_court_h", None)
        if image_to_court_h is None:
            return False
        return _point_inside_court_plane(point, image_to_court_h, self.config.court_filter_margin_cm)

    def _select_candidate(
        self,
        candidates: Sequence[_Candidate],
        frame_size: FrameSize | None,
    ) -> _Candidate | None:
        if not candidates:
            return None

        primary = max(candidates, key=lambda item: (item.score, -item.index))
        if not self.config.motion_consistency_enabled:
            return primary

        prediction = self._predict_next_visible_point()
        if prediction is None:
            return primary

        gate = self._motion_consistency_gate()
        stable_candidates = [candidate for candidate in candidates if _distance(candidate.point, prediction) <= gate]
        if not stable_candidates:
            if self._candidate_looks_like_drift(primary, prediction, gate, frame_size):
                return None
            return primary

        return max(
            stable_candidates,
            key=lambda item: self._motion_consistency_rank(item, prediction),
        )

    def _candidate_looks_like_drift(
        self,
        candidate: _Candidate,
        prediction: Point,
        gate: float,
        frame_size: FrameSize | None,
    ) -> bool:
        if not self.config.drift_guard_enabled:
            return False
        if self._missing_frames > 0:
            return False

        motion = self._recent_visible_motion()
        if motion is None:
            return False
        last_point, velocity, recent_speed = motion
        if recent_speed < max(0.0, float(self.config.drift_guard_min_history_speed_px)):
            return False

        displacement = (
            candidate.point[0] - last_point[0],
            candidate.point[1] - last_point[1],
        )
        displacement_length = _length(displacement)
        min_jump = max(0.0, float(self.config.drift_guard_min_jump_px))
        if displacement_length < min_jump:
            return False

        prediction_error = _distance(candidate.point, prediction)
        direction_gate = max(gate * max(0.0, float(self.config.drift_guard_direction_gate_scale)), min_jump)
        if prediction_error <= direction_gate:
            return False
        if self._drift_guard_near_top(last_point, candidate.point, prediction, frame_size):
            return False
        if displacement[1] <= -max(0.0, float(self.config.drift_guard_upward_impact_px)):
            return False

        cosine = _dot(displacement, velocity) / max(displacement_length * recent_speed, 1e-6)
        hard_gate = max(gate * max(0.0, float(self.config.drift_guard_hard_gate_scale)), min_jump * 2.0)
        return prediction_error >= hard_gate or cosine <= float(self.config.drift_guard_opposite_cosine)

    def _predict_next_visible_point(self) -> Point | None:
        visible_samples: list[_BufferedTrack] = []
        for sample in reversed(self._buffer):
            if sample.raw.visible and _track_point(sample.raw) is not None:
                visible_samples.append(sample)
                if len(visible_samples) >= 2:
                    break
        if len(visible_samples) < 2:
            return None

        last = visible_samples[0]
        previous = visible_samples[1]
        last_point = _track_point(last.raw)
        previous_point = _track_point(previous.raw)
        if last_point is None or previous_point is None:
            return None

        history_gap = max(1, last.frame_index - previous.frame_index)
        target_gap = max(1, self._frame_index - last.frame_index)
        velocity = (
            (last_point[0] - previous_point[0]) / float(history_gap),
            (last_point[1] - previous_point[1]) / float(history_gap),
        )
        return (
            last_point[0] + velocity[0] * float(target_gap),
            last_point[1] + velocity[1] * float(target_gap),
        )

    def _motion_consistency_gate(self) -> float:
        base_gate = max(0.0, float(self.config.motion_consistency_gate_px))
        recent_speed = self._recent_visible_speed_px_per_frame()
        return max(
            base_gate,
            recent_speed * max(0.0, float(self.config.motion_consistency_velocity_scale)),
        )

    def _recent_visible_speed_px_per_frame(self) -> float:
        motion = self._recent_visible_motion()
        return motion[2] if motion is not None else 0.0

    def _recent_visible_motion(self) -> tuple[Point, Point, float] | None:
        visible_samples: list[_BufferedTrack] = []
        for sample in reversed(self._buffer):
            if sample.raw.visible and _track_point(sample.raw) is not None:
                visible_samples.append(sample)
                if len(visible_samples) >= 2:
                    break
        if len(visible_samples) < 2:
            return None

        last = visible_samples[0]
        previous = visible_samples[1]
        last_point = _track_point(last.raw)
        previous_point = _track_point(previous.raw)
        if last_point is None or previous_point is None:
            return None

        frame_gap = max(1, last.frame_index - previous.frame_index)
        velocity = (
            (last_point[0] - previous_point[0]) / float(frame_gap),
            (last_point[1] - previous_point[1]) / float(frame_gap),
        )
        return last_point, velocity, _length(velocity)

    def _drift_guard_near_top(
        self,
        last_point: Point,
        candidate: Point,
        prediction: Point,
        frame_size: FrameSize | None,
    ) -> bool:
        top_band = max(0.0, float(self.config.drift_guard_top_band_px))
        if frame_size is not None:
            _, height = frame_size
            top_band = max(top_band, height * max(0.0, float(self.config.drift_guard_top_band_ratio)))
        return min(last_point[1], candidate[1], prediction[1]) <= top_band

    def _motion_consistency_rank(self, candidate: _Candidate, prediction: Point) -> tuple[float, float, int]:
        distance = _distance(candidate.point, prediction)
        distance_scale = max(1.0, float(self.config.motion_consistency_score_distance_px))
        return (
            candidate.score - distance / distance_scale,
            -distance,
            -candidate.index,
        )

    def _apply_tracknet_v3_repair(self, frame_size: FrameSize | None) -> None:
        if not self._buffer:
            return

        samples = list(self._buffer)
        xs = [_track_x(sample.raw) for sample in samples]
        ys = [_track_y(sample.raw) for sample in samples]
        visibility = [int(bool(sample.raw.visible)) for sample in samples]
        mask = generate_tracknet_v3_inpaint_mask(
            ys,
            visibility,
            height_threshold=self._inpaint_height_threshold(frame_size),
        )
        repaired_xs = linear_interpolate_masked_values(xs, mask)
        repaired_ys = linear_interpolate_masked_values(ys, mask)

        for index, sample in enumerate(samples):
            if not mask[index]:
                if not sample.inpaint_mask:
                    sample.repaired = _copy_track(sample.raw)
                continue
            if sample.inpaint_mask:
                continue
            sample.inpaint_mask = int(mask[index])
            score, heatmap_shape = self._inpaint_track_metadata(samples, index)
            sample.repaired = TrackResult(
                ball_xy=[float(repaired_xs[index]), float(repaired_ys[index])],
                visible=1,
                score=score,
                heatmap_shape=heatmap_shape,
            )

    def _inpaint_height_threshold(self, frame_size: FrameSize | None) -> float:
        if frame_size is None:
            return float(self.config.inpaint_top_threshold_px)
        _, height = frame_size
        return float(height) * max(0.0, float(self.config.inpaint_top_threshold_ratio))

    def _inpaint_track_metadata(
        self,
        samples: Sequence[_BufferedTrack],
        index: int,
    ) -> tuple[float, list[int]]:
        scores: list[float] = [float(self.config.inpaint_score)]
        heatmap_shape: list[int] = []
        for previous in range(index - 1, -1, -1):
            if samples[previous].raw.visible:
                scores.append(float(samples[previous].raw.score))
                heatmap_shape = list(samples[previous].raw.heatmap_shape)
                break
        for following in range(index + 1, len(samples)):
            if samples[following].raw.visible:
                scores.append(float(samples[following].raw.score))
                if not heatmap_shape:
                    heatmap_shape = list(samples[following].raw.heatmap_shape)
                break
        return max(0.0, min(scores)), heatmap_shape

    def _apply_causal_bridge(self, sample: _BufferedTrack, frame_size: FrameSize | None) -> None:
        if not self.config.causal_bridge_enabled:
            return
        if int(self.config.fixed_lag_frames) > 0:
            return
        if sample.raw.visible or sample.repaired.visible:
            return
        if self._missing_frames <= 0 or self._missing_frames > max(0, int(self.config.causal_bridge_frames)):
            return

        prediction = self._predict_next_visible_point()
        if prediction is None:
            return
        if frame_size is not None and not _point_inside_frame(prediction, frame_size):
            return
        if self._causal_bridge_top_exit_likely(prediction, frame_size):
            return

        sample.inpaint_mask = 2
        sample.repaired = TrackResult(
            ball_xy=[float(prediction[0]), float(prediction[1])],
            visible=1,
            score=max(0.0, min(float(self.config.causal_bridge_score), 1.0)),
            heatmap_shape=list(sample.raw.heatmap_shape),
        )

    def _causal_bridge_top_exit_likely(self, prediction: Point, frame_size: FrameSize | None) -> bool:
        visible_samples: list[_BufferedTrack] = []
        for sample in reversed(self._buffer):
            if sample.raw.visible and _track_point(sample.raw) is not None:
                visible_samples.append(sample)
                if len(visible_samples) >= 2:
                    break
        if not visible_samples:
            return False

        last_point = _track_point(visible_samples[0].raw)
        if last_point is None:
            return False
        top_threshold = self._inpaint_height_threshold(frame_size)
        if last_point[1] <= top_threshold:
            return True

        if len(visible_samples) < 2:
            return False
        previous_point = _track_point(visible_samples[1].raw)
        if previous_point is None:
            return False
        moving_up = last_point[1] < previous_point[1]
        return moving_up and prediction[1] <= top_threshold

    def _emit(self, current_sample: _BufferedTrack) -> tuple[TrackResult, int, int]:
        lag = max(0, int(self.config.fixed_lag_frames))
        if lag <= 0 or len(self._buffer) <= lag:
            self._trim_buffer()
            return _copy_track(current_sample.repaired), 0, current_sample.inpaint_mask

        emitted_sample = self._buffer.popleft()
        self._trim_buffer()
        return (
            _copy_track(emitted_sample.repaired),
            self._frame_index - emitted_sample.frame_index,
            emitted_sample.inpaint_mask,
        )

    def _trim_buffer(self) -> None:
        limit = max(2, int(self.config.buffer_frames))
        while len(self._buffer) > limit:
            self._buffer.popleft()

    def _action_reason(
        self,
        selected: _Candidate | None,
        emitted: TrackResult,
        source_offset: int,
        emitted_inpaint_mask: int,
    ) -> tuple[str, str]:
        if emitted.visible and emitted_inpaint_mask:
            if emitted_inpaint_mask == 2:
                return "inpaint", "tracknetv2_short_gap_bridge"
            return "inpaint", "tracknet_v3_linear_inpaint"
        if emitted.visible and source_offset > 0:
            return "accept", "tracknet_v3_lag_emit"
        if emitted.visible and selected is not None:
            return "accept", "tracknet_v3_candidate"
        return "reject", "missing_or_low_confidence"

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
        court_filter_active: bool,
        court_filtered_count: int,
        candidates: Sequence[_Candidate],
        selected: _Candidate | None,
        source_offset: int,
        emitted_inpaint_mask: int,
        last_visible_before: Point | None,
    ) -> None:
        if not self.debug_enabled:
            return
        width, height = frame_size if frame_size is not None else (0.0, 0.0)
        input_x = input_track.ball_xy[0] if input_track is not None and len(input_track.ball_xy) > 0 else -1.0
        input_y = input_track.ball_xy[1] if input_track is not None and len(input_track.ball_xy) > 1 else -1.0
        input_score = float(input_track.score) if input_track is not None else 0.0
        output_x = output_track.ball_xy[0] if len(output_track.ball_xy) > 0 else -1.0
        output_y = output_track.ball_xy[1] if len(output_track.ball_xy) > 1 else -1.0
        record: dict[str, object] = {
            "frame_index": self._frame_index,
            "action": action,
            "reason": reason,
            "raw_candidate_count": raw_candidate_count,
            "candidate_count": candidate_count,
            "court_filter_active": int(bool(court_filter_active)),
            "court_filtered_count": int(court_filtered_count),
            "selected_candidate_index": selected.index if selected is not None else -1,
            "selected_candidate_rank": f"{selected.score:.4f}" if selected is not None else "",
            "input_visible": int(bool(input_track.visible)) if input_track is not None else 0,
            "input_x": input_x,
            "input_y": input_y,
            "input_score": input_score,
            "output_visible": int(bool(output_track.visible)),
            "output_x": output_x,
            "output_y": output_y,
            "output_score": float(output_track.score),
            "locked_before": int(self._last_visible_point is not None),
            "locked_after": int(self._last_visible_point is not None),
            "missed_before": max(0, self._missing_frames - (1 if selected is None else 0)),
            "missed_after": self._missing_frames,
            "coast_before": 0,
            "coast_after": 0,
            "last_x_before": last_visible_before[0] if last_visible_before is not None else -1.0,
            "last_y_before": last_visible_before[1] if last_visible_before is not None else -1.0,
            "pred_x": -1.0,
            "pred_y": -1.0,
            "velocity_x_before": 0.0,
            "velocity_y_before": 0.0,
            "velocity_x_after": 0.0,
            "velocity_y_after": 0.0,
            "top_exit_remaining": 0,
            "frame_w": width,
            "frame_h": height,
            "dt": dt,
            "source_frame_offset": source_offset,
            "inpaint_mask": int(emitted_inpaint_mask),
            "candidates": self._format_candidates(candidates, frame_size),
        }
        self._last_debug_record = record
        self.debug_records.append(record)

    def _format_candidates(
        self,
        candidates: Sequence[_Candidate],
        frame_size: FrameSize | None,
    ) -> str:
        items = []
        for candidate in candidates[:5]:
            edge = "edge" if frame_size is not None and _near_edge(candidate.point, frame_size) else "mid"
            items.append(
                f"#{candidate.index}:({candidate.point[0]:.1f},{candidate.point[1]:.1f})"
                f"/s={candidate.score:.3f}/{edge}"
            )
        return " | ".join(items)

    def _invisible(self, *, score: float) -> TrackResult:
        return TrackResult(ball_xy=[-1.0, -1.0], visible=0, score=float(score), heatmap_shape=[])


def create_tracknet_v3_ball_track_filter(
    *,
    fps: float | None = None,
    debug_enabled: bool = False,
    fixed_lag_frames: int | None = None,
) -> BallTrackFilter:
    resolved_fps = float(fps) if fps is not None and fps > 0 else TrackNetV3TrajectoryFilterConfig.fps
    return BallTrackFilter(
        fps=resolved_fps,
        debug_enabled=debug_enabled,
        algorithm=TrackNetV3TrajectoryFilter(
            fps=resolved_fps,
            debug_enabled=debug_enabled,
            fixed_lag_frames=fixed_lag_frames,
        ),
    )


def generate_tracknet_v3_inpaint_mask(
    y_values: Sequence[float],
    visibility: Sequence[int],
    *,
    height_threshold: float = 30.0,
) -> list[int]:
    if len(y_values) != len(visibility):
        raise ValueError("Length of y_values and visibility should be the same")

    y = [float(value) for value in y_values]
    vis = [int(value) for value in visibility]
    inpaint_mask = [0 for _ in y]
    i = 0
    j = 0
    threshold = float(height_threshold)
    while j < len(vis):
        while i < len(vis) - 1 and vis[i] == 1:
            i += 1
        j = i
        while j < len(vis) - 1 and vis[j] == 0:
            j += 1
        if j == i:
            break
        if i == 0 and y[j] > threshold:
            for index in range(j):
                inpaint_mask[index] = 1
        elif (i > 1 and y[i - 1] > threshold) and (j < len(vis) and y[j] > threshold):
            for index in range(i, j):
                inpaint_mask[index] = 1
        i = j
    return inpaint_mask


def linear_interpolate_masked_values(target: Sequence[float], inpaint_mask: Sequence[int]) -> list[float]:
    if len(target) != len(inpaint_mask):
        raise ValueError("Length of target and inpaint_mask should be the same")

    values = [float(value) for value in target]
    mask = [int(value) for value in inpaint_mask]
    i = 0
    while i < len(mask):
        if mask[i] == 0:
            i += 1
            continue

        start = i
        while i < len(mask) and mask[i] == 1:
            i += 1
        end = i

        left_index = start - 1 if start > 0 else None
        right_index = end if end < len(mask) else None
        if left_index is None and right_index is None:
            continue
        if left_index is None:
            fill = values[right_index]
            for index in range(start, end):
                values[index] = fill
            continue
        if right_index is None:
            fill = values[left_index]
            for index in range(start, end):
                values[index] = fill
            continue

        left = values[left_index]
        right = values[right_index]
        span = end - start
        for offset, index in enumerate(range(start, end)):
            alpha = float(offset + 1) / float(span + 1)
            values[index] = left * (1.0 - alpha) + right * alpha
    return values


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


def _track_x(track: TrackResult) -> float:
    point = _track_point(track)
    return point[0] if point is not None and track.visible else -1.0


def _track_y(track: TrackResult) -> float:
    point = _track_point(track)
    return point[1] if point is not None and track.visible else -1.0


def _copy_track(track: TrackResult) -> TrackResult:
    return TrackResult(
        ball_xy=[float(value) for value in track.ball_xy[:2]] if len(track.ball_xy) >= 2 else [-1.0, -1.0],
        visible=int(bool(track.visible)),
        score=float(track.score),
        heatmap_shape=list(track.heatmap_shape),
    )


def _max_score(tracks: Sequence[TrackResult]) -> float:
    if not tracks:
        return 0.0
    return max(float(track.score) for track in tracks)


def _point_inside_frame(point: Point, frame_size: FrameSize) -> bool:
    width, height = frame_size
    return 0.0 <= point[0] < width and 0.0 <= point[1] < height


def _distance(a: Point, b: Point) -> float:
    return hypot(a[0] - b[0], a[1] - b[1])


def _length(vector: Point) -> float:
    return hypot(vector[0], vector[1])


def _dot(a: Point, b: Point) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _near_edge(point: Point, frame_size: FrameSize) -> bool:
    width, height = frame_size
    band = max(0.0, min(width, height) * 0.04)
    return point[0] <= band or point[0] >= width - band or point[1] <= band or point[1] >= height - band
