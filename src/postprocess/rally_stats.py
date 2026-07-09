from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from math import hypot
from typing import Any, Mapping, Sequence


PLAYER_KEYS = ("top", "bottom")
PLAYER_LABELS = {
    "top": "上方球员",
    "bottom": "下方球员",
}
COURT_WIDTH_CM = 610.0
COURT_LENGTH_CM = 1340.0


@dataclass
class _PlayerMotionState:
    last_point: tuple[float, float] | None = None
    last_timestamp_ms: int | None = None
    last_speed_mps: float = 0.0
    moving: bool = False
    high_intensity: bool = False
    current_continuous_m: float = 0.0
    total_distance_m: float = 0.0
    tracked_ms: int = 0
    speed_sum_mps: float = 0.0
    speed_samples: int = 0
    max_speed_mps: float = 0.0
    start_count: int = 0
    stop_count: int = 0
    high_intensity_count: int = 0
    max_continuous_m: float = 0.0
    forward_m: float = 0.0
    backward_m: float = 0.0
    left_m: float = 0.0
    right_m: float = 0.0
    stance_depth_sum_cm: float = 0.0
    stance_depth_samples: int = 0
    hit_count: int = 0
    passive_hit_count: int = 0
    zone_hits: Counter[str] = field(default_factory=Counter)


@dataclass
class _BallSpeedState:
    last_point: tuple[float, float] | None = None
    last_timestamp_ms: int | None = None
    current_speed_mps: float = 0.0
    current_valid: bool = False
    speed_sum_mps: float = 0.0
    speed_samples: int = 0
    max_speed_mps: float = 0.0
    recent_speed_mps: list[float] = field(default_factory=list)


class RallyStatsAccumulator:
    """Accumulate rally-level metrics from court-plane player and hit events."""

    def __init__(
        self,
        *,
        rally_id: str = "",
        rally_name: str = "",
        fps: float = 25.0,
        court_width_cm: float = COURT_WIDTH_CM,
        court_length_cm: float = COURT_LENGTH_CM,
        min_step_cm: float = 2.0,
        max_step_cm: float = 180.0,
        start_speed_mps: float = 1.20,
        stop_speed_mps: float = 0.35,
        high_intensity_speed_mps: float = 3.00,
        passive_hit_speed_mps: float = 0.75,
        ball_min_step_cm: float = 1.0,
        ball_max_speed_mps: float = 120.0,
        ball_speed_smoothing: float = 0.20,
        ball_speed_window_size: int = 5,
        ball_speed_outlier_delta_mps: float = 55.0,
        ball_speed_outlier_ratio: float = 4.0,
        ball_speed_max_accel_mps2: float = 120.0,
        start_visible_frames: int = 5,
        start_min_motion_px: float = 30.0,
        start_min_avg_ball_score: float = 0.40,
        freeze_after_rally_end: bool = True,
    ) -> None:
        self.rally_id = str(rally_id)
        self.rally_name = str(rally_name)
        self.fps = float(fps) if fps > 0 else 25.0
        self.court_width_cm = max(1.0, float(court_width_cm))
        self.court_length_cm = max(1.0, float(court_length_cm))
        self.min_step_cm = max(0.0, float(min_step_cm))
        self.max_step_cm = max(self.min_step_cm, float(max_step_cm))
        self.start_speed_mps = max(0.0, float(start_speed_mps))
        self.stop_speed_mps = max(0.0, float(stop_speed_mps))
        self.high_intensity_speed_mps = max(0.0, float(high_intensity_speed_mps))
        self.passive_hit_speed_mps = max(0.0, float(passive_hit_speed_mps))
        self.ball_min_step_cm = max(0.0, float(ball_min_step_cm))
        self.ball_max_speed_mps = max(0.0, float(ball_max_speed_mps))
        self.ball_speed_smoothing = max(0.0, min(float(ball_speed_smoothing), 1.0))
        self.ball_speed_window_size = max(1, int(ball_speed_window_size))
        self.ball_speed_outlier_delta_mps = max(0.0, float(ball_speed_outlier_delta_mps))
        self.ball_speed_outlier_ratio = max(1.0, float(ball_speed_outlier_ratio))
        self.ball_speed_max_accel_mps2 = max(0.0, float(ball_speed_max_accel_mps2))
        self.start_visible_frames = max(1, int(start_visible_frames))
        self.start_min_motion_px = max(0.0, float(start_min_motion_px))
        self.start_min_avg_ball_score = max(0.0, float(start_min_avg_ball_score))
        self.freeze_after_rally_end = bool(freeze_after_rally_end)

        self._players = {key: _PlayerMotionState() for key in PLAYER_KEYS}
        self._ball_speed = _BallSpeedState()
        self._frame_count = 0
        self._ball_visible_frames = 0
        self._pose_valid_frames = 0
        self._court_valid_frames = 0
        self._track_score_sum = 0.0
        self._track_score_samples = 0
        self._start_timestamp_ms: int | None = None
        self._last_timestamp_ms: int | None = None
        self._rally_start_timestamp_ms: int | None = None
        self._rally_end_timestamp_ms: int | None = None
        self._start_candidate_timestamp_ms: int | None = None
        self._start_candidate_point: tuple[float, float] | None = None
        self._start_candidate_visible_frames = 0
        self._start_candidate_score_sum = 0.0
        self._start_candidate_max_motion_px = 0.0
        self._landing_count = 0
        self._out_of_frame_count = 0
        self._hits: list[dict[str, Any]] = []
        self._hit_index_by_key: dict[tuple[str, int], int] = {}
        self._stroke_counts: Counter[str] = Counter()
        self._hit_confidence_sum = 0.0
        self._hit_confidence_samples = 0

    def update_frame(
        self,
        *,
        timestamp_ms: int,
        player_points: Mapping[int, Sequence[float]] | None,
        ball_visible: bool,
        ball_xy: Sequence[float] | None = None,
        ball_court_xy: Sequence[float] | None = None,
        ball_score: float = 0.0,
        court_valid: bool = False,
    ) -> None:
        timestamp_ms = max(0, int(timestamp_ms))
        if self.freeze_after_rally_end and self._rally_end_timestamp_ms is not None and timestamp_ms > self._rally_end_timestamp_ms:
            return
        if self._start_timestamp_ms is None:
            self._start_timestamp_ms = timestamp_ms
        self._last_timestamp_ms = timestamp_ms
        self._frame_count += 1
        self._ball_visible_frames += int(bool(ball_visible))
        self._court_valid_frames += int(bool(court_valid))
        self._track_score_sum += max(0.0, float(ball_score))
        self._track_score_samples += 1
        if self._should_track_rally_start(timestamp_ms):
            restarted = self._update_rally_start_candidate(timestamp_ms, bool(ball_visible), ball_xy, ball_score)
            if restarted:
                self._rally_end_timestamp_ms = None
        self._update_ball_speed(timestamp_ms, ball_court_xy if ball_visible else None)

        valid_points = self._clean_player_points(player_points)
        self._pose_valid_frames += int(bool(valid_points))
        seen_keys: set[str] = set()
        for player_index, point in valid_points.items():
            if 0 <= player_index < len(PLAYER_KEYS):
                player_key = PLAYER_KEYS[player_index]
                self._update_player_motion(player_key, point, timestamp_ms)
                seen_keys.add(player_key)

        for player_key, state in self._players.items():
            if player_key not in seen_keys:
                state.last_point = None
                state.last_timestamp_ms = None
                state.moving = False
                state.high_intensity = False
                state.current_continuous_m = 0.0

    def add_trajectory_event(
        self,
        event: object,
        *,
        ball_court_xy: Sequence[float] | None = None,
    ) -> None:
        if not isinstance(event, dict):
            return
        event_type = str(event.get("event_type", ""))
        if event_type not in {"hit", "landing", "out_of_frame"}:
            return
        timestamp_ms = max(0, int(event.get("timestamp_ms", 0)))
        if self.freeze_after_rally_end and self._rally_end_timestamp_ms is not None and timestamp_ms > self._rally_end_timestamp_ms:
            return
        if self._last_timestamp_ms is None or timestamp_ms > self._last_timestamp_ms:
            self._last_timestamp_ms = timestamp_ms
        frame_id = int(event.get("frame_id", -1))
        key = self._hit_key(frame_id, timestamp_ms)

        if event_type == "hit":
            if self._rally_start_timestamp_ms is None:
                self._rally_start_timestamp_ms = timestamp_ms
            elif (
                not self.freeze_after_rally_end
                and self._rally_end_timestamp_ms is not None
                and timestamp_ms > self._rally_end_timestamp_ms
            ):
                self._rally_start_timestamp_ms = timestamp_ms
                self._rally_end_timestamp_ms = None
                self._reset_rally_start_candidate()
            confidence = self._safe_float(event.get("confidence", 0.0))
            hit = self._ensure_hit_record(key, frame_id=frame_id, timestamp_ms=timestamp_ms)
            already_registered = bool(hit.get("_event_registered"))
            hit["event_confidence"] = confidence
            hit["source"] = "trajectory"
            if ball_court_xy is not None:
                hit["court_xy"] = self._clean_point(ball_court_xy)
            if not already_registered:
                self._sample_hit_confidence(confidence)
                hit["_event_registered"] = True
            return

        if event_type == "landing":
            self._landing_count += 1
            self._rally_end_timestamp_ms = timestamp_ms
            self._reset_rally_start_candidate()
        elif event_type == "out_of_frame":
            self._out_of_frame_count += 1

    def add_bst_prediction(self, prediction: object) -> None:
        if not isinstance(prediction, dict):
            return

        timestamp_ms = max(0, int(prediction.get("timestamp_ms", 0)))
        if self.freeze_after_rally_end and self._rally_end_timestamp_ms is not None and timestamp_ms > self._rally_end_timestamp_ms:
            return
        frame_id = int(prediction.get("event_frame_id", prediction.get("frame_id", -1)))
        key = self._hit_key(frame_id, timestamp_ms)
        hit = self._ensure_hit_record(key, frame_id=frame_id, timestamp_ms=timestamp_ms)

        class_label = str(prediction.get("pred_name", prediction.get("pred_display_name", "未知动作")))
        display_label = str(prediction.get("pred_display_name", class_label))
        player_key, class_stroke_label = self._parse_stroke_label(class_label)
        _display_player_key, display_stroke_label = self._parse_stroke_label(display_label)
        if player_key is None:
            player_key = _display_player_key
        stroke_label = display_stroke_label or class_stroke_label
        confidence = self._safe_float(prediction.get("confidence", 0.0))
        court_xy = self._clean_point(prediction.get("hit_court_xy"))
        if court_xy is None:
            court_xy = self._clean_point(hit.get("court_xy"))
        if court_xy is None:
            court_xy = self._estimate_hit_point_from_player(player_key)

        hit.update(
            {
                "source": "bst",
                "player": player_key,
                "player_label": PLAYER_LABELS.get(player_key, "") if player_key else "",
                "stroke": stroke_label,
                "raw_stroke": class_label,
                "confidence": confidence,
                "court_xy": court_xy,
                "zone": self._court_zone(player_key, court_xy),
                "used_homography": bool(prediction.get("used_homography")),
            }
        )
        if not bool(hit.get("_bst_registered")):
            self._register_player_hit(hit)
            self._stroke_counts[stroke_label] += 1
            self._sample_hit_confidence(confidence)
            hit["_bst_registered"] = True

    def summary(self) -> dict[str, Any]:
        duration_ms = self.duration_ms()
        player_summaries = {
            player_key: self._player_summary(player_key, state)
            for player_key, state in self._players.items()
        }
        hit_times = sorted(
            int(hit.get("timestamp_ms", 0))
            for hit in self._hits
            if int(hit.get("timestamp_ms", 0)) >= 0
        )
        avg_hit_interval_ms = 0.0
        if len(hit_times) >= 2:
            intervals = [right - left for left, right in zip(hit_times, hit_times[1:]) if right >= left]
            avg_hit_interval_ms = sum(intervals) / max(1, len(intervals))

        total_distance_m = sum(item["distance_m"] for item in player_summaries.values())
        duration_s = max(0.001, duration_ms / 1000.0)
        high_intensity_total = sum(item["high_intensity_count"] for item in player_summaries.values())
        motion_intensity_score = min(100.0, (total_distance_m / duration_s) * 18.0 + high_intensity_total * 4.0)

        return {
            "rally_id": self.rally_id,
            "rally_name": self.rally_name,
            "duration_ms": duration_ms,
            "duration_s": duration_ms / 1000.0,
            "rally_start_ms": self._rally_start_ms(),
            "rally_end_ms": self._rally_end_ms(),
            "rally_duration_s": self._rally_duration_s(),
            "rally_state": self._rally_state(),
            "frame_count": self._frame_count,
            "rally_hit_count": len(self._hits),
            "landing_count": self._landing_count,
            "out_of_frame_count": self._out_of_frame_count,
            "avg_hit_interval_ms": avg_hit_interval_ms,
            "players": player_summaries,
            "stroke_distribution": dict(self._stroke_counts),
            "hit_confidence_avg": self._hit_confidence_sum / max(1, self._hit_confidence_samples),
            "ball_speed": self._ball_speed_summary(),
            "motion_intensity_score": motion_intensity_score,
            "high_intensity_count": high_intensity_total,
            "data_reliability": {
                "ball_visible_rate": self._rate(self._ball_visible_frames, self._frame_count),
                "pose_valid_rate": self._rate(self._pose_valid_frames, self._frame_count),
                "court_valid_rate": self._rate(self._court_valid_frames, self._frame_count),
                "avg_ball_confidence": self._track_score_sum / max(1, self._track_score_samples),
            },
        }

    def details(self) -> dict[str, Any]:
        return {
            "hits": [self._public_hit(hit) for hit in self._hits],
            "players": {
                player_key: self._player_summary(player_key, state)
                for player_key, state in self._players.items()
            },
        }

    def export_record(self) -> dict[str, Any]:
        return {
            "rally_id": self.rally_id,
            "rally_name": self.rally_name,
            "summary": self.summary(),
            "details": self.details(),
        }

    def duration_ms(self) -> int:
        if self._start_timestamp_ms is None or self._last_timestamp_ms is None:
            return 0
        return max(0, int(self._last_timestamp_ms - self._start_timestamp_ms))

    def _update_player_motion(
        self,
        player_key: str,
        point: tuple[float, float],
        timestamp_ms: int,
    ) -> None:
        state = self._players[player_key]
        self._sample_stance_depth(player_key, state, point)
        if state.last_point is None or state.last_timestamp_ms is None:
            state.last_point = point
            state.last_timestamp_ms = timestamp_ms
            return

        dt_ms = max(0, timestamp_ms - state.last_timestamp_ms)
        step_cm = hypot(point[0] - state.last_point[0], point[1] - state.last_point[1])
        if dt_ms <= 0 or step_cm > self.max_step_cm:
            state.last_point = point
            state.last_timestamp_ms = timestamp_ms
            return

        dt_s = dt_ms / 1000.0
        state.tracked_ms += dt_ms
        step_m = step_cm / 100.0 if step_cm >= self.min_step_cm else 0.0
        speed_mps = step_m / max(dt_s, 1e-6)
        state.speed_sum_mps += speed_mps
        state.speed_samples += 1
        state.max_speed_mps = max(state.max_speed_mps, speed_mps)
        state.total_distance_m += step_m
        self._update_motion_counts(state, speed_mps, step_m)
        self._accumulate_direction(player_key, state, state.last_point, point, step_m)
        state.last_point = point
        state.last_timestamp_ms = timestamp_ms
        state.last_speed_mps = speed_mps

    def _update_ball_speed(
        self,
        timestamp_ms: int,
        ball_court_xy: Sequence[float] | None,
    ) -> None:
        point = self._clean_point(ball_court_xy)
        if point is None:
            self._ball_speed.last_point = None
            self._ball_speed.last_timestamp_ms = None
            self._ball_speed.current_speed_mps = 0.0
            self._ball_speed.current_valid = False
            self._ball_speed.recent_speed_mps.clear()
            return

        state = self._ball_speed
        if state.last_point is None or state.last_timestamp_ms is None:
            state.last_point = point
            state.last_timestamp_ms = timestamp_ms
            state.current_speed_mps = 0.0
            state.current_valid = False
            return

        dt_ms = max(0, timestamp_ms - state.last_timestamp_ms)
        step_cm = hypot(point[0] - state.last_point[0], point[1] - state.last_point[1])
        if dt_ms <= 0:
            state.current_valid = False
            return

        dt_s = dt_ms / 1000.0
        raw_speed_mps = 0.0 if step_cm < self.ball_min_step_cm else (step_cm / 100.0) / max(dt_s, 1e-6)
        if self._is_ball_speed_outlier(raw_speed_mps):
            state.current_valid = state.speed_samples > 0
            return

        state.last_point = point
        state.last_timestamp_ms = timestamp_ms
        self._append_ball_speed_sample(raw_speed_mps)
        target_speed_mps = self._median_ball_speed()

        if state.speed_samples <= 0 or not state.current_valid:
            stable_speed_mps = target_speed_mps
        else:
            alpha = self.ball_speed_smoothing
            stable_speed_mps = (alpha * target_speed_mps) + ((1.0 - alpha) * state.current_speed_mps)
            max_delta = self.ball_speed_max_accel_mps2 * dt_s
            if max_delta > 0.0:
                delta = max(-max_delta, min(max_delta, stable_speed_mps - state.current_speed_mps))
                stable_speed_mps = state.current_speed_mps + delta

        state.current_speed_mps = stable_speed_mps
        state.current_valid = True
        state.speed_sum_mps += stable_speed_mps
        state.speed_samples += 1
        state.max_speed_mps = max(state.max_speed_mps, stable_speed_mps)

    def _is_ball_speed_outlier(self, speed_mps: float) -> bool:
        if self.ball_max_speed_mps > 0.0 and speed_mps > self.ball_max_speed_mps:
            return True
        recent = self._ball_speed.recent_speed_mps
        if not recent:
            return False
        baseline = self._median_ball_speed()
        threshold = max(
            self.ball_speed_outlier_delta_mps,
            abs(baseline) * self.ball_speed_outlier_ratio,
        )
        return abs(speed_mps - baseline) > threshold

    def _append_ball_speed_sample(self, speed_mps: float) -> None:
        recent = self._ball_speed.recent_speed_mps
        recent.append(speed_mps)
        if len(recent) > self.ball_speed_window_size:
            del recent[0 : len(recent) - self.ball_speed_window_size]

    def _median_ball_speed(self) -> float:
        values = sorted(self._ball_speed.recent_speed_mps)
        if not values:
            return 0.0
        middle = len(values) // 2
        if len(values) % 2:
            return values[middle]
        return (values[middle - 1] + values[middle]) * 0.5

    def _ball_speed_summary(self) -> dict[str, Any]:
        state = self._ball_speed
        avg_speed_mps = state.speed_sum_mps / max(1, state.speed_samples)
        return {
            "current_mps": state.current_speed_mps,
            "current_kmh": state.current_speed_mps * 3.6,
            "avg_mps": avg_speed_mps,
            "avg_kmh": avg_speed_mps * 3.6,
            "max_mps": state.max_speed_mps,
            "max_kmh": state.max_speed_mps * 3.6,
            "samples": state.speed_samples,
            "current_valid": state.current_valid,
            "source": "court_plane",
            "filter": "median_ema_slew",
        }

    def _update_motion_counts(self, state: _PlayerMotionState, speed_mps: float, step_m: float) -> None:
        was_moving = state.moving
        now_moving = speed_mps >= self.start_speed_mps if not was_moving else speed_mps > self.stop_speed_mps
        if not was_moving and now_moving:
            state.start_count += 1
        if was_moving and not now_moving:
            state.stop_count += 1
        state.moving = now_moving

        was_high = state.high_intensity
        now_high = speed_mps >= self.high_intensity_speed_mps
        if not was_high and now_high:
            state.high_intensity_count += 1
        state.high_intensity = now_high

        if speed_mps > self.stop_speed_mps:
            state.current_continuous_m += step_m
            state.max_continuous_m = max(state.max_continuous_m, state.current_continuous_m)
        else:
            state.current_continuous_m = 0.0

    def _accumulate_direction(
        self,
        player_key: str,
        state: _PlayerMotionState,
        previous: tuple[float, float],
        current: tuple[float, float],
        step_m: float,
    ) -> None:
        if step_m <= 0.0:
            return
        dx = current[0] - previous[0]
        dy = current[1] - previous[1]
        abs_dx = abs(dx)
        abs_dy = abs(dy)
        if abs_dy >= abs_dx:
            toward_net = dy > 0 if player_key == "top" else dy < 0
            if toward_net:
                state.forward_m += step_m
            else:
                state.backward_m += step_m
        else:
            if dx < 0:
                state.left_m += step_m
            else:
                state.right_m += step_m

    def _sample_stance_depth(
        self,
        player_key: str,
        state: _PlayerMotionState,
        point: tuple[float, float],
    ) -> None:
        half = self.court_length_cm * 0.5
        y = max(0.0, min(self.court_length_cm, point[1]))
        depth = y if player_key == "top" else self.court_length_cm - y
        state.stance_depth_sum_cm += max(0.0, min(half, depth))
        state.stance_depth_samples += 1

    def _ensure_hit_record(
        self,
        key: tuple[str, int],
        *,
        frame_id: int,
        timestamp_ms: int,
    ) -> dict[str, Any]:
        existing = self._hit_index_by_key.get(key)
        if existing is not None:
            return self._hits[existing]
        hit = {
            "frame_id": frame_id,
            "timestamp_ms": timestamp_ms,
            "source": "",
            "player": "",
            "player_label": "",
            "stroke": "",
            "raw_stroke": "",
            "zone": "",
            "confidence": 0.0,
            "event_confidence": 0.0,
            "court_xy": None,
            "passive": False,
        }
        self._hit_index_by_key[key] = len(self._hits)
        self._hits.append(hit)
        return hit

    def _register_player_hit(self, hit: dict[str, Any]) -> None:
        if hit.get("_registered"):
            return
        player_key = str(hit.get("player", ""))
        if player_key not in self._players:
            return
        state = self._players[player_key]
        state.hit_count += 1
        zone = str(hit.get("zone") or "")
        if zone:
            state.zone_hits[zone] += 1
        passive = state.last_speed_mps <= self.passive_hit_speed_mps
        hit["passive"] = passive
        state.passive_hit_count += int(passive)
        hit["_registered"] = True

    def _public_hit(self, hit: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in hit.items()
            if not str(key).startswith("_")
        }

    def _rally_start_ms(self) -> int | None:
        return self._rally_start_timestamp_ms

    def _rally_end_ms(self) -> int | None:
        if self._rally_start_timestamp_ms is None:
            return None
        if self._rally_end_timestamp_ms is not None:
            return self._rally_end_timestamp_ms
        return self._last_timestamp_ms

    def _rally_duration_s(self) -> float:
        start_ms = self._rally_start_ms()
        end_ms = self._rally_end_ms()
        if start_ms is None or end_ms is None:
            return 0.0
        return max(0.0, (end_ms - start_ms) / 1000.0)

    def _rally_state(self) -> str:
        if self._rally_start_timestamp_ms is None:
            return "未开始"
        if self._rally_end_timestamp_ms is not None:
            return "回合结束"
        return "回合中"

    def _should_track_rally_start(self, timestamp_ms: int) -> bool:
        if self._rally_start_timestamp_ms is None:
            return True
        return (
            not self.freeze_after_rally_end
            and self._rally_end_timestamp_ms is not None
            and timestamp_ms > self._rally_end_timestamp_ms
        )

    def _update_rally_start_candidate(
        self,
        timestamp_ms: int,
        ball_visible: bool,
        ball_xy: Sequence[float] | None,
        ball_score: float,
    ) -> bool:
        point = self._clean_point(ball_xy)
        if not ball_visible or point is None:
            self._reset_rally_start_candidate()
            return False
        if self._start_candidate_point is None:
            self._start_candidate_timestamp_ms = timestamp_ms
            self._start_candidate_point = point
            self._start_candidate_visible_frames = 1
            self._start_candidate_score_sum = max(0.0, float(ball_score))
            self._start_candidate_max_motion_px = 0.0
            return False

        self._start_candidate_visible_frames += 1
        self._start_candidate_score_sum += max(0.0, float(ball_score))
        dx = point[0] - self._start_candidate_point[0]
        dy = point[1] - self._start_candidate_point[1]
        self._start_candidate_max_motion_px = max(self._start_candidate_max_motion_px, hypot(dx, dy))
        avg_score = self._start_candidate_score_sum / max(1, self._start_candidate_visible_frames)
        if (
            self._start_candidate_visible_frames >= self.start_visible_frames
            and self._start_candidate_max_motion_px >= self.start_min_motion_px
            and avg_score >= self.start_min_avg_ball_score
        ):
            self._rally_start_timestamp_ms = self._start_candidate_timestamp_ms
            self._reset_rally_start_candidate()
            return True
        return False

    def _reset_rally_start_candidate(self) -> None:
        self._start_candidate_timestamp_ms = None
        self._start_candidate_point = None
        self._start_candidate_visible_frames = 0
        self._start_candidate_score_sum = 0.0
        self._start_candidate_max_motion_px = 0.0

    def _player_summary(self, player_key: str, state: _PlayerMotionState) -> dict[str, Any]:
        avg_speed = (
            state.total_distance_m / max(1e-6, state.tracked_ms / 1000.0)
            if state.tracked_ms > 0
            else 0.0
        )
        total_fb = state.forward_m + state.backward_m
        total_lr = state.left_m + state.right_m
        avg_depth = (
            state.stance_depth_sum_cm / max(1, state.stance_depth_samples)
            if state.stance_depth_samples
            else 0.0
        )
        return {
            "label": PLAYER_LABELS[player_key],
            "distance_m": state.total_distance_m,
            "avg_speed_mps": avg_speed,
            "max_speed_mps": state.max_speed_mps,
            "stop_count": state.stop_count,
            "start_count": state.start_count,
            "hit_count": state.hit_count,
            "zone_hits": {
                "front": int(state.zone_hits.get("front", 0)),
                "mid": int(state.zone_hits.get("mid", 0)),
                "back": int(state.zone_hits.get("back", 0)),
            },
            "passive_hit_count": state.passive_hit_count,
            "high_intensity_count": state.high_intensity_count,
            "max_continuous_m": state.max_continuous_m,
            "front_back_movement_ratio": state.forward_m / total_fb if total_fb > 0 else 0.0,
            "left_right_movement_ratio": state.left_m / total_lr if total_lr > 0 else 0.0,
            "avg_stance_depth_cm": avg_depth,
        }

    def _court_zone(self, player_key: str | None, point: tuple[float, float] | None) -> str:
        if point is None:
            return ""
        y = max(0.0, min(self.court_length_cm, point[1]))
        half = self.court_length_cm * 0.5
        third = half / 3.0
        if player_key == "top":
            depth_from_back = max(0.0, min(half, y))
        elif player_key == "bottom":
            depth_from_back = max(0.0, min(half, self.court_length_cm - y))
        else:
            depth_from_back = max(0.0, min(self.court_length_cm, y))
            third = self.court_length_cm / 3.0
        if depth_from_back >= third * 2.0:
            return "front"
        if depth_from_back >= third:
            return "mid"
        return "back"

    def _estimate_hit_point_from_player(self, player_key: str | None) -> tuple[float, float] | None:
        if player_key not in self._players:
            return None
        return self._players[player_key].last_point

    def _parse_stroke_label(self, label: str) -> tuple[str | None, str]:
        text = label.strip() or "未知动作"
        lowered = text.lower()
        if lowered.startswith("top_"):
            return "top", text[4:] or text
        if lowered.startswith("bottom_"):
            return "bottom", text[7:] or text
        return None, text

    def _sample_hit_confidence(self, confidence: float) -> None:
        if confidence <= 0.0:
            return
        self._hit_confidence_sum += confidence
        self._hit_confidence_samples += 1

    def _hit_key(self, frame_id: int, timestamp_ms: int) -> tuple[str, int]:
        if frame_id >= 0:
            return "frame", int(frame_id)
        return "time", int(timestamp_ms)

    def _clean_player_points(
        self,
        player_points: Mapping[int, Sequence[float]] | None,
    ) -> dict[int, tuple[float, float]]:
        clean: dict[int, tuple[float, float]] = {}
        if not isinstance(player_points, Mapping):
            return clean
        for index, point in player_points.items():
            clean_point = self._clean_point(point)
            if clean_point is None:
                continue
            clean[int(index)] = clean_point
        return clean

    def _clean_point(self, point: object) -> tuple[float, float] | None:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            return None
        try:
            x = float(point[0])
            y = float(point[1])
        except (TypeError, ValueError):
            return None
        if not self._is_finite(x) or not self._is_finite(y):
            return None
        return x, y

    @staticmethod
    def _safe_float(value: object) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0
        if not RallyStatsAccumulator._is_finite(number):
            return 0.0
        return max(0.0, number)

    @staticmethod
    def _rate(count: int, total: int) -> float:
        return float(count) / float(total) if total > 0 else 0.0

    @staticmethod
    def _is_finite(value: float) -> bool:
        return value == value and value not in (float("inf"), float("-inf"))
