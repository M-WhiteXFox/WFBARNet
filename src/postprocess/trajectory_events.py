from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Sequence

import numpy as np

from src.utils.structures import FrameResult


@dataclass(slots=True)
class TrajectoryEventDetectorConfig:
    fps: float = 30.0
    speed_drop_ratio: float = 0.3
    min_speed_before_landing: float = 8.0
    max_speed_after_landing: float = 5.0
    high_speed_threshold: float = 8.0
    low_speed_threshold: float = 2.5
    low_speed_ratio: float = 0.7
    pre_window: int = 2
    post_window: int = 4
    hold_window: int = 3
    min_speed_at_hit: float = 8.0
    max_speed_at_hit: float = 220.0
    vy_reversal_threshold: float = 1.0
    min_hit_reversal_magnitude: float = 8.0
    min_hit_track_score: float = 0.48
    min_hit_neighbor_score: float = 0.35
    max_hit_neighbor_gap: int = 3
    hit_top_ignore_ratio: float = 0.08
    hit_top_ignore_px: float = 36.0
    acc_threshold: float = 3.0
    min_height_diff: float = 5.0
    min_peak_speed: float = 10.0
    min_speed_diff: float = 2.0
    min_visible_before: int = 3
    merge_window: int = 3
    future_check_window: int = 5
    edge_margin: float = 20.0
    history_frames: int = 180
    confirmation_frames: int = 1
    trajectory_end_missing_frames: int = 3
    visibility_drop_missing_frames: int = 3
    event_cooldown_seconds: float = 0.18
    max_event_lag_frames: int = 12


@dataclass(slots=True)
class _TrajectoryPoint:
    frame_id: int
    timestamp_ms: int
    x: float
    y: float
    visible: int
    score: float


class KinematicsCalculator:
    def __init__(self, fps: float = 30.0) -> None:
        self.fps = float(fps) if fps > 0 else 30.0

    def compute(
        self,
        x: Sequence[float],
        y: Sequence[float],
        visibility: Sequence[int],
    ) -> dict[str, np.ndarray]:
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        visibility_arr = np.asarray(visibility, dtype=np.float64)
        v_x = np.gradient(x_arr)
        v_y = np.gradient(y_arr)
        a_x = np.gradient(v_x)
        a_y = np.gradient(v_y)
        speed = np.sqrt(v_x**2 + v_y**2)
        direction = np.arctan2(v_y, v_x)
        speed_cubed = np.power(speed, 3)
        curvature = np.divide(
            np.abs(v_x * a_y - v_y * a_x),
            speed_cubed,
            out=np.zeros_like(speed),
            where=speed_cubed > 1e-6,
        )
        return {
            "v_x": v_x,
            "v_y": v_y,
            "a_x": a_x,
            "a_y": a_y,
            "speed": speed,
            "direction": direction,
            "curvature": curvature,
            "visibility": visibility_arr,
        }


class TrajectoryEventCandidateGenerator:
    def __init__(self, config: TrajectoryEventDetectorConfig | None = None) -> None:
        self.config = config or TrajectoryEventDetectorConfig()
        self.kinematics_calculator = KinematicsCalculator(self.config.fps)

    def generate(
        self,
        x: Sequence[float],
        y: Sequence[float],
        visibility: Sequence[int],
        *,
        img_height: int = 288,
        img_width: int = 512,
        include_trajectory_end: bool = True,
    ) -> list[dict[str, object]]:
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        vis = np.asarray(visibility, dtype=np.float64)
        n = len(x_arr)
        if n < self.config.min_visible_before + 2:
            return []

        kinematics = self.kinematics_calculator.compute(x_arr, y_arr, vis)
        a_x = np.nan_to_num(kinematics["a_x"], nan=0.0, posinf=0.0, neginf=0.0)
        a_y = np.nan_to_num(kinematics["a_y"], nan=0.0, posinf=0.0, neginf=0.0)
        speed = np.nan_to_num(kinematics["speed"], nan=0.0, posinf=0.0, neginf=0.0)
        acceleration = np.sqrt(a_x**2 + a_y**2)
        candidates: list[dict[str, object]] = []

        for t in range(self.config.min_visible_before, n - 1):
            if vis[t] == 0:
                continue
            recent_start = max(
                0,
                t - self.config.min_visible_before - max(0, int(self.config.max_hit_neighbor_gap)) + 1,
            )
            visible_count = int(np.sum(vis[recent_start : t + 1]))
            if visible_count < self.config.min_visible_before:
                continue

            primary: dict[str, object] | None = None
            primary_features: dict[str, object] = {}
            for event_type, rule, confidence, features in (
                ("hit", "vy_reversal", 0.85, self._check_vy_reversal(t, n, x_arr, y_arr, vis)),
                ("hit", "vx_reversal", 0.80, self._check_vx_reversal(t, n, x_arr, y_arr, vis)),
                ("landing", "speed_step", 0.90, self._check_speed_step(t, n, vis, speed)),
                ("landing", "low_speed_start", 0.85, self._check_low_speed_start(t, n, vis, speed)),
                ("landing", "speed_drop", 0.80, self._check_speed_drop(t, n, vis, speed)),
            ):
                if features is not None:
                    primary = {"event_type": event_type, "rule": rule, "confidence": confidence}
                    primary_features = features
                    break

            if primary is None:
                visibility_drop = self._check_visibility_drop(
                    t,
                    n,
                    x_arr,
                    y_arr,
                    vis,
                    img_height,
                    img_width,
                    visible_count,
                )
                if visibility_drop is not None:
                    primary = {
                        "event_type": visibility_drop["event_type"],
                        "rule": visibility_drop["rule"],
                        "confidence": visibility_drop["confidence"],
                    }
                    primary_features = visibility_drop["features"]  # type: ignore[assignment]

            auxiliary_rules: list[str] = []
            auxiliary_features: dict[str, object] = {}
            confidence_boost = 0.0
            acc_peak = self._check_acceleration_peak(t, n, vis, acceleration, speed)
            if acc_peak is not None:
                auxiliary_rules.append("acceleration_peak")
                auxiliary_features["acceleration_peak"] = acc_peak
                if primary is not None and primary["event_type"] == "hit":
                    confidence_boost += 0.05
            y_max = self._check_y_local_max(t, n, y_arr, vis)
            if y_max is not None:
                auxiliary_rules.append("y_local_max")
                auxiliary_features["y_local_max"] = y_max
                if primary is not None:
                    confidence_boost += 0.08 if primary["event_type"] == "landing" else 0.05
            speed_max = self._check_speed_local_max(t, n, speed, vis)
            if speed_max is not None:
                auxiliary_rules.append("speed_local_max")
                auxiliary_features["speed_local_max"] = speed_max
                if primary is not None and primary["event_type"] == "hit":
                    confidence_boost += 0.05

            if primary is None and not auxiliary_rules:
                continue
            if primary is None:
                continue
            candidate = {
                "frame": t,
                "x": float(x_arr[t]),
                "y": float(y_arr[t]),
                "event_type": primary["event_type"],
                "rule": primary["rule"],
                "confidence": min(0.95, float(primary["confidence"]) + confidence_boost),
                "all_rules": [str(primary["rule"])] + auxiliary_rules,
                "auxiliary_rules": auxiliary_rules,
                "features": {**primary_features, **auxiliary_features},
            }
            candidates.append(candidate)

        if include_trajectory_end:
            self._check_trajectory_end(x_arr, y_arr, vis, speed, candidates)
        merged = self._merge_nearby_candidates(candidates)
        merged.sort(key=lambda item: int(item["frame"]))
        return merged

    def _check_vy_reversal(
        self,
        t: int,
        n: int,
        x: np.ndarray,
        y: np.ndarray,
        visibility: np.ndarray,
    ) -> dict[str, float] | None:
        if t <= 0 or t + 1 >= n:
            return None
        prev_idx, next_idx = self._hit_neighbor_indices(t, n, visibility)
        if prev_idx is None or next_idx is None:
            return None
        next_gap = max(1, next_idx - t)
        vy_after = float((y[next_idx] - y[t]) / next_gap)
        if abs(vy_after) <= self.config.vy_reversal_threshold:
            return None
        before = self._before_velocity_with_sign(
            t,
            y,
            visibility,
            positive=vy_after < 0.0,
        )
        if before is None:
            return None
        vy_before, prev_gap = before
        vx_after = float((x[next_idx] - x[t]) / next_gap)
        speed_after = float(np.hypot(vx_after, vy_after))
        reversal_magnitude = abs(vy_before - vy_after)
        if (
            speed_after < self.config.min_speed_at_hit
            or speed_after > self.config.max_speed_at_hit
            or reversal_magnitude < self.config.min_hit_reversal_magnitude
        ):
            return None
        return {
            "vy_before": vy_before,
            "vy_after": vy_after,
            "speed_after": speed_after,
            "reversal_magnitude": reversal_magnitude,
            "prev_gap": float(prev_gap),
            "next_gap": float(next_idx - t),
        }

    def _check_vx_reversal(
        self,
        t: int,
        n: int,
        x: np.ndarray,
        y: np.ndarray,
        visibility: np.ndarray,
    ) -> dict[str, float] | None:
        if t <= 0 or t + 1 >= n:
            return None
        prev_idx, next_idx = self._hit_neighbor_indices(t, n, visibility)
        if prev_idx is None or next_idx is None:
            return None
        next_gap = max(1, next_idx - t)
        vx_after = float((x[next_idx] - x[t]) / next_gap)
        if abs(vx_after) <= self.config.vy_reversal_threshold:
            return None
        before = self._before_velocity_with_sign(
            t,
            x,
            visibility,
            positive=vx_after < 0.0,
        )
        if before is None:
            return None
        vx_before, prev_gap = before
        vy_after = float((y[next_idx] - y[t]) / next_gap)
        speed_after = float(np.hypot(vx_after, vy_after))
        reversal_magnitude = abs(vx_before - vx_after)
        if (
            speed_after < self.config.min_speed_at_hit
            or speed_after > self.config.max_speed_at_hit
            or reversal_magnitude < self.config.min_hit_reversal_magnitude
        ):
            return None
        return {
            "vx_before": vx_before,
            "vx_after": vx_after,
            "speed_after": speed_after,
            "reversal_magnitude": reversal_magnitude,
            "prev_gap": float(prev_gap),
            "next_gap": float(next_idx - t),
        }

    def _hit_neighbor_indices(
        self,
        t: int,
        n: int,
        visibility: np.ndarray,
    ) -> tuple[int | None, int | None]:
        max_gap = max(1, int(self.config.max_hit_neighbor_gap))
        return (
            self._nearest_visible_index(t, n, visibility, -1, max_gap),
            self._nearest_visible_index(t, n, visibility, 1, max_gap),
        )

    def _before_velocity_with_sign(
        self,
        t: int,
        values: np.ndarray,
        visibility: np.ndarray,
        *,
        positive: bool,
    ) -> tuple[float, int] | None:
        threshold = float(self.config.vy_reversal_threshold)
        best: tuple[float, int] | None = None
        max_gap = max(1, int(self.config.max_hit_neighbor_gap))
        for gap in range(1, max_gap + 1):
            idx = t - gap
            if idx < 0:
                break
            if visibility[idx] != 1:
                continue
            velocity = float((values[t] - values[idx]) / gap)
            if positive:
                if velocity <= threshold:
                    continue
                if best is None or velocity > best[0]:
                    best = (velocity, gap)
            else:
                if velocity >= -threshold:
                    continue
                if best is None or velocity < best[0]:
                    best = (velocity, gap)
        return best

    def _nearest_visible_index(
        self,
        t: int,
        n: int,
        visibility: np.ndarray,
        direction: int,
        max_gap: int,
    ) -> int | None:
        for offset in range(1, max_gap + 1):
            idx = t + direction * offset
            if idx < 0 or idx >= n:
                break
            if visibility[idx] == 1:
                return idx
        return None

    def _check_acceleration_peak(
        self,
        t: int,
        n: int,
        visibility: np.ndarray,
        acceleration: np.ndarray,
        speed: np.ndarray,
    ) -> dict[str, float] | None:
        if t <= 1 or t + 1 >= n or visibility[t - 1] != 1 or visibility[t] != 1 or visibility[t + 1] != 1:
            return None
        acc_before = float(acceleration[t - 1])
        acc_current = float(acceleration[t])
        acc_after = float(acceleration[t + 1])
        if not (acc_current > acc_before and acc_current > acc_after and acc_current > self.config.acc_threshold):
            return None
        return {
            "acc_before": acc_before,
            "acc_current": acc_current,
            "acc_after": acc_after,
            "speed": float(speed[t]),
            "acc_ratio": float(acc_current / (max(acc_before, acc_after) + 1e-6)),
        }

    def _check_speed_step(
        self,
        t: int,
        n: int,
        visibility: np.ndarray,
        speed: np.ndarray,
    ) -> dict[str, float] | None:
        if t <= 0 or t >= n or visibility[t] != 1 or visibility[t - 1] != 1:
            return None
        v_prev = float(speed[t - 1])
        v_curr = float(speed[t])
        if v_curr > self.config.low_speed_threshold:
            return None
        ratio = v_curr / (v_prev + 1e-6)
        if not (v_prev >= self.config.high_speed_threshold or ratio < self.config.speed_drop_ratio):
            return None
        post_end = min(n, t + self.config.post_window + 1)
        post_speeds = speed[t:post_end]
        post_vis = visibility[t:post_end]
        if len(post_speeds) == 0:
            return None
        visible_count = max(1, int(np.sum(post_vis)))
        low_ratio = float(np.sum((post_speeds <= self.config.low_speed_threshold) & (post_vis == 1))) / visible_count
        if low_ratio < self.config.low_speed_ratio:
            return None
        return {"v_prev": v_prev, "v_curr": v_curr, "ratio": float(ratio), "low_ratio": low_ratio}

    def _check_low_speed_start(
        self,
        t: int,
        n: int,
        visibility: np.ndarray,
        speed: np.ndarray,
    ) -> dict[str, float] | None:
        if t <= 0 or t >= n or visibility[t] != 1:
            return None
        v_curr = float(speed[t])
        v_prev = float(speed[t - 1]) if visibility[t - 1] == 1 else v_curr + 1e-6
        if v_curr > self.config.low_speed_threshold or v_prev <= self.config.low_speed_threshold:
            return None
        hold_end = min(n, t + self.config.hold_window)
        hold_speeds = speed[t:hold_end]
        hold_vis = visibility[t:hold_end]
        visible_count = max(1, int(np.sum(hold_vis)))
        low_ratio = float(np.sum((hold_speeds <= self.config.low_speed_threshold) & (hold_vis == 1))) / visible_count
        if low_ratio < self.config.low_speed_ratio:
            return None
        return {"v_prev": v_prev, "v_curr": v_curr, "low_ratio": low_ratio}

    def _check_speed_drop(
        self,
        t: int,
        n: int,
        visibility: np.ndarray,
        speed: np.ndarray,
    ) -> dict[str, float] | None:
        if t <= 0 or t + 2 >= n:
            return None
        speed_before = float(speed[t - 1])
        speed_current = float(speed[t])
        speed_after = float(speed[t + 1]) if visibility[t + 1] == 1 else 0.0
        if speed_before < self.config.min_speed_before_landing:
            return None
        speed_ratio = speed_current / (speed_before + 1e-6)
        if speed_ratio >= self.config.speed_drop_ratio or speed_after > self.config.max_speed_after_landing:
            return None
        check_end = min(n, t + self.config.future_check_window + 1)
        if check_end > t + 1:
            future_speeds = speed[t + 1 : check_end] * visibility[t + 1 : check_end]
            if len(future_speeds) > 0 and float(np.max(future_speeds)) > 10.0:
                return None
        return {
            "speed_ratio": float(speed_ratio),
            "speed_before": speed_before,
            "speed_current": speed_current,
            "speed_after": speed_after,
        }

    def _check_visibility_drop(
        self,
        t: int,
        n: int,
        x: np.ndarray,
        y: np.ndarray,
        visibility: np.ndarray,
        img_height: int,
        img_width: int,
        visible_count: int,
    ) -> dict[str, object] | None:
        if t + 1 >= n or visibility[t] != 1 or visibility[t + 1] != 0:
            return None
        missing_after = self._consecutive_missing_after(t, n, visibility)
        if missing_after < max(1, int(self.config.visibility_drop_missing_frames)):
            return None
        motion = self._previous_visible_motion(t, n, x, y, visibility)
        if motion is None:
            return None
        recent_speed = motion["speed_before"]
        if recent_speed < self.config.min_speed_before_landing:
            return None
        previous_v_y = motion["v_y"]
        previous_v_x = motion["v_x"]
        is_edge = (
            x[t] < self.config.edge_margin
            or x[t] > img_width - self.config.edge_margin
            or y[t] < self.config.edge_margin
            or y[t] > img_height - self.config.edge_margin
        )
        edge_distance = float(min(x[t], img_width - x[t], y[t], img_height - y[t]))
        moving_out = self._is_moving_out_of_frame(
            x[t],
            y[t],
            previous_v_x,
            previous_v_y,
            img_width,
            img_height,
        )
        features = {
            "speed_before": recent_speed,
            "v_y": previous_v_y,
            "visible_before": int(visible_count),
            "edge_distance": edge_distance,
            "missing_after": int(missing_after),
        }
        if is_edge and (moving_out or edge_distance <= self.config.edge_margin * 0.5):
            return {
                "event_type": "out_of_frame",
                "rule": "visibility_drop_edge",
                "confidence": 0.45,
                "features": features,
            }
        top_exit_band = max(self.config.edge_margin * 2.0, img_height * 0.25)
        if previous_v_y < -1.0 and y[t] <= top_exit_band:
            return {
                "event_type": "out_of_frame",
                "rule": "visibility_drop_upward",
                "confidence": 0.50,
                "features": features,
            }
        if y[t] < img_height * 0.15 and previous_v_y <= 0.0:
            return {
                "event_type": "out_of_frame",
                "rule": "visibility_drop_high_altitude",
                "confidence": 0.50,
                "features": features,
            }
        return {"event_type": "landing", "rule": "visibility_drop", "confidence": 0.55, "features": features}

    def _consecutive_missing_after(self, t: int, n: int, visibility: np.ndarray) -> int:
        count = 0
        for idx in range(t + 1, n):
            if visibility[idx] == 1:
                break
            count += 1
        return count

    def _previous_visible_motion(
        self,
        t: int,
        n: int,
        x: np.ndarray,
        y: np.ndarray,
        visibility: np.ndarray,
    ) -> dict[str, float] | None:
        prev_idx = self._nearest_visible_index(
            t,
            n,
            visibility,
            -1,
            max(self.config.min_visible_before, int(self.config.max_hit_neighbor_gap)),
        )
        if prev_idx is None:
            return None
        gap = max(1, t - prev_idx)
        v_x = float((x[t] - x[prev_idx]) / gap)
        v_y = float((y[t] - y[prev_idx]) / gap)
        return {"v_x": v_x, "v_y": v_y, "speed_before": float(np.hypot(v_x, v_y))}

    def _is_moving_out_of_frame(
        self,
        x: float,
        y: float,
        v_x: float,
        v_y: float,
        img_width: int,
        img_height: int,
    ) -> bool:
        margin = self.config.edge_margin
        return (
            (x < margin and v_x < 0.0)
            or (x > img_width - margin and v_x > 0.0)
            or (y < margin and v_y < 0.0)
            or (y > img_height - margin and v_y > 0.0)
        )

    def _check_y_local_max(
        self,
        t: int,
        n: int,
        y: np.ndarray,
        visibility: np.ndarray,
    ) -> dict[str, float] | None:
        if t <= 0 or t + 1 >= n or visibility[t - 1] != 1 or visibility[t] != 1 or visibility[t + 1] != 1:
            return None
        if not (y[t] > y[t - 1] and y[t] > y[t + 1]):
            return None
        height_diff = float(min(y[t] - y[t - 1], y[t] - y[t + 1]))
        if height_diff <= self.config.min_height_diff:
            return None
        return {
            "y_before": float(y[t - 1]),
            "y_current": float(y[t]),
            "y_after": float(y[t + 1]),
            "height_diff": height_diff,
        }

    def _check_speed_local_max(
        self,
        t: int,
        n: int,
        speed: np.ndarray,
        visibility: np.ndarray,
    ) -> dict[str, float] | None:
        if t <= 0 or t + 1 >= n or visibility[t - 1] != 1 or visibility[t] != 1 or visibility[t + 1] != 1:
            return None
        if not (speed[t] > speed[t - 1] and speed[t] > speed[t + 1]):
            return None
        speed_diff = float(min(speed[t] - speed[t - 1], speed[t] - speed[t + 1]))
        if speed[t] <= self.config.min_peak_speed or speed_diff <= self.config.min_speed_diff:
            return None
        return {
            "speed_before": float(speed[t - 1]),
            "speed_current": float(speed[t]),
            "speed_after": float(speed[t + 1]),
            "speed_diff": speed_diff,
        }

    def _check_trajectory_end(
        self,
        x: np.ndarray,
        y: np.ndarray,
        visibility: np.ndarray,
        speed: np.ndarray,
        candidates: list[dict[str, object]],
    ) -> None:
        visible_indices = np.where(visibility == 1)[0]
        if len(visible_indices) == 0:
            return
        last_visible_idx = int(visible_indices[-1])
        if last_visible_idx <= self.config.min_visible_before:
            return
        if any(
            abs(int(candidate["frame"]) - last_visible_idx) <= 3 and float(candidate.get("confidence", 0.0)) > 0.65
            for candidate in candidates
        ):
            return
        tail_start = self._find_tail_low_speed_start(speed, visibility, last_visible_idx)
        if tail_start is not None:
            candidates.append(
                {
                    "frame": tail_start,
                    "x": float(x[tail_start]),
                    "y": float(y[tail_start]),
                    "event_type": "landing",
                    "rule": "trajectory_end",
                    "confidence": 0.88,
                    "all_rules": ["trajectory_end"],
                    "auxiliary_rules": [],
                    "features": {
                        "speed": float(speed[tail_start]),
                        "recent_avg_speed": float(np.mean(speed[max(0, tail_start - 3) : tail_start + 1])),
                        "landing_type": "tail_low_speed_start",
                    },
                }
            )
            return
        candidates.append(
            {
                "frame": last_visible_idx,
                "x": float(x[last_visible_idx]),
                "y": float(y[last_visible_idx]),
                "event_type": "landing",
                "rule": "trajectory_end",
                "confidence": 0.65,
                "all_rules": ["trajectory_end"],
                "auxiliary_rules": [],
                "features": {
                    "speed": float(speed[last_visible_idx]),
                    "recent_avg_speed": float(np.mean(speed[max(0, last_visible_idx - 3) : last_visible_idx + 1])),
                    "landing_type": "tail_fallback",
                },
            }
        )

    def _find_tail_low_speed_start(
        self,
        speed: np.ndarray,
        visibility: np.ndarray,
        last_visible_idx: int,
    ) -> int | None:
        idx = last_visible_idx
        while idx >= 0:
            if visibility[idx] != 1:
                idx -= 1
                continue
            if speed[idx] <= self.config.low_speed_threshold:
                idx -= 1
                continue
            break
        tail_start = idx + 1
        if tail_start > last_visible_idx:
            return None
        low_len = int(
            np.sum(
                (visibility[tail_start : last_visible_idx + 1] == 1)
                & (speed[tail_start : last_visible_idx + 1] <= self.config.low_speed_threshold)
            )
        )
        if low_len < self.config.hold_window:
            return None
        return tail_start

    def _merge_nearby_candidates(self, candidates: list[dict[str, object]]) -> list[dict[str, object]]:
        if len(candidates) <= 1:
            return candidates
        rule_priority = {
            "vy_reversal": 7,
            "vx_reversal": 6,
            "acceleration_peak": 5,
            "speed_step": 5,
            "low_speed_start": 5,
            "speed_drop": 4,
            "visibility_drop": 3,
            "visibility_drop_edge": 3,
            "visibility_drop_upward": 3,
            "visibility_drop_high_altitude": 3,
            "trajectory_end": 3,
            "y_local_max": 2,
            "speed_local_max": 2,
        }
        candidates.sort(key=lambda item: int(item["frame"]))
        merged: list[dict[str, object]] = []
        i = 0
        while i < len(candidates):
            group = [candidates[i]]
            base_frame = int(candidates[i]["frame"])
            j = i + 1
            while j < len(candidates) and int(candidates[j]["frame"]) - base_frame <= self.config.merge_window:
                group.append(candidates[j])
                j += 1
            merged.append(
                max(
                    group,
                    key=lambda item: (
                        rule_priority.get(str(item["rule"]), 0),
                        float(item["confidence"]),
                    ),
                )
            )
            i = j
        return merged


class RealtimeTrajectoryEventDetector:
    def __init__(self, config: TrajectoryEventDetectorConfig | None = None, *, fps: float | None = None) -> None:
        self.config = config or TrajectoryEventDetectorConfig()
        if fps is not None and fps > 0:
            self.config.fps = float(fps)
        self.generator = TrajectoryEventCandidateGenerator(self.config)
        self._history: deque[_TrajectoryPoint] = deque(maxlen=max(8, int(self.config.history_frames)))
        self._emitted: set[tuple[str, int]] = set()
        self._last_event_time_ms: dict[str, int] = {}
        self._last_event: dict[str, object] | None = None

    def reset(self) -> None:
        self._history.clear()
        self._emitted.clear()
        self._last_event_time_ms.clear()
        self._last_event = None

    def update(
        self,
        result: FrameResult,
        *,
        timestamp_ms: int | None = None,
        frame_shape: Sequence[int] | None = None,
    ) -> dict[str, object] | None:
        timestamp = int(timestamp_ms) if timestamp_ms is not None else self._timestamp_ms(result.frame_id)
        self._history.append(self._point_from_result(result, timestamp))
        self._last_event = None
        if len(self._history) < self.config.min_visible_before + self.config.confirmation_frames + 2:
            return None

        samples = list(self._history)
        x = [sample.x for sample in samples]
        y = [sample.y for sample in samples]
        visibility = [sample.visible for sample in samples]
        img_width, img_height = _frame_size(frame_shape)
        include_trajectory_end = self._tail_missing_frames(samples) >= self.config.trajectory_end_missing_frames
        candidates = self.generator.generate(
            x,
            y,
            visibility,
            img_height=img_height,
            img_width=img_width,
            include_trajectory_end=include_trajectory_end,
        )
        for candidate in candidates:
            local_index = int(candidate["frame"])
            if local_index >= len(samples):
                continue
            lag_frames = len(samples) - 1 - local_index
            if lag_frames < self.config.confirmation_frames:
                continue
            max_lag = int(self.config.max_event_lag_frames)
            if max_lag >= 0 and lag_frames > max_lag:
                continue
            event = self._event_from_candidate(candidate, samples[local_index])
            if str(event["event_type"]) == "hit" and not self._is_hit_event_valid(
                event,
                samples,
                local_index,
                img_height,
            ):
                continue
            event_key = (str(event["event_type"]), int(event["frame_id"]))
            if event_key in self._emitted:
                continue
            if self._in_cooldown(event):
                continue
            self._emitted.add(event_key)
            self._last_event_time_ms[str(event["event_type"])] = int(event["timestamp_ms"])
            self._last_event = event
            return dict(event)
        return None

    def last_event(self) -> dict[str, object] | None:
        if self._last_event is None:
            return None
        return dict(self._last_event)

    def _timestamp_ms(self, frame_id: int) -> int:
        fps = self.config.fps if self.config.fps > 0 else 30.0
        return int(round(max(0.0, float(frame_id) / fps) * 1000.0))

    def _point_from_result(self, result: FrameResult, timestamp_ms: int) -> _TrajectoryPoint:
        visible = int(bool(result.track.visible))
        x, y = -1.0, -1.0
        if visible and len(result.track.ball_xy) >= 2:
            try:
                raw_x = float(result.track.ball_xy[0])
                raw_y = float(result.track.ball_xy[1])
            except (TypeError, ValueError):
                visible = 0
            else:
                if isfinite(raw_x) and isfinite(raw_y):
                    x, y = raw_x, raw_y
                else:
                    visible = 0
        return _TrajectoryPoint(
            frame_id=int(result.frame_id),
            timestamp_ms=timestamp_ms,
            x=x,
            y=y,
            visible=visible,
            score=float(result.track.score),
        )

    def _tail_missing_frames(self, samples: Sequence[_TrajectoryPoint]) -> int:
        count = 0
        for sample in reversed(samples):
            if sample.visible:
                break
            count += 1
        return count

    def _event_from_candidate(
        self,
        candidate: dict[str, object],
        sample: _TrajectoryPoint,
    ) -> dict[str, object]:
        return {
            "event_type": str(candidate.get("event_type", "unknown")),
            "frame_id": int(sample.frame_id),
            "timestamp_ms": int(sample.timestamp_ms),
            "ball_xy": [float(candidate.get("x", sample.x)), float(candidate.get("y", sample.y))],
            "rule": str(candidate.get("rule", "")),
            "confidence": float(candidate.get("confidence", 0.0)),
            "all_rules": [str(item) for item in _as_sequence(candidate.get("all_rules"))],
            "auxiliary_rules": [str(item) for item in _as_sequence(candidate.get("auxiliary_rules"))],
            "features": _json_safe(candidate.get("features", {})),
        }

    def _is_hit_event_valid(
        self,
        event: dict[str, object],
        samples: Sequence[_TrajectoryPoint],
        local_index: int,
        img_height: int,
    ) -> bool:
        rule = str(event.get("rule", ""))
        if rule not in {"vy_reversal", "vx_reversal"}:
            return False
        sample = samples[local_index]
        if sample.score < self.config.min_hit_track_score:
            return False
        for direction in (-1, 1):
            neighbor_index = self._nearest_visible_sample_index(samples, local_index, direction)
            if neighbor_index is None:
                return False
            neighbor = samples[neighbor_index]
            if neighbor.score < self.config.min_hit_neighbor_score:
                return False
        ball_xy = event.get("ball_xy", [-1.0, -1.0])
        if isinstance(ball_xy, (list, tuple)) and len(ball_xy) >= 2:
            try:
                y = float(ball_xy[1])
            except (TypeError, ValueError):
                return False
            top_band = max(
                float(self.config.hit_top_ignore_px),
                float(img_height) * max(0.0, float(self.config.hit_top_ignore_ratio)),
            )
            if y <= top_band:
                return False
        return True

    def _nearest_visible_sample_index(
        self,
        samples: Sequence[_TrajectoryPoint],
        local_index: int,
        direction: int,
    ) -> int | None:
        max_gap = max(1, int(self.config.max_hit_neighbor_gap))
        for offset in range(1, max_gap + 1):
            idx = local_index + direction * offset
            if idx < 0 or idx >= len(samples):
                break
            if samples[idx].visible:
                return idx
        return None

    def _in_cooldown(self, event: dict[str, object]) -> bool:
        event_type = str(event["event_type"])
        timestamp_ms = int(event["timestamp_ms"])
        last_time = self._last_event_time_ms.get(event_type)
        if last_time is None:
            return False
        cooldown_ms = int(round(max(0.0, float(self.config.event_cooldown_seconds)) * 1000.0))
        return timestamp_ms - last_time < cooldown_ms


def _frame_size(frame_shape: Sequence[int] | None) -> tuple[int, int]:
    if frame_shape is None or len(frame_shape) < 2:
        return 512, 288
    height = int(frame_shape[0])
    width = int(frame_shape[1])
    if width <= 0 or height <= 0:
        return 512, 288
    return width, height


def _as_sequence(value: object) -> Sequence[object]:
    if isinstance(value, (list, tuple)):
        return value
    return []


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return float(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)
