from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import acos, degrees, hypot
from pathlib import Path

import cv2
import numpy as np

from src.utils.structures import FrameResult, PersonPoseResult, TrackResult


DEFAULT_SKELETON = [
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
]
BALL_COLOR = (0, 0, 255)
TRAIL_COLOR = (0, 220, 255)
HIT_COLOR = (0, 0, 255)


@dataclass(frozen=True)
class _ArmMotion:
    person_id: int
    wrist_index: int
    wrist: tuple[float, float]
    elbow: tuple[float, float] | None
    shoulder: tuple[float, float] | None
    wrist_speed: float
    extension: float


@dataclass
class TrackTrailRenderer:
    fps: float = 25.0
    history_seconds: float = 0.5
    current_radius: int = 8
    trail_radius: int = 4
    hit_marker_seconds: float = 2.0
    hit_marker_radius: int = 7
    hit_min_speed_px_per_sec: float = 500.0
    hit_min_turn_deg: float = 85.0
    hit_speed_change_min_turn_deg: float = 45.0
    hit_min_speed_change_ratio: float = 1.7
    hit_cooldown_seconds: float = 0.18
    hit_max_gap_seconds: float = 0.16
    hit_top_exit_band_px: float = 36.0
    hit_top_exit_band_ratio: float = 0.08
    hit_pose_assist_score: float = 0.60
    hit_pose_assist_strong_score: float = 0.78
    hit_pose_assist_max_ball_wrist_px: float = 96.0
    hit_pose_assist_min_wrist_speed_px_per_sec: float = 220.0
    hit_pose_assist_relaxed_turn_deg: float = 55.0
    hit_pose_assist_relaxed_min_speed_px_per_sec: float = 360.0
    hit_pose_assist_relaxed_speed_change_ratio: float = 1.25
    hit_floor_bounce_min_vertical_px: float = 10.0
    hit_floor_bounce_min_vertical_ratio: float = 0.45
    hit_floor_bounce_max_rebound_speed_ratio: float = 1.35
    _points: deque[tuple[float, int, float, float, float, int, bool, float]] = field(default_factory=deque)
    _hit_markers: deque[tuple[float, float, float]] = field(default_factory=deque)
    _last_hit_time_s: float = -999.0
    _last_hit_event: dict[str, object] | None = None
    _segment_id: int = 0
    _last_visible_timestamp_s: float | None = None
    _last_arm_states: dict[tuple[int, int], tuple[float, float, float]] = field(default_factory=dict)

    def draw(
        self,
        frame: np.ndarray,
        result: FrameResult,
        *,
        timestamp_ms: int | None = None,
    ) -> np.ndarray:
        canvas = frame.copy()
        return self.draw_on(canvas, result, timestamp_ms=timestamp_ms)

    def draw_on(
        self,
        canvas: np.ndarray,
        result: FrameResult,
        *,
        timestamp_ms: int | None = None,
    ) -> np.ndarray:
        _draw_pose(canvas, result)
        timestamp_s = self.update_hit_detection(result, timestamp_ms=timestamp_ms, frame_shape=canvas.shape)
        self._draw_trail(canvas, timestamp_s)
        self._draw_current(canvas, result.track)
        self._draw_hit_markers(canvas)
        return canvas

    def update_hit_detection(
        self,
        result: FrameResult,
        *,
        timestamp_ms: int | None = None,
        frame_shape: tuple[int, ...] | None = None,
    ) -> float:
        timestamp_s = self._timestamp_seconds(result.frame_id, timestamp_ms)
        self._last_hit_event = None
        arm_motions = self._update_arm_motion(result.pose, timestamp_s)
        if result.track.visible:
            if (
                self._last_visible_timestamp_s is not None
                and timestamp_s - self._last_visible_timestamp_s > self.hit_max_gap_seconds
            ):
                self._segment_id += 1
            x, y = map(float, result.track.ball_xy)
            occluded = self._point_is_person_occluded((x, y), result)
            pose_score = self._pose_hit_score((x, y), arm_motions)
            self._points.append(
                (
                    timestamp_s,
                    int(result.frame_id),
                    x,
                    y,
                    float(result.track.score),
                    self._segment_id,
                    occluded,
                    pose_score,
                )
            )
            self._last_visible_timestamp_s = timestamp_s
            if self._is_hit_event(timestamp_s, frame_shape):
                hit_time_s, hit_frame_id, hit_x, hit_y, *_ = self._points[-2]
                self._hit_markers.append((timestamp_s, hit_x, hit_y))
                self._last_hit_event = {
                    "frame_id": int(hit_frame_id),
                    "timestamp_ms": int(round(hit_time_s * 1000.0)),
                    "ball_xy": [float(hit_x), float(hit_y)],
                }
        elif self._last_visible_timestamp_s is not None:
            self._segment_id += 1
            self._last_visible_timestamp_s = None
        self._prune(timestamp_s)
        return timestamp_s

    def last_hit_event(self) -> dict[str, object] | None:
        if self._last_hit_event is None:
            return None
        return dict(self._last_hit_event)

    def _timestamp_seconds(self, frame_id: int, timestamp_ms: int | None) -> float:
        if timestamp_ms is not None:
            return max(0.0, float(timestamp_ms) / 1000.0)
        fps = self.fps if self.fps > 0 else 25.0
        return max(0.0, float(frame_id) / fps)

    def _prune(self, timestamp_s: float) -> None:
        cutoff = timestamp_s - max(0.0, self.history_seconds)
        while self._points and self._points[0][0] < cutoff:
            self._points.popleft()
        hit_cutoff = timestamp_s - max(0.0, self.hit_marker_seconds)
        while self._hit_markers and self._hit_markers[0][0] <= hit_cutoff:
            self._hit_markers.popleft()

    def _draw_trail(self, canvas: np.ndarray, timestamp_s: float) -> None:
        for point_time, _frame_id, x, y, *_ in self._points:
            age = max(0.0, timestamp_s - point_time)
            fade = max(0.15, 1.0 - age / max(self.history_seconds, 1e-6))
            radius = max(2, int(round(self.trail_radius + fade * 2)))
            thickness = 1 if fade < 0.55 else 2
            color = tuple(int(channel * fade) for channel in TRAIL_COLOR)
            cv2.circle(canvas, (int(round(x)), int(round(y))), radius, color, thickness)

    def _draw_current(self, canvas: np.ndarray, track: TrackResult) -> None:
        if not track.visible:
            return
        x, y = map(int, map(round, track.ball_xy))
        cv2.circle(canvas, (x, y), self.current_radius, BALL_COLOR, 2)
        cv2.circle(canvas, (x, y), self.current_radius + 6, TRAIL_COLOR, 1)
        cv2.putText(
            canvas,
            f"{track.score:.2f}",
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            BALL_COLOR,
            1,
        )

    def _is_hit_event(self, timestamp_s: float, frame_shape: tuple[int, ...] | None) -> bool:
        if len(self._points) < 3:
            return False
        if timestamp_s - self._last_hit_time_s < self.hit_cooldown_seconds:
            return False

        prev, mid, current = self._points[-3], self._points[-2], self._points[-1]
        if len({prev[5], mid[5], current[5]}) > 1:
            return False
        dt_before = mid[0] - prev[0]
        dt_after = current[0] - mid[0]
        if dt_before <= 1e-6 or dt_after <= 1e-6:
            return False

        vx_before = mid[2] - prev[2]
        vy_before = mid[3] - prev[3]
        vx_after = current[2] - mid[2]
        vy_after = current[3] - mid[3]
        dist_before = hypot(vx_before, vy_before)
        dist_after = hypot(vx_after, vy_after)
        if min(dist_before, dist_after) < 3.0:
            return False

        speed_before = dist_before / dt_before
        speed_after = dist_after / dt_after
        pose_score = max(prev[7], mid[7], current[7])
        has_pose_assist = pose_score >= self.hit_pose_assist_score
        min_speed = (
            self.hit_pose_assist_relaxed_min_speed_px_per_sec
            if has_pose_assist
            else self.hit_min_speed_px_per_sec
        )
        if max(speed_before, speed_after) < min_speed:
            return False
        if self._looks_like_top_exit(prev, mid, current, frame_shape):
            return False
        if self._looks_like_floor_bounce(
            prev,
            mid,
            current,
            speed_before=speed_before,
            speed_after=speed_after,
            dist_before=dist_before,
            dist_after=dist_after,
        ) and pose_score < self.hit_pose_assist_strong_score:
            return False
        if self._looks_like_person_occlusion(prev, mid, current) and pose_score < self.hit_pose_assist_strong_score:
            return False

        turn_cos = (vx_before * vx_after + vy_before * vy_after) / (dist_before * dist_after)
        turn_deg = degrees(acos(max(-1.0, min(1.0, turn_cos))))
        slower_speed = max(min(speed_before, speed_after), 1e-6)
        speed_change = max(speed_before, speed_after) / slower_speed

        is_direction_change = turn_deg >= self.hit_min_turn_deg
        is_speed_snap = (
            turn_deg >= self.hit_speed_change_min_turn_deg
            and speed_change >= self.hit_min_speed_change_ratio
        )
        is_pose_assisted = has_pose_assist and (
            turn_deg >= self.hit_pose_assist_relaxed_turn_deg
            or speed_change >= self.hit_pose_assist_relaxed_speed_change_ratio
        )
        if not (is_direction_change or is_speed_snap or is_pose_assisted):
            return False

        self._last_hit_time_s = timestamp_s
        return True

    def _looks_like_top_exit(
        self,
        prev: tuple[float, int, float, float, float, int, bool, float],
        mid: tuple[float, int, float, float, float, int, bool, float],
        current: tuple[float, int, float, float, float, int, bool, float],
        frame_shape: tuple[int, ...] | None,
    ) -> bool:
        if frame_shape is None or len(frame_shape) < 2:
            return False
        height = max(1.0, float(frame_shape[0]))
        top_band = max(float(self.hit_top_exit_band_px), height * max(0.0, float(self.hit_top_exit_band_ratio)))
        upward_motion = prev[3] - mid[3]
        min_upward_motion = max(6.0, height * 0.006)
        near_top = min(prev[3], mid[3], current[3]) <= top_band
        candidate_near_top = mid[3] <= top_band
        return near_top and candidate_near_top and upward_motion >= min_upward_motion

    def _looks_like_person_occlusion(
        self,
        prev: tuple[float, int, float, float, float, int, bool, float],
        mid: tuple[float, int, float, float, float, int, bool, float],
        current: tuple[float, int, float, float, float, int, bool, float],
    ) -> bool:
        return bool(mid[6] and (prev[6] or current[6]))

    def _looks_like_floor_bounce(
        self,
        prev: tuple[float, int, float, float, float, int, bool, float],
        mid: tuple[float, int, float, float, float, int, bool, float],
        current: tuple[float, int, float, float, float, int, bool, float],
        *,
        speed_before: float,
        speed_after: float,
        dist_before: float,
        dist_after: float,
    ) -> bool:
        min_vertical = max(0.0, float(self.hit_floor_bounce_min_vertical_px))
        vy_before = mid[3] - prev[3]
        vy_after = current[3] - mid[3]
        down_then_up = vy_before >= min_vertical and vy_after <= -min_vertical
        if not down_then_up:
            return False

        vertex_is_lowest = mid[3] >= prev[3] + min_vertical and mid[3] >= current[3] + min_vertical
        if not vertex_is_lowest:
            return False

        min_vertical_ratio = max(0.0, min(1.0, float(self.hit_floor_bounce_min_vertical_ratio)))
        vertical_before = abs(vy_before) / max(dist_before, 1e-6)
        vertical_after = abs(vy_after) / max(dist_after, 1e-6)
        if min(vertical_before, vertical_after) < min_vertical_ratio:
            return False

        max_rebound_ratio = max(0.1, float(self.hit_floor_bounce_max_rebound_speed_ratio))
        return speed_after <= speed_before * max_rebound_ratio

    def _update_arm_motion(self, poses: list[PersonPoseResult], timestamp_s: float) -> list[_ArmMotion]:
        motions: list[_ArmMotion] = []
        seen_keys: set[tuple[int, int]] = set()
        for pose_index, person in enumerate(poses):
            try:
                person_id = int(getattr(person, "person_id", pose_index))
            except (TypeError, ValueError):
                person_id = pose_index
            for shoulder_index, elbow_index, wrist_index in ((5, 7, 9), (6, 8, 10)):
                wrist = self._pose_keypoint(person, wrist_index, min_score=0.20)
                if wrist is None:
                    continue
                elbow = self._pose_keypoint(person, elbow_index, min_score=0.15)
                shoulder = self._pose_keypoint(person, shoulder_index, min_score=0.15)
                state_key = (person_id, wrist_index)
                previous = self._last_arm_states.get(state_key)
                wrist_speed = 0.0
                if previous is not None:
                    dt = max(timestamp_s - previous[0], 1e-6)
                    wrist_speed = hypot(wrist[0] - previous[1], wrist[1] - previous[2]) / dt
                extension = self._arm_extension(shoulder, elbow, wrist)
                motions.append(
                    _ArmMotion(
                        person_id=person_id,
                        wrist_index=wrist_index,
                        wrist=wrist,
                        elbow=elbow,
                        shoulder=shoulder,
                        wrist_speed=wrist_speed,
                        extension=extension,
                    )
                )
                self._last_arm_states[state_key] = (timestamp_s, wrist[0], wrist[1])
                seen_keys.add(state_key)

        stale_cutoff = timestamp_s - 1.0
        for key, state in list(self._last_arm_states.items()):
            if key not in seen_keys and state[0] < stale_cutoff:
                del self._last_arm_states[key]
        return motions

    def _pose_hit_score(self, ball: tuple[float, float], arm_motions: list[_ArmMotion]) -> float:
        best_score = 0.0
        max_distance = max(1.0, float(self.hit_pose_assist_max_ball_wrist_px))
        for motion in arm_motions:
            wrist_distance = hypot(ball[0] - motion.wrist[0], ball[1] - motion.wrist[1])
            if wrist_distance > max_distance:
                continue
            proximity_score = 1.0 - wrist_distance / max_distance
            speed_score = min(
                1.0,
                motion.wrist_speed / max(1.0, float(self.hit_pose_assist_min_wrist_speed_px_per_sec) * 2.0),
            )
            extension_score = max(0.0, min(1.0, (motion.extension - 0.55) / 0.35))
            score = 0.40 * proximity_score + 0.45 * speed_score + 0.15 * extension_score
            best_score = max(best_score, score)
        return best_score

    @staticmethod
    def _arm_extension(
        shoulder: tuple[float, float] | None,
        elbow: tuple[float, float] | None,
        wrist: tuple[float, float],
    ) -> float:
        if shoulder is None or elbow is None:
            return 0.0
        upper = hypot(elbow[0] - shoulder[0], elbow[1] - shoulder[1])
        lower = hypot(wrist[0] - elbow[0], wrist[1] - elbow[1])
        full = upper + lower
        if full <= 1e-6:
            return 0.0
        reach = hypot(wrist[0] - shoulder[0], wrist[1] - shoulder[1])
        return max(0.0, min(1.0, reach / full))

    @staticmethod
    def _pose_keypoint(
        person: PersonPoseResult,
        index: int,
        *,
        min_score: float,
    ) -> tuple[float, float] | None:
        if index >= len(person.keypoints):
            return None
        if index < len(person.scores) and float(person.scores[index]) < min_score:
            return None
        point = person.keypoints[index]
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            return None
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, ValueError):
            return None
        if x != x or y != y or x in (float("inf"), float("-inf")) or y in (float("inf"), float("-inf")):
            return None
        return x, y

    def _point_is_person_occluded(self, point: tuple[float, float], result: FrameResult) -> bool:
        x, y = point
        for person in result.pose:
            bbox = getattr(person, "bbox", None)
            if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                continue
            try:
                x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
            except (TypeError, ValueError):
                continue
            if x2 <= x1 or y2 <= y1:
                continue
            if x1 <= x <= x2 and y1 <= y <= y2:
                return True
        return False

    def _draw_hit_marker(self, canvas: np.ndarray, point: tuple[float, float]) -> None:
        x, y = map(int, map(round, point))
        cv2.circle(canvas, (x, y), self.hit_marker_radius, HIT_COLOR, -1)

    def _draw_hit_markers(self, canvas: np.ndarray) -> None:
        for _, x, y in self._hit_markers:
            self._draw_hit_marker(canvas, (x, y))


def _draw_pose(canvas: np.ndarray, result: FrameResult) -> None:
    for person in result.pose:
        x1, y1, x2, y2 = map(int, person.bbox)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 180, 255), 2)
        for x, y in person.keypoints:
            cv2.circle(canvas, (int(x), int(y)), 4, (0, 255, 0), -1)
        for a, b in DEFAULT_SKELETON:
            if a < len(person.keypoints) and b < len(person.keypoints):
                p1 = tuple(map(int, person.keypoints[a]))
                p2 = tuple(map(int, person.keypoints[b]))
                cv2.line(canvas, p1, p2, (255, 180, 0), 2)


def draw_result(frame: np.ndarray, result: FrameResult) -> np.ndarray:
    canvas = frame.copy()
    _draw_pose(canvas, result)
    if result.track.visible:
        x, y = map(int, result.track.ball_xy)
        cv2.circle(canvas, (x, y), 8, BALL_COLOR, 2)
        cv2.putText(canvas, f"{result.track.score:.2f}", (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BALL_COLOR, 1)
    return canvas


def save_visualization_video(frames: list[np.ndarray], results: list[FrameResult], path: Path, fps: float = 25.0) -> None:
    if not frames:
        return
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    trail_renderer = TrackTrailRenderer(fps=fps)
    for frame, result in zip(frames, results):
        writer.write(trail_renderer.draw(frame, result))
    writer.release()
