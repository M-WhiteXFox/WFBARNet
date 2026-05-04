from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import hypot
from pathlib import Path

import cv2
import numpy as np

from src.utils.structures import FrameResult, TrackResult


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
LANDING_COLOR = (0, 255, 0)
OUT_OF_FRAME_COLOR = (255, 0, 255)
MARKER_OUTLINE_COLOR = (255, 255, 255)

_TrailPoint = tuple[float, int, float, float, int]


@dataclass
class TrackTrailRenderer:
    fps: float = 25.0
    history_seconds: float = 0.5
    current_radius: int = 8
    trail_radius: int = 4
    trail_break_threshold_px: float = 80.0
    segment_gap_seconds: float = 0.16
    event_marker_seconds: float = 2.0
    event_marker_radius: int = 8
    _points: deque[_TrailPoint] = field(default_factory=deque)
    _event_markers: deque[tuple[float, float, float, str]] = field(default_factory=deque)
    _segment_id: int = 0
    _last_visible_timestamp_s: float | None = None

    def draw(
        self,
        frame: np.ndarray,
        result: FrameResult,
        *,
        timestamp_ms: int | None = None,
        trajectory_event: object | None = None,
    ) -> np.ndarray:
        canvas = frame.copy()
        return self.draw_on(
            canvas,
            result,
            timestamp_ms=timestamp_ms,
            trajectory_event=trajectory_event,
        )

    def draw_on(
        self,
        canvas: np.ndarray,
        result: FrameResult,
        *,
        timestamp_ms: int | None = None,
        trajectory_event: object | None = None,
    ) -> np.ndarray:
        _draw_pose(canvas, result)
        timestamp_s = self.update_track_history(result, timestamp_ms=timestamp_ms)
        self.add_trajectory_event(trajectory_event)
        self._draw_trail(canvas, timestamp_s)
        self._draw_current(canvas, result.track)
        self._draw_event_markers(canvas)
        return canvas

    def update_track_history(
        self,
        result: FrameResult,
        *,
        timestamp_ms: int | None = None,
    ) -> float:
        timestamp_s = self._timestamp_seconds(result.frame_id, timestamp_ms)
        if result.track.visible:
            if (
                self._last_visible_timestamp_s is not None
                and timestamp_s - self._last_visible_timestamp_s > self.segment_gap_seconds
            ):
                self._segment_id += 1
            x, y = map(float, result.track.ball_xy)
            self._points.append((timestamp_s, int(result.frame_id), x, y, self._segment_id))
            self._last_visible_timestamp_s = timestamp_s
        elif self._last_visible_timestamp_s is not None:
            self._segment_id += 1
            self._last_visible_timestamp_s = None
        self._prune(timestamp_s)
        return timestamp_s

    def add_trajectory_event(self, event: object) -> None:
        if not isinstance(event, dict):
            return
        event_type = str(event.get("event_type", ""))
        if event_type not in {"hit", "landing", "out_of_frame"}:
            return
        ball_xy = event.get("ball_xy", [-1.0, -1.0])
        if not isinstance(ball_xy, (list, tuple)) or len(ball_xy) < 2:
            return
        try:
            x = float(ball_xy[0])
            y = float(ball_xy[1])
        except (TypeError, ValueError):
            return
        if not np.isfinite(x) or not np.isfinite(y):
            return
        try:
            timestamp_ms = int(event.get("timestamp_ms", 0))
        except (TypeError, ValueError):
            timestamp_ms = 0
        timestamp_s = max(0.0, float(timestamp_ms) / 1000.0)
        self._event_markers.append((timestamp_s, x, y, event_type))

    def _timestamp_seconds(self, frame_id: int, timestamp_ms: int | None) -> float:
        if timestamp_ms is not None:
            return max(0.0, float(timestamp_ms) / 1000.0)
        fps = self.fps if self.fps > 0 else 25.0
        return max(0.0, float(frame_id) / fps)

    def _prune(self, timestamp_s: float) -> None:
        cutoff = timestamp_s - max(0.0, self.history_seconds)
        while self._points and self._points[0][0] < cutoff:
            self._points.popleft()
        event_cutoff = timestamp_s - max(0.0, self.event_marker_seconds)
        while self._event_markers and self._event_markers[0][0] <= event_cutoff:
            self._event_markers.popleft()

    def _draw_trail(self, canvas: np.ndarray, timestamp_s: float) -> None:
        trail_points = list(self._points)
        for previous, current in zip(trail_points, trail_points[1:]):
            _, _prev_frame_id, prev_x, prev_y, prev_segment_id = previous
            point_time, _frame_id, x, y, segment_id = current
            if segment_id != prev_segment_id:
                continue
            if hypot(x - prev_x, y - prev_y) > self.trail_break_threshold_px:
                continue
            age = max(0.0, timestamp_s - point_time)
            fade = max(0.15, 1.0 - age / max(self.history_seconds, 1e-6))
            color = tuple(int(channel * fade) for channel in TRAIL_COLOR)
            cv2.line(
                canvas,
                (int(round(prev_x)), int(round(prev_y))),
                (int(round(x)), int(round(y))),
                color,
                2,
            )

        for point_time, _frame_id, x, y, _segment_id in self._points:
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

    def _draw_event_markers(self, canvas: np.ndarray) -> None:
        for _, x, y, event_type in self._event_markers:
            self._draw_event_marker(canvas, (x, y), event_type)

    def _draw_event_marker(self, canvas: np.ndarray, point: tuple[float, float], event_type: str) -> None:
        x, y = self._marker_point(canvas, point)
        radius = max(3, int(self.event_marker_radius))
        if event_type == "landing":
            pts = np.array(
                [[x, y - radius], [x + radius, y], [x, y + radius], [x - radius, y]],
                dtype=np.int32,
            )
            cv2.fillConvexPoly(canvas, pts, LANDING_COLOR)
            cv2.polylines(canvas, [pts], isClosed=True, color=MARKER_OUTLINE_COLOR, thickness=1)
            return
        if event_type == "out_of_frame":
            cv2.line(canvas, (x - radius, y - radius), (x + radius, y + radius), OUT_OF_FRAME_COLOR, 3)
            cv2.line(canvas, (x - radius, y + radius), (x + radius, y - radius), OUT_OF_FRAME_COLOR, 3)
            cv2.circle(canvas, (x, y), radius + 3, MARKER_OUTLINE_COLOR, 1)
            return
        cv2.circle(canvas, (x, y), radius, HIT_COLOR, -1)
        cv2.circle(canvas, (x, y), radius + 2, MARKER_OUTLINE_COLOR, 1)

    def _marker_point(self, canvas: np.ndarray, point: tuple[float, float]) -> tuple[int, int]:
        height, width = canvas.shape[:2]
        x = int(round(point[0]))
        y = int(round(point[1]))
        if width > 0:
            x = max(0, min(width - 1, x))
        if height > 0:
            y = max(0, min(height - 1, y))
        return x, y


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
