from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.utils.structures import FrameResult


DEFAULT_SKELETON = [(0, 1), (1, 2), (1, 3)]


def draw_result(frame: np.ndarray, result: FrameResult) -> np.ndarray:
    canvas = frame.copy()
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

    if result.track.visible:
        x, y = map(int, result.track.ball_xy)
        cv2.circle(canvas, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(canvas, f"{result.track.score:.2f}", (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return canvas


def save_visualization_video(frames: list[np.ndarray], results: list[FrameResult], path: Path, fps: float = 25.0) -> None:
    if not frames:
        return
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame, result in zip(frames, results):
        writer.write(draw_result(frame, result))
    writer.release()
