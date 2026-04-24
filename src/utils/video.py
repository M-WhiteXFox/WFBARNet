from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoMetadata:
    fps: float
    width: int
    height: int
    frame_count: int


def probe_video(source: str) -> VideoMetadata:
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video source: {source}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    cap.release()
    return VideoMetadata(fps=fps, width=width, height=height, frame_count=frame_count)


def load_frames(source: str) -> list[np.ndarray]:
    path = Path(source)
    if path.is_dir():
        images = sorted(path.glob("*"))
        frames = []
        for item in images:
            frame = cv2.imread(str(item))
            if frame is not None:
                frames.append(frame)
        return frames

    cap = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def iter_frame_windows(frames: list[np.ndarray]) -> Iterator[tuple[int, np.ndarray, list[np.ndarray]]]:
    total = len(frames)
    for idx, frame in enumerate(frames):
        prev_idx = max(0, idx - 1)
        next_idx = min(total - 1, idx + 1)
        yield idx, frame, [frames[prev_idx], frame, frames[next_idx]]


def iter_video_frames(source: str, max_frames: int | None = None) -> Iterator[tuple[int, np.ndarray]]:
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video source: {source}")
    frame_id = 0
    try:
        while True:
            if max_frames is not None and frame_id >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            yield frame_id, frame
            frame_id += 1
    finally:
        cap.release()


def iter_video_frame_windows(
    source: str,
    max_frames: int | None = None,
) -> Iterator[tuple[int, np.ndarray, list[np.ndarray]]]:
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video source: {source}")
    try:
        ok, first_frame = cap.read()
        if not ok:
            return

        ok, second_frame = cap.read()
        if not ok:
            second_frame = first_frame.copy()
            next_is_real = False
        else:
            next_is_real = True

        prev_frame = first_frame.copy()
        curr_frame = first_frame
        next_frame = second_frame
        frame_id = 0

        while True:
            if max_frames is not None and frame_id >= max_frames:
                break
            yield frame_id, curr_frame, [prev_frame, curr_frame, next_frame]
            if not next_is_real:
                break

            prev_frame = curr_frame
            curr_frame = next_frame
            frame_id += 1
            ok, incoming = cap.read()
            if ok:
                next_frame = incoming
                next_is_real = True
            else:
                next_frame = curr_frame.copy()
                next_is_real = False
    finally:
        cap.release()
