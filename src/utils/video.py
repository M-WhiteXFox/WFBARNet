from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


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
