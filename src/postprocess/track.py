from __future__ import annotations

import cv2
import numpy as np

from src.preprocess.track import TrackPreprocessMeta
from src.utils.structures import TrackResult


def _extract_ball_center(mask: np.ndarray) -> tuple[float, float] | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    return float(x + w / 2.0), float(y + h / 2.0)


def decode_track_heatmap(
    heatmaps: np.ndarray,
    meta: TrackPreprocessMeta,
    score_thr: float,
) -> TrackResult:
    if heatmaps.ndim == 4:
        heatmap = heatmaps[0, 1]
    elif heatmaps.ndim == 3:
        heatmap = heatmaps[1]
    else:
        raise ValueError(f"Unexpected heatmap shape: {heatmaps.shape}")

    score = float(np.max(heatmap))
    binary_mask = (heatmap > score_thr).astype(np.uint8) * 255
    center = _extract_ball_center(binary_mask)
    if center is None:
        return TrackResult(
            ball_xy=[-1.0, -1.0],
            visible=0,
            score=score,
            heatmap_shape=list(heatmap.shape),
        )

    x, y = center
    ball_xy = [x * meta.scale_x, y * meta.scale_y]
    return TrackResult(
        ball_xy=ball_xy,
        visible=1,
        score=score,
        heatmap_shape=list(heatmap.shape),
    )
