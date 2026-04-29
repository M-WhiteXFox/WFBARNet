from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.preprocess.track import TrackPreprocessMeta
from src.utils.structures import TrackResult


@dataclass(slots=True)
class _HeatmapCandidate:
    center: tuple[float, float]
    score: float
    rank: tuple[float, float, float, float]


def _extract_ball_candidates(
    heatmap: np.ndarray,
    mask: np.ndarray,
    *,
    max_candidates: int | None = None,
) -> list[_HeatmapCandidate]:
    labels_count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    if labels_count <= 1:
        return []

    candidates: list[_HeatmapCandidate] = []
    for label_id in range(1, labels_count):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area <= 0:
            continue

        component_mask = labels == label_id
        values = heatmap[component_mask].astype(np.float64, copy=False)
        peak = float(values.max(initial=0.0))
        total = float(values.sum())
        mean = total / float(area)

        ys, xs = np.nonzero(component_mask)
        if total > 1e-8:
            weights = values / total
            center = (float(np.sum(xs * weights)), float(np.sum(ys * weights)))
        else:
            cx, cy = centroids[label_id]
            center = (float(cx), float(cy))

        width = max(int(stats[label_id, cv2.CC_STAT_WIDTH]), 1)
        height = max(int(stats[label_id, cv2.CC_STAT_HEIGHT]), 1)
        compactness = float(area) / float(width * height)
        rank = (peak, mean, min(float(area), 24.0), compactness)
        candidates.append(_HeatmapCandidate(center=center, score=peak, rank=rank))

    candidates.sort(key=lambda item: item.rank, reverse=True)
    if max_candidates is not None:
        candidates = candidates[:max(1, int(max_candidates))]
    return candidates


def _extract_ball_candidate(heatmap: np.ndarray, mask: np.ndarray) -> _HeatmapCandidate | None:
    candidates = _extract_ball_candidates(heatmap, mask, max_candidates=1)
    return candidates[0] if candidates else None


def _decode_single_heatmap(
    heatmap: np.ndarray,
    meta: TrackPreprocessMeta,
    score_thr: float,
) -> TrackResult:
    score = float(np.max(heatmap))
    binary_mask = (heatmap > score_thr).astype(np.uint8) * 255
    candidate = _extract_ball_candidate(heatmap, binary_mask)
    if candidate is None:
        return TrackResult(
            ball_xy=[-1.0, -1.0],
            visible=0,
            score=score,
            heatmap_shape=list(heatmap.shape),
        )

    x, y = candidate.center
    ball_xy = [x * meta.scale_x, y * meta.scale_y]
    return TrackResult(
        ball_xy=ball_xy,
        visible=1,
        score=candidate.score,
        heatmap_shape=list(heatmap.shape),
    )


def decode_track_heatmap(
    heatmaps: np.ndarray,
    meta: TrackPreprocessMeta,
    score_thr: float,
) -> TrackResult:
    heatmap = _select_track_heatmap_plane(heatmaps)
    return _decode_single_heatmap(heatmap, meta, score_thr)


def decode_track_heatmap_candidates(
    heatmaps: np.ndarray,
    meta: TrackPreprocessMeta,
    score_thr: float,
    *,
    max_candidates: int = 5,
    candidate_score_thr: float | None = None,
) -> list[TrackResult]:
    heatmap = _select_track_heatmap_plane(heatmaps)
    score = float(np.max(heatmap))
    threshold = score_thr if candidate_score_thr is None else float(candidate_score_thr)
    threshold = max(0.0, min(float(score_thr), threshold))
    binary_mask = (heatmap > threshold).astype(np.uint8) * 255
    candidates = _extract_ball_candidates(heatmap, binary_mask, max_candidates=max_candidates)
    if not candidates:
        return [
            TrackResult(
                ball_xy=[-1.0, -1.0],
                visible=0,
                score=score,
                heatmap_shape=list(heatmap.shape),
            )
        ]

    return [
        TrackResult(
            ball_xy=[candidate.center[0] * meta.scale_x, candidate.center[1] * meta.scale_y],
            visible=1,
            score=candidate.score,
            heatmap_shape=list(heatmap.shape),
        )
        for candidate in candidates
    ]


def _select_track_heatmap_plane(heatmaps: np.ndarray) -> np.ndarray:
    if heatmaps.ndim == 4:
        return heatmaps[0, 1]
    elif heatmaps.ndim == 3:
        if heatmaps.shape[0] == 2:
            return heatmaps[1]
        if heatmaps.shape[0] == 1:
            return heatmaps[0]
        return heatmaps[1]
    elif heatmaps.ndim == 2:
        return heatmaps
    else:
        raise ValueError(f"Unexpected heatmap shape: {heatmaps.shape}")


def decode_track_heatmap_batch(
    batch_heatmaps: np.ndarray,
    metas: list[TrackPreprocessMeta],
    score_thr: float,
) -> list[TrackResult]:
    heatmap_planes = _select_track_heatmap_batch_planes(batch_heatmaps)
    batch_size = heatmap_planes.shape[0]
    if batch_size != len(metas):
        raise ValueError(f"Heatmap batch size {batch_size} doesn't match metas length {len(metas)}")

    results = []
    for i in range(batch_size):
        heatmap = heatmap_planes[i]
        meta = metas[i]

        results.append(_decode_single_heatmap(heatmap, meta, score_thr))

    return results


def decode_track_heatmap_candidate_batch(
    batch_heatmaps: np.ndarray,
    metas: list[TrackPreprocessMeta],
    score_thr: float,
    *,
    max_candidates: int = 5,
    candidate_score_thr: float | None = None,
) -> list[list[TrackResult]]:
    heatmap_planes = _select_track_heatmap_batch_planes(batch_heatmaps)
    batch_size = heatmap_planes.shape[0]
    if batch_size != len(metas):
        raise ValueError(f"Heatmap batch size {batch_size} doesn't match metas length {len(metas)}")

    return [
        decode_track_heatmap_candidates(
            heatmap_planes[index],
            metas[index],
            score_thr,
            max_candidates=max_candidates,
            candidate_score_thr=candidate_score_thr,
        )
        for index in range(batch_size)
    ]


def _select_track_heatmap_batch_planes(batch_heatmaps: np.ndarray) -> np.ndarray:
    if batch_heatmaps.ndim == 4:
        return batch_heatmaps[:, 1]
    if batch_heatmaps.ndim == 3:
        return batch_heatmaps
    raise ValueError(f"Batch heatmaps must be 3D or 4D, got {batch_heatmaps.ndim}D")
