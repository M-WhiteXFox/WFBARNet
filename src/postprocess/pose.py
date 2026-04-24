from __future__ import annotations

from typing import Iterable

from src.preprocess.pose import PosePreprocessMeta
from src.utils.structures import PersonPoseResult


def restore_keypoints(
    keypoints: Iterable[Iterable[float]],
    meta: PosePreprocessMeta,
) -> list[list[float]]:
    restored = []
    for x, y in keypoints:
        restored.append([float(x) * meta.scale_x, float(y) * meta.scale_y])
    return restored


def restore_bbox(bbox: Iterable[float], meta: PosePreprocessMeta) -> list[float]:
    x1, y1, x2, y2 = bbox
    return [
        float(x1) * meta.scale_x,
        float(y1) * meta.scale_y,
        float(x2) * meta.scale_x,
        float(y2) * meta.scale_y,
    ]


def build_pose_result(
    person_id: int,
    bbox: list[float],
    keypoints: list[list[float]],
    scores: list[float],
    meta: PosePreprocessMeta | None = None,
) -> PersonPoseResult:
    restored_bbox = restore_bbox(bbox, meta) if meta is not None else [float(x) for x in bbox]
    restored_kpts = restore_keypoints(keypoints, meta) if meta is not None else [
        [float(x), float(y)] for x, y in keypoints
    ]
    person_score = float(sum(scores) / max(len(scores), 1))
    return PersonPoseResult(
        person_id=person_id,
        bbox=restored_bbox,
        keypoints=restored_kpts,
        scores=[float(x) for x in scores],
        person_score=person_score,
    )
