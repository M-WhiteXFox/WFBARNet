from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from src.utils.structures import FrameResult


def export_json(results: list[FrameResult], path: Path) -> None:
    path.write_text(json.dumps([item.to_dict() for item in results], ensure_ascii=False, indent=2), encoding="utf-8")


def export_csv(results: list[FrameResult], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["frame_id", "person_count", "ball_x", "ball_y", "ball_visible", "ball_score"],
        )
        writer.writeheader()
        for item in results:
            writer.writerow(
                {
                    "frame_id": item.frame_id,
                    "person_count": len(item.pose),
                    "ball_x": item.track.ball_xy[0],
                    "ball_y": item.track.ball_xy[1],
                    "ball_visible": item.track.visible,
                    "ball_score": item.track.score,
                }
            )


def export_npy(results: list[FrameResult], path: Path) -> None:
    max_persons = max((len(item.pose) for item in results), default=0)
    max_kpts = max((len(person.keypoints) for item in results for person in item.pose), default=0)
    frames = len(results)
    keypoints = np.zeros((frames, max_persons, max_kpts, 2), dtype=np.float32)
    keypoint_scores = np.zeros((frames, max_persons, max_kpts), dtype=np.float32)
    bboxes = np.zeros((frames, max_persons, 4), dtype=np.float32)
    person_scores = np.zeros((frames, max_persons), dtype=np.float32)
    ball_xy = np.zeros((frames, 2), dtype=np.float32)
    ball_visible = np.zeros((frames,), dtype=np.int32)
    ball_score = np.zeros((frames,), dtype=np.float32)

    for frame_idx, item in enumerate(results):
        ball_xy[frame_idx] = np.asarray(item.track.ball_xy, dtype=np.float32)
        ball_visible[frame_idx] = item.track.visible
        ball_score[frame_idx] = item.track.score
        for person_idx, person in enumerate(item.pose):
            bboxes[frame_idx, person_idx] = np.asarray(person.bbox, dtype=np.float32)
            person_scores[frame_idx, person_idx] = person.person_score
            for kp_idx, kp in enumerate(person.keypoints):
                keypoints[frame_idx, person_idx, kp_idx] = np.asarray(kp, dtype=np.float32)
                if kp_idx < len(person.scores):
                    keypoint_scores[frame_idx, person_idx, kp_idx] = person.scores[kp_idx]

    np.save(
        path,
        {
            "frame_ids": np.arange(frames, dtype=np.int32),
            "keypoints": keypoints,
            "keypoint_scores": keypoint_scores,
            "bboxes": bboxes,
            "person_scores": person_scores,
            "ball_xy": ball_xy,
            "ball_visible": ball_visible,
            "ball_score": ball_score,
        },
        allow_pickle=True,
    )
