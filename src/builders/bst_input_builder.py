from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.structures import FrameResult


@dataclass
class BSTInputBuilder:
    normalize: bool = False

    def build(self, results: list[FrameResult]) -> dict[str, Any]:
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

        if self.normalize and frames > 0 and max_persons > 0:
            width = np.maximum(bboxes[..., 2] - bboxes[..., 0], 1.0)
            height = np.maximum(bboxes[..., 3] - bboxes[..., 1], 1.0)
            center_x = (bboxes[..., 0] + bboxes[..., 2]) / 2.0
            center_y = (bboxes[..., 1] + bboxes[..., 3]) / 2.0
            for p in range(max_persons):
                keypoints[:, p, :, 0] = (keypoints[:, p, :, 0] - center_x[:, p : p + 1]) / width[:, p : p + 1]
                keypoints[:, p, :, 1] = (keypoints[:, p, :, 1] - center_y[:, p : p + 1]) / height[:, p : p + 1]

        ball_features = np.concatenate(
            [
                ball_xy,
                ball_visible[:, None].astype(np.float32),
                ball_score[:, None],
            ],
            axis=1,
        )

        return {
            "frame_ids": np.arange(frames, dtype=np.int32),
            "keypoints": keypoints,
            "keypoint_scores": keypoint_scores,
            "bboxes": bboxes,
            "person_scores": person_scores,
            "ball_xy": ball_xy,
            "ball_visible": ball_visible,
            "ball_score": ball_score,
            "ball_features": ball_features,
            "sequence_length": np.asarray([frames], dtype=np.int32),
            "player_slots": np.asarray([max_persons], dtype=np.int32),
            "joint_count": np.asarray([max_kpts], dtype=np.int32),
            "format_note": np.asarray(
                ["Generic BST-ready multimodal package. Adjust final tensor order to your exact BST implementation."],
                dtype=object,
            ),
        }

    def save(self, results: list[FrameResult], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.build(results), allow_pickle=True)
