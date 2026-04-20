from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class MMPoseInferenceItem:
    bbox: list[float]
    keypoints: list[list[float]]
    scores: list[float]


class MMPoseBackend:
    def __init__(
        self,
        model_config: str | None,
        model_weight: str | None,
        device: str,
        bbox_mode: str = "whole_image",
        det_config: str | None = None,
        det_weight: str | None = None,
        conf_thr: float = 0.3,
        max_persons: int = 2,
    ) -> None:
        self.model_config = model_config
        self.model_weight = model_weight
        self.device = device
        self.bbox_mode = bbox_mode
        self.det_config = det_config
        self.det_weight = det_weight
        self.conf_thr = conf_thr
        self.max_persons = max_persons
        self.model = self._try_init()

    def _try_init(self) -> Any:
        try:
            from mmpose.apis import init_model  # type: ignore

            if not self.model_config or not self.model_weight:
                return None
            return init_model(self.model_config, self.model_weight, device=self.device)
        except Exception:
            return None

    def infer(self, image: np.ndarray) -> list[MMPoseInferenceItem]:
        h, w = image.shape[:2]
        if self.model is None:
            bbox = [0.0, 0.0, float(w), float(h)]
            keypoints = [[w * 0.5, h * 0.2], [w * 0.5, h * 0.35], [w * 0.45, h * 0.55], [w * 0.55, h * 0.55]]
            scores = [0.6] * len(keypoints)
            return [MMPoseInferenceItem(bbox=bbox, keypoints=keypoints, scores=scores)]

        try:
            from mmpose.apis import inference_topdown  # type: ignore
        except Exception:
            bbox = [0.0, 0.0, float(w), float(h)]
            keypoints = [[w * 0.5, h * 0.2], [w * 0.5, h * 0.35], [w * 0.45, h * 0.55], [w * 0.55, h * 0.55]]
            scores = [0.6] * len(keypoints)
            return [MMPoseInferenceItem(bbox=bbox, keypoints=keypoints, scores=scores)]

        bboxes = self._build_bboxes(image)
        results = inference_topdown(self.model, image, bboxes=bboxes)
        items: list[MMPoseInferenceItem] = []
        for result in results[: self.max_persons]:
            pred = result.pred_instances
            keypoints = pred.keypoints[0].tolist()
            scores = pred.keypoint_scores[0].tolist()
            bbox = pred.bboxes[0].tolist() if hasattr(pred, "bboxes") else bboxes[0]["bbox"]
            items.append(MMPoseInferenceItem(bbox=bbox, keypoints=keypoints, scores=scores))
        return items

    def _build_bboxes(self, image: np.ndarray) -> list[dict[str, list[float]]]:
        h, w = image.shape[:2]
        if self.bbox_mode == "split_two":
            return [
                {"bbox": [0.0, 0.0, float(w) * 0.5, float(h)]},
                {"bbox": [float(w) * 0.5, 0.0, float(w), float(h)]},
            ]
        return [{"bbox": [0.0, 0.0, float(w), float(h)]}]
