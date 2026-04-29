from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class MMPoseInferenceItem:
    bbox: list[float]
    keypoints: list[list[float]]
    scores: list[float]
    coordinate_space: str = "original"


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
        allow_dummy: bool = False,
    ) -> None:
        self.model_config = model_config
        self.model_weight = model_weight
        self.device = device
        self.bbox_mode = bbox_mode
        self.det_config = det_config
        self.det_weight = det_weight
        self.conf_thr = conf_thr
        self.max_persons = max_persons
        self.allow_dummy = allow_dummy
        self.model = self._try_init()

    def _try_init(self) -> Any:
        if self.allow_dummy:
            return None
        if not self.model_config or not self.model_weight:
            raise ValueError("MMPose backend requires both model_config and model_weight.")
        if not Path(self.model_config).exists():
            raise FileNotFoundError(f"MMPose config file not found: {self.model_config}")
        if not Path(self.model_weight).exists():
            raise FileNotFoundError(f"MMPose weight file not found: {self.model_weight}")
        try:
            from mmpose.apis import init_model  # type: ignore

            return init_model(self.model_config, self.model_weight, device=self.device)
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize MMPose backend: {exc}") from exc

    def infer(self, image: np.ndarray, court_prediction: object | None = None) -> list[MMPoseInferenceItem]:
        del court_prediction
        h, w = image.shape[:2]
        if self.model is None:
            bbox = [0.0, 0.0, float(w), float(h)]
            keypoints = [[w * 0.5, h * 0.2], [w * 0.5, h * 0.35], [w * 0.45, h * 0.55], [w * 0.55, h * 0.55]]
            scores = [0.6] * len(keypoints)
            return [MMPoseInferenceItem(bbox=bbox, keypoints=keypoints, scores=scores)]

        try:
            from mmpose.apis import inference_topdown  # type: ignore
        except Exception as exc:
            raise RuntimeError(f"MMPose inference API is unavailable: {exc}") from exc

        bboxes = self._build_bboxes(image)
        results = inference_topdown(self.model, image, bboxes=np.asarray(bboxes, dtype=np.float32))
        items: list[MMPoseInferenceItem] = []
        for idx, result in enumerate(results[: self.max_persons]):
            pred = result.pred_instances
            keypoints = pred.keypoints[0].tolist()
            scores = pred.keypoint_scores[0].tolist()
            bbox = pred.bboxes[0].tolist() if hasattr(pred, "bboxes") else bboxes[idx]
            items.append(MMPoseInferenceItem(bbox=bbox, keypoints=keypoints, scores=scores))
        return items

    def _build_bboxes(self, image: np.ndarray) -> list[list[float]]:
        h, w = image.shape[:2]
        if self.bbox_mode == "detector":
            raise NotImplementedError("Detector-based MMPose bboxes are not wired yet. Use whole_image or split_two.")
        if self.bbox_mode == "split_two":
            return [
                [0.0, 0.0, float(w) * 0.5, float(h)],
                [float(w) * 0.5, 0.0, float(w), float(h)],
            ]
        return [[0.0, 0.0, float(w), float(h)]]
