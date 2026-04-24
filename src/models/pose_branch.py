from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.models.mmpose_backend import MMPoseBackend
from src.postprocess.pose import build_pose_result
from src.preprocess.pose import preprocess_pose_frame
from src.utils.device import resolve_device
from src.utils.structures import PersonPoseResult


@dataclass
class PoseBranch:
    backend: str = "mmpose"
    device: str = "cpu"
    model_config: str | None = None
    model_weight: str | None = None
    det_config: str | None = None
    det_weight: str | None = None
    bbox_mode: str = "whole_image"
    input_size: tuple[int, int] = (192, 256)
    conf_thr: float = 0.3
    max_persons: int = 2

    def __post_init__(self) -> None:
        self.device = resolve_device(self.device)
        backend = self.backend.strip().lower()
        if backend not in {"mmpose", "dummy"}:
            raise ValueError(f"Unsupported pose backend: {self.backend!r}. Supported values are 'mmpose' and 'dummy'.")
        self.backend_impl = MMPoseBackend(
            model_config=self.model_config,
            model_weight=self.model_weight,
            device=self.device,
            bbox_mode=self.bbox_mode,
            det_config=self.det_config,
            det_weight=self.det_weight,
            conf_thr=self.conf_thr,
            max_persons=self.max_persons,
            allow_dummy=backend == "dummy",
        )

    def infer(self, image: np.ndarray) -> list[PersonPoseResult]:
        _, meta = preprocess_pose_frame(image, self.input_size, self.device)
        raw_items = self.backend_impl.infer(image)
        outputs: list[PersonPoseResult] = []
        for idx, item in enumerate(raw_items):
            outputs.append(
                build_pose_result(
                    person_id=idx,
                    bbox=item.bbox,
                    keypoints=item.keypoints,
                    scores=item.scores,
                    meta=meta if item.coordinate_space == "resized" else None,
                )
            )
        return outputs
