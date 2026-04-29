from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.models.mmpose_backend import MMPoseBackend
from src.models.yolo_pose_backend import YoloPoseBackend
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
    yolo_imgsz: int | None = None
    yolo_crop_pose: bool = False
    yolo_crop_imgsz: int | None = None
    yolo_crop_padding: float = 0.25
    yolo_crop_min_box_conf: float = 0.45
    yolo_max_pose_crops: int | None = None
    yolo_court_filter: bool = False
    yolo_court_required: bool = False
    yolo_court_margin: float = 30.0
    yolo_person_model_weight: str | None = None

    def __post_init__(self) -> None:
        self.device = resolve_device(self.device)
        self.backend_name = self.backend.strip().lower().replace("_", "-")
        if self.backend_name in {"mmpose", "dummy"}:
            self.backend_impl = MMPoseBackend(
                model_config=self.model_config,
                model_weight=self.model_weight,
                device=self.device,
                bbox_mode=self.bbox_mode,
                det_config=self.det_config,
                det_weight=self.det_weight,
                conf_thr=self.conf_thr,
                max_persons=self.max_persons,
                allow_dummy=self.backend_name == "dummy",
            )
            return

        if self.backend_name in {"yolo26s-pose", "yolo-pose", "ultralytics", "ultralytics-pose"}:
            self.backend_impl = YoloPoseBackend(
                model_weight=self.model_weight,
                device=self.device,
                conf_thr=self.conf_thr,
                max_persons=self.max_persons,
                imgsz=self.yolo_imgsz,
                crop_pose=self.yolo_crop_pose,
                crop_imgsz=self.yolo_crop_imgsz,
                crop_padding=self.yolo_crop_padding,
                crop_min_box_conf=self.yolo_crop_min_box_conf,
                max_pose_crops=self.yolo_max_pose_crops,
                court_filter=self.yolo_court_filter,
                court_required=self.yolo_court_required,
                court_margin=self.yolo_court_margin,
                person_model_weight=self.yolo_person_model_weight,
            )
            return

        raise ValueError(
            f"Unsupported pose backend: {self.backend!r}. "
            "Supported values are 'mmpose', 'dummy', and 'yolo26s-pose'."
        )

    def infer(self, image: np.ndarray, court_prediction: object | None = None) -> list[PersonPoseResult]:
        raw_items = self.backend_impl.infer(image, court_prediction=court_prediction)
        meta = None
        if any(item.coordinate_space == "resized" for item in raw_items):
            _, meta = preprocess_pose_frame(image, self.input_size, self.device)
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
