from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any

import numpy as np

from src.models.mmpose_backend import MMPoseInferenceItem

COURT_WIDTH = 610.0
COURT_LENGTH = 1340.0


@dataclass(slots=True)
class YoloPoseBackend:
    model_weight: str | None
    device: str
    conf_thr: float = 0.3
    max_persons: int = 2
    imgsz: int | None = None
    crop_pose: bool = False
    crop_imgsz: int | None = None
    crop_padding: float = 0.25
    crop_min_box_conf: float = 0.45
    crop_refine_score_thr: float = 0.65
    crop_refine_min_strong_keypoints: int = 10
    max_pose_crops: int | None = None
    court_filter: bool = False
    court_required: bool = False
    court_margin: float = 30.0
    person_model_weight: str | None = None
    model: Any = field(init=False, repr=False)
    person_model: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.model_weight:
            raise ValueError("YOLO pose backend requires model_weight, e.g. assets/weights/pose/yolo26s-pose.pt")
        if not Path(self.model_weight).exists():
            raise FileNotFoundError(f"YOLO pose weight file not found: {self.model_weight}")
        self.model = self._load_model(self.model_weight)
        self.person_model = self._load_person_model()

    def _load_model(self, model_weight: str) -> Any:
        self._configure_ultralytics()
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Ultralytics is required for YOLO pose inference. Install it with `pip install ultralytics`."
            ) from exc
        self._patch_pose26_head()
        return YOLO(model_weight)

    def _load_person_model(self) -> Any:
        if not self.person_model_weight:
            return self.model
        person_path = Path(self.person_model_weight)
        if not person_path.exists():
            raise FileNotFoundError(f"YOLO person detector weight file not found: {self.person_model_weight}")
        return self._load_model(str(person_path))

    def _configure_ultralytics(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        config_dir = project_root / ".ultralytics"
        config_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))

    def _patch_pose26_head(self) -> None:
        try:
            from ultralytics.nn.modules import head  # type: ignore
            import torch
        except Exception:
            return
        if not hasattr(head, "Pose26") and hasattr(head, "Pose"):
            class Pose26(head.Pose):
                def forward(self, x):
                    bs = x[0].shape[0]
                    kpt_features = [self.cv4[i](x[i]) for i in range(self.nl)]
                    kpt = torch.cat(
                        [self.cv4_kpts[i](kpt_features[i]).view(bs, self.nk, -1) for i in range(self.nl)],
                        -1,
                    )
                    detections = head.Detect.forward(self, x)
                    if self.training:
                        return detections, kpt
                    pred_kpt = self.kpts_decode(bs, kpt)
                    if self.export:
                        return torch.cat([detections, pred_kpt], 1)
                    return torch.cat([detections[0], pred_kpt], 1), (detections[1], kpt)

            head.Pose26 = Pose26
        if not hasattr(head, "RealNVP"):
            class RealNVP(torch.nn.Module):
                def forward(self, x, *args, **kwargs):
                    return x

                def inverse(self, x, *args, **kwargs):
                    return x

            head.RealNVP = RealNVP

    def infer(
        self,
        image: np.ndarray,
        court_prediction: object | None = None,
    ) -> list[MMPoseInferenceItem]:
        detect_conf = self.crop_min_box_conf if self.crop_pose else self.conf_thr
        result = self._predict_person(image, imgsz=self.imgsz, conf=detect_conf)

        boxes = self._boxes(result)
        box_scores = self._box_scores(result, len(boxes))
        keypoints, scores = self._keypoints(result)
        if not boxes:
            return []

        keep_indices = self._court_box_indices(boxes, court_prediction)
        if keep_indices is None:
            if self.court_filter and self.court_required:
                return []
        else:
            boxes = [boxes[index] for index in keep_indices]
            box_scores = [box_scores[index] for index in keep_indices]
            keypoints = [keypoints[index] if index < len(keypoints) else [] for index in keep_indices]
            scores = [scores[index] if index < len(scores) else [] for index in keep_indices]
            if not boxes:
                return []

        if self.crop_pose:
            crop_items = self._infer_crop_poses(image, boxes, box_scores, keypoints, scores)
            if crop_items:
                return crop_items

        if not keypoints:
            return []
        return self._items_from_candidates(boxes, box_scores, keypoints, scores, self.max_persons)

    def _predict(self, image: np.ndarray, *, imgsz: int | None, conf: float) -> Any:
        return self._predict_model(self.model, image, imgsz=imgsz, conf=conf)

    def _predict_person(self, image: np.ndarray, *, imgsz: int | None, conf: float) -> Any:
        return self._predict_model(self.person_model, image, imgsz=imgsz, conf=conf, classes=[0])

    def _predict_model(
        self,
        model: Any,
        image: np.ndarray,
        *,
        imgsz: int | None,
        conf: float,
        classes: list[int] | None = None,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "device": self._ultralytics_device(),
            "conf": conf,
            "verbose": False,
        }
        if imgsz is not None and int(imgsz) > 0:
            kwargs["imgsz"] = int(imgsz)
        if classes is not None:
            kwargs["classes"] = classes
        return model.predict(image, **kwargs)[0]

    def _court_box_indices(
        self,
        boxes: list[list[float]],
        court_prediction: object | None,
    ) -> list[int] | None:
        if not self.court_filter:
            return None
        image_to_court_h = extract_image_to_court_h(court_prediction)
        if image_to_court_h is None:
            return None
        return filter_boxes_by_court(boxes, image_to_court_h, self.court_margin)

    def _items_from_candidates(
        self,
        boxes: list[list[float]],
        box_scores: list[float],
        keypoints: list[list[list[float]]],
        scores: list[list[float]],
        limit: int,
    ) -> list[MMPoseInferenceItem]:
        items: list[MMPoseInferenceItem] = []
        count = min(len(boxes), len(keypoints), len(scores), len(box_scores))
        order = sorted(
            range(count),
            key=lambda idx: _candidate_rank(boxes[idx], box_scores[idx], scores[idx]),
            reverse=True,
        )
        for idx in order[: max(0, int(limit))]:
            if idx >= len(keypoints):
                continue
            items.append(
                MMPoseInferenceItem(
                    bbox=boxes[idx],
                    keypoints=keypoints[idx],
                    scores=scores[idx],
                    coordinate_space="original",
                )
            )
        return items

    def _infer_crop_poses(
        self,
        image: np.ndarray,
        boxes: list[list[float]],
        box_scores: list[float],
        keypoints: list[list[list[float]]],
        scores: list[list[float]],
    ) -> list[MMPoseInferenceItem]:
        frame_shape = image.shape
        count = min(len(boxes), len(box_scores))
        if count <= 0:
            return []

        crop_limit = self.max_persons if self.max_pose_crops is None else self.max_pose_crops
        candidate_indices = [
            idx
            for idx in range(count)
            if float(box_scores[idx]) >= float(self.crop_min_box_conf)
        ]
        candidate_indices.sort(
            key=lambda idx: _candidate_rank(
                boxes[idx],
                box_scores[idx],
                scores[idx] if idx < len(scores) else [],
            ),
            reverse=True,
        )

        items: list[MMPoseInferenceItem] = []
        for idx in candidate_indices[: max(0, int(crop_limit))]:
            fallback_item = self._original_item(boxes, keypoints, scores, idx)
            keypoint_scores = scores[idx] if idx < len(scores) else []
            if fallback_item is not None and not needs_crop_refine(
                keypoint_scores,
                self.crop_refine_score_thr,
                self.crop_refine_min_strong_keypoints,
            ):
                items.append(fallback_item)
                continue

            crop_rect = expanded_crop_rect(boxes[idx], frame_shape, self.crop_padding)
            if crop_rect is None:
                if fallback_item is not None:
                    items.append(fallback_item)
                continue

            x1, y1, x2, y2 = crop_rect
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                if fallback_item is not None:
                    items.append(fallback_item)
                continue

            try:
                crop_result = self._predict(crop, imgsz=self.crop_imgsz, conf=max(0.05, self.conf_thr * 0.75))
            except Exception:
                if fallback_item is not None:
                    items.append(fallback_item)
                continue

            crop_boxes = self._boxes(crop_result)
            crop_box_scores = self._box_scores(crop_result, len(crop_boxes))
            crop_keypoints, crop_scores = self._keypoints(crop_result)
            crop_item = self._best_crop_item(
                crop_boxes,
                crop_box_scores,
                crop_keypoints,
                crop_scores,
                crop.shape,
            )
            if crop_item is None:
                if fallback_item is not None:
                    items.append(fallback_item)
                continue

            items.append(translate_item(crop_item, float(x1), float(y1)))

        return items[: self.max_persons]

    def _original_item(
        self,
        boxes: list[list[float]],
        keypoints: list[list[list[float]]],
        scores: list[list[float]],
        idx: int,
    ) -> MMPoseInferenceItem | None:
        if idx >= len(boxes) or idx >= len(keypoints) or idx >= len(scores):
            return None
        if not keypoints[idx] or not scores[idx]:
            return None
        return MMPoseInferenceItem(
            bbox=boxes[idx],
            keypoints=keypoints[idx],
            scores=scores[idx],
            coordinate_space="original",
        )

    def _best_crop_item(
        self,
        boxes: list[list[float]],
        box_scores: list[float],
        keypoints: list[list[list[float]]],
        scores: list[list[float]],
        crop_shape: tuple[int, ...],
    ) -> MMPoseInferenceItem | None:
        count = min(len(boxes), len(box_scores), len(keypoints), len(scores))
        if count <= 0:
            return None

        height, width = crop_shape[:2]
        crop_center = (float(width) * 0.5, float(height) * 0.5)
        order = sorted(
            range(count),
            key=lambda idx: (
                _candidate_rank(boxes[idx], box_scores[idx], scores[idx])
                - _center_distance(boxes[idx], crop_center) / max(float(width), float(height), 1.0)
            ),
            reverse=True,
        )
        idx = order[0]
        return MMPoseInferenceItem(
            bbox=boxes[idx],
            keypoints=keypoints[idx],
            scores=scores[idx],
            coordinate_space="original",
        )

    def _ultralytics_device(self) -> str:
        if self.device == "cpu":
            return "cpu"
        if self.device.startswith("cuda:"):
            return self.device.split(":", 1)[1]
        if self.device.startswith("cuda"):
            return "0"
        return self.device

    def _boxes(self, result: Any) -> list[list[float]]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or getattr(boxes, "xyxy", None) is None:
            return []
        xyxy = boxes.xyxy.detach().cpu().numpy()
        return [[float(v) for v in row[:4]] for row in xyxy]

    def _box_scores(self, result: Any, count: int) -> list[float]:
        boxes = getattr(result, "boxes", None)
        conf = getattr(boxes, "conf", None) if boxes is not None else None
        if conf is None:
            return [1.0] * max(0, int(count))
        values = conf.detach().cpu().numpy().reshape(-1)
        output = [float(value) for value in values[:count]]
        if len(output) < count:
            output.extend([1.0] * (count - len(output)))
        return output

    def _keypoints(self, result: Any) -> tuple[list[list[list[float]]], list[list[float]]]:
        kpts = getattr(result, "keypoints", None)
        if kpts is None or getattr(kpts, "xy", None) is None:
            return [], []

        xy = kpts.xy.detach().cpu().numpy()
        conf = getattr(kpts, "conf", None)
        if conf is not None:
            score_array = conf.detach().cpu().numpy()
        else:
            score_array = np.ones(xy.shape[:2], dtype=np.float32)

        keypoints = [
            [[float(x), float(y)] for x, y in person]
            for person in xy
        ]
        scores = [
            [float(score) for score in person_scores]
            for person_scores in score_array
        ]
        return keypoints, scores


def _box_area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _candidate_rank(box: list[float], box_score: float, keypoint_scores: list[float]) -> float:
    keypoint_score = float(sum(keypoint_scores) / max(len(keypoint_scores), 1)) if keypoint_scores else 0.0
    area_bias = min(_box_area(box), 250000.0) / 250000.0
    return float(box_score) * 1.5 + keypoint_score + area_bias * 0.05


def needs_crop_refine(
    keypoint_scores: list[float],
    score_thr: float,
    min_strong_keypoints: int,
) -> bool:
    if not keypoint_scores:
        return True
    finite_scores = [float(score) for score in keypoint_scores if np.isfinite(score)]
    if not finite_scores:
        return True
    mean_score = sum(finite_scores) / len(finite_scores)
    strong_count = sum(1 for score in finite_scores if score >= float(score_thr))
    return mean_score < float(score_thr) or strong_count < int(min_strong_keypoints)


def expanded_crop_rect(
    box: list[float],
    frame_shape: tuple[int, ...],
    padding: float,
) -> tuple[int, int, int, int] | None:
    if len(frame_shape) < 2 or len(box) < 4:
        return None
    height, width = int(frame_shape[0]), int(frame_shape[1])
    if height <= 0 or width <= 0:
        return None

    x1, y1, x2, y2 = [float(value) for value in box[:4]]
    if not np.isfinite([x1, y1, x2, y2]).all() or x2 <= x1 or y2 <= y1:
        return None

    pad = max(0.0, float(padding))
    box_w = x2 - x1
    box_h = y2 - y1
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    crop_w = box_w * (1.0 + pad * 2.0)
    crop_h = box_h * (1.0 + pad * 2.0)
    side = max(crop_w, crop_h)
    crop_w = max(crop_w, side * 0.55)
    crop_h = max(crop_h, side)

    left = max(0, int(np.floor(cx - crop_w * 0.5)))
    top = max(0, int(np.floor(cy - crop_h * 0.5)))
    right = min(width, int(np.ceil(cx + crop_w * 0.5)))
    bottom = min(height, int(np.ceil(cy + crop_h * 0.5)))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def filter_boxes_by_court(
    boxes: list[list[float]],
    image_to_court_h: np.ndarray,
    margin: float,
) -> list[int]:
    keep: list[int] = []
    for index, box in enumerate(boxes):
        anchor = box_anchor_point(box)
        if anchor is None:
            continue
        court_xy = project_image_point(image_to_court_h, anchor)
        if court_xy is None:
            continue
        if point_in_court(court_xy, margin):
            keep.append(index)
    return keep


def extract_image_to_court_h(court_prediction: object | None) -> np.ndarray | None:
    if court_prediction is None:
        return None
    valid = prediction_value(court_prediction, "valid", False)
    if not valid:
        return None
    raw_h = prediction_value(court_prediction, "image_to_court_h", None)
    if raw_h is None:
        return None
    try:
        h = np.asarray(raw_h, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if h.shape != (3, 3) or not np.isfinite(h).all():
        return None
    return h


def prediction_value(prediction: object, key: str, default: object) -> object:
    if isinstance(prediction, dict):
        return prediction.get(key, default)
    return getattr(prediction, key, default)


def box_anchor_point(box: list[float]) -> tuple[float, float] | None:
    if len(box) < 4:
        return None
    x1, y1, x2, y2 = [float(value) for value in box[:4]]
    if not np.isfinite([x1, y1, x2, y2]).all() or x2 <= x1 or y2 <= y1:
        return None
    return (x1 + x2) * 0.5, y2


def project_image_point(
    image_to_court_h: np.ndarray,
    point: tuple[float, float],
) -> tuple[float, float] | None:
    image_point = np.array([point[0], point[1], 1.0], dtype=np.float64)
    court_point = image_to_court_h @ image_point
    denom = float(court_point[2])
    if abs(denom) < 1e-9:
        return None
    x = float(court_point[0] / denom)
    y = float(court_point[1] / denom)
    if not np.isfinite([x, y]).all():
        return None
    return x, y


def point_in_court(court_xy: tuple[float, float], margin: float) -> bool:
    x, y = court_xy
    allowed_margin = max(0.0, float(margin))
    if x < -allowed_margin or x > COURT_WIDTH + allowed_margin:
        return False
    if y < -allowed_margin or y > COURT_LENGTH + allowed_margin:
        return False
    return True


def translate_item(item: MMPoseInferenceItem, offset_x: float, offset_y: float) -> MMPoseInferenceItem:
    return MMPoseInferenceItem(
        bbox=[
            float(item.bbox[0]) + offset_x,
            float(item.bbox[1]) + offset_y,
            float(item.bbox[2]) + offset_x,
            float(item.bbox[3]) + offset_y,
        ],
        keypoints=[
            [float(point[0]) + offset_x, float(point[1]) + offset_y]
            for point in item.keypoints
        ],
        scores=list(item.scores),
        coordinate_space=item.coordinate_space,
    )


def _center_distance(box: list[float], center: tuple[float, float]) -> float:
    box_center_x = (float(box[0]) + float(box[2])) * 0.5
    box_center_y = (float(box[1]) + float(box[3])) * 0.5
    return float(np.hypot(box_center_x - center[0], box_center_y - center[1]))
