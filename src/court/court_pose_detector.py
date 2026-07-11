from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from src.court import opencv_court_homography_core as _court_core
from src.court.monotrack_court_detector import MonoTrackCourtLineConfig, detect_monotrack_court_lines
from src.court.shuttlecourt_seg_detector import (
    PROJECT_ROOT,
    ShuttleCourtSegConfig,
    ShuttleCourtSegLineDetector,
    _detection_from_quad,
    _score_segmentation_candidate,
    _to_numpy,
)


DEFAULT_COURT_POSE_WEIGHTS = PROJECT_ROOT / "assets" / "weights" / "court_pose" / "best.pt"


@dataclass(slots=True)
class CourtPoseConfig(ShuttleCourtSegConfig):
    weights: str = "assets/weights/court_pose/best.pt"
    imgsz: int = 512
    max_det: int = 3
    seg_roi_dilate_px: int = 28
    seg_line_min_area_ratio: float = 0.45
    snap_search_px: float = 72.0
    max_refine_corner_shift_ratio: float = 0.08
    min_pose_keypoint_conf: float = 0.10
    min_pose_line_support: float = 0.08
    min_pose_snap_points: int = 10
    max_pose_refine_shift_ratio: float = 0.10
    coarse_confidence_scale: float = 0.62
    corner_snap: bool = True
    corner_snap_radius: int = 130
    corner_snap_max_shift: int = 110
    corner_snap_min_line_length: int = 28
    corner_snap_max_gap: int = 14
    corner_snap_hough_threshold: int = 12
    corner_snap_angle_tol: float = 38.0
    corner_snap_min_angle_separation: float = 22.0
    corner_snap_nearest_white_radius: int = 8
    corner_snap_min_white_support: float = 0.035
    corner_snap_strong_prior_support: float = 0.28
    corner_snap_max_strong_prior_shift: int = 18
    corner_snap_edge_band: int = 110
    corner_snap_edge_extension: int = 180
    corner_snap_cross_search_radius: int = 72
    corner_snap_ray_length: int = 64
    corner_snap_min_ray_support: float = 0.16
    corner_snap_min_centerline_score: float = 0.20
    min_pose_corner_snaps: int = 2
    max_pose_corner_snap_shift_ratio: float = 0.07


class CourtPoseLineDetector(ShuttleCourtSegLineDetector):
    """Use YOLO Pose as a court ROI prior, then fit the actual white court lines."""

    def __init__(self, config: CourtPoseConfig | None = None, *, model: Any | None = None) -> None:
        super().__init__(config or CourtPoseConfig(), model=model)
        self.config: CourtPoseConfig
        self._monotrack_args = SimpleNamespace(**asdict(MonoTrackCourtLineConfig()))

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model

        weights = resolve_court_pose_weights(self.config.weights)
        config_dir = PROJECT_ROOT / ".ultralytics"
        config_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Missing dependency: ultralytics.") from exc

        self._weights_path = weights
        self._model = YOLO(str(weights))
        return self._model

    def _detect(
        self,
        frame: np.ndarray,
        *,
        previous: _court_core.CourtLineDetection | None,
    ) -> _court_core.CourtLineDetection | None:
        model = self._ensure_model()
        results = model.predict(
            frame,
            imgsz=int(self.config.imgsz),
            conf=float(self.config.conf),
            iou=float(self.config.iou),
            max_det=max(1, int(self.config.max_det)),
            device=self._device,
            verbose=False,
        )
        if not results:
            return None

        result = results[0]
        keypoints = getattr(result, "keypoints", None)
        if keypoints is None:
            return None
        all_points = _to_numpy(getattr(keypoints, "xy", None))
        if all_points.ndim != 3 or all_points.shape[1] < 4:
            return None

        all_keypoint_conf = _to_numpy(getattr(keypoints, "conf", None))
        boxes = getattr(result, "boxes", None)
        box_confidences = _to_numpy(getattr(boxes, "conf", None)) if boxes is not None else np.empty((0,))
        frame_area = float(max(frame.shape[0] * frame.shape[1], 1))
        line_mask, green_mask = _court_core.create_white_line_mask(frame, self._args)

        best: _court_core.CourtLineDetection | None = None
        best_rank = -1.0
        for index, raw_points in enumerate(all_points):
            raw_quad = _court_core.order_quad_points(np.asarray(raw_points[:4], dtype=np.float32))
            if raw_quad is None or not np.isfinite(raw_quad).all():
                continue
            area = _court_core.polygon_area(raw_quad)
            area_ratio = area / frame_area
            if area_ratio < max(0.0, float(self.config.min_mask_area_ratio)):
                continue

            box_confidence = float(box_confidences[index]) if index < len(box_confidences) else 1.0
            keypoint_scores = (
                np.asarray(all_keypoint_conf[index][:4], dtype=np.float32)
                if all_keypoint_conf.ndim == 2 and index < len(all_keypoint_conf)
                else np.ones((4,), dtype=np.float32)
            )
            finite_scores = keypoint_scores[np.isfinite(keypoint_scores)]
            keypoint_confidence = float(finite_scores.mean()) if finite_scores.size else 0.0
            if finite_scores.size and float(finite_scores.min()) < float(self.config.min_pose_keypoint_conf):
                continue

            detection, corner_snap_count, corner_snap_shift = self._detection_from_monotrack(
                frame,
                raw_quad,
                previous=previous,
                box_confidence=box_confidence,
                area_ratio=area_ratio,
                line_mask=line_mask,
                green_mask=green_mask,
            )
            if detection is None:
                detection = _detection_from_quad(
                    quad=raw_quad.copy(),
                    confidence=box_confidence,
                    area_ratio=area_ratio,
                    polygon_points=4,
                    class_id=0,
                    rejected_masks=0,
                    line_mask=line_mask,
                    green_mask=green_mask,
                    args=self._args,
                )
                corner_snap_count = 0
                corner_snap_shift = 0.0
            if detection is None:
                continue

            mean_shift = float(np.mean(np.linalg.norm(detection.corners - raw_quad, axis=1)))
            max_shift = float(np.hypot(frame.shape[1], frame.shape[0])) * float(
                self.config.max_pose_refine_shift_ratio
            )
            if mean_shift > max_shift:
                detection = self._coarse_detection(
                    raw_quad,
                    box_confidence=box_confidence,
                    area_ratio=area_ratio,
                    line_mask=line_mask,
                    green_mask=green_mask,
                )
                if detection is None:
                    continue
                mean_shift = 0.0
                corner_snap_count = 0
                corner_snap_shift = 0.0

            homography_refined = bool(detection.components.get("refine_accepted", 0.0))
            monotrack_fused = bool(detection.components.get("pose_monotrack_fused", 0.0))
            white_line_refined = monotrack_fused or (
                homography_refined
                and detection.mask_support >= float(self.config.min_pose_line_support)
                and detection.snap_points >= int(self.config.min_pose_snap_points)
            ) or corner_snap_count >= int(self.config.min_pose_corner_snaps)
            rank, fused_confidence, components, reason = _score_segmentation_candidate(
                quad=detection.corners,
                frame_shape=frame.shape,
                area_ratio=area_ratio,
                box_confidence=box_confidence,
                previous=previous,
                line_support=detection.mask_support,
                green_side_support=detection.green_side_support,
                snap_points=detection.snap_points,
                args=self._args,
            )
            if white_line_refined:
                fused_confidence = max(fused_confidence, 0.65 * detection.confidence + 0.35 * fused_confidence)
                rank += 0.20 * detection.confidence
                detection.scheme = "court_pose_white_line"
                detection.reason = f"pose ROI white-line refinement: {reason}"
            else:
                fused_confidence *= float(self.config.coarse_confidence_scale)
                rank *= float(self.config.coarse_confidence_scale)
                detection.scheme = "court_pose_coarse"
                detection.reason = f"pose coarse fallback; insufficient white-line evidence: {reason}"

            detection.confidence = float(np.clip(fused_confidence, 0.0, 1.0))
            detection.components.update(components)
            detection.components.update(
                {
                    "pose_candidate_index": float(index),
                    "pose_box_confidence": float(np.clip(box_confidence, 0.0, 1.0)),
                    "pose_keypoint_confidence": float(np.clip(keypoint_confidence, 0.0, 1.0)),
                    "pose_white_line_refined": 1.0 if white_line_refined else 0.0,
                    "pose_refine_mean_shift_px": mean_shift,
                    "pose_corner_snap_count": float(corner_snap_count),
                    "pose_corner_snap_mean_shift_px": float(corner_snap_shift),
                }
            )
            if rank > best_rank:
                best_rank = rank
                best = detection
        return best

    def _detection_from_monotrack(
        self,
        frame: np.ndarray,
        raw_quad: np.ndarray,
        *,
        previous: _court_core.CourtLineDetection | None,
        box_confidence: float,
        area_ratio: float,
        line_mask: np.ndarray,
        green_mask: np.ndarray,
    ) -> tuple[_court_core.CourtLineDetection | None, int, float]:
        monotrack = detect_monotrack_court_lines(frame, previous, self._monotrack_args)
        if monotrack is None:
            return None, 0, 0.0

        outer_quad, flags = _court_core.snap_corners_by_outer_edge_fits(
            line_mask,
            monotrack.corners,
            self._args,
            prefer_outer_cluster=True,
        )
        fused_quad, selected_flags = _select_pose_consistent_outer_corners(
            monotrack.corners,
            outer_quad,
            flags,
            raw_quad,
        )
        fused_quad = _complete_outer_sides_from_vanishing_point(monotrack.corners, fused_quad, selected_flags)
        ordered = _court_core.order_quad_points(fused_quad)
        if ordered is None:
            ordered = monotrack.corners.copy()
            selected_flags = [False, False, False, False]

        snap_count = int(sum(selected_flags))
        mean_shift = float(np.mean(np.linalg.norm(ordered - monotrack.corners, axis=1)))
        max_shift = float(np.hypot(frame.shape[1], frame.shape[0])) * float(self.config.max_pose_corner_snap_shift_ratio)
        if mean_shift > max_shift:
            ordered = monotrack.corners.copy()
            snap_count = 0
            mean_shift = 0.0

        fused = self._coarse_detection(
            ordered,
            box_confidence=box_confidence,
            area_ratio=area_ratio,
            line_mask=line_mask,
            green_mask=green_mask,
        )
        if fused is None:
            return monotrack, 0, 0.0
        if fused.mask_support + 0.03 < monotrack.mask_support * 0.70:
            return monotrack, 0, 0.0

        fused.confidence = float(np.clip(0.70 * monotrack.confidence + 0.30 * box_confidence, 0.0, 1.0))
        fused.line_count = monotrack.line_count
        fused.merged_line_count = monotrack.merged_line_count
        fused.intersection_count = monotrack.intersection_count
        fused.debug_segments = monotrack.debug_segments
        fused.debug_merged_lines = monotrack.debug_merged_lines
        fused.snap_points = max(monotrack.snap_points, snap_count)
        fused.snap_mean_shift = mean_shift
        fused.components.update(monotrack.components)
        fused.components.update(
            {
                "pose_monotrack_fused": 1.0,
                "pose_outer_corner_count": float(snap_count),
                "pose_outer_mean_shift_px": mean_shift,
            }
        )
        fused.scheme = "court_pose_white_line"
        fused.reason = "pose-constrained MonoTrack white-line fit"
        return fused, snap_count, mean_shift

    def _coarse_detection(
        self,
        raw_quad: np.ndarray,
        *,
        box_confidence: float,
        area_ratio: float,
        line_mask: np.ndarray,
        green_mask: np.ndarray,
    ) -> _court_core.CourtLineDetection | None:
        coarse_args = SimpleNamespace(**vars(self._args))
        coarse_args.refine_homography = False
        return _detection_from_quad(
            quad=raw_quad.copy(),
            confidence=box_confidence,
            area_ratio=area_ratio,
            polygon_points=4,
            class_id=0,
            rejected_masks=0,
            line_mask=line_mask,
            green_mask=green_mask,
            args=coarse_args,
        )

    def _build_prediction(self, **kwargs: Any):
        prediction = super()._build_prediction(**kwargs)
        if prediction.scheme == "shuttlecourt_seg":
            prediction.scheme = "court_pose"
        prediction.status = prediction.status.replace("segmentation", "pose")
        return prediction


def _select_pose_consistent_outer_corners(
    base_quad: np.ndarray,
    outer_quad: np.ndarray,
    outer_flags: list[bool],
    pose_quad: np.ndarray,
) -> tuple[np.ndarray, list[bool]]:
    selected = np.asarray(base_quad, dtype=np.float32).copy()
    selected_flags = [False, False, False, False]
    for index, available in enumerate(outer_flags):
        if not available:
            continue
        base_distance = float(np.linalg.norm(base_quad[index] - pose_quad[index]))
        outer_distance = float(np.linalg.norm(outer_quad[index] - pose_quad[index]))
        if outer_distance + 4.0 < base_distance:
            selected[index] = outer_quad[index]
            selected_flags[index] = True
    return selected, selected_flags


def _complete_outer_sides_from_vanishing_point(
    base_quad: np.ndarray,
    selected_quad: np.ndarray,
    selected_flags: list[bool],
) -> np.ndarray:
    base = np.asarray(base_quad, dtype=np.float64)
    completed = np.asarray(selected_quad, dtype=np.float64).copy()
    vertical_vanishing = _line_intersection(_line(base[0], base[3]), _line(base[1], base[2]))
    if vertical_vanishing is None:
        return completed.astype(np.float32)

    top_line = _line(completed[0], completed[1])
    bottom_line = _line(completed[3], completed[2])
    side_length = max(
        float(np.linalg.norm(base[3] - base[0])),
        float(np.linalg.norm(base[2] - base[1])),
        1.0,
    )
    max_completion_shift = side_length * 0.22
    for top_index, bottom_index in ((0, 3), (1, 2)):
        if selected_flags[bottom_index] and not selected_flags[top_index]:
            candidate = _line_intersection(_line(completed[bottom_index], vertical_vanishing), top_line)
            if candidate is not None and float(np.linalg.norm(candidate - base[top_index])) <= max_completion_shift:
                completed[top_index] = candidate
                selected_flags[top_index] = True
        elif selected_flags[top_index] and not selected_flags[bottom_index]:
            candidate = _line_intersection(_line(completed[top_index], vertical_vanishing), bottom_line)
            if candidate is not None and float(np.linalg.norm(candidate - base[bottom_index])) <= max_completion_shift:
                completed[bottom_index] = candidate
                selected_flags[bottom_index] = True
    return completed.astype(np.float32)


def _line(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.cross(
        np.asarray([float(first[0]), float(first[1]), 1.0], dtype=np.float64),
        np.asarray([float(second[0]), float(second[1]), 1.0], dtype=np.float64),
    )


def _line_intersection(first: np.ndarray, second: np.ndarray) -> np.ndarray | None:
    point = np.cross(first, second)
    if abs(float(point[2])) < 1e-8 or not np.isfinite(point).all():
        return None
    result = point[:2] / point[2]
    return result if np.isfinite(result).all() else None


def resolve_court_pose_weights(raw: str | Path) -> Path:
    requested = Path(raw).expanduser()
    candidates = [requested] if requested.is_absolute() else [PROJECT_ROOT / requested]
    candidates.extend(
        [
            DEFAULT_COURT_POSE_WEIGHTS,
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("Could not find court pose weights. Searched: " + "; ".join(str(path) for path in candidates))
