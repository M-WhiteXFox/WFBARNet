from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np

from src.court import opencv_court_homography_core as _court_core
from src.court.opencv_court_detector import CourtLinePrediction


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(slots=True)
class ShuttleCourtSegConfig:
    weights: str = "weights/shttlecourtnet"
    device: str = "auto"
    imgsz: int = 416
    conf: float = 0.25
    iou: float = 0.70
    max_det: int = 3
    retina_masks: bool = True
    redetect_interval: float = 4.0
    reliable_conf: float = 0.75
    medium_conf: float = 0.55
    smooth_alpha_reliable: float = 0.45
    smooth_alpha_medium: float = 0.20
    min_mask_area_ratio: float = 0.025
    small_candidate_area_ratio: float = 0.12
    small_candidate_min_line_support: float = 0.04
    approx_epsilon_ratio: float = 0.02
    seg_roi_dilate_px: int = 18
    seg_line_min_area_ratio: float = 0.45
    white_s_max: int = 130
    white_v_min: int = 120
    white_chroma_max: int = 96
    line_response_percentile: float = 91.0
    line_response_min: int = 72
    line_local_bg_ksize: int = 31
    use_green_roi: bool = True
    green_h_min: int = 30
    green_h_max: int = 100
    green_s_min: int = 70
    green_v_min: int = 35
    white_green_pair_offset_px: int = 8
    keep_all_green_rois: bool = False
    detect_max_width: int = 960
    hough_threshold: int = 45
    min_line_length_ratio: float = 0.055
    max_line_gap_ratio: float = 0.025
    angle_bin_deg: float = 5.0
    angle_tol_deg: float = 16.0
    min_angle_separation_deg: float = 25.0
    merge_rho_px: float = 18.0
    max_lines_per_family: int = 3
    point_scheme: str = "auto"
    refine_homography: bool = True
    snap_search_px: float = 18.0
    snap_response_threshold: float = 0.18
    max_refine_corner_shift_ratio: float = 0.025
    green_side_offset_px: float = 14.0
    min_outer_width_ratio: float = 0.08
    min_outer_depth_ratio: float = 0.08
    min_outer_width_depth_ratio: float = 0.18
    max_outer_width_depth_ratio: float = 5.5
    max_transverse_angle_deg: float = 35.0
    jump_ratio_hard: float = 0.18


class ShuttleCourtSegLineDetector:
    def __init__(self, config: ShuttleCourtSegConfig | None = None, *, model: Any | None = None) -> None:
        self.config = config or ShuttleCourtSegConfig()
        self._args = SimpleNamespace(**asdict(self.config))
        self._state = _court_core.TrackingState()
        self._latest_prediction: CourtLinePrediction | None = None
        self._model = model
        self._weights_path: Path | None = None
        self._device = _resolve_device(self.config.device)

    def reset(self) -> None:
        self._state = _court_core.TrackingState()
        self._latest_prediction = None

    def predict(
        self,
        frame: np.ndarray,
        frame_id: int,
        timestamp_ms: int,
        *,
        force: bool = False,
    ) -> CourtLinePrediction:
        if frame is None or frame.ndim < 2:
            raise ValueError("ShuttleCourt segmentation prediction expects a BGR image frame.")

        timestamp_ms = max(0, int(timestamp_ms))
        timestamp_s = timestamp_ms / 1000.0
        should_attempt = bool(force) or _court_core.should_redetect(self._state, int(frame_id), timestamp_s, self._args)
        candidate = None
        detect_ms = 0.0

        if should_attempt:
            started_at = perf_counter()
            candidate = self._detect(frame, previous=None if force else self._state.current)
            detect_ms = (perf_counter() - started_at) * 1000.0
            _court_core.update_tracking_state(
                self._state,
                candidate,
                self._args,
                int(frame_id),
                timestamp_s,
            )
        else:
            candidate = self._state.last_candidate

        prediction = self._build_prediction(
            frame=frame,
            frame_id=int(frame_id),
            timestamp_ms=timestamp_ms,
            attempted=should_attempt,
            candidate=candidate,
            detect_ms=detect_ms,
        )
        self._latest_prediction = prediction
        return prediction

    def latest_prediction(self) -> CourtLinePrediction | None:
        return self._latest_prediction

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model

        weights = resolve_shuttlecourt_weights(self.config.weights)
        config_dir = PROJECT_ROOT / ".ultralytics"
        config_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Missing dependency: ultralytics. Install it with `python -m pip install ultralytics`.") from exc

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
            retina_masks=bool(self.config.retina_masks),
            verbose=False,
        )
        if not results:
            return None

        result = results[0]
        masks = getattr(result, "masks", None)
        if masks is None:
            return None

        polygons = list(getattr(masks, "xy", []) or [])
        if not polygons:
            return None

        boxes = getattr(result, "boxes", None)
        confidences = _to_numpy(getattr(boxes, "conf", None)) if boxes is not None else np.empty((0,), dtype=np.float32)
        classes = _to_numpy(getattr(boxes, "cls", None)) if boxes is not None else np.empty((0,), dtype=np.float32)
        frame_area = float(max(frame.shape[0] * frame.shape[1], 1))
        min_area = frame_area * max(0.0, float(self.config.min_mask_area_ratio))
        line_mask, green_mask = _court_core.create_white_line_mask(frame, self._args)

        best_detection: _court_core.CourtLineDetection | None = None
        best_rank = -1.0
        rejected = 0
        for index, raw_polygon in enumerate(polygons):
            polygon = np.asarray(raw_polygon, dtype=np.float32).reshape(-1, 2)
            if polygon.shape[0] < 3 or not np.isfinite(polygon).all():
                rejected += 1
                continue
            area = abs(float(cv2.contourArea(polygon.reshape(-1, 1, 2))))
            if area < min_area:
                rejected += 1
                continue
            confidence = float(confidences[index]) if index < len(confidences) else 1.0
            class_id = int(classes[index]) if index < len(classes) else 0
            detection = _detection_from_segmentation_lines(
                polygon=polygon,
                frame_shape=frame.shape,
                confidence=confidence,
                area_ratio=area / frame_area,
                polygon_points=len(polygon),
                class_id=class_id,
                rejected_masks=rejected,
                line_mask=line_mask,
                green_mask=green_mask,
                previous=previous,
                args=self._args,
            )
            if detection is None:
                quad = _quad_from_polygon(polygon, frame.shape, self._args)
                if quad is None:
                    rejected += 1
                    continue
                detection = _detection_from_quad(
                    quad=quad,
                    confidence=confidence,
                    area_ratio=area / frame_area,
                    polygon_points=len(polygon),
                    class_id=class_id,
                    rejected_masks=rejected,
                    line_mask=line_mask,
                    green_mask=green_mask,
                    args=self._args,
                )
            if detection is None:
                rejected += 1
                continue

            line_fit = bool(detection.components.get("seg_line_fit"))
            line_confidence = float(detection.confidence)
            rank, fused_confidence, components, reason = _score_segmentation_candidate(
                quad=detection.corners,
                frame_shape=frame.shape,
                area_ratio=area / frame_area,
                box_confidence=confidence,
                previous=previous,
                line_support=detection.mask_support,
                green_side_support=detection.green_side_support,
                snap_points=detection.snap_points,
                args=self._args,
            )
            if line_fit:
                fused_confidence = max(fused_confidence, 0.65 * line_confidence + 0.35 * fused_confidence)
                rank += 0.20 * line_confidence
                reason = f"segmentation ROI white-line fit: {reason}"
            detection.confidence = fused_confidence
            detection.reason = reason
            detection.components.update(components)
            detection.components["candidate_index"] = float(index)
            detection.components["candidate_rank"] = float(rank)
            if rank > best_rank:
                best_rank = rank
                best_detection = detection

        return best_detection

    def _build_prediction(
        self,
        *,
        frame: np.ndarray,
        frame_id: int,
        timestamp_ms: int,
        attempted: bool,
        candidate: Any,
        detect_ms: float,
    ) -> CourtLinePrediction:
        current = self._state.current
        valid = current is not None
        candidate_confidence = _clean_float(getattr(candidate, "confidence", None)) if candidate is not None else None
        updated = bool(attempted and self._state.last_update_type in {"reliable update", "medium startup", "medium smooth"})
        status = _status_text(attempted, self._state.last_update_type, valid, candidate)
        height, width = frame.shape[:2]

        if current is None:
            return CourtLinePrediction(
                frame_id=frame_id,
                timestamp_ms=timestamp_ms,
                source_size=(int(width), int(height)),
                valid=False,
                attempted=attempted,
                updated=False,
                update_type=self._state.last_update_type,
                status=status,
                confidence=0.0,
                candidate_confidence=candidate_confidence,
                reason=getattr(candidate, "reason", self._state.last_update_type) if candidate is not None else self._state.last_update_type,
                scheme="shuttlecourt_seg",
                corners=[],
                keypoints=[],
                court_to_image_h=[],
                image_to_court_h=[],
                projected_lines={},
                metrics=_metrics_from_detection(candidate),
                detect_ms=float(detect_ms),
                rejected_count=int(self._state.rejected_count),
            )

        return CourtLinePrediction(
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            source_size=(int(width), int(height)),
            valid=True,
            attempted=attempted,
            updated=updated,
            update_type=self._state.last_update_type,
            status=status,
            confidence=float(_clean_float(current.confidence) or 0.0),
            candidate_confidence=candidate_confidence,
            reason=str(current.reason),
            scheme=str(current.scheme),
            corners=_points_to_list(current.corners),
            keypoints=_keypoints_to_list(current.keypoint_names, current.keypoints),
            court_to_image_h=_matrix_to_list(current.court_to_image_h),
            image_to_court_h=_matrix_to_list(current.image_to_court_h),
            projected_lines=_projected_lines_to_list(current.projected_lines),
            metrics=_metrics_from_detection(current),
            detect_ms=float(detect_ms),
            rejected_count=int(self._state.rejected_count),
        )


def resolve_shuttlecourt_weights(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend(
            [
                PROJECT_ROOT / path,
                PROJECT_ROOT / "weights" / "shttlecourtnet" / path.name,
                PROJECT_ROOT / "weights" / "ShuttleCourtNet" / path.name,
                PROJECT_ROOT / "assets" / "weights" / "ShuttleCourtNet" / path.name,
            ]
        )

    for candidate in candidates:
        if candidate.is_file():
            return candidate
        if candidate.is_dir():
            matches = sorted(candidate.glob("*.pt"), key=lambda item: item.stat().st_mtime, reverse=True)
            if matches:
                return matches[0]

    fallback_dirs = (
        PROJECT_ROOT / "weights" / "shttlecourtnet",
        PROJECT_ROOT / "weights" / "ShuttleCourtNet",
        PROJECT_ROOT / "assets" / "weights" / "ShuttleCourtNet",
    )
    for directory in fallback_dirs:
        exact = directory / "ShuttleCourt.pt"
        if exact.is_file():
            return exact
        if directory.is_dir():
            matches = sorted(directory.glob("*.pt"), key=lambda item: item.stat().st_mtime, reverse=True)
            if matches:
                return matches[0]

    searched = [str(candidate) for candidate in candidates]
    searched.extend(str(directory) for directory in fallback_dirs)
    raise FileNotFoundError("Could not find ShuttleCourtNet weights. Searched: " + "; ".join(searched))


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _to_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.array([])
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _detection_from_segmentation_lines(
    *,
    polygon: np.ndarray,
    frame_shape: tuple[int, ...],
    confidence: float,
    area_ratio: float,
    polygon_points: int,
    class_id: int,
    rejected_masks: int,
    line_mask: np.ndarray,
    green_mask: np.ndarray,
    previous: _court_core.CourtLineDetection | None,
    args: SimpleNamespace,
) -> _court_core.CourtLineDetection | None:
    roi_mask = _segmentation_roi_mask(polygon, frame_shape, args)
    if roi_mask is None or cv2.countNonZero(roi_mask) <= 0:
        return None

    line_roi = cv2.bitwise_and(line_mask, roi_mask)
    if cv2.countNonZero(line_roi) <= 0:
        return None

    green_roi = cv2.bitwise_and(green_mask, roi_mask) if green_mask.size else green_mask
    segments, _ = _court_core.detect_hough_segments(line_roi, args)
    if len(segments) < 4:
        return None

    best = _best_line_detection_from_masks(
        segments=segments,
        line_mask=line_roi,
        green_mask=green_roi,
        previous=previous,
        args=args,
    )
    if best is None:
        return None

    polygon_area = abs(float(cv2.contourArea(np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2))))
    line_area = _court_core.polygon_area(best.corners)
    area_ratio_to_seg = line_area / max(1.0, polygon_area)
    if area_ratio_to_seg < max(0.0, float(getattr(args, "seg_line_min_area_ratio", 0.0))):
        return None

    roi_support = _projected_lines_roi_support(best.projected_lines, roi_mask)
    best.scheme = "shuttlecourt_seg"
    best.reason = f"segmentation ROI white-line fit: {best.reason}"
    best.components.update(
        {
            "segmentation_area": float(np.clip(area_ratio, 0.0, 1.0)),
            "box_confidence": float(np.clip(confidence, 0.0, 1.0)),
            "polygon_points": float(polygon_points),
            "class_id": float(class_id),
            "rejected_masks": float(rejected_masks),
            "seg_line_fit": 1.0,
            "seg_roi_support": float(roi_support),
            "seg_line_area_ratio": float(area_ratio_to_seg),
        }
    )
    if roi_support < 0.35:
        best.confidence *= 0.70
        best.reason = f"{best.reason}; weak segmentation ROI support"
    return best


def _best_line_detection_from_masks(
    *,
    segments: list[_court_core.LineSegment],
    line_mask: np.ndarray,
    green_mask: np.ndarray,
    previous: _court_core.CourtLineDetection | None,
    args: SimpleNamespace,
) -> _court_core.CourtLineDetection | None:
    family_a_segments, family_b_segments, angles = _court_core.choose_direction_families(segments, args)
    best: _court_core.CourtLineDetection | None = None
    if angles is not None:
        merged_a = _court_core.merge_line_family(family_a_segments, angles[0], args)
        merged_b = _court_core.merge_line_family(family_b_segments, angles[1], args)
        if len(merged_a) >= 2 and len(merged_b) >= 2:
            intersections = _court_core.all_family_intersections(merged_a, merged_b, line_mask.shape)
            if len(intersections) >= 4:
                best = _court_core.find_best_court_quad(
                    family_a=merged_a,
                    family_b=merged_b,
                    segments=segments,
                    intersections=intersections,
                    mask=line_mask,
                    green_mask=green_mask,
                    previous=previous,
                    args=args,
                )

    if best is None or best.confidence < float(args.medium_conf):
        clusters = _court_core.choose_angle_clusters(segments, args, max_clusters=4)
        best_three = _court_core.find_best_court_quad_three_family(
            clusters=clusters,
            segments=segments,
            mask=line_mask,
            green_mask=green_mask,
            previous=previous,
            args=args,
        )
        if best_three is not None and (best is None or best_three.confidence > best.confidence):
            best = best_three
    return best


def _detection_from_quad(
    *,
    quad: np.ndarray,
    confidence: float,
    area_ratio: float,
    polygon_points: int,
    class_id: int,
    rejected_masks: int,
    line_mask: np.ndarray,
    green_mask: np.ndarray,
    args: SimpleNamespace,
) -> _court_core.CourtLineDetection | None:
    court_to_image_h, image_to_court_h = _court_core.compute_homographies(quad)
    if court_to_image_h is None or image_to_court_h is None:
        return None

    snap_points = 0
    snap_mean_shift = 0.0
    refined_h, snap_points, snap_mean_shift = _court_core.refine_homography_with_white_lines(
        court_to_image_h,
        line_mask,
        args,
    )
    refined_corners = _court_core.project_points(_court_core.STANDARD_COURT_TEMPLATE.corners, refined_h)
    frame_diag = float(np.hypot(line_mask.shape[1], line_mask.shape[0]))
    refine_corner_shift = float(np.mean(np.linalg.norm(refined_corners - quad, axis=1)))
    max_refine_shift = max(24.0, frame_diag * max(0.0, float(args.max_refine_corner_shift_ratio)))
    refine_accepted = False
    if (
        _court_core.is_convex_quad(refined_corners)
        and _court_core.polygon_area(refined_corners) > 1.0
        and refine_corner_shift <= max_refine_shift
    ):
        refine_accepted = True
        court_to_image_h = refined_h
        quad = refined_corners.astype(np.float32)
        _, image_to_court_h = _court_core.compute_homographies(quad)
        if image_to_court_h is None:
            return None

    names_and_points = _court_core.template_keypoints_for_scheme("8")
    keypoint_names = [item[0] for item in names_and_points]
    template_points = np.asarray([item[1] for item in names_and_points], dtype=np.float32)
    keypoints = _court_core.project_points(template_points, court_to_image_h)
    side_lengths = np.linalg.norm(quad - np.roll(quad, -1, axis=0), axis=1)
    projected_lines = _court_core.project_template_lines(court_to_image_h)
    mask_support, supported_lines = _court_core.projected_template_support(projected_lines, line_mask)
    green_side_support = _court_core.outer_line_green_side_support(
        court_to_image_h,
        green_mask,
        offset_px=float(args.green_side_offset_px),
    )
    keypoint_scheme, keypoint_names, keypoints, supported_keypoints = _court_core.select_keypoints(
        court_to_image_h,
        [],
        line_mask,
        args,
    )
    return _court_core.CourtLineDetection(
        corners=quad.astype(np.float32),
        keypoints=keypoints.astype(np.float32),
        keypoint_names=keypoint_names,
        court_to_image_h=court_to_image_h,
        image_to_court_h=image_to_court_h,
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        components={
            "segmentation_area": float(np.clip(area_ratio, 0.0, 1.0)),
            "box_confidence": float(np.clip(confidence, 0.0, 1.0)),
            "polygon_points": float(polygon_points),
            "class_id": float(class_id),
            "rejected_masks": float(rejected_masks),
            "refine_corner_shift": float(refine_corner_shift),
            "refine_accepted": 1.0 if refine_accepted else 0.0,
            "keypoint_scheme": 6.0 if keypoint_scheme == "6" else 8.0,
        },
        line_count=0,
        merged_line_count=0,
        intersection_count=0,
        supported_keypoints=supported_keypoints,
        avg_line_length=float(np.mean(side_lengths)) if side_lengths.size else 0.0,
        mask_support=float(mask_support),
        green_side_support=float(green_side_support),
        snap_points=int(max(snap_points, supported_lines)) if refine_accepted else 0,
        snap_mean_shift=float(snap_mean_shift) if refine_accepted else 0.0,
        scheme="shuttlecourt_seg",
        reason="YOLO segmentation mask",
        projected_lines=projected_lines,
        debug_segments=[],
        debug_merged_lines=[],
    )


def _score_segmentation_candidate(
    *,
    quad: np.ndarray,
    frame_shape: tuple[int, ...],
    area_ratio: float,
    box_confidence: float,
    previous: _court_core.CourtLineDetection | None,
    line_support: float,
    green_side_support: float,
    snap_points: int,
    args: SimpleNamespace,
) -> tuple[float, float, dict[str, float], str]:
    geometry_score = _court_core.quad_geometry_score(quad, frame_shape)
    shape_score = _court_core.court_shape_sanity_score(quad, frame_shape, args)
    bounds_score = _court_core.quad_bounds_score(quad, frame_shape)
    area_score = _segmentation_area_score(area_ratio)
    center_score = _main_court_center_score(quad, frame_shape)
    temporal_score = _temporal_stability_score(quad, previous, frame_shape)
    line_score = float(np.clip(line_support / 0.28, 0.0, 1.0))
    green_score = float(np.clip(green_side_support / 0.55, 0.0, 1.0))
    snap_score = float(np.clip(float(snap_points) / 35.0, 0.0, 1.0))
    box_score = float(np.clip(box_confidence, 0.0, 1.0))

    quality = (
        0.18 * geometry_score
        + 0.16 * shape_score
        + 0.10 * bounds_score
        + 0.10 * area_score
        + 0.14 * center_score
        + 0.08 * temporal_score
        + 0.12 * line_score
        + 0.08 * green_score
        + 0.04 * snap_score
    )
    rank = quality * (0.65 + 0.35 * box_score)
    fused_confidence = 0.50 * box_score + 0.50 * quality

    reason = "YOLO segmentation mask"
    if shape_score < 0.55:
        fused_confidence *= 0.45
        rank *= 0.45
        reason = "segmentation candidate has implausible court shape"
    elif bounds_score < 0.75:
        fused_confidence *= 0.65
        rank *= 0.65
        reason = "segmentation candidate is partly out of bounds"
    elif geometry_score < 0.30:
        fused_confidence *= 0.55
        rank *= 0.55
        reason = "segmentation candidate has weak quadrilateral geometry"
    elif (
        area_ratio < max(float(args.min_mask_area_ratio), float(args.small_candidate_area_ratio))
        and line_support < float(args.small_candidate_min_line_support)
    ):
        fused_confidence *= 0.30
        rank *= 0.30
        reason = "segmentation candidate is too small and lacks white-line support"
    elif center_score < 0.20 and previous is None:
        fused_confidence *= 0.75
        rank *= 0.75
        reason = "segmentation candidate is far from the main court region"

    components = {
        "seg_geometry": float(geometry_score),
        "seg_shape": float(shape_score),
        "seg_bounds": float(bounds_score),
        "seg_area_score": float(area_score),
        "seg_center": float(center_score),
        "seg_temporal": float(temporal_score),
        "seg_line_support": float(line_score),
        "seg_green_sides": float(green_score),
        "seg_snap_points": float(snap_score),
        "seg_quality": float(np.clip(quality, 0.0, 1.0)),
        "seg_area_ratio": float(np.clip(area_ratio, 0.0, 1.0)),
        "seg_line_support_raw": float(np.clip(line_support, 0.0, 1.0)),
    }
    return float(rank), float(np.clip(fused_confidence, 0.0, 1.0)), components, reason


def _segmentation_area_score(area_ratio: float) -> float:
    area = float(np.clip(area_ratio, 0.0, 1.0))
    if area <= 0.0:
        return 0.0
    if area < 0.035:
        return float(np.clip(area / 0.035, 0.0, 1.0))
    if area > 0.88:
        return float(np.clip((1.0 - area) / 0.12, 0.0, 1.0))
    return 1.0


def _main_court_center_score(quad: np.ndarray, frame_shape: tuple[int, ...]) -> float:
    height, width = frame_shape[:2]
    if width <= 0 or height <= 0:
        return 0.0
    center = np.mean(np.asarray(quad, dtype=np.float32).reshape(4, 2), axis=0)
    dx = float(center[0] / max(1.0, float(width)) - 0.5)
    dy = float(center[1] / max(1.0, float(height)) - 0.62)
    distance = float(np.hypot(dx * 1.15, dy))
    return float(np.clip(1.0 - distance * 1.85, 0.0, 1.0))


def _temporal_stability_score(
    quad: np.ndarray,
    previous: _court_core.CourtLineDetection | None,
    frame_shape: tuple[int, ...],
) -> float:
    if previous is None or previous.corners.shape != (4, 2):
        return 0.75
    height, width = frame_shape[:2]
    diag = max(1.0, float(np.hypot(width, height)))
    mean_shift = float(np.mean(np.linalg.norm(quad.astype(np.float32) - previous.corners.astype(np.float32), axis=1)))
    return float(np.clip(1.0 - mean_shift / (diag * 0.20), 0.0, 1.0))


def _quad_from_polygon(
    polygon: np.ndarray,
    frame_shape: tuple[int, ...],
    args: SimpleNamespace,
) -> np.ndarray | None:
    height, width = frame_shape[:2]
    points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2).copy()
    points[:, 0] = np.clip(points[:, 0], 0.0, max(0.0, float(width - 1)))
    points[:, 1] = np.clip(points[:, 1], 0.0, max(0.0, float(height - 1)))
    if points.shape[0] < 4:
        return None

    hull = cv2.convexHull(points.reshape(-1, 1, 2))
    if hull.shape[0] < 4:
        return None

    perimeter = float(cv2.arcLength(hull, True))
    base_ratio = max(0.001, float(args.approx_epsilon_ratio))
    ratios = (base_ratio * 0.5, base_ratio, base_ratio * 1.5, base_ratio * 2.0, base_ratio * 3.0, 0.08)
    for ratio in ratios:
        approx = cv2.approxPolyDP(hull, max(1.0, perimeter * ratio), True).reshape(-1, 2).astype(np.float32)
        if approx.shape[0] != 4:
            continue
        ordered = _order_quad(approx)
        if _is_reasonable_quad(ordered, frame_shape):
            return ordered

    rect = cv2.minAreaRect(hull)
    box = cv2.boxPoints(rect).astype(np.float32)
    ordered_box = _order_quad(box)
    if _is_reasonable_quad(ordered_box, frame_shape):
        return ordered_box
    return None


def _segmentation_roi_mask(
    polygon: np.ndarray,
    frame_shape: tuple[int, ...],
    args: SimpleNamespace,
) -> np.ndarray | None:
    height, width = frame_shape[:2]
    points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2).copy()
    if points.shape[0] < 3 or not np.isfinite(points).all():
        return None
    points[:, 0] = np.clip(points[:, 0], 0.0, max(0.0, float(width - 1)))
    points[:, 1] = np.clip(points[:, 1], 0.0, max(0.0, float(height - 1)))
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [points.astype(np.int32).reshape(-1, 1, 2)], 255)
    dilation = max(0, int(getattr(args, "seg_roi_dilate_px", 0)))
    if dilation > 0:
        kernel_size = dilation * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def _projected_lines_roi_support(projected_lines: dict[str, np.ndarray], roi_mask: np.ndarray) -> float:
    if not projected_lines:
        return 0.0
    supports: list[float] = []
    for points in projected_lines.values():
        points_array = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if len(points_array) < 2:
            continue
        total = 0
        inside = 0
        for p1, p2 in zip(points_array, points_array[1:]):
            length = float(np.linalg.norm(p2 - p1))
            samples = max(4, min(80, int(length / 12.0)))
            for t in np.linspace(0.0, 1.0, samples, dtype=np.float32):
                point = p1 * (1.0 - t) + p2 * t
                total += 1
                if _court_core.sample_mask(roi_mask, point) > 0:
                    inside += 1
        if total:
            supports.append(inside / float(total))
    return float(np.mean(supports)) if supports else 0.0


def _order_quad(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    y_sorted = np.argsort(pts[:, 1])
    top = y_sorted[:2]
    bottom = y_sorted[2:]
    top = top[np.argsort(pts[top, 0])]
    bottom = bottom[np.argsort(pts[bottom, 0])]
    order = np.array([top[0], top[1], bottom[1], bottom[0]], dtype=np.int32)
    ordered = pts[order]

    if not _is_convex(ordered):
        sums = pts[:, 0] + pts[:, 1]
        diffs = pts[:, 1] - pts[:, 0]
        fallback = np.array(
            [int(np.argmin(sums)), int(np.argmin(diffs)), int(np.argmax(sums)), int(np.argmax(diffs))],
            dtype=np.int32,
        )
        if len(set(int(index) for index in fallback)) == 4:
            ordered = pts[fallback]
    return ordered.astype(np.float32)


def _is_reasonable_quad(points: np.ndarray, frame_shape: tuple[int, ...]) -> bool:
    if points.shape != (4, 2) or not np.isfinite(points).all() or not _is_convex(points):
        return False
    height, width = frame_shape[:2]
    frame_area = float(max(width * height, 1))
    area = abs(float(cv2.contourArea(points.reshape(-1, 1, 2))))
    if area < frame_area * 0.002:
        return False
    side_lengths = np.linalg.norm(points - np.roll(points, -1, axis=0), axis=1)
    return bool(np.min(side_lengths) >= max(8.0, float(np.hypot(width, height)) * 0.006))


def _is_convex(points: np.ndarray) -> bool:
    contour = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return bool(cv2.isContourConvex(contour))


def _status_text(attempted: bool, update_type: str, valid: bool, candidate: Any) -> str:
    if not attempted:
        return "reuse current" if valid else "waiting for first detection"
    if candidate is None:
        return "no segmentation mask; reusing previous" if valid else "no segmentation mask"
    if update_type == "rejected":
        return "segmentation candidate rejected; reusing previous" if valid else "segmentation candidate rejected"
    return update_type


def _clean_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _points_to_list(points: Any) -> list[list[float]]:
    array = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    output: list[list[float]] = []
    for x, y in array:
        clean_x = _clean_float(x)
        clean_y = _clean_float(y)
        if clean_x is not None and clean_y is not None:
            output.append([clean_x, clean_y])
    return output


def _keypoints_to_list(names: list[str], points: Any) -> list[dict[str, Any]]:
    point_list = _points_to_list(points)
    return [
        {
            "name": str(names[index]) if index < len(names) else f"keypoint_{index}",
            "point": point,
        }
        for index, point in enumerate(point_list)
    ]


def _matrix_to_list(matrix: Any) -> list[list[float]]:
    array = np.asarray(matrix, dtype=np.float64)
    if array.shape != (3, 3) or not np.isfinite(array).all():
        return []
    return [[float(value) for value in row] for row in array.tolist()]


def _projected_lines_to_list(lines: dict[str, Any]) -> dict[str, list[list[float]]]:
    return {str(name): _points_to_list(points) for name, points in lines.items()}


def _metrics_from_detection(detection: Any) -> dict[str, Any]:
    if detection is None:
        return {}
    return {
        "line_count": int(getattr(detection, "line_count", 0)),
        "merged_line_count": int(getattr(detection, "merged_line_count", 0)),
        "intersection_count": int(getattr(detection, "intersection_count", 0)),
        "supported_keypoints": int(getattr(detection, "supported_keypoints", 0)),
        "avg_line_length": float(_clean_float(getattr(detection, "avg_line_length", 0.0)) or 0.0),
        "mask_support": float(_clean_float(getattr(detection, "mask_support", 0.0)) or 0.0),
        "green_side_support": float(_clean_float(getattr(detection, "green_side_support", 0.0)) or 0.0),
        "snap_points": int(getattr(detection, "snap_points", 0)),
        "snap_mean_shift": float(_clean_float(getattr(detection, "snap_mean_shift", 0.0)) or 0.0),
        "components": {
            str(name): float(value)
            for name, value in dict(getattr(detection, "components", {}) or {}).items()
            if _clean_float(value) is not None
        },
    }
