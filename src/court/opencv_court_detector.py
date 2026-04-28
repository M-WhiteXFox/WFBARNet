from __future__ import annotations

from dataclasses import asdict, dataclass
from types import SimpleNamespace
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from src.court import opencv_court_homography_core as _opencv_court


@dataclass(slots=True)
class OpenCVCourtLineConfig:
    redetect_interval: float = 4.0
    detect_max_width: int = 960
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
    max_refine_corner_shift_ratio: float = 0.045
    green_side_offset_px: float = 14.0
    reliable_conf: float = 0.75
    medium_conf: float = 0.55
    smooth_alpha_reliable: float = 0.45
    smooth_alpha_medium: float = 0.20
    jump_ratio_hard: float = 0.18
    mask_alpha: float = 0.14
    line_thickness: int = 3
    point_radius: int = 5
    show_labels: bool = False
    draw_debug_lines: bool = False


@dataclass(slots=True)
class CourtLinePrediction:
    frame_id: int
    timestamp_ms: int
    source_size: tuple[int, int]
    valid: bool
    attempted: bool
    updated: bool
    update_type: str
    status: str
    confidence: float
    candidate_confidence: float | None
    reason: str
    scheme: str
    corners: list[list[float]]
    keypoints: list[dict[str, Any]]
    court_to_image_h: list[list[float]]
    image_to_court_h: list[list[float]]
    projected_lines: dict[str, list[list[float]]]
    metrics: dict[str, Any]
    detect_ms: float
    rejected_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "timestamp_ms": self.timestamp_ms,
            "source_size": list(self.source_size),
            "valid": self.valid,
            "attempted": self.attempted,
            "updated": self.updated,
            "update_type": self.update_type,
            "status": self.status,
            "confidence": self.confidence,
            "candidate_confidence": self.candidate_confidence,
            "reason": self.reason,
            "scheme": self.scheme,
            "corners": self.corners,
            "keypoints": self.keypoints,
            "court_to_image_h": self.court_to_image_h,
            "image_to_court_h": self.image_to_court_h,
            "projected_lines": self.projected_lines,
            "metrics": self.metrics,
            "detect_ms": self.detect_ms,
            "rejected_count": self.rejected_count,
        }


class OpenCVCourtLineDetector:
    def __init__(self, config: OpenCVCourtLineConfig | None = None) -> None:
        self.config = config or OpenCVCourtLineConfig()
        self._args = SimpleNamespace(**asdict(self.config))
        self._state = _opencv_court.TrackingState()
        self._latest_prediction: CourtLinePrediction | None = None

    def reset(self) -> None:
        self._state = _opencv_court.TrackingState()
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
            raise ValueError("Court line prediction expects a BGR image frame.")

        timestamp_ms = max(0, int(timestamp_ms))
        timestamp_s = timestamp_ms / 1000.0
        should_attempt = bool(force) or _opencv_court.should_redetect(self._state, int(frame_id), timestamp_s, self._args)
        candidate = None
        detect_ms = 0.0

        if should_attempt:
            started_at = perf_counter()
            candidate = _opencv_court.detect_court_lines(frame, self._state.current, self._args)
            detect_ms = (perf_counter() - started_at) * 1000.0
            _opencv_court.update_tracking_state(
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
        candidate_confidence = _clean_float(candidate.confidence) if candidate is not None else None
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
                scheme="",
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


def draw_court_prediction(
    frame: np.ndarray,
    prediction: CourtLinePrediction,
    *,
    mask_alpha: float = 0.14,
    line_thickness: int = 3,
    show_keypoints: bool = False,
) -> np.ndarray:
    if not prediction.valid or not prediction.projected_lines:
        return frame.copy()

    canvas = frame.copy()
    outer = prediction.projected_lines.get("doubles_outer")
    if outer and mask_alpha > 0.0:
        polygon = np.asarray(outer, dtype=np.float32).reshape(-1, 2)
        if polygon.shape[0] >= 3 and np.isfinite(polygon).all():
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [polygon.astype(np.int32).reshape(-1, 1, 2)], (35, 210, 90))
            alpha = float(np.clip(mask_alpha, 0.0, 1.0))
            cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0, canvas)

    thickness = max(1, int(line_thickness))
    for name, points in prediction.projected_lines.items():
        if len(points) < 2:
            continue
        line_points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if not np.isfinite(line_points).all():
            continue
        pts = line_points.astype(np.int32).reshape(-1, 1, 2)
        closed = name == "doubles_outer"
        cv2.polylines(canvas, [pts], closed, (0, 65, 25), thickness + 3, lineType=cv2.LINE_AA)
        cv2.polylines(canvas, [pts], closed, (40, 245, 110), thickness + (1 if closed else 0), lineType=cv2.LINE_AA)

    if show_keypoints:
        for item in prediction.keypoints:
            point = item.get("point")
            if not point or len(point) < 2:
                continue
            x, y = int(round(float(point[0]))), int(round(float(point[1])))
            cv2.circle(canvas, (x, y), 5, (255, 245, 80), -1, lineType=cv2.LINE_AA)

    return canvas


def _status_text(attempted: bool, update_type: str, valid: bool, candidate: Any) -> str:
    if not attempted:
        return "reuse current" if valid else "waiting for first detection"
    if candidate is None:
        return "no candidate; reusing previous" if valid else "no candidate"
    if update_type == "rejected":
        return "candidate rejected; reusing previous" if valid else "candidate rejected"
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
