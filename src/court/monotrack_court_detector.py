from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass
from types import SimpleNamespace
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from src.court import opencv_court_homography_core as _court_core
from src.court.opencv_court_detector import CourtLinePrediction


@dataclass(slots=True)
class MonoTrackCourtLineConfig:
    redetect_interval: float = 4.0
    detect_max_width: int = 960
    luminance_threshold: int = 80
    diff_threshold: int = 20
    ridge_offset_px: int = 4
    gradient_kernel_size: int = 3
    structure_kernel_size: int = 21
    hough_threshold: int = 50
    hough_min_line_length: int = 50
    hough_max_line_gap: int = 10
    angle_bin_deg: float = 5.0
    angle_tol_deg: float = 16.0
    min_angle_separation_deg: float = 25.0
    merge_rho_px: float = 16.0
    max_lines_per_family: int = 3
    model_sample_step_px: float = 8.0
    model_sample_radius_px: int = 2
    point_scheme: str = "auto"
    refine_homography: bool = True
    snap_search_px: float = 18.0
    snap_response_threshold: float = 0.18
    max_refine_corner_shift_ratio: float = 0.045
    green_side_offset_px: float = 14.0
    min_outer_width_ratio: float = 0.08
    min_outer_depth_ratio: float = 0.08
    min_outer_width_depth_ratio: float = 0.18
    max_outer_width_depth_ratio: float = 5.5
    max_transverse_angle_deg: float = 35.0
    reliable_conf: float = 0.68
    medium_conf: float = 0.48
    smooth_alpha_reliable: float = 0.45
    smooth_alpha_medium: float = 0.20
    jump_ratio_hard: float = 0.18


_TEMPLATE_H_LINES = (
    ("top_base", 0.0),
    ("top_doubles_long_service", _court_core.DOUBLES_LONG_SERVICE_FROM_BACK),
    ("top_short_service", _court_core.NET_Y - _court_core.SHORT_SERVICE_FROM_NET),
    ("net", _court_core.NET_Y),
    ("bottom_short_service", _court_core.NET_Y + _court_core.SHORT_SERVICE_FROM_NET),
    ("bottom_doubles_long_service", _court_core.COURT_LENGTH - _court_core.DOUBLES_LONG_SERVICE_FROM_BACK),
    ("bottom_base", _court_core.COURT_LENGTH),
)
_TEMPLATE_V_LINES = (
    ("left_side", 0.0),
    ("left_singles", _court_core.SINGLES_MARGIN),
    ("center", _court_core.COURT_WIDTH / 2.0),
    ("right_singles", _court_core.COURT_WIDTH - _court_core.SINGLES_MARGIN),
    ("right_side", _court_core.COURT_WIDTH),
)
_TEMPLATE_LINE_PAIR_COMBINATIONS = tuple(
    (h_pair, v_pair)
    for h_pair in itertools.combinations(_TEMPLATE_H_LINES, 2)
    for v_pair in itertools.combinations(_TEMPLATE_V_LINES, 2)
)


class MonoTrackCourtLineDetector:
    def __init__(self, config: MonoTrackCourtLineConfig | None = None) -> None:
        self.config = config or MonoTrackCourtLineConfig()
        self._args = SimpleNamespace(**asdict(self.config))
        self._state = _court_core.TrackingState()
        self._latest_prediction: CourtLinePrediction | None = None

    def reset(self) -> None:
        self._state = _court_core.TrackingState()
        self._latest_prediction = None

    def latest_prediction(self) -> CourtLinePrediction | None:
        return self._latest_prediction

    def predict(
        self,
        frame: np.ndarray,
        frame_id: int,
        timestamp_ms: int,
        *,
        force: bool = False,
    ) -> CourtLinePrediction:
        if frame is None or frame.ndim < 2:
            raise ValueError("MonoTrack court line prediction expects a BGR image frame.")

        timestamp_ms = max(0, int(timestamp_ms))
        timestamp_s = timestamp_ms / 1000.0
        should_attempt = bool(force) or _court_core.should_redetect(
            self._state,
            int(frame_id),
            timestamp_s,
            self._args,
        )
        candidate = None
        detect_ms = 0.0

        if should_attempt:
            started_at = perf_counter()
            candidate = detect_monotrack_court_lines(frame, self._state.current, self._args)
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


def detect_monotrack_court_lines(
    frame: np.ndarray,
    previous: _court_core.CourtLineDetection | None,
    args: SimpleNamespace,
    *,
    roi_mask: np.ndarray | None = None,
) -> _court_core.CourtLineDetection | None:
    detect_frame, scale = _court_core.resize_for_detection(frame, int(args.detect_max_width))
    previous_small = (
        _court_core.scale_detection(previous, scale, detect_frame.shape)
        if previous is not None and scale != 1.0
        else previous
    )
    mask = create_monotrack_line_mask(detect_frame, args)
    if roi_mask is not None:
        roi = np.asarray(roi_mask, dtype=np.uint8)
        if roi.ndim == 3:
            roi = roi[..., 0]
        if roi.shape != detect_frame.shape[:2]:
            roi = cv2.resize(
                roi,
                (detect_frame.shape[1], detect_frame.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        roi = np.where(roi > 0, 255, 0).astype(np.uint8)
        mask = cv2.bitwise_and(mask, roi)
    segments = detect_monotrack_hough_segments(mask, args)
    if len(segments) < 4:
        return None

    best_three_family = fit_monotrack_three_family(segments, mask, previous_small, args)
    if best_three_family is not None and best_three_family.confidence >= float(args.medium_conf):
        return _court_core.scale_detection(best_three_family, 1.0 / scale, frame.shape) if scale != 1.0 else best_three_family

    family_a, family_b, angles = _court_core.choose_direction_families(segments, args)
    if angles is None:
        return _court_core.scale_detection(best_three_family, 1.0 / scale, frame.shape) if scale != 1.0 else best_three_family
    merged_a = _court_core.merge_line_family(family_a, angles[0], args)
    merged_b = _court_core.merge_line_family(family_b, angles[1], args)
    if len(merged_a) < 2 or len(merged_b) < 2:
        return _court_core.scale_detection(best_three_family, 1.0 / scale, frame.shape) if scale != 1.0 else best_three_family

    best_small = fit_monotrack_template(
        merged_a,
        merged_b,
        segments,
        mask,
        previous_small,
        args,
    )
    if best_small is None:
        best_small = best_three_family
    elif best_three_family is not None and best_three_family.confidence > best_small.confidence:
        best_small = best_three_family
    return _court_core.scale_detection(best_small, 1.0 / scale, frame.shape) if scale != 1.0 else best_small


def fit_monotrack_three_family(
    segments: list[_court_core.LineSegment],
    mask: np.ndarray,
    previous: _court_core.CourtLineDetection | None,
    args: SimpleNamespace,
) -> _court_core.CourtLineDetection | None:
    clusters = _court_core.choose_angle_clusters(segments, args, max_clusters=4)
    if len(clusters) < 3:
        return None
    green_mask = np.zeros(mask.shape, dtype=np.uint8)
    detection = _court_core.find_best_court_quad_three_family(clusters, segments, mask, green_mask, previous, args)
    if detection is None:
        return None
    detection.scheme = "monotrack"
    detection.components = dict(detection.components)
    detection.components["monotrack_three_family"] = 1.0
    return detection


def create_monotrack_line_mask(frame: np.ndarray, args: SimpleNamespace) -> np.ndarray:
    if frame.ndim == 2:
        luminance = frame
    else:
        luminance = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    luminance_i = luminance.astype(np.int16)
    offset = max(1, int(args.ridge_offset_px))
    mask = np.zeros(luminance.shape, dtype=np.uint8)
    if luminance.shape[0] <= offset * 2 or luminance.shape[1] <= offset * 2:
        return mask

    center = luminance_i[offset:-offset, offset:-offset]
    left = luminance_i[offset:-offset, 0 : -2 * offset]
    right = luminance_i[offset:-offset, 2 * offset :]
    top = luminance_i[0 : -2 * offset, offset:-offset]
    bottom = luminance_i[2 * offset :, offset:-offset]
    bright = center >= int(args.luminance_threshold)
    x_ridge = (center - left > int(args.diff_threshold)) & (center - right > int(args.diff_threshold))
    y_ridge = (center - top > int(args.diff_threshold)) & (center - bottom > int(args.diff_threshold))
    mask[offset:-offset, offset:-offset] = np.where(bright & (x_ridge | y_ridge), 255, 0).astype(np.uint8)
    return filter_monotrack_line_pixels(mask, luminance, args)


def filter_monotrack_line_pixels(mask: np.ndarray, luminance: np.ndarray, args: SimpleNamespace) -> np.ndarray:
    float_image = luminance.astype(np.float32)
    float_image = cv2.GaussianBlur(float_image, (5, 5), 0)
    ksize = max(1, int(args.gradient_kernel_size))
    if ksize % 2 == 0:
        ksize += 1
    dx = cv2.Sobel(float_image, cv2.CV_32F, 1, 0, ksize=ksize)
    dy = cv2.Sobel(float_image, cv2.CV_32F, 0, 1, ksize=ksize)
    dx2 = dx * dx
    dxy = dx * dy
    dy2 = dy * dy
    kernel_size = max(3, int(args.structure_kernel_size))
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
    a = cv2.filter2D(dx2, -1, kernel)
    b = cv2.filter2D(dxy, -1, kernel)
    d = cv2.filter2D(dy2, -1, kernel)
    trace = a + d
    root = np.sqrt(np.maximum(0.0, (a - d) * (a - d) + 4.0 * b * b))
    lambda_max = (trace + root) * 0.5
    lambda_min = (trace - root) * 0.5
    directional = lambda_max > 4.0 * np.maximum(lambda_min, 1e-6)
    return np.where((mask > 0) & directional, 255, 0).astype(np.uint8)


def detect_monotrack_hough_segments(mask: np.ndarray, args: SimpleNamespace) -> list[_court_core.LineSegment]:
    raw_lines = cv2.HoughLinesP(
        mask,
        rho=1,
        theta=np.pi / 180.0,
        threshold=max(1, int(args.hough_threshold)),
        minLineLength=max(1, int(args.hough_min_line_length)),
        maxLineGap=max(1, int(args.hough_max_line_gap)),
    )
    segments: list[_court_core.LineSegment] = []
    if raw_lines is None:
        return segments
    for raw in raw_lines.reshape(-1, 4):
        p1 = np.array([float(raw[0]), float(raw[1])], dtype=np.float32)
        p2 = np.array([float(raw[2]), float(raw[3])], dtype=np.float32)
        length = float(np.linalg.norm(p2 - p1))
        if length <= 1.0:
            continue
        segments.append(
            _court_core.LineSegment(
                p1=p1,
                p2=p2,
                angle_deg=_court_core.line_angle_deg(p1, p2),
                length=length,
            )
        )
    segments.sort(key=lambda item: item.length, reverse=True)
    return segments


def fit_monotrack_template(
    family_a: list[_court_core.MergedLine],
    family_b: list[_court_core.MergedLine],
    segments: list[_court_core.LineSegment],
    mask: np.ndarray,
    previous: _court_core.CourtLineDetection | None,
    args: SimpleNamespace,
) -> _court_core.CourtLineDetection | None:
    candidates = [
        sorted(family_a, key=lambda line: line.length, reverse=True)[: max(2, int(args.max_lines_per_family))],
        sorted(family_b, key=lambda line: line.length, reverse=True)[: max(2, int(args.max_lines_per_family))],
    ]
    best: _court_core.CourtLineDetection | None = None
    best_model_score = -1.0e18
    for image_h_lines, image_v_lines in (candidates, list(reversed(candidates))):
        for image_h_pair in itertools.combinations(image_h_lines, 2):
            for image_v_pair in itertools.combinations(image_v_lines, 2):
                for ordered_h_pair in _line_pair_orders(image_h_pair):
                    for ordered_v_pair in _line_pair_orders(image_v_pair):
                        image_quad = _intersections_for_pairs(ordered_h_pair, ordered_v_pair)
                        if image_quad is None:
                            continue
                        for template_h_pair, template_v_pair in _TEMPLATE_LINE_PAIR_COMBINATIONS:
                            template_quad = _template_quad(template_h_pair, template_v_pair)
                            court_to_image_h = cv2.getPerspectiveTransform(template_quad, image_quad)
                            if court_to_image_h is None or not np.isfinite(court_to_image_h).all():
                                continue
                            corners = _court_core.project_points(
                                _court_core.STANDARD_COURT_TEMPLATE.corners,
                                court_to_image_h,
                            )
                            if not _candidate_corners_plausible(corners, mask.shape, args):
                                continue
                            model_score = score_projected_template(court_to_image_h, mask, args)
                            if model_score <= best_model_score:
                                continue
                            detection = build_monotrack_detection(
                                court_to_image_h,
                                corners,
                                segments,
                                image_h_lines + image_v_lines,
                                mask,
                                previous,
                                args,
                                model_score,
                            )
                            if detection is None:
                                continue
                            best = detection
                            best_model_score = model_score
    return best


def build_monotrack_detection(
    court_to_image_h: np.ndarray,
    corners: np.ndarray,
    segments: list[_court_core.LineSegment],
    merged_lines: list[_court_core.MergedLine],
    mask: np.ndarray,
    previous: _court_core.CourtLineDetection | None,
    args: SimpleNamespace,
    model_score: float,
) -> _court_core.CourtLineDetection | None:
    _, image_to_court_h = _court_core.compute_homographies(corners)
    if image_to_court_h is None:
        return None
    projected_lines = _court_core.project_template_lines(court_to_image_h)
    mask_support, supported_lines = _court_core.projected_template_support(projected_lines, mask)
    names_and_points = _court_core.template_keypoints_for_scheme("6")
    names = [item[0] for item in names_and_points]
    template_points = np.asarray([item[1] for item in names_and_points], dtype=np.float32)
    keypoints = _court_core.project_points(template_points, court_to_image_h)
    supported_keypoints = _count_supported_keypoints(keypoints, mask)
    avg_length = float(np.mean([seg.length for seg in segments[: min(12, len(segments))]])) if segments else 0.0
    detection = _court_core.CourtLineDetection(
        corners=corners.astype(np.float32),
        keypoints=keypoints.astype(np.float32),
        keypoint_names=names,
        court_to_image_h=court_to_image_h.astype(np.float64),
        image_to_court_h=image_to_court_h.astype(np.float64),
        confidence=0.0,
        components={},
        line_count=len(segments),
        merged_line_count=len(merged_lines),
        intersection_count=0,
        supported_keypoints=supported_keypoints,
        avg_line_length=avg_length,
        mask_support=mask_support,
        green_side_support=0.55,
        snap_points=supported_lines,
        snap_mean_shift=0.0,
        scheme="monotrack",
        reason="candidate",
        projected_lines=projected_lines,
        debug_segments=segments[:60],
        debug_merged_lines=merged_lines,
    )
    confidence, components, reason = _court_core.score_court_detection(detection, previous, mask.shape[:2], args)
    components.update(
        _court_core.projected_singles_support_components(
            projected_lines,
            mask,
            mask_support,
        )
    )
    components.update(
        _court_core.projected_outer_support_components(
            projected_lines,
            mask,
        )
    )
    components["monotrack_model"] = float(np.clip(model_score / max(1.0, np.hypot(mask.shape[1], mask.shape[0])), 0.0, 1.0))
    detection.confidence = confidence
    detection.components = components
    detection.reason = reason
    return detection


def score_projected_template(court_to_image_h: np.ndarray, mask: np.ndarray, args: SimpleNamespace) -> float:
    projected_lines = _court_core.project_template_lines(court_to_image_h)
    score = 0.0
    for name, points in projected_lines.items():
        if name == "doubles_outer":
            polyline = np.vstack([points, points[0]])
            weight = 1.25
        elif "singles" in name:
            polyline = points
            weight = 2.0
        else:
            polyline = points
            weight = 1.0
        for p1, p2 in zip(polyline[:-1], polyline[1:]):
            score += score_line_segment(mask, p1, p2, weight, args)
    return score


def score_line_segment(
    mask: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    weight: float,
    args: SimpleNamespace,
) -> float:
    if not np.isfinite(p1).all() or not np.isfinite(p2).all():
        return -1.0e6
    height, width = mask.shape[:2]
    length = float(np.linalg.norm(p2 - p1))
    if length <= 1.0:
        return 0.0
    step = max(1.0, float(args.model_sample_step_px))
    samples = max(2, int(length / step))
    radius = max(0, int(args.model_sample_radius_px))
    hits = 0
    total = 0
    for alpha in np.linspace(0.0, 1.0, samples, dtype=np.float32):
        point = p1 * (1.0 - alpha) + p2 * alpha
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))
        if x < 0 or x >= width or y < 0 or y >= height:
            continue
        total += 1
        patch = mask[
            max(0, y - radius) : min(height, y + radius + 1),
            max(0, x - radius) : min(width, x + radius + 1),
        ]
        if cv2.countNonZero(patch) > 0:
            hits += 1
    if total == 0:
        return -length * 0.5
    misses = total - hits
    return float(weight) * (hits - 0.5 * misses / max(float(weight), 1.0))


def _candidate_corners_plausible(corners: np.ndarray, frame_shape: tuple[int, ...], args: SimpleNamespace) -> bool:
    if corners.shape != (4, 2) or not np.isfinite(corners).all():
        return False
    if not _court_core.is_convex_quad(corners):
        return False
    if _court_core.quad_geometry_score(corners, frame_shape) <= 0.05:
        return False
    if _court_core.court_shape_sanity_score(corners, frame_shape, args) < 0.35:
        return False
    height, width = frame_shape[:2]
    margin_x = width * 0.25
    margin_y = height * 0.25
    return bool(
        np.all(corners[:, 0] >= -margin_x)
        and np.all(corners[:, 0] <= width + margin_x)
        and np.all(corners[:, 1] >= -margin_y)
        and np.all(corners[:, 1] <= height + margin_y)
    )


def _line_pair_orders(pair: tuple[_court_core.MergedLine, _court_core.MergedLine]) -> tuple[tuple[_court_core.MergedLine, _court_core.MergedLine], ...]:
    return (pair, (pair[1], pair[0]))


def _intersections_for_pairs(
    h_pair: tuple[_court_core.MergedLine, _court_core.MergedLine],
    v_pair: tuple[_court_core.MergedLine, _court_core.MergedLine],
) -> np.ndarray | None:
    points = [
        _court_core.intersect_merged_lines(h_pair[0], v_pair[0]),
        _court_core.intersect_merged_lines(h_pair[0], v_pair[1]),
        _court_core.intersect_merged_lines(h_pair[1], v_pair[1]),
        _court_core.intersect_merged_lines(h_pair[1], v_pair[0]),
    ]
    if any(point is None for point in points):
        return None
    quad = np.asarray(points, dtype=np.float32)
    if not np.isfinite(quad).all():
        return None
    return quad


def _template_quad(
    h_pair: tuple[tuple[str, float], tuple[str, float]],
    v_pair: tuple[tuple[str, float], tuple[str, float]],
) -> np.ndarray:
    y1 = float(h_pair[0][1])
    y2 = float(h_pair[1][1])
    x1 = float(v_pair[0][1])
    x2 = float(v_pair[1][1])
    return np.asarray([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


def _count_supported_keypoints(points: np.ndarray, mask: np.ndarray) -> int:
    dilated = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1)
    return sum(1 for point in points if _court_core.point_patch_support(dilated, point, radius=7) >= 0.04)


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
