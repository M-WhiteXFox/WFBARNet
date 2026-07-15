from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_SOURCE = PROJECT_ROOT / "videos" / "MVI_0212.MP4"

COURT_WIDTH = 610.0
COURT_LENGTH = 1340.0
SINGLES_MARGIN = 46.0
NET_Y = COURT_LENGTH / 2.0
SHORT_SERVICE_FROM_NET = 198.0
DOUBLES_LONG_SERVICE_FROM_BACK = 76.0


@dataclass(frozen=True, slots=True)
class CourtTemplate:
    width: float
    length: float
    corners: np.ndarray
    mask_polygon: np.ndarray
    lines: tuple[tuple[str, np.ndarray], ...]
    keypoints_8: tuple[tuple[str, tuple[float, float]], ...]
    keypoints_6: tuple[tuple[str, tuple[float, float]], ...]


@dataclass(slots=True)
class LineSegment:
    p1: np.ndarray
    p2: np.ndarray
    angle_deg: float
    length: float

    @property
    def midpoint(self) -> np.ndarray:
        return (self.p1 + self.p2) * 0.5


@dataclass(slots=True)
class MergedLine:
    theta_deg: float
    rho: float
    length: float
    count: int


@dataclass(slots=True)
class CourtLineDetection:
    corners: np.ndarray
    keypoints: np.ndarray
    keypoint_names: list[str]
    court_to_image_h: np.ndarray
    image_to_court_h: np.ndarray
    confidence: float
    components: dict[str, float]
    line_count: int
    merged_line_count: int
    intersection_count: int
    supported_keypoints: int
    avg_line_length: float
    mask_support: float
    green_side_support: float
    snap_points: int
    snap_mean_shift: float
    scheme: str
    reason: str
    projected_lines: dict[str, np.ndarray] = field(default_factory=dict)
    debug_segments: list[LineSegment] = field(default_factory=list)
    debug_merged_lines: list[MergedLine] = field(default_factory=list)
    last_update_frame: int = 0
    last_update_time: float = 0.0


@dataclass(slots=True)
class TrackingState:
    current: CourtLineDetection | None = None
    last_candidate: CourtLineDetection | None = None
    startup_candidate: CourtLineDetection | None = None
    startup_candidate_count: int = 0
    last_attempt_frame: int = -1
    last_attempt_time: float = 0.0
    last_update_type: str = "none"
    rejected_count: int = 0


def build_standard_court_template() -> CourtTemplate:
    width = COURT_WIDTH
    length = COURT_LENGTH
    singles_left = SINGLES_MARGIN
    singles_right = width - SINGLES_MARGIN
    center_x = width / 2.0
    top_short = NET_Y - SHORT_SERVICE_FROM_NET
    bottom_short = NET_Y + SHORT_SERVICE_FROM_NET
    top_long = DOUBLES_LONG_SERVICE_FROM_BACK
    bottom_long = length - DOUBLES_LONG_SERVICE_FROM_BACK

    corners = np.array(
        [[0.0, 0.0], [width, 0.0], [width, length], [0.0, length]],
        dtype=np.float32,
    )
    lines: tuple[tuple[str, np.ndarray], ...] = (
        ("doubles_outer", corners),
        ("singles_left_sideline", np.array([[singles_left, 0.0], [singles_left, length]], dtype=np.float32)),
        ("singles_right_sideline", np.array([[singles_right, 0.0], [singles_right, length]], dtype=np.float32)),
        ("top_short_service", np.array([[0.0, top_short], [width, top_short]], dtype=np.float32)),
        ("bottom_short_service", np.array([[0.0, bottom_short], [width, bottom_short]], dtype=np.float32)),
        ("top_doubles_long_service", np.array([[0.0, top_long], [width, top_long]], dtype=np.float32)),
        ("bottom_doubles_long_service", np.array([[0.0, bottom_long], [width, bottom_long]], dtype=np.float32)),
        ("top_center_service", np.array([[center_x, 0.0], [center_x, top_short]], dtype=np.float32)),
        ("bottom_center_service", np.array([[center_x, bottom_short], [center_x, length]], dtype=np.float32)),
    )
    keypoints_8 = (
        ("outer_tl", (0.0, 0.0)),
        ("outer_tr", (width, 0.0)),
        ("outer_br", (width, length)),
        ("outer_bl", (0.0, length)),
        ("top_short_l", (0.0, top_short)),
        ("top_short_r", (width, top_short)),
        ("bottom_short_r", (width, bottom_short)),
        ("bottom_short_l", (0.0, bottom_short)),
    )
    keypoints_6 = (
        ("outer_tl", (0.0, 0.0)),
        ("outer_tr", (width, 0.0)),
        ("outer_br", (width, length)),
        ("outer_bl", (0.0, length)),
        ("top_center_service", (center_x, top_short)),
        ("bottom_center_service", (center_x, bottom_short)),
    )
    return CourtTemplate(
        width=width,
        length=length,
        corners=corners,
        mask_polygon=corners,
        lines=lines,
        keypoints_8=keypoints_8,
        keypoints_6=keypoints_6,
    )


STANDARD_COURT_TEMPLATE = build_standard_court_template()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Traditional OpenCV badminton court white-line detection and homography demo."
    )
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="Video path or camera index.")
    parser.add_argument("--save-video", default="", help="Optional output mp4 path.")
    parser.add_argument("--calibration", default="court_calibration.json", help="Path to save/load court calibration json.")
    parser.add_argument("--preview-template", action="store_true", default=True, help="Preview full projected court template after 4 points are available.")
    parser.add_argument("--no-preview-template", dest="preview_template", action="store_false", help="Start with full projected court template preview disabled.")
    parser.add_argument("--manual-first", action="store_true", help="Start in manual point selection mode instead of auto detection.")
    parser.add_argument("--load-calibration", action="store_true", help="Load calibration json on startup if available.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames. 0 means all frames.")
    parser.add_argument("--no-display", action="store_true", help="Run without an OpenCV window.")
    parser.add_argument("--realtime-playback", action="store_true", help="Sleep to match source FPS for video files.")
    parser.add_argument("--redetect-interval", type=float, default=4.0, help="Seconds between line redetections.")
    parser.add_argument("--detect-max-width", type=int, default=960, help="Resize detection frame to this max width.")

    parser.add_argument("--white-s-max", type=int, default=130, help="HSV saturation upper bound for white lines.")
    parser.add_argument("--white-v-min", type=int, default=120, help="HSV value lower bound for white lines.")
    parser.add_argument("--white-chroma-max", type=int, default=96, help="Lab chroma distance upper bound for white lines.")
    parser.add_argument("--line-response-percentile", type=float, default=91.0)
    parser.add_argument("--line-response-min", type=int, default=72)
    parser.add_argument("--line-local-bg-ksize", type=int, default=31)
    parser.add_argument("--no-green-roi", dest="use_green_roi", action="store_false", help="Disable green-court ROI.")
    parser.set_defaults(use_green_roi=True)
    parser.add_argument("--green-h-min", type=int, default=30)
    parser.add_argument("--green-h-max", type=int, default=100)
    parser.add_argument("--green-s-min", type=int, default=70)
    parser.add_argument("--green-v-min", type=int, default=35)
    parser.add_argument("--white-green-pair-offset-px", type=int, default=8)
    parser.add_argument(
        "--keep-all-green-rois",
        action="store_true",
        help="Use every green court-like component instead of focusing the central primary court.",
    )

    parser.add_argument("--hough-threshold", type=int, default=45)
    parser.add_argument("--min-line-length-ratio", type=float, default=0.055)
    parser.add_argument("--max-line-gap-ratio", type=float, default=0.025)
    parser.add_argument("--angle-bin-deg", type=float, default=5.0)
    parser.add_argument("--angle-tol-deg", type=float, default=16.0)
    parser.add_argument("--min-angle-separation-deg", type=float, default=25.0)
    parser.add_argument("--merge-rho-px", type=float, default=18.0)
    parser.add_argument("--max-lines-per-family", type=int, default=3)
    parser.add_argument("--point-scheme", choices=("auto", "8", "6"), default="auto")
    parser.add_argument("--no-refine-homography", dest="refine_homography", action="store_false")
    parser.set_defaults(refine_homography=True)
    parser.add_argument("--snap-search-px", type=float, default=18.0)
    parser.add_argument("--snap-response-threshold", type=float, default=0.18)
    parser.add_argument("--max-refine-corner-shift-ratio", type=float, default=0.045)
    parser.add_argument("--green-side-offset-px", type=float, default=14.0)
    parser.add_argument("--min-outer-width-ratio", type=float, default=0.08)
    parser.add_argument("--min-outer-depth-ratio", type=float, default=0.08)
    parser.add_argument("--min-outer-width-depth-ratio", type=float, default=0.18)
    parser.add_argument("--max-outer-width-depth-ratio", type=float, default=5.5)
    parser.add_argument("--max-transverse-angle-deg", type=float, default=35.0)
    parser.add_argument("--no-corner-snap", dest="corner_snap", action="store_false", help="Disable local white-line intersection snapping for the four editable corners.")
    parser.set_defaults(corner_snap=True)
    parser.add_argument("--corner-snap-radius", type=int, default=80, help="Search radius in pixels around each editable corner.")
    parser.add_argument("--corner-snap-max-shift", type=int, default=48, help="Maximum pixel shift allowed when snapping a corner.")
    parser.add_argument("--corner-snap-min-line-length", type=int, default=28, help="Minimum local Hough segment length used for corner snapping.")
    parser.add_argument("--corner-snap-max-gap", type=int, default=12, help="Maximum local Hough gap used for corner snapping.")
    parser.add_argument("--corner-snap-hough-threshold", type=int, default=12, help="Local Hough threshold used for corner snapping.")
    parser.add_argument("--corner-snap-angle-tol", type=float, default=38.0, help="Angle tolerance against the current quadrilateral edge directions.")
    parser.add_argument("--corner-snap-min-angle-separation", type=float, default=22.0, help="Minimum angle separation between the two local lines at a corner.")
    parser.add_argument("--corner-snap-nearest-white-radius", type=int, default=6, help="Final snapped corner must be within this radius of a white mask pixel.")
    parser.add_argument("--corner-snap-min-white-support", type=float, default=0.035, help="Minimum local white mask support required at a snapped corner.")
    parser.add_argument("--corner-snap-strong-prior-support", type=float, default=0.28, help="If the original corner already has this much white support, reject large snap jumps.")
    parser.add_argument("--corner-snap-max-strong-prior-shift", type=int, default=18, help="Maximum snap shift allowed when the original corner is already well supported by white pixels.")
    parser.add_argument("--corner-snap-edge-band", type=int, default=82, help="Pixel band around each current outer edge used to refit the true white boundary line.")
    parser.add_argument("--corner-snap-edge-extension", type=int, default=160, help="Extra pixels past each current edge endpoint when refitting boundary lines.")
    parser.add_argument("--corner-snap-cross-search-radius", type=int, default=58, help="Search radius for the final two-direction white cross feature.")
    parser.add_argument("--corner-snap-ray-length", type=int, default=56, help="Ray length used to verify white support in both court-edge directions.")
    parser.add_argument("--corner-snap-min-ray-support", type=float, default=0.18, help="Minimum one-sided white support required for each of the two corner directions.")
    parser.add_argument("--corner-snap-min-centerline-score", type=float, default=0.22, help="Reject final corner pixels too close to the edge of a white stripe.")

    parser.add_argument("--reliable-conf", type=float, default=0.75)
    parser.add_argument("--medium-conf", type=float, default=0.55)
    parser.add_argument("--smooth-alpha-reliable", type=float, default=0.45)
    parser.add_argument("--smooth-alpha-medium", type=float, default=0.20)
    parser.add_argument("--jump-ratio-hard", type=float, default=0.18)

    parser.add_argument("--mask-alpha", type=float, default=0.14)
    parser.add_argument("--line-thickness", type=int, default=3)
    parser.add_argument("--point-radius", type=int, default=5)
    parser.add_argument("--draw-debug-lines", action="store_true", help="Draw merged Hough line families.")
    parser.add_argument("--show-labels", action="store_true", help="Draw detected keypoint labels.")
    parser.add_argument("--log-every", type=int, default=30)
    return parser.parse_args()


def open_source(source: str) -> cv2.VideoCapture:
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"Source video not found: {path}")
        cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")
    return cap


def create_writer(path: str, fps: float, size: tuple[int, int]) -> cv2.VideoWriter | None:
    if not path:
        return None
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {output_path}")
    return writer


def resize_for_detection(frame: np.ndarray, max_width: int) -> tuple[np.ndarray, float]:
    if max_width <= 0 or frame.shape[1] <= max_width:
        return frame, 1.0
    scale = max_width / float(frame.shape[1])
    resized = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return resized, scale


def angle_distance_deg(a: float, b: float) -> float:
    diff = abs(a - b) % 180.0
    return min(diff, 180.0 - diff)


def line_angle_deg(p1: np.ndarray, p2: np.ndarray) -> float:
    angle = math.degrees(math.atan2(float(p2[1] - p1[1]), float(p2[0] - p1[0])))
    return angle % 180.0


def line_normal(theta_deg: float) -> np.ndarray:
    theta = math.radians(theta_deg)
    return np.array([-math.sin(theta), math.cos(theta)], dtype=np.float32)


def line_direction(theta_deg: float) -> np.ndarray:
    theta = math.radians(theta_deg)
    return np.array([math.cos(theta), math.sin(theta)], dtype=np.float32)


def polygon_area(points: np.ndarray) -> float:
    return float(cv2.contourArea(np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)))


def is_convex_quad(points: np.ndarray) -> bool:
    return bool(cv2.isContourConvex(np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)))


def order_quad_points(points: np.ndarray) -> np.ndarray | None:
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape != (4, 2) or not np.isfinite(pts).all():
        return None

    y_sorted = np.argsort(pts[:, 1])
    top = y_sorted[:2]
    bottom = y_sorted[2:]
    top = top[np.argsort(pts[top, 0])]
    bottom = bottom[np.argsort(pts[bottom, 0])]
    ordered = pts[[top[0], top[1], bottom[1], bottom[0]]]
    if polygon_area(ordered) >= 1.0 and is_convex_quad(ordered):
        return ordered

    sums = pts[:, 0] + pts[:, 1]
    diffs = pts[:, 1] - pts[:, 0]
    fallback_idx = [int(np.argmin(sums)), int(np.argmin(diffs)), int(np.argmax(sums)), int(np.argmax(diffs))]
    if len(set(fallback_idx)) != 4:
        return None
    ordered = pts[fallback_idx]
    if polygon_area(ordered) < 1.0 or not is_convex_quad(ordered):
        return None
    return ordered.astype(np.float32)


def project_points(points: np.ndarray, h: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, h).reshape(-1, 2)


def point_to_segment_distance(point: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    point = np.asarray(point, dtype=np.float32)
    p1 = np.asarray(p1, dtype=np.float32)
    p2 = np.asarray(p2, dtype=np.float32)
    direction = p2 - p1
    denom = float(np.dot(direction, direction))
    if denom <= 1e-6:
        return float(np.linalg.norm(point - p1))
    t = float(np.clip(np.dot(point - p1, direction) / denom, 0.0, 1.0))
    closest = p1 + direction * t
    return float(np.linalg.norm(point - closest))


def point_to_line_distance(point: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    point = np.asarray(point, dtype=np.float32)
    p1 = np.asarray(p1, dtype=np.float32)
    p2 = np.asarray(p2, dtype=np.float32)
    direction = p2 - p1
    length = float(np.linalg.norm(direction))
    if length <= 1e-6:
        return float(np.linalg.norm(point - p1))
    cross = float(direction[0] * (point[1] - p1[1]) - direction[1] * (point[0] - p1[0]))
    return abs(cross) / length


def nearest_mask_pixel(mask: np.ndarray, point: np.ndarray, radius: int) -> np.ndarray | None:
    height, width = mask.shape[:2]
    x = int(round(float(point[0])))
    y = int(round(float(point[1])))
    search_radius = max(1, int(radius))
    x1 = max(0, x - search_radius)
    x2 = min(width, x + search_radius + 1)
    y1 = max(0, y - search_radius)
    y2 = min(height, y + search_radius + 1)
    patch = mask[y1:y2, x1:x2]
    ys, xs = np.where(patch > 0)
    if len(xs) == 0:
        return None
    coords = np.stack([xs + x1, ys + y1], axis=1).astype(np.float32)
    distances = np.linalg.norm(coords - np.asarray(point, dtype=np.float32).reshape(1, 2), axis=1)
    return coords[int(np.argmin(distances))]


def ray_mask_support(mask: np.ndarray, point: np.ndarray, angle_deg: float, args: argparse.Namespace) -> float:
    height, width = mask.shape[:2]
    direction = line_direction(angle_deg)
    length = max(12.0, float(args.corner_snap_ray_length))
    samples = max(5, int(length / 5.0))
    patch_radius = 2
    best = 0.0
    for sign in (-1.0, 1.0):
        hits = 0
        total = 0
        for distance in np.linspace(4.0, length, samples, dtype=np.float32):
            sample = point + direction * float(sign * distance)
            x = int(round(float(sample[0])))
            y = int(round(float(sample[1])))
            if x < 0 or x >= width or y < 0 or y >= height:
                continue
            total += 1
            patch = mask[
                max(0, y - patch_radius) : min(height, y + patch_radius + 1),
                max(0, x - patch_radius) : min(width, x + patch_radius + 1),
            ]
            if cv2.countNonZero(patch) > 0:
                hits += 1
        if total > 0:
            best = max(best, hits / float(total))
    return best


def refine_to_white_cross_feature(
    mask: np.ndarray,
    initial_point: np.ndarray,
    angle_a: float,
    angle_b: float,
    args: argparse.Namespace,
) -> np.ndarray | None:
    height, width = mask.shape[:2]
    x = int(round(float(initial_point[0])))
    y = int(round(float(initial_point[1])))
    search_radius = max(3, int(args.corner_snap_cross_search_radius))
    x1 = max(0, x - search_radius)
    x2 = min(width, x + search_radius + 1)
    y1 = max(0, y - search_radius)
    y2 = min(height, y + search_radius + 1)
    patch = mask[y1:y2, x1:x2]
    ys, xs = np.where(patch > 0)
    if len(xs) == 0:
        return None
    harris = cv2.cornerHarris(np.float32(patch), blockSize=5, ksize=3, k=0.04)
    harris = cv2.dilate(harris, None)
    harris[patch == 0] = 0.0
    max_harris = float(np.max(harris)) if harris.size else 0.0
    distance_transform = cv2.distanceTransform(patch, cv2.DIST_L2, 3)
    max_distance = float(np.max(distance_transform)) if distance_transform.size else 0.0

    local_coords = np.stack([xs, ys], axis=1).astype(np.int32)
    global_coords = np.stack([xs + x1, ys + y1], axis=1).astype(np.float32)
    harris_values = harris[ys, xs].astype(np.float32) if max_harris > 0 else np.zeros((len(xs),), dtype=np.float32)
    centerline_values = distance_transform[ys, xs].astype(np.float32) if max_distance > 0 else np.zeros((len(xs),), dtype=np.float32)
    distances = np.linalg.norm(global_coords - np.asarray(initial_point, dtype=np.float32).reshape(1, 2), axis=1)
    pre_scores = (
        harris_values / max(max_harris, 1e-6)
        + 0.65 * centerline_values / max(max_distance, 1e-6)
        + 0.18 * np.clip(1.0 - distances / max(1.0, search_radius), 0.0, 1.0)
    )
    order = np.argsort(pre_scores)[::-1][: min(220, len(pre_scores))]
    coords = global_coords[order]
    local_coords = local_coords[order]
    min_ray_support = max(0.0, float(args.corner_snap_min_ray_support))
    min_centerline_score = max(0.0, float(args.corner_snap_min_centerline_score))
    best_point: np.ndarray | None = None
    best_score = -1.0
    for point, local_point in zip(coords, local_coords):
        local_x = int(local_point[0])
        local_y = int(local_point[1])
        centerline_score = float(np.clip(float(distance_transform[local_y, local_x]) / max(max_distance, 1e-6), 0.0, 1.0)) if max_distance > 0 else 0.0
        if centerline_score < min_centerline_score:
            continue
        support_a = ray_mask_support(mask, point, angle_a, args)
        support_b = ray_mask_support(mask, point, angle_b, args)
        min_support = min(support_a, support_b)
        if min_support < min_ray_support:
            continue
        distance = float(np.linalg.norm(point - initial_point))
        distance_score = float(np.clip(1.0 - distance / max(1.0, search_radius), 0.0, 1.0))
        patch_score = float(np.clip(point_patch_support(mask, point, radius=4) / 0.12, 0.0, 1.0))
        corner_score = float(np.clip(float(harris[local_y, local_x]) / max(max_harris, 1e-6), 0.0, 1.0)) if max_harris > 0 else 0.0
        score = (
            0.34 * min_support
            + 0.17 * ((support_a + support_b) * 0.5)
            + 0.12 * distance_score
            + 0.08 * patch_score
            + 0.17 * corner_score
            + 0.12 * centerline_score
        )
        if score > best_score:
            best_score = score
            best_point = point
    return best_point.astype(np.float32) if best_point is not None else None


def compute_homographies(corners: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    court_to_image_h, _ = cv2.findHomography(STANDARD_COURT_TEMPLATE.corners, corners, 0)
    image_to_court_h, _ = cv2.findHomography(corners, STANDARD_COURT_TEMPLATE.corners, 0)
    if court_to_image_h is None or image_to_court_h is None:
        return None, None
    if not np.isfinite(court_to_image_h).all() or not np.isfinite(image_to_court_h).all():
        return None, None
    return court_to_image_h.astype(np.float64), image_to_court_h.astype(np.float64)


def project_template_lines(h: np.ndarray) -> dict[str, np.ndarray]:
    return {name: project_points(line, h) for name, line in STANDARD_COURT_TEMPLATE.lines}


def template_keypoints_for_scheme(scheme: str) -> tuple[tuple[str, tuple[float, float]], ...]:
    if scheme == "6":
        return STANDARD_COURT_TEMPLATE.keypoints_6
    return STANDARD_COURT_TEMPLATE.keypoints_8


def create_white_line_mask(frame: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    if args.use_green_roi:
        lower_green = np.array([int(args.green_h_min), int(args.green_s_min), int(args.green_v_min)], dtype=np.uint8)
        upper_green = np.array([int(args.green_h_max), 255, 255], dtype=np.uint8)
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        green_mask = cv2.morphologyEx(
            green_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
            iterations=2,
        )
        green_mask = keep_large_components(green_mask, min_area=max(200, int(frame.shape[0] * frame.shape[1] * 0.01)))
        if not args.keep_all_green_rois:
            green_mask = keep_primary_green_component(green_mask)
        if cv2.countNonZero(green_mask) > frame.shape[0] * frame.shape[1] * 0.03:
            green_roi = cv2.dilate(
                green_mask,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)),
                iterations=1,
            )
        else:
            green_roi = np.full(frame.shape[:2], 255, dtype=np.uint8)
    else:
        green_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        green_roi = np.full(frame.shape[:2], 255, dtype=np.uint8)

    white_mask = build_white_line_feature_mask(frame, hsv, green_mask, green_roi, args)
    white_mask = cv2.morphologyEx(
        white_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )
    white_mask = cv2.morphologyEx(
        white_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
        iterations=2,
    )
    white_mask = keep_large_components(white_mask, min_area=max(16, int(frame.shape[0] * frame.shape[1] * 0.00003)))
    return white_mask, green_mask


def normalize_to_u8(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    if max_value <= min_value + 1e-6:
        return np.zeros(values.shape, dtype=np.uint8)
    normalized = (values - min_value) * (255.0 / (max_value - min_value))
    return np.clip(normalized, 0, 255).astype(np.uint8)


def odd_kernel_size(value: int, minimum: int = 3) -> int:
    size = max(minimum, int(value))
    return size if size % 2 == 1 else size + 1


def build_white_line_feature_mask(
    frame: np.ndarray,
    hsv: np.ndarray,
    green_mask: np.ndarray,
    green_roi: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    """Model white court lines as bright, low-chroma ridges embedded in green court pixels."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    bg_ksize = odd_kernel_size(int(args.line_local_bg_ksize), minimum=9)
    local_background = cv2.GaussianBlur(enhanced_l, (bg_ksize, bg_ksize), 0)
    local_contrast = cv2.subtract(enhanced_l, local_background)

    top_hat_parts = []
    for kernel in (
        cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 17)),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
    ):
        top_hat_parts.append(cv2.morphologyEx(enhanced_l, cv2.MORPH_TOPHAT, kernel))
    top_hat = np.maximum.reduce(top_hat_parts)

    lab_chroma = np.abs(a_channel.astype(np.int16) - 128) + np.abs(b_channel.astype(np.int16) - 128)
    low_chroma = np.clip(255.0 - lab_chroma.astype(np.float32) * 2.2, 0, 255).astype(np.uint8)
    low_saturation = cv2.subtract(np.full_like(saturation, 255), saturation)

    contrast_score = normalize_to_u8(local_contrast)
    top_hat_score = normalize_to_u8(top_hat)
    response = (
        0.42 * contrast_score.astype(np.float32)
        + 0.34 * top_hat_score.astype(np.float32)
        + 0.16 * low_saturation.astype(np.float32)
        + 0.08 * low_chroma.astype(np.float32)
    )
    response = np.clip(response, 0, 255).astype(np.uint8)

    roi_values = response[green_roi > 0]
    if roi_values.size:
        adaptive_threshold = float(np.percentile(roi_values, float(args.line_response_percentile)))
    else:
        adaptive_threshold = float(args.line_response_min)
    threshold = int(np.clip(max(float(args.line_response_min), adaptive_threshold), 0, 255))
    _, response_mask = cv2.threshold(response, threshold, 255, cv2.THRESH_BINARY)

    low_chroma_mask = (lab_chroma <= int(args.white_chroma_max)).astype(np.uint8) * 255
    white_gate = cv2.inRange(
        hsv,
        np.array([0, 0, max(0, int(args.white_v_min) - 25)], dtype=np.uint8),
        np.array([179, int(args.white_s_max), 255], dtype=np.uint8),
    )
    white_gate = cv2.bitwise_or(white_gate, cv2.bitwise_and(response_mask, low_chroma_mask))
    response_mask = cv2.bitwise_and(response_mask, white_gate)
    response_mask = cv2.bitwise_and(response_mask, green_roi)

    if cv2.countNonZero(green_mask) > 0:
        paired_green = paired_green_support_mask(green_mask, int(args.white_green_pair_offset_px))
        response_mask = cv2.bitwise_and(response_mask, paired_green)
    return response_mask


def shift_mask(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    height, width = mask.shape[:2]
    shifted = np.zeros_like(mask)
    src_x1 = max(0, -dx)
    src_x2 = min(width, width - dx)
    dst_x1 = max(0, dx)
    dst_x2 = min(width, width + dx)
    src_y1 = max(0, -dy)
    src_y2 = min(height, height - dy)
    dst_y1 = max(0, dy)
    dst_y2 = min(height, height + dy)
    if src_x1 >= src_x2 or src_y1 >= src_y2:
        return shifted
    shifted[dst_y1:dst_y2, dst_x1:dst_x2] = mask[src_y1:src_y2, src_x1:src_x2]
    return shifted


def paired_green_support_mask(green_mask: np.ndarray, offset_px: int) -> np.ndarray:
    offset = max(3, int(offset_px))
    green = cv2.dilate(green_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    pairs = [
        ((0, -offset), (0, offset)),
        ((-offset, 0), (offset, 0)),
        ((-offset, -offset), (offset, offset)),
        ((-offset, offset), (offset, -offset)),
    ]
    support = np.zeros_like(green)
    for (dx1, dy1), (dx2, dy2) in pairs:
        side_a = shift_mask(green, dx1, dy1)
        side_b = shift_mask(green, dx2, dy2)
        support = cv2.bitwise_or(support, cv2.bitwise_and(side_a, side_b))
    return support


def keep_primary_green_component(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 2:
        return mask
    height, width = mask.shape[:2]
    image_center = np.array([width * 0.5, height * 0.62], dtype=np.float32)
    diag = float(np.hypot(width, height))
    best_label = -1
    best_score = -1.0
    for label in range(1, num_labels):
        area = float(stats[label, cv2.CC_STAT_AREA])
        centroid = np.asarray(centroids[label], dtype=np.float32)
        center_distance = float(np.linalg.norm(centroid - image_center)) / max(1.0, diag)
        center_score = float(np.clip(1.0 - center_distance * 2.2, 0.15, 1.0))
        score = area * center_score
        if score > best_score:
            best_score = score
            best_label = label
    out = np.zeros_like(mask)
    if best_label > 0:
        out[labels == best_label] = 255
    return out


def keep_large_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    out = np.zeros_like(mask)
    for label in range(1, num_labels):
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_area:
            out[labels == label] = 255
    return out


def detect_hough_segments(mask: np.ndarray, args: argparse.Namespace) -> tuple[list[LineSegment], np.ndarray]:
    height, width = mask.shape[:2]
    diag = float(np.hypot(width, height))
    edges = cv2.Canny(mask, 50, 150, apertureSize=3)
    raw_lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=max(1, int(args.hough_threshold)),
        minLineLength=max(20, int(diag * float(args.min_line_length_ratio))),
        maxLineGap=max(4, int(diag * float(args.max_line_gap_ratio))),
    )
    segments: list[LineSegment] = []
    if raw_lines is None:
        return segments, edges
    min_len = max(16.0, diag * float(args.min_line_length_ratio) * 0.6)
    for raw in raw_lines.reshape(-1, 4):
        p1 = np.array([float(raw[0]), float(raw[1])], dtype=np.float32)
        p2 = np.array([float(raw[2]), float(raw[3])], dtype=np.float32)
        length = float(np.linalg.norm(p2 - p1))
        if length < min_len:
            continue
        segments.append(LineSegment(p1=p1, p2=p2, angle_deg=line_angle_deg(p1, p2), length=length))
    segments.sort(key=lambda item: item.length, reverse=True)
    return segments, edges


def choose_direction_families(
    segments: list[LineSegment],
    args: argparse.Namespace,
) -> tuple[list[LineSegment], list[LineSegment], tuple[float, float] | None]:
    if len(segments) < 4:
        return [], [], None

    bin_size = max(1.0, float(args.angle_bin_deg))
    bin_count = max(18, int(round(180.0 / bin_size)))
    hist = np.zeros((bin_count,), dtype=np.float32)
    for seg in segments:
        index = int(round((seg.angle_deg % 180.0) / bin_size)) % bin_count
        hist[index] += float(seg.length)

    smoothed = hist.copy()
    for shift in (-1, 1):
        smoothed += np.roll(hist, shift) * 0.45
    candidate_bins = np.argsort(smoothed)[::-1][: min(12, bin_count)]
    candidate_angles = [float((index * bin_size) % 180.0) for index in candidate_bins if smoothed[index] > 0]

    best_pair: tuple[float, float] | None = None
    best_score = -1.0
    min_sep = float(args.min_angle_separation_deg)
    angle_tol = float(args.angle_tol_deg)
    for angle_a, angle_b in itertools.combinations(candidate_angles, 2):
        separation = angle_distance_deg(angle_a, angle_b)
        if separation < min_sep or separation > 180.0 - min_sep:
            continue
        score = 0.0
        for seg in segments:
            dist_a = angle_distance_deg(seg.angle_deg, angle_a)
            dist_b = angle_distance_deg(seg.angle_deg, angle_b)
            if min(dist_a, dist_b) <= angle_tol:
                score += seg.length
        if score > best_score:
            best_score = score
            best_pair = (angle_a, angle_b)

    if best_pair is None:
        return [], [], None

    family_a: list[LineSegment] = []
    family_b: list[LineSegment] = []
    angle_a, angle_b = best_pair
    for seg in segments:
        dist_a = angle_distance_deg(seg.angle_deg, angle_a)
        dist_b = angle_distance_deg(seg.angle_deg, angle_b)
        if min(dist_a, dist_b) > angle_tol:
            continue
        if dist_a <= dist_b:
            family_a.append(seg)
        else:
            family_b.append(seg)

    if len(family_a) < 2 or len(family_b) < 2:
        return [], [], None
    return family_a, family_b, best_pair


def choose_angle_clusters(
    segments: list[LineSegment],
    args: argparse.Namespace,
    max_clusters: int = 4,
) -> list[tuple[float, list[LineSegment]]]:
    if len(segments) < 4:
        return []
    bin_size = max(1.0, float(args.angle_bin_deg))
    bin_count = max(18, int(round(180.0 / bin_size)))
    hist = np.zeros((bin_count,), dtype=np.float32)
    for seg in segments:
        index = int(round((seg.angle_deg % 180.0) / bin_size)) % bin_count
        hist[index] += float(seg.length)
    smoothed = hist.copy()
    for shift in (-1, 1):
        smoothed += np.roll(hist, shift) * 0.45

    peak_angles: list[float] = []
    min_sep = float(args.min_angle_separation_deg)
    for index in np.argsort(smoothed)[::-1]:
        if smoothed[index] <= 0:
            break
        angle = float((int(index) * bin_size) % 180.0)
        if all(angle_distance_deg(angle, existing) >= min_sep for existing in peak_angles):
            peak_angles.append(angle)
        if len(peak_angles) >= max_clusters:
            break

    clusters: list[tuple[float, list[LineSegment]]] = []
    angle_tol = float(args.angle_tol_deg)
    for angle in peak_angles:
        cluster = [seg for seg in segments if angle_distance_deg(seg.angle_deg, angle) <= angle_tol]
        if cluster:
            clusters.append((angle, cluster))
    clusters.sort(key=lambda item: sum(seg.length for seg in item[1]), reverse=True)
    return clusters


def merge_line_family(segments: list[LineSegment], theta_deg: float, args: argparse.Namespace) -> list[MergedLine]:
    if not segments:
        return []
    normal = line_normal(theta_deg)
    rows: list[tuple[float, LineSegment]] = []
    for seg in segments:
        rho = float(np.dot(normal, seg.midpoint))
        rows.append((rho, seg))
    rows.sort(key=lambda item: item[0])

    merged: list[MergedLine] = []
    current: list[tuple[float, LineSegment]] = []
    threshold = max(4.0, float(args.merge_rho_px))

    def flush(group: list[tuple[float, LineSegment]]) -> None:
        if not group:
            return
        weights = np.array([max(1.0, item[1].length) for item in group], dtype=np.float32)
        rhos = np.array([item[0] for item in group], dtype=np.float32)
        rho = float(np.average(rhos, weights=weights))
        length = float(np.sum(weights))
        merged.append(MergedLine(theta_deg=theta_deg, rho=rho, length=length, count=len(group)))

    last_rho: float | None = None
    for rho, seg in rows:
        if last_rho is None or abs(rho - last_rho) <= threshold:
            current.append((rho, seg))
        else:
            flush(current)
            current = [(rho, seg)]
        last_rho = rho if last_rho is None else 0.65 * last_rho + 0.35 * rho
    flush(current)
    merged.sort(key=lambda line: line.length, reverse=True)
    return merged


def intersect_merged_lines(line_a: MergedLine, line_b: MergedLine) -> np.ndarray | None:
    normal_a = line_normal(line_a.theta_deg).astype(np.float64)
    normal_b = line_normal(line_b.theta_deg).astype(np.float64)
    matrix = np.vstack([normal_a, normal_b])
    det = float(np.linalg.det(matrix))
    if abs(det) < 1e-5:
        return None
    rhs = np.array([line_a.rho, line_b.rho], dtype=np.float64)
    point = np.linalg.solve(matrix, rhs)
    if not np.isfinite(point).all():
        return None
    return point.astype(np.float32)


def all_family_intersections(
    family_a: list[MergedLine],
    family_b: list[MergedLine],
    frame_shape: tuple[int, ...],
    margin_ratio: float = 0.12,
) -> list[np.ndarray]:
    height, width = frame_shape[:2]
    margin_x = width * margin_ratio
    margin_y = height * margin_ratio
    points: list[np.ndarray] = []
    for line_a in family_a:
        for line_b in family_b:
            point = intersect_merged_lines(line_a, line_b)
            if point is None:
                continue
            if -margin_x <= point[0] <= width + margin_x and -margin_y <= point[1] <= height + margin_y:
                points.append(point)
    return points


def line_to_frame_points(line: MergedLine, frame_shape: tuple[int, ...]) -> tuple[tuple[int, int], tuple[int, int]] | None:
    height, width = frame_shape[:2]
    direction = line_direction(line.theta_deg).astype(np.float64)
    normal = line_normal(line.theta_deg).astype(np.float64)
    base = normal * float(line.rho)
    candidates: list[np.ndarray] = []
    if abs(direction[0]) > 1e-8:
        for x in (0.0, float(width - 1)):
            t = (x - base[0]) / direction[0]
            y = base[1] + t * direction[1]
            if -1.0 <= y <= height:
                candidates.append(np.array([x, y], dtype=np.float64))
    if abs(direction[1]) > 1e-8:
        for y in (0.0, float(height - 1)):
            t = (y - base[1]) / direction[1]
            x = base[0] + t * direction[0]
            if -1.0 <= x <= width:
                candidates.append(np.array([x, y], dtype=np.float64))
    if len(candidates) < 2:
        return None
    best_pair = max(itertools.combinations(candidates, 2), key=lambda pair: float(np.linalg.norm(pair[0] - pair[1])))
    p1 = tuple(np.round(best_pair[0]).astype(int).tolist())
    p2 = tuple(np.round(best_pair[1]).astype(int).tolist())
    return p1, p2


def quad_bounds_score(points: np.ndarray, frame_shape: tuple[int, ...]) -> float:
    height, width = frame_shape[:2]
    inside = (
        (points[:, 0] >= -width * 0.04)
        & (points[:, 0] <= width * 1.04)
        & (points[:, 1] >= -height * 0.04)
        & (points[:, 1] <= height * 1.04)
    )
    return float(np.mean(inside.astype(np.float32)))


def quad_geometry_score(points: np.ndarray, frame_shape: tuple[int, ...]) -> float:
    if points.shape != (4, 2) or not np.isfinite(points).all() or not is_convex_quad(points):
        return 0.0
    height, width = frame_shape[:2]
    frame_area = max(1.0, float(width * height))
    area_ratio = polygon_area(points) / frame_area
    if area_ratio <= 0.002:
        return 0.0
    area_score = float(np.clip((area_ratio - 0.01) / 0.24, 0.0, 1.0))
    side_lengths = np.linalg.norm(points - np.roll(points, -1, axis=0), axis=1)
    min_side = float(np.min(side_lengths))
    diag = float(np.hypot(width, height))
    side_score = float(np.clip(min_side / max(1.0, diag * 0.10), 0.0, 1.0))
    return 0.55 * area_score + 0.45 * side_score


def court_shape_sanity_score(points: np.ndarray, frame_shape: tuple[int, ...], args: argparse.Namespace) -> float:
    if points.shape != (4, 2) or not np.isfinite(points).all() or not is_convex_quad(points):
        return 0.0
    height, width = frame_shape[:2]
    diag = max(1.0, float(np.hypot(width, height)))
    edge_lengths = np.linalg.norm(points - np.roll(points, -1, axis=0), axis=1)
    top_width, right_depth, bottom_width, left_depth = [float(value) for value in edge_lengths]
    court_width_px = max(top_width, bottom_width)
    court_depth_px = max(left_depth, right_depth)
    if court_width_px <= 1.0 or court_depth_px <= 1.0:
        return 0.0

    min_width_ratio = max(0.01, float(getattr(args, "min_outer_width_ratio", 0.08)))
    min_depth_ratio = max(0.01, float(getattr(args, "min_outer_depth_ratio", 0.08)))
    min_aspect = max(0.01, float(getattr(args, "min_outer_width_depth_ratio", 0.18)))
    max_aspect = max(min_aspect + 0.01, float(getattr(args, "max_outer_width_depth_ratio", 5.5)))

    width_score = float(np.clip(court_width_px / (diag * min_width_ratio), 0.0, 1.0))
    depth_score = float(np.clip(court_depth_px / (diag * min_depth_ratio), 0.0, 1.0))
    aspect = court_width_px / court_depth_px
    if aspect < min_aspect:
        aspect_score = float(np.clip(aspect / min_aspect, 0.0, 1.0))
    elif aspect > max_aspect:
        aspect_score = float(np.clip(max_aspect / aspect, 0.0, 1.0))
    else:
        aspect_score = 1.0
    return min(width_score, depth_score, aspect_score)


def is_likely_transverse_family(theta_deg: float, args: argparse.Namespace) -> bool:
    max_angle = float(np.clip(getattr(args, "max_transverse_angle_deg", 35.0), 0.0, 90.0))
    return angle_distance_deg(float(theta_deg), 0.0) <= max_angle


def projected_line_mask_support(mask: np.ndarray, line_points: np.ndarray, sample_step: float = 12.0, radius: int = 2) -> float:
    if len(line_points) < 2 or not np.isfinite(line_points).all():
        return 0.0
    height, width = mask.shape[:2]
    hits = 0
    total = 0
    for p1, p2 in zip(line_points[:-1], line_points[1:]):
        length = float(np.linalg.norm(p2 - p1))
        samples = max(2, int(length / max(1.0, sample_step)))
        for t in np.linspace(0.0, 1.0, samples, dtype=np.float32):
            point = p1 * (1.0 - t) + p2 * t
            x = int(round(float(point[0])))
            y = int(round(float(point[1])))
            if x < 0 or x >= width or y < 0 or y >= height:
                continue
            x1 = max(0, x - radius)
            x2 = min(width, x + radius + 1)
            y1 = max(0, y - radius)
            y2 = min(height, y + radius + 1)
            total += 1
            if cv2.countNonZero(mask[y1:y2, x1:x2]) > 0:
                hits += 1
    if total == 0:
        return 0.0
    return hits / float(total)


def projected_template_support(projected_lines: dict[str, np.ndarray], mask: np.ndarray) -> tuple[float, int]:
    if not projected_lines:
        return 0.0, 0
    supports: list[float] = []
    supported_lines = 0
    dilated = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    for _, points in projected_lines.items():
        score = projected_line_mask_support(dilated, points)
        supports.append(score)
        if score >= 0.18:
            supported_lines += 1
    if not supports:
        return 0.0, 0
    return float(np.mean(supports)), supported_lines


def projected_singles_support_components(
    projected_lines: dict[str, np.ndarray],
    mask: np.ndarray,
    aggregate_support: float,
) -> dict[str, float]:
    dilated = cv2.dilate(
        mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    left_support = projected_line_mask_support(
        dilated,
        np.asarray(projected_lines.get("singles_left_sideline", []), dtype=np.float32),
    )
    right_support = projected_line_mask_support(
        dilated,
        np.asarray(projected_lines.get("singles_right_sideline", []), dtype=np.float32),
    )
    min_support = min(left_support, right_support)
    return {
        "singles_left_support": float(left_support),
        "singles_right_support": float(right_support),
        "singles_min_support": float(min_support),
        "singles_support_ratio": float(min_support / max(float(aggregate_support), 1.0e-6)),
    }


def projected_outer_support_components(
    projected_lines: dict[str, np.ndarray],
    mask: np.ndarray,
    *,
    sample_step: float = 8.0,
    radius: int = 3,
) -> dict[str, float]:
    """Measure white-line support independently on all four doubles-court edges."""
    edge_names = ("top", "right", "bottom", "left")
    supports = {f"outer_{name}_support": 0.0 for name in edge_names}
    raw_outer = np.asarray(projected_lines.get("doubles_outer", []), dtype=np.float32)
    if raw_outer.size != 8:
        supports.update({"outer_min_support": 0.0, "outer_mean_support": 0.0})
        return supports
    outer = raw_outer.reshape(4, 2)
    if not np.isfinite(outer).all() or mask.size == 0:
        supports.update({"outer_min_support": 0.0, "outer_mean_support": 0.0})
        return supports

    edge_values: list[float] = []
    for index, name in enumerate(edge_names):
        edge = np.asarray([outer[index], outer[(index + 1) % 4]], dtype=np.float32)
        value = projected_line_mask_support(
            mask,
            edge,
            sample_step=sample_step,
            radius=radius,
        )
        supports[f"outer_{name}_support"] = float(value)
        edge_values.append(float(value))
    supports["outer_min_support"] = float(np.min(edge_values))
    supports["outer_mean_support"] = float(np.mean(edge_values))
    return supports


def sample_mask(mask: np.ndarray, point: np.ndarray) -> float:
    height, width = mask.shape[:2]
    x = int(round(float(point[0])))
    y = int(round(float(point[1])))
    if x < 0 or x >= width or y < 0 or y >= height:
        return 0.0
    return float(mask[y, x]) / 255.0


def outer_line_green_side_support(
    court_to_image_h: np.ndarray,
    green_mask: np.ndarray,
    offset_px: float,
) -> float:
    """Check that the projected outer court lines are surrounded by green on both sides.

    A common false positive is the green mat / wood floor border: it is long and straight,
    but one side is not green. True court white lines usually have green on both sides.
    """
    if green_mask.size == 0 or cv2.countNonZero(green_mask) == 0:
        return 0.55

    green = cv2.dilate(green_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1)
    projected = project_points(STANDARD_COURT_TEMPLATE.corners, court_to_image_h)
    scores: list[float] = []
    offset = max(4.0, float(offset_px))
    for p1, p2 in zip(projected, np.roll(projected, -1, axis=0)):
        direction = p2 - p1
        length = float(np.linalg.norm(direction))
        if length < 1.0:
            continue
        direction /= length
        normal = np.array([-direction[1], direction[0]], dtype=np.float32)
        samples = max(8, min(80, int(length / 18.0)))
        side_a = 0
        side_b = 0
        total = 0
        for t in np.linspace(0.08, 0.92, samples, dtype=np.float32):
            point = p1 * (1.0 - t) + p2 * t
            total += 1
            if sample_mask(green, point + normal * offset) > 0:
                side_a += 1
            if sample_mask(green, point - normal * offset) > 0:
                side_b += 1
        if total:
            a_ratio = side_a / float(total)
            b_ratio = side_b / float(total)
            scores.append(min(a_ratio, b_ratio))
    if not scores:
        return 0.0
    return float(np.mean(scores))


def refine_homography_with_white_lines(
    court_to_image_h: np.ndarray,
    mask: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, int, float]:
    """Snap projected template samples to nearby white-line pixels and refit H."""
    if not args.refine_homography:
        return court_to_image_h, 0, 0.0

    search_px = max(4.0, float(args.snap_search_px))
    response_threshold = float(np.clip(args.snap_response_threshold, 0.0, 1.0))
    response = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    response = cv2.GaussianBlur(response, (7, 7), 0)
    template_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    shifts: list[float] = []

    for _, template_line in STANDARD_COURT_TEMPLATE.lines:
        projected = project_points(template_line, court_to_image_h)
        p1, p2 = projected[0], projected[-1]
        direction = p2 - p1
        length = float(np.linalg.norm(direction))
        if length < 1.0:
            continue
        direction /= length
        normal = np.array([-direction[1], direction[0]], dtype=np.float32)
        samples = max(5, min(48, int(length / 28.0)))
        offsets = np.linspace(-search_px, search_px, int(search_px * 2.0) + 1, dtype=np.float32)
        for t in np.linspace(0.08, 0.92, samples, dtype=np.float32):
            template_point = template_line[0] * (1.0 - t) + template_line[-1] * t
            projected_point = p1 * (1.0 - t) + p2 * t
            best_score = 0.0
            best_point: np.ndarray | None = None
            best_shift = 0.0
            for offset in offsets:
                candidate = projected_point + normal * float(offset)
                score = sample_mask(response, candidate)
                if score > best_score:
                    best_score = score
                    best_point = candidate
                    best_shift = float(abs(offset))
            if best_point is None or best_score < response_threshold:
                continue
            template_points.append(template_point.astype(np.float32))
            image_points.append(best_point.astype(np.float32))
            shifts.append(best_shift)

    if len(template_points) < 12:
        return court_to_image_h, len(template_points), float(np.mean(shifts)) if shifts else 0.0

    src = np.asarray(template_points, dtype=np.float32)
    dst = np.asarray(image_points, dtype=np.float32)
    refined_h, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if refined_h is None or not np.isfinite(refined_h).all():
        return court_to_image_h, len(template_points), float(np.mean(shifts)) if shifts else 0.0
    inlier_count = int(np.sum(inliers)) if inliers is not None else len(template_points)
    if inlier_count < 10:
        return court_to_image_h, len(template_points), float(np.mean(shifts)) if shifts else 0.0
    return refined_h.astype(np.float64), inlier_count, float(np.mean(shifts)) if shifts else 0.0


def point_patch_support(mask: np.ndarray, point: np.ndarray, radius: int = 6) -> float:
    height, width = mask.shape[:2]
    x = int(round(float(point[0])))
    y = int(round(float(point[1])))
    if x < 0 or x >= width or y < 0 or y >= height:
        return 0.0
    x1 = max(0, x - radius)
    x2 = min(width, x + radius + 1)
    y1 = max(0, y - radius)
    y2 = min(height, y + radius + 1)
    patch = mask[y1:y2, x1:x2]
    if patch.size == 0:
        return 0.0
    return float(cv2.countNonZero(patch)) / float(patch.size)


def select_keypoints(
    court_to_image_h: np.ndarray,
    intersections: list[np.ndarray],
    mask: np.ndarray,
    args: argparse.Namespace,
) -> tuple[str, list[str], np.ndarray, int]:
    schemes = ["8", "6"] if args.point_scheme == "auto" else [args.point_scheme]
    best: tuple[str, list[str], np.ndarray, int] | None = None
    best_ratio = -1.0
    intersection_array = np.asarray(intersections, dtype=np.float32) if intersections else np.empty((0, 2), dtype=np.float32)
    tol = max(10.0, float(np.hypot(mask.shape[1], mask.shape[0])) * 0.015)
    dilated = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1)

    for scheme in schemes:
        names_and_points = template_keypoints_for_scheme(scheme)
        names = [item[0] for item in names_and_points]
        template_points = np.asarray([item[1] for item in names_and_points], dtype=np.float32)
        image_points = project_points(template_points, court_to_image_h)
        supported = 0
        for point in image_points:
            near_intersection = False
            if intersection_array.size:
                distances = np.linalg.norm(intersection_array - point.reshape(1, 2), axis=1)
                near_intersection = bool(np.min(distances) <= tol)
            white_support = point_patch_support(dilated, point, radius=7) >= 0.05
            if near_intersection or white_support:
                supported += 1
        ratio = supported / float(len(names))
        if ratio > best_ratio:
            best_ratio = ratio
            best = (scheme, names, image_points, supported)
    if best is None:
        names_and_points = template_keypoints_for_scheme("6")
        names = [item[0] for item in names_and_points]
        template_points = np.asarray([item[1] for item in names_and_points], dtype=np.float32)
        return "6", names, project_points(template_points, court_to_image_h), 0
    return best


def structure_score_from_homography(corners: np.ndarray, court_to_image_h: np.ndarray, frame_shape: tuple[int, ...]) -> float:
    projected_corners = project_points(STANDARD_COURT_TEMPLATE.corners, court_to_image_h)
    reproj = float(np.mean(np.linalg.norm(projected_corners - corners, axis=1)))
    reproj_score = float(np.clip(1.0 - reproj / 12.0, 0.0, 1.0))
    height, width = frame_shape[:2]
    projected_center = project_points(np.array([[COURT_WIDTH * 0.5, COURT_LENGTH * 0.5]], dtype=np.float32), court_to_image_h)[0]
    center_ok = 0.0 <= projected_center[0] <= width and 0.0 <= projected_center[1] <= height
    return 0.75 * reproj_score + 0.25 * (1.0 if center_ok else 0.35)


def score_court_detection(
    detection: CourtLineDetection,
    previous: CourtLineDetection | None,
    frame_shape: tuple[int, ...],
    args: argparse.Namespace,
) -> tuple[float, dict[str, float], str]:
    """Score a white-line court detection from geometry, support, and temporal stability."""
    height, width = frame_shape[:2]
    diag = float(np.hypot(width, height))
    line_count_score = float(np.clip(detection.line_count / 18.0, 0.0, 1.0))
    merged_count_score = float(np.clip(detection.merged_line_count / 8.0, 0.0, 1.0))
    effective_line_score = 0.55 * line_count_score + 0.45 * merged_count_score
    keypoint_score = float(np.clip(detection.supported_keypoints / max(1.0, float(len(detection.keypoint_names))), 0.0, 1.0))
    quad_score = quad_geometry_score(detection.corners, frame_shape)
    shape_score = court_shape_sanity_score(detection.corners, frame_shape, args)
    bounds_score = quad_bounds_score(detection.corners, frame_shape)
    length_score = float(np.clip(detection.avg_line_length / max(1.0, diag * 0.18), 0.0, 1.0))
    structure_score = structure_score_from_homography(detection.corners, detection.court_to_image_h, frame_shape)
    support_score = float(np.clip(detection.mask_support / 0.36, 0.0, 1.0))
    green_side_score = float(np.clip(detection.green_side_support / 0.55, 0.0, 1.0))
    snap_score = float(np.clip(detection.snap_points / 35.0, 0.0, 1.0))

    if previous is None:
        stability_score = 0.82
        jump_ratio = 0.0
    else:
        offsets = np.linalg.norm(detection.corners - previous.corners, axis=1)
        mean_offset = float(np.mean(offsets))
        jump_ratio = mean_offset / max(1.0, diag)
        stability_score = float(np.clip(1.0 - (jump_ratio / max(0.01, float(args.jump_ratio_hard))), 0.0, 1.0))

    components = {
        "line_count": effective_line_score,
        "keypoints": keypoint_score,
        "quad": quad_score,
        "shape": shape_score,
        "bounds": bounds_score,
        "stability": stability_score,
        "line_length": length_score,
        "structure": structure_score,
        "mask_support": support_score,
        "green_sides": green_side_score,
        "snap_points": snap_score,
        "jump_ratio": jump_ratio,
    }

    confidence = (
        0.10 * effective_line_score
        + 0.15 * keypoint_score
        + 0.13 * quad_score
        + 0.07 * bounds_score
        + 0.12 * stability_score
        + 0.08 * length_score
        + 0.10 * structure_score
        + 0.10 * support_score
        + 0.12 * green_side_score
        + 0.03 * snap_score
    )

    reason = "ok"
    if shape_score < 0.55:
        confidence *= 0.25
        reason = "implausible court shape"
    elif bounds_score <= 0.75:
        confidence *= 0.5
        reason = "corner out of bounds"
    elif quad_score < 0.25:
        confidence *= 0.35
        reason = "weak quadrilateral"
    elif keypoint_score < 0.45:
        confidence *= 0.65
        reason = "few supported intersections"
    elif green_side_score < 0.38:
        confidence *= 0.65
        reason = "outer lines look like court-edge/floor border"
    elif previous is not None and jump_ratio > float(args.jump_ratio_hard):
        confidence *= 0.45
        reason = "large temporal jump"

    return float(np.clip(confidence, 0.0, 1.0)), components, reason


def build_detection_from_corners(
    corners: np.ndarray,
    segments: list[LineSegment],
    merged_lines: list[MergedLine],
    intersections: list[np.ndarray],
    mask: np.ndarray,
    green_mask: np.ndarray,
    previous: CourtLineDetection | None,
    args: argparse.Namespace,
) -> CourtLineDetection | None:
    court_to_image_h, image_to_court_h = compute_homographies(corners)
    if court_to_image_h is None or image_to_court_h is None:
        return None
    refined_h, snap_points, snap_mean_shift = refine_homography_with_white_lines(court_to_image_h, mask, args)
    refined_corners = project_points(STANDARD_COURT_TEMPLATE.corners, refined_h)
    frame_diag = float(np.hypot(mask.shape[1], mask.shape[0]))
    refine_corner_shift = float(np.mean(np.linalg.norm(refined_corners - corners, axis=1)))
    max_refine_shift = max(8.0, frame_diag * max(0.0, float(args.max_refine_corner_shift_ratio)))
    if (
        is_convex_quad(refined_corners)
        and polygon_area(refined_corners) > 1.0
        and refine_corner_shift <= max_refine_shift
    ):
        court_to_image_h = refined_h
        corners = refined_corners.astype(np.float32)
        _, image_to_court_h = compute_homographies(corners)
        if image_to_court_h is None:
            return None
    projected_lines = project_template_lines(court_to_image_h)
    mask_support, _ = projected_template_support(projected_lines, mask)
    green_side_support = outer_line_green_side_support(
        court_to_image_h,
        green_mask,
        offset_px=float(args.green_side_offset_px),
    )
    scheme, names, keypoints, supported_keypoints = select_keypoints(court_to_image_h, intersections, mask, args)
    avg_length = float(np.mean([seg.length for seg in segments[: min(12, len(segments))]])) if segments else 0.0
    detection = CourtLineDetection(
        corners=corners.astype(np.float32),
        keypoints=keypoints.astype(np.float32),
        keypoint_names=names,
        court_to_image_h=court_to_image_h,
        image_to_court_h=image_to_court_h,
        confidence=0.0,
        components={},
        line_count=len(segments),
        merged_line_count=len(merged_lines),
        intersection_count=len(intersections),
        supported_keypoints=supported_keypoints,
        avg_line_length=avg_length,
        mask_support=mask_support,
        green_side_support=green_side_support,
        snap_points=snap_points,
        snap_mean_shift=snap_mean_shift,
        scheme=scheme,
        reason="candidate",
        projected_lines=projected_lines,
        debug_segments=segments[:60],
        debug_merged_lines=merged_lines,
    )
    confidence, components, reason = score_court_detection(detection, previous, mask.shape[:2], args)
    components.update(
        projected_singles_support_components(
            projected_lines,
            mask,
            mask_support,
        )
    )
    components.update(projected_outer_support_components(projected_lines, mask))
    detection.confidence = confidence
    detection.components = components
    detection.reason = reason
    return detection


def find_best_court_quad(
    family_a: list[MergedLine],
    family_b: list[MergedLine],
    segments: list[LineSegment],
    intersections: list[np.ndarray],
    mask: np.ndarray,
    green_mask: np.ndarray,
    previous: CourtLineDetection | None,
    args: argparse.Namespace,
) -> CourtLineDetection | None:
    height, width = mask.shape[:2]
    family_a = sorted(family_a, key=lambda line: line.length, reverse=True)[: max(2, int(args.max_lines_per_family))]
    family_b = sorted(family_b, key=lambda line: line.length, reverse=True)[: max(2, int(args.max_lines_per_family))]
    merged_lines = family_a + family_b
    best: CourtLineDetection | None = None

    for pair_a in itertools.combinations(family_a, 2):
        for pair_b in itertools.combinations(family_b, 2):
            raw_points: list[np.ndarray] = []
            for line_a in pair_a:
                for line_b in pair_b:
                    point = intersect_merged_lines(line_a, line_b)
                    if point is not None:
                        raw_points.append(point)
            if len(raw_points) != 4:
                continue
            corners = order_quad_points(np.asarray(raw_points, dtype=np.float32))
            if corners is None:
                continue
            margin_x = width * 0.15
            margin_y = height * 0.15
            if np.any(corners[:, 0] < -margin_x) or np.any(corners[:, 0] > width + margin_x):
                continue
            if np.any(corners[:, 1] < -margin_y) or np.any(corners[:, 1] > height + margin_y):
                continue
            if quad_geometry_score(corners, mask.shape[:2]) <= 0.05:
                continue
            detection = build_detection_from_corners(
                corners=corners,
                segments=segments,
                merged_lines=merged_lines,
                intersections=intersections,
                mask=mask,
                green_mask=green_mask,
                previous=previous,
                args=args,
            )
            if detection is None:
                continue
            if best is None or detection.confidence > best.confidence:
                best = detection
    return best


def collect_cross_family_intersections(
    line_groups: list[list[MergedLine]],
    frame_shape: tuple[int, ...],
) -> list[np.ndarray]:
    points: list[np.ndarray] = []
    for group_a, group_b in itertools.combinations(line_groups, 2):
        points.extend(all_family_intersections(group_a, group_b, frame_shape))
    return points


def find_best_court_quad_three_family(
    clusters: list[tuple[float, list[LineSegment]]],
    segments: list[LineSegment],
    mask: np.ndarray,
    green_mask: np.ndarray,
    previous: CourtLineDetection | None,
    args: argparse.Namespace,
) -> CourtLineDetection | None:
    merged_clusters: list[tuple[float, list[MergedLine]]] = []
    for angle, cluster_segments in clusters:
        merged = merge_line_family(cluster_segments, angle, args)
        if merged:
            merged_clusters.append((angle, merged[: max(1, int(args.max_lines_per_family))]))
    if len(merged_clusters) < 3:
        return None

    height, width = mask.shape[:2]
    best: CourtLineDetection | None = None
    for cross_index, (cross_angle, cross_lines) in enumerate(merged_clusters):
        if not is_likely_transverse_family(cross_angle, args):
            continue
        if len(cross_lines) < 2:
            continue
        side_indices = [index for index in range(len(merged_clusters)) if index != cross_index]
        for side_left_index, side_right_index in itertools.combinations(side_indices, 2):
            side_left = merged_clusters[side_left_index][1]
            side_right = merged_clusters[side_right_index][1]
            if not side_left or not side_right:
                continue
            merged_lines = cross_lines + side_left + side_right
            intersections = collect_cross_family_intersections([cross_lines, side_left, side_right], mask.shape[:2])
            for cross_pair in itertools.combinations(cross_lines, 2):
                for line_left in side_left:
                    for line_right in side_right:
                        raw_points: list[np.ndarray] = []
                        for cross_line in cross_pair:
                            for side_line in (line_left, line_right):
                                point = intersect_merged_lines(cross_line, side_line)
                                if point is not None:
                                    raw_points.append(point)
                        if len(raw_points) != 4:
                            continue
                        corners = order_quad_points(np.asarray(raw_points, dtype=np.float32))
                        if corners is None:
                            continue
                        margin_x = width * 0.15
                        margin_y = height * 0.15
                        if np.any(corners[:, 0] < -margin_x) or np.any(corners[:, 0] > width + margin_x):
                            continue
                        if np.any(corners[:, 1] < -margin_y) or np.any(corners[:, 1] > height + margin_y):
                            continue
                        if quad_geometry_score(corners, mask.shape[:2]) <= 0.05:
                            continue
                        detection = build_detection_from_corners(
                            corners=corners,
                            segments=segments,
                            merged_lines=merged_lines,
                            intersections=intersections,
                            mask=mask,
                            green_mask=green_mask,
                            previous=previous,
                            args=args,
                        )
                        if detection is None:
                            continue
                        if best is None or detection.confidence > best.confidence:
                            best = detection
    return best


def scale_detection(detection: CourtLineDetection | None, scale: float, frame_shape: tuple[int, ...]) -> CourtLineDetection | None:
    if detection is None or abs(scale - 1.0) < 1e-6:
        return detection
    corners = detection.corners * scale
    court_to_image_h, image_to_court_h = compute_homographies(corners)
    if court_to_image_h is None or image_to_court_h is None:
        return None
    keypoints = detection.keypoints * scale
    projected_lines = project_template_lines(court_to_image_h)
    scaled_segments = [
        LineSegment(p1=seg.p1 * scale, p2=seg.p2 * scale, angle_deg=seg.angle_deg, length=seg.length * scale)
        for seg in detection.debug_segments
    ]
    scaled_merged = [
        MergedLine(theta_deg=line.theta_deg, rho=line.rho * scale, length=line.length * scale, count=line.count)
        for line in detection.debug_merged_lines
    ]
    return CourtLineDetection(
        corners=corners.astype(np.float32),
        keypoints=keypoints.astype(np.float32),
        keypoint_names=list(detection.keypoint_names),
        court_to_image_h=court_to_image_h,
        image_to_court_h=image_to_court_h,
        confidence=detection.confidence,
        components=dict(detection.components),
        line_count=detection.line_count,
        merged_line_count=detection.merged_line_count,
        intersection_count=detection.intersection_count,
        supported_keypoints=detection.supported_keypoints,
        avg_line_length=detection.avg_line_length * scale,
        mask_support=detection.mask_support,
        green_side_support=detection.green_side_support,
        snap_points=detection.snap_points,
        snap_mean_shift=detection.snap_mean_shift * scale,
        scheme=detection.scheme,
        reason=detection.reason,
        projected_lines=projected_lines,
        debug_segments=scaled_segments,
        debug_merged_lines=scaled_merged,
        last_update_frame=detection.last_update_frame,
        last_update_time=detection.last_update_time,
    )


def detect_court_lines(
    frame: np.ndarray,
    previous: CourtLineDetection | None,
    args: argparse.Namespace,
) -> CourtLineDetection | None:
    detect_frame, scale = resize_for_detection(frame, int(args.detect_max_width))
    previous_small = scale_detection(previous, scale, detect_frame.shape) if previous is not None and scale != 1.0 else previous
    mask, green_mask = create_white_line_mask(detect_frame, args)
    segments, _ = detect_hough_segments(mask, args)
    if len(segments) < 4:
        return None
    family_a_segments, family_b_segments, angles = choose_direction_families(segments, args)
    if angles is None:
        return None
    merged_a = merge_line_family(family_a_segments, angles[0], args)
    merged_b = merge_line_family(family_b_segments, angles[1], args)
    if len(merged_a) < 2 or len(merged_b) < 2:
        clusters = choose_angle_clusters(segments, args, max_clusters=4)
        best_small = find_best_court_quad_three_family(
            clusters=clusters,
            segments=segments,
            mask=mask,
            green_mask=green_mask,
            previous=previous_small,
            args=args,
        )
        return scale_detection(best_small, 1.0 / scale, frame.shape) if scale != 1.0 else best_small
    intersections = all_family_intersections(merged_a, merged_b, detect_frame.shape)
    if len(intersections) < 4:
        clusters = choose_angle_clusters(segments, args, max_clusters=4)
        best_small = find_best_court_quad_three_family(
            clusters=clusters,
            segments=segments,
            mask=mask,
            green_mask=green_mask,
            previous=previous_small,
            args=args,
        )
        return scale_detection(best_small, 1.0 / scale, frame.shape) if scale != 1.0 else best_small
    best_small = find_best_court_quad(
        family_a=merged_a,
        family_b=merged_b,
        segments=segments,
        intersections=intersections,
        mask=mask,
        green_mask=green_mask,
        previous=previous_small,
        args=args,
    )
    if best_small is None or best_small.confidence < float(args.medium_conf):
        clusters = choose_angle_clusters(segments, args, max_clusters=4)
        best_three = find_best_court_quad_three_family(
            clusters=clusters,
            segments=segments,
            mask=mask,
            green_mask=green_mask,
            previous=previous_small,
            args=args,
        )
        if best_three is not None and (best_small is None or best_three.confidence > best_small.confidence):
            best_small = best_three
    if best_small is None:
        return None
    return scale_detection(best_small, 1.0 / scale, frame.shape) if scale != 1.0 else best_small


def blend_detections(
    previous: CourtLineDetection | None,
    candidate: CourtLineDetection,
    alpha: float,
    frame_id: int,
    timestamp: float,
    update_type: str,
) -> CourtLineDetection:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if previous is None:
        corners = candidate.corners.copy()
    else:
        corners = alpha * candidate.corners + (1.0 - alpha) * previous.corners

    court_to_image_h, image_to_court_h = compute_homographies(corners)
    if court_to_image_h is None or image_to_court_h is None:
        return candidate
    scheme = candidate.scheme
    names_and_points = template_keypoints_for_scheme(scheme)
    names = [item[0] for item in names_and_points]
    template_points = np.asarray([item[1] for item in names_and_points], dtype=np.float32)
    keypoints = project_points(template_points, court_to_image_h)
    blended = CourtLineDetection(
        corners=corners.astype(np.float32),
        keypoints=keypoints.astype(np.float32),
        keypoint_names=names,
        court_to_image_h=court_to_image_h,
        image_to_court_h=image_to_court_h,
        confidence=candidate.confidence,
        components=dict(candidate.components),
        line_count=candidate.line_count,
        merged_line_count=candidate.merged_line_count,
        intersection_count=candidate.intersection_count,
        supported_keypoints=candidate.supported_keypoints,
        avg_line_length=candidate.avg_line_length,
        mask_support=candidate.mask_support,
        green_side_support=candidate.green_side_support,
        snap_points=candidate.snap_points,
        snap_mean_shift=candidate.snap_mean_shift,
        scheme=scheme,
        reason=f"{update_type}: {candidate.reason}",
        projected_lines=project_template_lines(court_to_image_h),
        debug_segments=candidate.debug_segments,
        debug_merged_lines=candidate.debug_merged_lines,
        last_update_frame=frame_id,
        last_update_time=timestamp,
    )
    return blended


def update_tracking_state(
    state: TrackingState,
    candidate: CourtLineDetection | None,
    args: argparse.Namespace,
    frame_id: int,
    timestamp: float,
) -> None:
    state.last_attempt_frame = frame_id
    state.last_attempt_time = timestamp
    state.last_candidate = candidate
    if candidate is None:
        state.startup_candidate = None
        state.startup_candidate_count = 0
        state.last_update_type = "no candidate"
        state.rejected_count += 1
        return

    if (
        state.current is not None
        and state.current.scheme == "court_pose_white_line"
        and candidate.scheme == "court_pose_coarse"
    ):
        state.startup_candidate = None
        state.startup_candidate_count = 0
        state.last_update_type = "quality downgrade rejected"
        state.rejected_count += 1
        return

    if candidate.confidence >= float(args.reliable_conf):
        state.startup_candidate = None
        state.startup_candidate_count = 0
        quality_upgrade = bool(
            state.current is not None
            and state.current.scheme == "court_pose_coarse"
            and candidate.scheme == "court_pose_white_line"
        )
        alpha = 1.0 if quality_upgrade else float(args.smooth_alpha_reliable)
        state.current = blend_detections(state.current, candidate, alpha, frame_id, timestamp, "reliable")
        state.last_update_type = "reliable update"
        state.rejected_count = 0
    elif candidate.confidence >= float(args.medium_conf):
        alpha = float(args.smooth_alpha_medium)
        if state.current is None:
            confirm_frames = max(0, int(getattr(args, "coarse_startup_confirm_frames", 0)))
            if candidate.scheme == "court_pose_coarse" and confirm_frames > 1:
                max_shift_ratio = max(
                    0.0,
                    float(getattr(args, "coarse_startup_max_corner_shift_ratio", 0.03)),
                )
                if state.startup_candidate is None or not _detections_geometry_consistent(
                    state.startup_candidate,
                    candidate,
                    max_shift_ratio,
                ):
                    state.startup_candidate = candidate
                    state.startup_candidate_count = 1
                else:
                    state.startup_candidate_count += 1
                if state.startup_candidate_count >= confirm_frames:
                    state.current = blend_detections(
                        None,
                        candidate,
                        1.0,
                        frame_id,
                        timestamp,
                        "medium startup",
                    )
                    state.startup_candidate = None
                    state.startup_candidate_count = 0
                    state.last_update_type = "medium startup"
                    state.rejected_count = 0
                    return
                state.last_update_type = f"coarse confirmation {state.startup_candidate_count}/{confirm_frames}"
                return
            state.startup_candidate = None
            state.startup_candidate_count = 0
            state.last_update_type = "rejected"
            state.rejected_count += 1
            return
        else:
            state.startup_candidate = None
            state.startup_candidate_count = 0
            state.current = blend_detections(state.current, candidate, alpha, frame_id, timestamp, "medium smooth")
            state.last_update_type = "medium smooth"
        state.rejected_count = 0
    else:
        state.startup_candidate = None
        state.startup_candidate_count = 0
        state.last_update_type = "rejected"
        state.rejected_count += 1


def _detections_geometry_consistent(
    reference: CourtLineDetection,
    candidate: CourtLineDetection,
    max_corner_shift_ratio: float,
) -> bool:
    reference_corners = np.asarray(reference.corners, dtype=np.float32).reshape(-1, 2)
    candidate_corners = np.asarray(candidate.corners, dtype=np.float32).reshape(-1, 2)
    if reference_corners.shape != (4, 2) or candidate_corners.shape != (4, 2):
        return False
    if not np.isfinite(reference_corners).all() or not np.isfinite(candidate_corners).all():
        return False
    reference_scale = max(
        float(np.linalg.norm(reference_corners[2] - reference_corners[0])),
        float(np.linalg.norm(reference_corners[3] - reference_corners[1])),
        1.0,
    )
    max_shift = float(np.max(np.linalg.norm(candidate_corners - reference_corners, axis=1)))
    return (max_shift / reference_scale) <= max_corner_shift_ratio


def should_redetect(state: TrackingState, frame_id: int, timestamp: float, args: argparse.Namespace) -> bool:
    if frame_id == 0 or state.current is None:
        return True
    interval = max(0.1, float(args.redetect_interval))
    if state.current.scheme == "court_pose_coarse":
        interval = min(
            interval,
            max(0.1, float(getattr(args, "coarse_redetect_interval", interval))),
        )
    return (timestamp - state.last_attempt_time) >= interval


def draw_text(
    canvas: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float = 0.65,
    color: tuple[int, int, int] = (245, 245, 245),
    thickness: int = 2,
) -> None:
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (8, 8, 8), thickness + 2, cv2.LINE_AA)
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


class FourPointEditor:
    def __init__(self, window_name: str):
        self.window_name = window_name
        self.points = np.empty((0, 2), dtype=np.float32)
        self.drag_index: int | None = None
        self.drag_radius = 18.0
        self.manual_mode = False
        self.edited = False
        self.user_modified = False
        self.snap_pending = False
        self.confirmed = False

    @property
    def point_count(self) -> int:
        return int(len(self.points))

    def set_points(self, points: np.ndarray | None) -> None:
        if points is None:
            self.clear()
            return
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if len(pts) >= 4:
            ordered = order_quad_points(pts[:4])
            if ordered is not None:
                pts = ordered
        self.points = pts[:4].copy()
        self.drag_index = None
        self.edited = False
        self.user_modified = False
        self.snap_pending = False
        self.confirmed = False

    def _nearest_index(self, x: int, y: int) -> int | None:
        if self.point_count == 0:
            return None
        point = np.array([float(x), float(y)], dtype=np.float32)
        distances = np.linalg.norm(self.points - point, axis=1)
        index = int(np.argmin(distances))
        return index if float(distances[index]) <= self.drag_radius else None

    def _set_point(self, index: int, x: int, y: int) -> None:
        self.points[index] = np.array([float(x), float(y)], dtype=np.float32)
        self.edited = True
        self.user_modified = True
        self.snap_pending = True
        self.confirmed = False

    def _sort_if_complete(self) -> None:
        if self.point_count != 4:
            return
        ordered = order_quad_points(self.points)
        if ordered is not None:
            self.points = ordered

    def mouse_callback(self, event, x, y, flags, param) -> None:
        del flags, param
        if event == cv2.EVENT_LBUTTONDOWN:
            nearest = self._nearest_index(x, y)
            if nearest is not None:
                self.drag_index = nearest
                return
            if self.manual_mode or self.point_count < 4:
                if self.point_count < 4:
                    self.points = np.vstack([self.points, np.array([[float(x), float(y)]], dtype=np.float32)])
                    self.edited = True
                    self.user_modified = True
                    self.snap_pending = True
                    self.confirmed = False
                    if self.point_count == 4:
                        self._sort_if_complete()
                        self.manual_mode = False
                return

        if event == cv2.EVENT_MOUSEMOVE and self.drag_index is not None:
            self._set_point(self.drag_index, x, y)
            return

        if event == cv2.EVENT_LBUTTONUP and self.drag_index is not None:
            self._set_point(self.drag_index, x, y)
            self.drag_index = None
            self._sort_if_complete()

    def draw(self, frame: np.ndarray) -> np.ndarray:
        if self.point_count >= 2:
            pts = np.round(self.points).astype(np.int32).reshape(-1, 1, 2)
            closed = self.point_count == 4
            cv2.polylines(frame, [pts], closed, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.polylines(frame, [pts], closed, (0, 220, 255), 2, cv2.LINE_AA)

        for index, point in enumerate(self.points):
            center = tuple(np.round(point).astype(int))
            cv2.circle(frame, center, 9, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(frame, center, 7, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, center, 2, (0, 80, 255), -1, cv2.LINE_AA)
            draw_text(frame, str(index), (center[0] + 12, center[1] - 10), 0.55, (0, 255, 255))
        return frame

    def get_points(self) -> np.ndarray | None:
        if self.point_count != 4:
            return None
        ordered = order_quad_points(self.points)
        return ordered.copy() if ordered is not None else None

    def clear(self) -> None:
        self.points = np.empty((0, 2), dtype=np.float32)
        self.drag_index = None
        self.edited = False
        self.user_modified = False
        self.snap_pending = False
        self.confirmed = False

    def set_manual_mode(self, enabled: bool) -> None:
        self.manual_mode = bool(enabled)
        self.drag_index = None


def compute_homography_from_points(image_points: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    ordered = order_quad_points(np.asarray(image_points, dtype=np.float32).reshape(-1, 2))
    if ordered is None:
        return None
    court_to_image_h, image_to_court_h = compute_homographies(ordered)
    if court_to_image_h is None or image_to_court_h is None:
        return None
    return court_to_image_h, image_to_court_h


def build_detection_from_manual_points(
    image_points: np.ndarray,
    frame_id: int,
    timestamp: float,
    reason: str,
) -> CourtLineDetection | None:
    ordered = order_quad_points(np.asarray(image_points, dtype=np.float32).reshape(-1, 2))
    if ordered is None:
        return None
    homography = compute_homography_from_points(ordered)
    if homography is None:
        return None
    court_to_image_h, image_to_court_h = homography
    names_and_points = template_keypoints_for_scheme("8")
    names = [item[0] for item in names_and_points]
    template_points = np.asarray([item[1] for item in names_and_points], dtype=np.float32)
    keypoints = project_points(template_points, court_to_image_h)
    projected_lines = project_template_lines(court_to_image_h)
    return CourtLineDetection(
        corners=ordered.astype(np.float32),
        keypoints=keypoints.astype(np.float32),
        keypoint_names=names,
        court_to_image_h=court_to_image_h,
        image_to_court_h=image_to_court_h,
        confidence=1.0,
        components={"manual": 1.0},
        line_count=0,
        merged_line_count=0,
        intersection_count=4,
        supported_keypoints=len(names),
        avg_line_length=0.0,
        mask_support=1.0,
        green_side_support=1.0,
        snap_points=0,
        snap_mean_shift=0.0,
        scheme="manual",
        reason=reason,
        projected_lines=projected_lines,
        last_update_frame=frame_id,
        last_update_time=timestamp,
    )


def editor_has_manual_priority(editor: FourPointEditor) -> bool:
    return bool(editor.confirmed or editor.user_modified or editor.manual_mode)


def sync_editor_from_current_detection(
    frame: np.ndarray,
    state: TrackingState,
    editor: FourPointEditor,
    args: argparse.Namespace,
    frame_id: int,
    timestamp: float,
) -> int:
    if state.current is None or editor_has_manual_priority(editor):
        return 0
    editor.set_points(state.current.corners)
    snap_count = snap_editor_points_to_frame(frame, editor, args)
    image_points = editor.get_points()
    if image_points is not None:
        snapped_detection = build_detection_from_manual_points(image_points, frame_id, timestamp, "auto stable snapped")
        if snapped_detection is not None:
            snapped_detection.confidence = state.current.confidence
            snapped_detection.components = dict(state.current.components)
            snapped_detection.line_count = state.current.line_count
            snapped_detection.merged_line_count = state.current.merged_line_count
            snapped_detection.intersection_count = state.current.intersection_count
            snapped_detection.supported_keypoints = state.current.supported_keypoints
            snapped_detection.avg_line_length = state.current.avg_line_length
            snapped_detection.mask_support = state.current.mask_support
            snapped_detection.green_side_support = state.current.green_side_support
            snapped_detection.snap_points = state.current.snap_points
            snapped_detection.snap_mean_shift = state.current.snap_mean_shift
            snapped_detection.scheme = state.current.scheme
            snapped_detection.reason = f"{state.current.reason}; editor snap {snap_count}/4"
            snapped_detection.debug_segments = state.current.debug_segments
            snapped_detection.debug_merged_lines = state.current.debug_merged_lines
            state.current = snapped_detection
    return snap_count


def run_auto_redetect(
    frame: np.ndarray,
    state: TrackingState,
    editor: FourPointEditor,
    args: argparse.Namespace,
    frame_id: int,
    timestamp: float,
) -> tuple[float, str, int]:
    detect_start = time.perf_counter()
    previous = state.current
    candidate = detect_court_lines(frame, previous, args)
    detect_elapsed = time.perf_counter() - detect_start
    detect_fps = 1.0 / detect_elapsed if detect_elapsed > 1e-4 else 0.0

    if editor_has_manual_priority(editor):
        state.last_attempt_frame = frame_id
        state.last_attempt_time = timestamp
        state.last_candidate = candidate
        state.last_update_type = "suggestion only"
        if candidate is None:
            state.rejected_count += 1
            return detect_fps, "auto suggestion: no candidate", 0
        return detect_fps, f"auto suggestion score={candidate.confidence:.2f}", 0

    update_tracking_state(state, candidate, args, frame_id, timestamp)
    snap_count = sync_editor_from_current_detection(frame, state, editor, args, frame_id, timestamp)
    if candidate is None:
        return detect_fps, "auto update: no candidate", snap_count
    current_score = state.current.confidence if state.current is not None else float("nan")
    return detect_fps, f"auto update candidate={candidate.confidence:.2f} current={current_score:.2f} snapped {snap_count}/4", snap_count


def save_calibration(path: str, source: str, frame_shape: tuple[int, int], image_points: np.ndarray) -> None:
    homography = compute_homography_from_points(image_points)
    if homography is None:
        raise ValueError("Need four valid ordered image points before saving calibration.")
    court_to_image_h, image_to_court_h = homography
    ordered = order_quad_points(np.asarray(image_points, dtype=np.float32).reshape(-1, 2))
    if ordered is None:
        raise ValueError("Invalid image point quadrilateral.")
    height, width = int(frame_shape[0]), int(frame_shape[1])
    payload = {
        "source": str(source),
        "frame_width": width,
        "frame_height": height,
        "image_points_order": ["top_left", "top_right", "bottom_right", "bottom_left"],
        "image_points": ordered.astype(float).tolist(),
        "court_points": STANDARD_COURT_TEMPLATE.corners.astype(float).tolist(),
        "court_to_image_h": court_to_image_h.astype(float).tolist(),
        "image_to_court_h": image_to_court_h.astype(float).tolist(),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_calibration(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    input_path = Path(path)
    if not input_path.is_file():
        return None
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        image_points = np.asarray(payload["image_points"], dtype=np.float32).reshape(4, 2)
    except (KeyError, ValueError, json.JSONDecodeError, OSError) as exc:
        print(f"[load] invalid calibration file: {input_path} ({exc})")
        return None
    ordered = order_quad_points(image_points)
    if ordered is None:
        print(f"[load] invalid four-point geometry: {input_path}")
        return None
    homography = compute_homography_from_points(ordered)
    if homography is None:
        print(f"[load] could not compute homography: {input_path}")
        return None
    court_to_image_h, image_to_court_h = homography
    return ordered, court_to_image_h, image_to_court_h


def draw_projected_template(frame: np.ndarray, image_points: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    homography = compute_homography_from_points(image_points)
    if homography is None:
        return frame
    court_to_image_h, _ = homography
    thickness = max(1, int(args.line_thickness))
    for name, line in STANDARD_COURT_TEMPLATE.lines:
        projected = project_points(line, court_to_image_h)
        if len(projected) < 2 or not np.isfinite(projected).all():
            continue
        closed = name == "doubles_outer"
        pts = np.round(projected).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], closed, (0, 60, 20), thickness + 3, cv2.LINE_AA)
        cv2.polylines(frame, [pts], closed, (40, 245, 105), thickness + (1 if closed else 0), cv2.LINE_AA)
    return frame


def create_corner_snap_mask(frame: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    strict_mask, green_mask = create_white_line_mask(frame, args)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_chroma = np.sqrt((lab[:, :, 1] - 128.0) ** 2 + (lab[:, :, 2] - 128.0) ** 2)
    low_chroma = (lab_chroma <= min(72.0, float(args.white_chroma_max))).astype(np.uint8) * 255
    strict_mask = cv2.bitwise_and(strict_mask, low_chroma)
    loose_white = cv2.inRange(
        hsv,
        np.array([0, 0, max(0, int(args.white_v_min) - 25)], dtype=np.uint8),
        np.array([179, min(150, int(args.white_s_max) + 15), 255], dtype=np.uint8),
    )
    loose_white = cv2.bitwise_and(loose_white, low_chroma)
    if cv2.countNonZero(green_mask) > 0:
        green_context = cv2.dilate(
            green_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
            iterations=2,
        )
        loose_white = cv2.bitwise_and(loose_white, green_context)
    snap_mask = cv2.bitwise_or(strict_mask, loose_white)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    snap_mask = cv2.morphologyEx(snap_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    snap_mask = cv2.morphologyEx(snap_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return snap_mask


def detect_local_snap_segments(mask: np.ndarray, point: np.ndarray, args: argparse.Namespace) -> list[LineSegment]:
    height, width = mask.shape[:2]
    radius = max(24, int(args.corner_snap_radius))
    x = int(round(float(point[0])))
    y = int(round(float(point[1])))
    x1 = max(0, x - radius)
    x2 = min(width, x + radius + 1)
    y1 = max(0, y - radius)
    y2 = min(height, y + radius + 1)
    if x2 - x1 < 16 or y2 - y1 < 16:
        return []

    roi = mask[y1:y2, x1:x2]
    roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    edges = cv2.Canny(roi, 40, 140, apertureSize=3)
    raw_lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=max(4, int(args.corner_snap_hough_threshold)),
        minLineLength=max(10, int(args.corner_snap_min_line_length)),
        maxLineGap=max(3, int(args.corner_snap_max_gap)),
    )
    if raw_lines is None:
        return []

    segments: list[LineSegment] = []
    for raw in raw_lines.reshape(-1, 4):
        p1 = np.array([float(raw[0] + x1), float(raw[1] + y1)], dtype=np.float32)
        p2 = np.array([float(raw[2] + x1), float(raw[3] + y1)], dtype=np.float32)
        length = float(np.linalg.norm(p2 - p1))
        if length < max(10.0, float(args.corner_snap_min_line_length)):
            continue
        if point_to_line_distance(point, p1, p2) > radius * 0.55:
            continue
        if point_to_segment_distance(point, p1, p2) > radius * 0.95:
            continue
        segments.append(LineSegment(p1=p1, p2=p2, angle_deg=line_angle_deg(p1, p2), length=length))
    segments.sort(key=lambda item: item.length, reverse=True)
    return segments[:18]


def intersect_segment_infinite_lines(a: LineSegment, b: LineSegment) -> np.ndarray | None:
    normal_a = line_normal(a.angle_deg).astype(np.float64)
    normal_b = line_normal(b.angle_deg).astype(np.float64)
    matrix = np.vstack([normal_a, normal_b])
    if abs(float(np.linalg.det(matrix))) < 1e-6:
        return None
    rhs = np.array([float(np.dot(normal_a, a.p1)), float(np.dot(normal_b, b.p1))], dtype=np.float64)
    point = np.linalg.solve(matrix, rhs)
    return point.astype(np.float32) if np.isfinite(point).all() else None


def edge_projection(point: np.ndarray, edge_start: np.ndarray, edge_unit: np.ndarray) -> float:
    return float(np.dot(np.asarray(point, dtype=np.float32) - edge_start, edge_unit))


def fit_outer_edge_line(
    mask: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    args: argparse.Namespace,
    court_center: np.ndarray | None = None,
    prefer_outer_cluster: bool = False,
) -> LineSegment | None:
    height, width = mask.shape[:2]
    p1 = np.asarray(p1, dtype=np.float32)
    p2 = np.asarray(p2, dtype=np.float32)
    edge_vec = p2 - p1
    edge_len = float(np.linalg.norm(edge_vec))
    if edge_len < 30.0:
        return None
    edge_unit = edge_vec / edge_len
    expected_angle = line_angle_deg(p1, p2)
    band = max(12.0, float(args.corner_snap_edge_band))
    extension = max(20.0, float(args.corner_snap_edge_extension))

    x1 = max(0, int(np.floor(min(p1[0], p2[0]) - band - extension)))
    x2 = min(width, int(np.ceil(max(p1[0], p2[0]) + band + extension + 1)))
    y1 = max(0, int(np.floor(min(p1[1], p2[1]) - band - extension)))
    y2 = min(height, int(np.ceil(max(p1[1], p2[1]) + band + extension + 1)))
    if x2 - x1 < 20 or y2 - y1 < 20:
        return None

    roi = mask[y1:y2, x1:x2]
    roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    edges = cv2.Canny(roi, 40, 140, apertureSize=3)
    raw_lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=max(8, int(args.corner_snap_hough_threshold)),
        minLineLength=max(18, int(args.corner_snap_min_line_length)),
        maxLineGap=max(4, int(args.corner_snap_max_gap)),
    )
    if raw_lines is None:
        return None

    selected: list[tuple[float, float, float, LineSegment]] = []
    for raw in raw_lines.reshape(-1, 4):
        q1 = np.array([float(raw[0] + x1), float(raw[1] + y1)], dtype=np.float32)
        q2 = np.array([float(raw[2] + x1), float(raw[3] + y1)], dtype=np.float32)
        length = float(np.linalg.norm(q2 - q1))
        if length < max(12.0, float(args.corner_snap_min_line_length)):
            continue
        angle = line_angle_deg(q1, q2)
        if angle_distance_deg(angle, expected_angle) > max(16.0, min(35.0, float(args.corner_snap_angle_tol))):
            continue
        midpoint = (q1 + q2) * 0.5
        projection = edge_projection(midpoint, p1, edge_unit)
        if projection < -extension or projection > edge_len + extension:
            continue
        distance = point_to_line_distance(midpoint, p1, p2)
        if distance > band:
            continue
        outward_score = point_to_line_distance(court_center, q1, q2) if court_center is not None else 0.0
        selected.append((outward_score, distance, -length, LineSegment(p1=q1, p2=q2, angle_deg=angle, length=length)))

    if not selected:
        return None
    if prefer_outer_cluster:
        selected.sort(key=lambda item: (-item[0], item[1], item[2]))
        max_outward_score = selected[0][0]
        outward_margin = max(8.0, band * 0.18)
        outer_cluster = [item for item in selected if item[0] >= max_outward_score - outward_margin]
        if len(outer_cluster) >= 2:
            outer_cluster.sort(key=lambda item: (item[1], item[2]))
            kept = [item[3] for item in outer_cluster[:10]]
        else:
            kept = [item[3] for item in selected[:8]]
    else:
        selected.sort(key=lambda item: (item[1], item[2]))
        kept = [item[3] for item in selected[:8]]
    fit_points = np.asarray([point for seg in kept for point in (seg.p1, seg.p2)], dtype=np.float32)
    if len(fit_points) < 4:
        return None
    vx, vy, x0, y0 = cv2.fitLine(fit_points.reshape(-1, 1, 2), cv2.DIST_HUBER, 0, 0.01, 0.01).reshape(4)
    direction = np.array([float(vx), float(vy)], dtype=np.float32)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        return None
    direction /= norm
    if float(np.dot(direction, edge_unit)) < 0:
        direction *= -1.0
    center = np.array([float(x0), float(y0)], dtype=np.float32)
    fit_p1 = center - direction * edge_len * 0.5
    fit_p2 = center + direction * edge_len * 0.5
    return LineSegment(p1=fit_p1, p2=fit_p2, angle_deg=line_angle_deg(fit_p1, fit_p2), length=edge_len)


def snap_corners_by_outer_edge_fits(
    mask: np.ndarray,
    points: np.ndarray,
    args: argparse.Namespace,
    *,
    prefer_outer_cluster: bool = False,
) -> tuple[np.ndarray, list[bool]]:
    ordered = order_quad_points(points)
    if ordered is None:
        return points, [False, False, False, False]

    edge_pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
    court_center = np.mean(ordered, axis=0).astype(np.float32)
    fitted_edges = [
        fit_outer_edge_line(
            mask,
            ordered[start],
            ordered[end],
            args,
            court_center,
            prefer_outer_cluster=prefer_outer_cluster,
        )
        for index, (start, end) in enumerate(edge_pairs)
    ]
    snapped = ordered.copy()
    flags = [False, False, False, False]
    adjacent_edges = [(3, 0), (0, 1), (1, 2), (2, 3)]
    max_shift = max(6.0, float(args.corner_snap_max_shift))
    nearest_white_radius = max(1, int(args.corner_snap_nearest_white_radius))
    min_white_support = max(0.0, float(args.corner_snap_min_white_support))

    for corner_index, (edge_a_index, edge_b_index) in enumerate(adjacent_edges):
        edge_a = fitted_edges[edge_a_index]
        edge_b = fitted_edges[edge_b_index]
        if edge_a is None or edge_b is None:
            continue
        intersection = intersect_segment_infinite_lines(edge_a, edge_b)
        if intersection is None:
            continue
        if float(np.linalg.norm(intersection - ordered[corner_index])) > max_shift:
            continue
        white_point = nearest_mask_pixel(mask, intersection, nearest_white_radius)
        if white_point is None:
            continue
        cross_point = refine_to_white_cross_feature(mask, white_point, edge_a.angle_deg, edge_b.angle_deg, args)
        if cross_point is None:
            continue
        if point_patch_support(mask, cross_point, radius=4) < min_white_support:
            continue
        snapped[corner_index] = cross_point.astype(np.float32)
        flags[corner_index] = True

    resorted = order_quad_points(snapped)
    if resorted is None:
        return ordered, [False, False, False, False]
    return resorted, flags


def corner_expected_angles(points: np.ndarray, index: int) -> tuple[float, float]:
    neighbor_indices = ((1, 3), (0, 2), (1, 3), (0, 2))
    first, second = neighbor_indices[index]
    return line_angle_deg(points[index], points[first]), line_angle_deg(points[index], points[second])


def corner_angle_match_score(angle_a: float, angle_b: float, expected_a: float, expected_b: float, tolerance: float) -> float:
    tolerance = max(1.0, float(tolerance))

    def one_score(angle: float, expected: float) -> float:
        return float(np.clip(1.0 - angle_distance_deg(angle, expected) / tolerance, 0.0, 1.0))

    direct = 0.5 * (one_score(angle_a, expected_a) + one_score(angle_b, expected_b))
    swapped = 0.5 * (one_score(angle_a, expected_b) + one_score(angle_b, expected_a))
    return max(direct, swapped)


def snap_single_corner_to_white_intersection(
    mask: np.ndarray,
    points: np.ndarray,
    index: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, bool]:
    original = points[index].astype(np.float32)
    segments = detect_local_snap_segments(mask, original, args)
    if len(segments) < 2:
        return original, False

    height, width = mask.shape[:2]
    radius = max(24.0, float(args.corner_snap_radius))
    max_shift = max(6.0, min(radius, float(args.corner_snap_max_shift)))
    nearest_white_radius = max(1, int(args.corner_snap_nearest_white_radius))
    min_white_support = max(0.0, float(args.corner_snap_min_white_support))
    min_angle_separation = max(5.0, float(args.corner_snap_min_angle_separation))
    expected_a, expected_b = corner_expected_angles(points, index)
    angle_tolerance = max(12.0, float(args.corner_snap_angle_tol))
    dilated = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)

    best_point: np.ndarray | None = None
    best_score = -1.0
    for line_a, line_b in itertools.combinations(segments, 2):
        angle_separation = angle_distance_deg(line_a.angle_deg, line_b.angle_deg)
        if angle_separation < min_angle_separation:
            continue
        intersection = intersect_segment_infinite_lines(line_a, line_b)
        if intersection is None:
            continue
        if intersection[0] < -2 or intersection[0] > width + 2 or intersection[1] < -2 or intersection[1] > height + 2:
            continue
        shift = float(np.linalg.norm(intersection - original))
        if shift > max_shift:
            continue
        white_point = nearest_mask_pixel(mask, intersection, nearest_white_radius)
        if white_point is None:
            continue
        cross_point = refine_to_white_cross_feature(mask, white_point, line_a.angle_deg, line_b.angle_deg, args)
        if cross_point is None:
            continue
        white_point = cross_point
        final_shift = float(np.linalg.norm(white_point - original))
        if final_shift > max_shift:
            continue
        white_support = point_patch_support(mask, white_point, radius=4)
        if white_support < min_white_support:
            continue

        angle_score = corner_angle_match_score(line_a.angle_deg, line_b.angle_deg, expected_a, expected_b, angle_tolerance)
        if angle_score < 0.18:
            continue
        length_score = float(np.clip((line_a.length + line_b.length) / max(1.0, 2.0 * radius), 0.0, 1.0))
        shift_score = float(np.clip(1.0 - final_shift / max_shift, 0.0, 1.0))
        patch_score = float(np.clip(point_patch_support(dilated, white_point, radius=6) / 0.10, 0.0, 1.0))
        geometric_score = float(np.clip(1.0 - np.linalg.norm(white_point - intersection) / max(1.0, nearest_white_radius), 0.0, 1.0))
        score = 0.34 * angle_score + 0.22 * length_score + 0.20 * shift_score + 0.14 * patch_score + 0.10 * geometric_score
        if score > best_score:
            best_score = score
            best_point = white_point

    if best_point is None:
        return original, False
    return best_point.astype(np.float32), True


def snap_corners_to_white_intersections(
    frame: np.ndarray,
    image_points: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[bool]]:
    ordered = order_quad_points(np.asarray(image_points, dtype=np.float32).reshape(-1, 2))
    if ordered is None or not bool(args.corner_snap):
        return np.asarray(image_points, dtype=np.float32).reshape(-1, 2), [False, False, False, False]

    mask = create_corner_snap_mask(frame, args)
    snapped, edge_flags = snap_corners_by_outer_edge_fits(mask, ordered, args)
    ordered_snapped = order_quad_points(snapped)
    snapped = ordered_snapped if ordered_snapped is not None else ordered.copy()
    flags: list[bool] = list(edge_flags)
    for index in range(4):
        if flags[index]:
            continue
        point, used = snap_single_corner_to_white_intersection(mask, snapped, index, args)
        snapped[index] = point
        flags[index] = used

    strong_prior_support = max(0.0, float(args.corner_snap_strong_prior_support))
    max_strong_prior_shift = max(1.0, float(args.corner_snap_max_strong_prior_shift))
    for index in range(4):
        shift = float(np.linalg.norm(snapped[index] - ordered[index]))
        prior_support = point_patch_support(mask, ordered[index], radius=4)
        if prior_support >= strong_prior_support and shift > max_strong_prior_shift:
            white_point = nearest_mask_pixel(mask, ordered[index], max(2, int(args.corner_snap_nearest_white_radius)))
            snapped[index] = white_point.astype(np.float32) if white_point is not None else ordered[index]
            flags[index] = True

    resorted = order_quad_points(snapped)
    if resorted is None:
        return ordered, [False, False, False, False]
    return resorted, flags


def snap_editor_points_to_frame(frame: np.ndarray, editor: FourPointEditor, args: argparse.Namespace) -> int:
    image_points = editor.get_points()
    if image_points is None or not bool(args.corner_snap):
        editor.snap_pending = False
        return 0
    snapped, flags = snap_corners_to_white_intersections(frame, image_points, args)
    snap_count = int(sum(flags))
    if snap_count > 0:
        was_user_modified = editor.user_modified
        was_confirmed = editor.confirmed
        editor.set_points(snapped)
        editor.user_modified = was_user_modified
        editor.confirmed = was_confirmed
    editor.snap_pending = False
    return snap_count


def draw_projected_court(canvas: np.ndarray, detection: CourtLineDetection, args: argparse.Namespace) -> None:
    if args.mask_alpha > 0:
        mask_polygon = project_points(STANDARD_COURT_TEMPLATE.mask_polygon, detection.court_to_image_h)
        if mask_polygon.shape[0] >= 3 and np.isfinite(mask_polygon).all():
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [np.asarray(mask_polygon, dtype=np.int32).reshape(-1, 1, 2)], (35, 210, 90))
            alpha = float(np.clip(args.mask_alpha, 0.0, 1.0))
            cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0, canvas)

    line_color = (40, 245, 110)
    shadow = (0, 65, 25)
    thickness = max(1, int(args.line_thickness))
    for name, points in detection.projected_lines.items():
        if len(points) < 2 or not np.isfinite(points).all():
            continue
        closed = name == "doubles_outer"
        pts = np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], closed, shadow, thickness + 3, lineType=cv2.LINE_AA)
        cv2.polylines(canvas, [pts], closed, line_color, thickness + (1 if closed else 0), lineType=cv2.LINE_AA)


def draw_debug_lines(canvas: np.ndarray, detection: CourtLineDetection, args: argparse.Namespace) -> None:
    if not args.draw_debug_lines:
        return
    colors = [(255, 160, 40), (60, 180, 255)]
    for index, line in enumerate(detection.debug_merged_lines):
        endpoints = line_to_frame_points(line, canvas.shape)
        if endpoints is None:
            continue
        cv2.line(canvas, endpoints[0], endpoints[1], colors[index % 2], 1, cv2.LINE_AA)


def draw_keypoints(canvas: np.ndarray, detection: CourtLineDetection, args: argparse.Namespace) -> None:
    radius = max(3, int(args.point_radius))
    for index, point in enumerate(detection.keypoints):
        if not np.isfinite(point).all():
            continue
        x, y = int(round(float(point[0]))), int(round(float(point[1])))
        if x < -20 or x > canvas.shape[1] + 20 or y < -20 or y > canvas.shape[0] + 20:
            continue
        cv2.circle(canvas, (x, y), radius + 2, (15, 15, 15), -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (x, y), radius, (255, 245, 80), -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (x, y), radius + 2, (255, 245, 80), 1, lineType=cv2.LINE_AA)
        if args.show_labels and index < len(detection.keypoint_names):
            cv2.putText(
                canvas,
                detection.keypoint_names[index],
                (x + 7, y - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 245, 80),
                1,
                cv2.LINE_AA,
            )


def format_float(value: float, digits: int = 2) -> str:
    if not np.isfinite(value):
        return "--"
    return f"{value:.{digits}f}"


def draw_visualization(
    frame: np.ndarray,
    state: TrackingState,
    editor: FourPointEditor,
    fps_text: str,
    args: argparse.Namespace,
    preview_template: bool,
    debug_enabled: bool,
    detection_status: str,
) -> np.ndarray:
    canvas = frame.copy()
    image_points = editor.get_points()
    current = state.current
    final_points = image_points if image_points is not None else (current.corners if current is not None else None)
    homography_valid = final_points is not None and compute_homography_from_points(final_points) is not None

    if preview_template and final_points is not None:
        canvas = draw_projected_template(canvas, final_points, args)

    candidate = state.last_candidate
    if debug_enabled and candidate is not None:
        for line in candidate.debug_merged_lines:
            endpoints = line_to_frame_points(line, canvas.shape)
            if endpoints is not None:
                cv2.line(canvas, endpoints[0], endpoints[1], (255, 170, 40), 1, cv2.LINE_AA)

    candidate_conf = candidate.confidence if candidate is not None else float("nan")
    current_conf = current.confidence if current is not None else float("nan")
    status_color = (210, 245, 210) if homography_valid else (80, 120, 255)
    editor.draw(canvas)
    draw_text(
        canvas,
        f"points {editor.point_count}/4 | H {'valid' if homography_valid else 'invalid'} | preview {'on' if preview_template else 'off'} | manual {'on' if editor.manual_mode else 'off'} | confirmed {'yes' if editor.confirmed else 'no'}",
        (24, 38),
        0.66,
        status_color,
    )
    draw_text(
        canvas,
        f"current {format_float(current_conf)} | candidate {format_float(candidate_conf)} | {state.last_update_type} | {detection_status}",
        (24, 70),
        0.58,
    )
    final_detection = current if current is not None else candidate
    if final_detection is not None:
        draw_text(
            canvas,
            (
                f"scheme {final_detection.scheme} | lines {final_detection.line_count}/merged {final_detection.merged_line_count} | "
                f"kp {final_detection.supported_keypoints}/{len(final_detection.keypoint_names)} | {final_detection.reason}"
            ),
            (24, 102),
            0.55,
        )
        draw_text(canvas, fps_text, (24, 134), 0.55)
        draw_text(canvas, "Drag points | a snap | r redetect | m manual | Enter confirm | s save | l load | v preview | q quit", (24, 166), 0.52, (245, 245, 180))
    else:
        draw_text(canvas, fps_text, (24, 102), 0.55)
        draw_text(canvas, "Drag points | a snap | r redetect | m manual | Enter confirm | s save | l load | v preview | q quit", (24, 134), 0.52, (245, 245, 180))
    return canvas


def main() -> None:
    args = parse_args()
    cap = open_source(args.source)
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    writer = create_writer(args.save_video, source_fps, (width, height)) if width and height else None

    print("=" * 76)
    print("OpenCV badminton court four-point calibration demo")
    print(f"source   : {args.source}")
    print(f"redetect : every {float(args.redetect_interval):.1f}s; manual/confirmed points keep priority")
    print("template : 610 x 1340 standard court coordinates")
    print("H        : image_to_court_h maps image pixels to template coordinates")
    print("keys     : Drag points | a snap | r redetect | m manual | Enter confirm | s save | l load | v preview | q quit")
    print("=" * 76)

    window_name = "OpenCV court homography"
    editor = FourPointEditor(window_name)
    preview_template = bool(args.preview_template)
    debug_enabled = bool(args.draw_debug_lines)
    detection_status = "auto detection not run"
    last_detect_fps = 0.0
    state = TrackingState()

    if args.load_calibration:
        loaded = load_calibration(args.calibration)
        if loaded is not None:
            image_points, court_to_image_h, image_to_court_h = loaded
            del court_to_image_h, image_to_court_h
            editor.set_points(image_points)
            editor.confirmed = True
            editor.user_modified = True
            state.current = build_detection_from_manual_points(image_points, 0, 0.0, "loaded calibration")
            state.last_update_type = "loaded calibration"
            detection_status = f"loaded calibration {args.calibration}"
            print(f"[load] {args.calibration}")
        else:
            print(f"[load] no usable calibration at {args.calibration}")

    if args.manual_first and editor.point_count < 4:
        editor.clear()
        editor.set_manual_mode(True)
        detection_status = "manual mode"

    if not args.no_display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, editor.mouse_callback)

    frame_id = 0
    max_frames = max(0, int(args.max_frames))
    log_every = max(1, int(args.log_every))
    started_at = time.perf_counter()
    last_frame_at = started_at

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if max_frames and frame_id >= max_frames:
                break

            source_timestamp = frame_id / source_fps if not args.source.isdigit() else time.perf_counter() - started_at
            if should_redetect(state, frame_id, source_timestamp, args):
                last_detect_fps, detection_status, snap_count = run_auto_redetect(
                    frame,
                    state,
                    editor,
                    args,
                    frame_id,
                    source_timestamp,
                )
                candidate = state.last_candidate
                score = f"{candidate.confidence:.2f}" if candidate is not None else "--"
                print(
                    f"[auto] {detection_status} | candidate={score} | "
                    f"current={'yes' if state.current is not None else 'no'} | snapped={snap_count}/4"
                )

            if editor.snap_pending and editor.drag_index is None and editor.point_count == 4:
                snap_count = snap_editor_points_to_frame(frame, editor, args)
                detection_status = f"manual/drag snapped {snap_count}/4"
                image_points = editor.get_points()
                if image_points is not None:
                    state.current = build_detection_from_manual_points(
                        image_points,
                        frame_id,
                        source_timestamp,
                        "manual/drag snapped",
                    )
                    state.last_update_type = "manual current"
                if snap_count > 0:
                    print(f"[snap] manual/drag snapped {snap_count}/4 corners to white-line intersections")

            avg_fps = (frame_id + 1) / max(time.perf_counter() - started_at, 1e-6)
            fps_text = f"detect {last_detect_fps:.1f} FPS | avg {avg_fps:.1f} FPS | frame {frame_id}"

            vis = draw_visualization(frame, state, editor, fps_text, args, preview_template, debug_enabled, detection_status)
            if writer is not None:
                writer.write(vis)

            if not args.no_display:
                cv2.imshow(window_name, vis)
                wait_ms = 1
                if args.realtime_playback:
                    frame_interval = 1.0 / source_fps
                    spent = time.perf_counter() - last_frame_at
                    wait_ms = max(1, int((frame_interval - spent) * 1000))
                key = cv2.waitKey(wait_ms)
                key_code = key & 0xFF if key >= 0 else -1
                if key_code in (ord("q"), 27):
                    break
                if key_code == ord("r"):
                    last_detect_fps, detection_status, snap_count = run_auto_redetect(
                        frame,
                        state,
                        editor,
                        args,
                        frame_id,
                        source_timestamp,
                    )
                    candidate = state.last_candidate
                    score = f"{candidate.confidence:.2f}" if candidate is not None else "--"
                    if editor_has_manual_priority(editor):
                        print(f"[redetect] suggestion only | confidence={score}; current manual points were not overwritten")
                    else:
                        print(f"[redetect] {detection_status} | confidence={score} | snapped={snap_count}/4")
                elif key_code == ord("m"):
                    editor.clear()
                    editor.set_manual_mode(True)
                    detection_status = "manual mode"
                    print("[manual] click four corners: top-left, top-right, bottom-right, bottom-left")
                elif key_code == ord("a"):
                    snap_count = snap_editor_points_to_frame(frame, editor, args)
                    detection_status = f"manual snap {snap_count}/4"
                    image_points = editor.get_points()
                    if image_points is not None:
                        state.current = build_detection_from_manual_points(
                            image_points,
                            frame_id,
                            source_timestamp,
                            "manual snap",
                        )
                        state.last_update_type = "manual current"
                        editor.user_modified = True
                    print(f"[snap] snapped {snap_count}/4 corners to white-line intersections")
                elif key_code in (13, 10):
                    if editor.snap_pending and editor.drag_index is None and editor.point_count == 4:
                        snap_count = snap_editor_points_to_frame(frame, editor, args)
                        detection_status = f"confirm snap {snap_count}/4"
                    image_points = editor.get_points()
                    homography = compute_homography_from_points(image_points) if image_points is not None else None
                    if image_points is None or homography is None:
                        print("[confirm] need four valid points before computing Homography")
                    else:
                        court_to_image_h, image_to_court_h = homography
                        editor.confirmed = True
                        editor.user_modified = True
                        state.current = build_detection_from_manual_points(
                            image_points,
                            frame_id,
                            source_timestamp,
                            "manual confirmed",
                        )
                        state.last_update_type = "manual confirmed"
                        print("[confirm] image_points [top_left, top_right, bottom_right, bottom_left]:")
                        print(np.array2string(image_points, precision=3, suppress_small=True))
                        print("[confirm] court_to_image_h:")
                        print(np.array2string(court_to_image_h, precision=6, suppress_small=True))
                        print("[confirm] image_to_court_h:")
                        print(np.array2string(image_to_court_h, precision=6, suppress_small=True))
                elif key_code == ord("s"):
                    if editor.snap_pending and editor.drag_index is None and editor.point_count == 4:
                        snap_count = snap_editor_points_to_frame(frame, editor, args)
                        detection_status = f"save snap {snap_count}/4"
                    image_points = editor.get_points()
                    if image_points is None:
                        print("[save] need four valid points before saving")
                    else:
                        try:
                            save_calibration(args.calibration, args.source, frame.shape[:2], image_points)
                            editor.confirmed = True
                            editor.user_modified = True
                            state.current = build_detection_from_manual_points(
                                image_points,
                                frame_id,
                                source_timestamp,
                                "manual saved",
                            )
                            state.last_update_type = "manual saved"
                            print(f"[save] {args.calibration}")
                        except (ValueError, OSError) as exc:
                            print(f"[save] failed: {exc}")
                elif key_code == ord("l"):
                    loaded = load_calibration(args.calibration)
                    if loaded is None:
                        print(f"[load] no usable calibration at {args.calibration}")
                    else:
                        image_points, court_to_image_h, image_to_court_h = loaded
                        del court_to_image_h, image_to_court_h
                        editor.set_points(image_points)
                        editor.confirmed = True
                        editor.user_modified = True
                        editor.set_manual_mode(False)
                        state.current = build_detection_from_manual_points(
                            image_points,
                            frame_id,
                            source_timestamp,
                            "loaded calibration",
                        )
                        state.last_update_type = "loaded calibration"
                        detection_status = f"loaded calibration {args.calibration}"
                        print(f"[load] {args.calibration}")
                elif key_code == ord("v"):
                    preview_template = not preview_template
                    print(f"[preview] {'on' if preview_template else 'off'}")
                elif key_code == ord("d"):
                    debug_enabled = not debug_enabled
                    print(f"[debug] {'on' if debug_enabled else 'off'}")
                last_frame_at = time.perf_counter()

            if frame_id % log_every == 0:
                current_conf = state.current.confidence if state.current is not None else float("nan")
                candidate_conf = state.last_candidate.confidence if state.last_candidate is not None else float("nan")
                print(
                    f"[frame {frame_id:06d}] points={editor.point_count}/4 "
                    f"current={format_float(current_conf)} candidate={format_float(candidate_conf)} "
                    f"preview={'on' if preview_template else 'off'} | {fps_text}"
                )

            frame_id += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyWindow(window_name)

    print(f"[done] processed {frame_id} frames")
    if args.save_video:
        print(f"[out] {args.save_video}")


if __name__ == "__main__":
    main()
