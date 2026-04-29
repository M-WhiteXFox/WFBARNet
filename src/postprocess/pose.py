from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from src.preprocess.pose import PosePreprocessMeta
from src.utils.structures import PersonPoseResult

COURT_WIDTH = 610.0
COURT_LENGTH = 1340.0
COURT_NET_Y = COURT_LENGTH / 2.0
COURT_HALF_ORDER = ("top", "bottom")


def restore_keypoints(
    keypoints: Iterable[Iterable[float]],
    meta: PosePreprocessMeta,
) -> list[list[float]]:
    restored = []
    for x, y in keypoints:
        restored.append([float(x) * meta.scale_x, float(y) * meta.scale_y])
    return restored


def restore_bbox(bbox: Iterable[float], meta: PosePreprocessMeta) -> list[float]:
    x1, y1, x2, y2 = bbox
    return [
        float(x1) * meta.scale_x,
        float(y1) * meta.scale_y,
        float(x2) * meta.scale_x,
        float(y2) * meta.scale_y,
    ]


def build_pose_result(
    person_id: int,
    bbox: list[float],
    keypoints: list[list[float]],
    scores: list[float],
    meta: PosePreprocessMeta | None = None,
) -> PersonPoseResult:
    restored_bbox = restore_bbox(bbox, meta) if meta is not None else [float(x) for x in bbox]
    restored_kpts = restore_keypoints(keypoints, meta) if meta is not None else [
        [float(x), float(y)] for x, y in keypoints
    ]
    person_score = float(sum(scores) / max(len(scores), 1))
    return PersonPoseResult(
        person_id=person_id,
        bbox=restored_bbox,
        keypoints=restored_kpts,
        scores=[float(x) for x in scores],
        person_score=person_score,
    )


def filter_pose_results_by_court_halves(
    poses: list[PersonPoseResult],
    court_prediction: Any | None,
    *,
    max_per_half: int = 1,
    court_margin: float = 30.0,
) -> list[PersonPoseResult]:
    """Keep at most one pose in each half using the court homography."""
    limit = max(0, int(max_per_half)) * 2
    if not poses or limit <= 0:
        return []

    image_to_court_h = _extract_image_to_court_h(court_prediction)
    if image_to_court_h is None:
        return _renumber_pose_results(poses[:limit])

    by_half: dict[str, list[tuple[float, PersonPoseResult]]] = {"top": [], "bottom": []}
    for pose in poses:
        anchor = _pose_anchor_point(pose)
        if anchor is None:
            continue
        court_xy = _project_image_point(image_to_court_h, anchor)
        if court_xy is None:
            continue

        half = _court_half(court_xy, court_margin)
        if half is None:
            continue

        score = _pose_half_score(pose, court_xy)
        by_half[half].append((score, pose))

    selected: list[PersonPoseResult] = []
    for half in ("top", "bottom"):
        candidates = sorted(by_half[half], key=lambda item: item[0], reverse=True)
        selected.extend(pose for _, pose in candidates[:max_per_half])

    return _renumber_pose_results(selected)


@dataclass(slots=True)
class _PoseTrack:
    half: str
    pose: PersonPoseResult | None = None
    velocity_xy: tuple[float, float] = (0.0, 0.0)
    missed_frames: int = 0
    age: int = 0


class CourtPoseTargetTracker:
    """Detection-assisted tracker that keeps one pose target in each court half."""

    def __init__(
        self,
        *,
        max_missing_frames: int = 24,
        court_margin: float = 30.0,
        detection_smoothing: float = 0.68,
        velocity_smoothing: float = 0.55,
        court_required: bool = False,
        predict_missing_motion: bool = True,
        motion_prediction_scale: float = 1.0,
    ) -> None:
        self.max_missing_frames = max(0, int(max_missing_frames))
        self.court_margin = max(0.0, float(court_margin))
        self.detection_smoothing = min(1.0, max(0.0, float(detection_smoothing)))
        self.velocity_smoothing = min(1.0, max(0.0, float(velocity_smoothing)))
        self.court_required = bool(court_required)
        self.predict_missing_motion = bool(predict_missing_motion)
        self.motion_prediction_scale = min(1.0, max(0.0, float(motion_prediction_scale)))
        self._tracks = {half: _PoseTrack(half=half) for half in COURT_HALF_ORDER}

    def reset(self) -> None:
        for half in COURT_HALF_ORDER:
            self._tracks[half] = _PoseTrack(half=half)

    def update(
        self,
        detections: list[PersonPoseResult],
        court_prediction: Any | None,
        *,
        frame_shape: tuple[int, ...] | None = None,
    ) -> list[PersonPoseResult]:
        image_to_court_h = _extract_image_to_court_h(court_prediction)
        if image_to_court_h is None:
            if self.court_required:
                self.reset()
                return []
            self._update_without_court(detections, frame_shape=frame_shape)
            return self._active_outputs()

        candidates_by_half = self._split_detections_by_half(detections, image_to_court_h)
        for half in COURT_HALF_ORDER:
            track = self._tracks[half]
            candidate = self._best_candidate_for_track(track, candidates_by_half[half])
            if candidate is None:
                self._predict_track(track, frame_shape=frame_shape)
            else:
                self._correct_track(track, candidate, frame_shape=frame_shape)
            if track.pose is not None and not _pose_is_in_court_area(
                track.pose,
                image_to_court_h,
                self.court_margin,
            ):
                self._clear_track(track)
        return self._active_outputs()

    def _split_detections_by_half(
        self,
        detections: list[PersonPoseResult],
        image_to_court_h: np.ndarray,
    ) -> dict[str, list[PersonPoseResult]]:
        candidates_by_half: dict[str, list[PersonPoseResult]] = {half: [] for half in COURT_HALF_ORDER}
        for pose in detections:
            anchor = _pose_anchor_point(pose)
            if anchor is None:
                continue
            court_xy = _project_image_point(image_to_court_h, anchor)
            if court_xy is None:
                continue
            half = _court_half(court_xy, self.court_margin)
            if half is not None:
                candidates_by_half[half].append(pose)
        return candidates_by_half

    def _update_without_court(
        self,
        detections: list[PersonPoseResult],
        *,
        frame_shape: tuple[int, ...] | None,
    ) -> None:
        remaining = list(detections)
        assigned: dict[str, PersonPoseResult] = {}

        for half in COURT_HALF_ORDER:
            track = self._tracks[half]
            if track.pose is None or not remaining:
                continue
            best = self._best_candidate_for_track(track, remaining)
            if best is not None:
                assigned[half] = best
                remaining.remove(best)

        for half in COURT_HALF_ORDER:
            if half in assigned or not remaining:
                continue
            track = self._tracks[half]
            if track.pose is not None and track.missed_frames <= self.max_missing_frames:
                continue
            best = max(remaining, key=lambda pose: float(pose.person_score))
            assigned[half] = best
            remaining.remove(best)

        for half in COURT_HALF_ORDER:
            track = self._tracks[half]
            if half in assigned:
                self._correct_track(track, assigned[half], frame_shape=frame_shape)
            else:
                self._predict_track(track, frame_shape=frame_shape)

    def _best_candidate_for_track(
        self,
        track: _PoseTrack,
        candidates: list[PersonPoseResult],
    ) -> PersonPoseResult | None:
        if not candidates:
            return None
        if track.pose is None or track.missed_frames > self.max_missing_frames:
            return max(candidates, key=lambda pose: float(pose.person_score))
        prediction_steps = self._prediction_steps_for_match(track)
        ranked = sorted(
            candidates,
            key=lambda pose: _track_match_score(track, pose, prediction_steps=prediction_steps),
            reverse=True,
        )
        for candidate in ranked:
            if _track_match_allowed(track, candidate, prediction_steps=prediction_steps):
                return candidate
        return None

    def _correct_track(
        self,
        track: _PoseTrack,
        detection: PersonPoseResult,
        *,
        frame_shape: tuple[int, ...] | None,
    ) -> None:
        detection = _clamp_pose(_copy_pose(detection), frame_shape)
        if track.pose is None or track.missed_frames > self.max_missing_frames:
            track.pose = detection
            track.velocity_xy = (0.0, 0.0)
            track.missed_frames = 0
            track.age += 1
            return

        previous_center = np.array(_bbox_center(track.pose.bbox), dtype=np.float64)
        detection_center = np.array(_bbox_center(detection.bbox), dtype=np.float64)
        elapsed_frames = max(1.0, float(track.missed_frames + 1))
        observed_velocity = (detection_center - previous_center) / elapsed_frames
        current_velocity = np.array(track.velocity_xy, dtype=np.float64)
        velocity = self.velocity_smoothing * current_velocity + (1.0 - self.velocity_smoothing) * observed_velocity
        velocity = _limit_velocity(velocity, detection.bbox)

        base_pose = (
            _shift_pose(
                track.pose,
                track.velocity_xy[0] * self.motion_prediction_scale,
                track.velocity_xy[1] * self.motion_prediction_scale,
                frame_shape,
            )
            if self.predict_missing_motion
            else track.pose
        )
        detection_alpha = _adaptive_detection_alpha(detection, self.detection_smoothing)
        track.pose = _blend_pose(base_pose, detection, detection_alpha, frame_shape)
        track.velocity_xy = (float(velocity[0]), float(velocity[1]))
        track.missed_frames = 0
        track.age += 1

    def _predict_track(
        self,
        track: _PoseTrack,
        *,
        frame_shape: tuple[int, ...] | None,
    ) -> None:
        if track.pose is None:
            return
        track.missed_frames += 1
        if track.missed_frames > self.max_missing_frames:
            track.pose = None
            track.velocity_xy = (0.0, 0.0)
            return

        dx, dy = track.velocity_xy
        if self.predict_missing_motion:
            track.pose = _shift_pose(
                track.pose,
                dx * self.motion_prediction_scale,
                dy * self.motion_prediction_scale,
                frame_shape,
            )
        track.pose.person_score = max(0.01, float(track.pose.person_score) * 0.96)
        track.velocity_xy = (dx * 0.88, dy * 0.88)
        track.age += 1

    def _clear_track(self, track: _PoseTrack) -> None:
        track.pose = None
        track.velocity_xy = (0.0, 0.0)
        track.missed_frames = self.max_missing_frames + 1

    def _prediction_steps_for_match(self, track: _PoseTrack) -> float:
        if self.predict_missing_motion:
            return self.motion_prediction_scale
        return float(max(1, track.missed_frames + 1))

    def _active_outputs(self) -> list[PersonPoseResult]:
        outputs: list[PersonPoseResult] = []
        for person_id, half in enumerate(COURT_HALF_ORDER):
            track = self._tracks[half]
            if track.pose is not None and track.missed_frames <= self.max_missing_frames:
                outputs.append(_copy_pose(track.pose, person_id=person_id))
        return outputs


def _extract_image_to_court_h(court_prediction: Any | None) -> np.ndarray | None:
    if court_prediction is None:
        return None

    valid = _prediction_value(court_prediction, "valid", False)
    if not valid:
        return None

    raw_h = _prediction_value(court_prediction, "image_to_court_h", None)
    if raw_h is None:
        return None

    try:
        h = np.asarray(raw_h, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if h.shape != (3, 3) or not np.isfinite(h).all():
        return None
    return h


def _prediction_value(prediction: Any, key: str, default: Any) -> Any:
    if isinstance(prediction, dict):
        return prediction.get(key, default)
    return getattr(prediction, key, default)


def _pose_anchor_point(pose: PersonPoseResult) -> tuple[float, float] | None:
    if len(pose.bbox) < 4:
        return None
    x1, y1, x2, y2 = [float(value) for value in pose.bbox[:4]]
    if not np.isfinite([x1, y1, x2, y2]).all() or x2 <= x1 or y2 <= y1:
        return None
    return (x1 + x2) * 0.5, y2


def _project_image_point(
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


def _court_half(court_xy: tuple[float, float], margin: float) -> str | None:
    x, y = court_xy
    if x < -margin or x > COURT_WIDTH + margin:
        return None
    if y < -margin or y > COURT_LENGTH + margin:
        return None
    return "top" if y < COURT_NET_Y else "bottom"


def _pose_is_in_court_area(
    pose: PersonPoseResult,
    image_to_court_h: np.ndarray,
    margin: float,
) -> bool:
    anchor = _pose_anchor_point(pose)
    if anchor is None:
        return False
    court_xy = _project_image_point(image_to_court_h, anchor)
    if court_xy is None:
        return False
    return _court_half(court_xy, margin) is not None


def _pose_half_score(pose: PersonPoseResult, court_xy: tuple[float, float]) -> float:
    x, y = court_xy
    outside = max(0.0, -x, x - COURT_WIDTH, -y, y - COURT_LENGTH)
    center_bias = abs(x - COURT_WIDTH * 0.5) * 0.0002
    return float(pose.person_score) - outside * 0.01 - center_bias


def _renumber_pose_results(poses: list[PersonPoseResult]) -> list[PersonPoseResult]:
    return [
        _copy_pose(pose, person_id=index)
        for index, pose in enumerate(poses)
    ]


def _copy_pose(pose: PersonPoseResult, *, person_id: int | None = None) -> PersonPoseResult:
    return PersonPoseResult(
        person_id=pose.person_id if person_id is None else person_id,
        bbox=list(pose.bbox),
        keypoints=[list(point) for point in pose.keypoints],
        scores=list(pose.scores),
        person_score=float(pose.person_score),
    )


def _track_match_score(
    track: _PoseTrack,
    detection: PersonPoseResult,
    *,
    prediction_steps: float = 1.0,
) -> float:
    if track.pose is None:
        return float(detection.person_score)
    dx = track.velocity_xy[0] * max(0.0, float(prediction_steps))
    dy = track.velocity_xy[1] * max(0.0, float(prediction_steps))
    predicted_bbox = _shift_bbox(track.pose.bbox, dx, dy)
    iou = _bbox_iou(predicted_bbox, detection.bbox)
    predicted_center = np.array(_bbox_center(predicted_bbox), dtype=np.float64)
    detection_center = np.array(_bbox_center(detection.bbox), dtype=np.float64)
    center_distance = float(np.linalg.norm(detection_center - predicted_center))
    scale = _match_distance_limit(predicted_bbox, detection.bbox, track.missed_frames)
    distance_penalty = center_distance / scale
    return float(detection.person_score) + iou * 1.2 - distance_penalty * 0.35


def _track_match_allowed(
    track: _PoseTrack,
    detection: PersonPoseResult,
    *,
    prediction_steps: float = 1.0,
) -> bool:
    if track.pose is None:
        return True
    dx = track.velocity_xy[0] * max(0.0, float(prediction_steps))
    dy = track.velocity_xy[1] * max(0.0, float(prediction_steps))
    predicted_bbox = _shift_bbox(track.pose.bbox, dx, dy)
    iou = _bbox_iou(predicted_bbox, detection.bbox)
    predicted_center = np.array(_bbox_center(predicted_bbox), dtype=np.float64)
    detection_center = np.array(_bbox_center(detection.bbox), dtype=np.float64)
    center_distance = float(np.linalg.norm(detection_center - predicted_center))
    limit = _match_distance_limit(predicted_bbox, detection.bbox, track.missed_frames)
    return center_distance <= limit or (iou >= 0.08 and center_distance <= limit * 1.35)


def _match_distance_limit(
    previous_bbox: list[float],
    detection_bbox: list[float],
    missed_frames: int,
) -> float:
    height = max(_bbox_height(previous_bbox), _bbox_height(detection_bbox), 1.0)
    width = max(_bbox_width(previous_bbox), _bbox_width(detection_bbox), 1.0)
    base = max(24.0, min(150.0, max(height * 0.75, width * 2.0)))
    gap_bonus = min(max(0, int(missed_frames)), 4) * 12.0
    return base + gap_bonus


def _adaptive_detection_alpha(detection: PersonPoseResult, base_alpha: float) -> float:
    alpha = min(1.0, max(0.0, float(base_alpha)))
    diagonal = _bbox_diagonal(detection.bbox)
    if diagonal < 90.0:
        return min(alpha, 0.62)
    if diagonal < 140.0:
        return min(alpha, 0.70)
    return alpha


def _limit_velocity(velocity: np.ndarray, bbox: list[float]) -> np.ndarray:
    speed = float(np.linalg.norm(velocity))
    if speed <= 1e-6:
        return velocity
    limit = max(4.0, min(48.0, _bbox_diagonal(bbox) * 0.32))
    if speed <= limit:
        return velocity
    return velocity * (limit / speed)


def _blend_pose(
    predicted: PersonPoseResult,
    detection: PersonPoseResult,
    detection_alpha: float,
    frame_shape: tuple[int, ...] | None,
) -> PersonPoseResult:
    alpha = min(1.0, max(0.0, detection_alpha))
    beta = 1.0 - alpha
    bbox = [
        beta * float(old_value) + alpha * float(new_value)
        for old_value, new_value in zip(predicted.bbox, detection.bbox)
    ]

    if len(predicted.keypoints) == len(detection.keypoints):
        keypoints = [
            [
                beta * float(old_point[0]) + alpha * float(new_point[0]),
                beta * float(old_point[1]) + alpha * float(new_point[1]),
            ]
            for old_point, new_point in zip(predicted.keypoints, detection.keypoints)
        ]
    else:
        keypoints = [list(point) for point in detection.keypoints]

    scores = list(detection.scores) if detection.scores else list(predicted.scores)
    person_score = max(float(predicted.person_score) * 0.90, float(detection.person_score))
    return _clamp_pose(
        PersonPoseResult(
            person_id=detection.person_id,
            bbox=bbox,
            keypoints=keypoints,
            scores=scores,
            person_score=person_score,
        ),
        frame_shape,
    )


def _shift_pose(
    pose: PersonPoseResult,
    dx: float,
    dy: float,
    frame_shape: tuple[int, ...] | None,
) -> PersonPoseResult:
    shifted = PersonPoseResult(
        person_id=pose.person_id,
        bbox=_shift_bbox(pose.bbox, dx, dy),
        keypoints=[
            [float(point[0]) + dx, float(point[1]) + dy]
            for point in pose.keypoints
        ],
        scores=list(pose.scores),
        person_score=float(pose.person_score),
    )
    return _clamp_pose(shifted, frame_shape)


def _shift_bbox(bbox: list[float], dx: float, dy: float) -> list[float]:
    return [
        float(bbox[0]) + dx,
        float(bbox[1]) + dy,
        float(bbox[2]) + dx,
        float(bbox[3]) + dy,
    ]


def _clamp_pose(
    pose: PersonPoseResult,
    frame_shape: tuple[int, ...] | None,
) -> PersonPoseResult:
    if frame_shape is None or len(frame_shape) < 2:
        return pose
    height = max(1.0, float(frame_shape[0]))
    width = max(1.0, float(frame_shape[1]))
    pose.bbox = _clamp_bbox(pose.bbox, width, height)
    pose.keypoints = [
        [
            min(width - 1.0, max(0.0, float(point[0]))),
            min(height - 1.0, max(0.0, float(point[1]))),
        ]
        for point in pose.keypoints
    ]
    return pose


def _clamp_bbox(bbox: list[float], width: float, height: float) -> list[float]:
    x1 = min(width - 1.0, max(0.0, float(bbox[0])))
    y1 = min(height - 1.0, max(0.0, float(bbox[1])))
    x2 = min(width - 1.0, max(0.0, float(bbox[2])))
    y2 = min(height - 1.0, max(0.0, float(bbox[3])))
    if x2 <= x1:
        if x1 >= width - 1.0:
            x1 = max(0.0, width - 2.0)
        x2 = min(width - 1.0, x1 + 1.0)
    if y2 <= y1:
        if y1 >= height - 1.0:
            y1 = max(0.0, height - 2.0)
        y2 = min(height - 1.0, y1 + 1.0)
    return [x1, y1, x2, y2]


def _bbox_center(bbox: list[float]) -> tuple[float, float]:
    return (float(bbox[0]) + float(bbox[2])) * 0.5, (float(bbox[1]) + float(bbox[3])) * 0.5


def _bbox_diagonal(bbox: list[float]) -> float:
    width = max(0.0, float(bbox[2]) - float(bbox[0]))
    height = max(0.0, float(bbox[3]) - float(bbox[1]))
    return float(np.hypot(width, height))


def _bbox_width(bbox: list[float]) -> float:
    return max(0.0, float(bbox[2]) - float(bbox[0]))


def _bbox_height(bbox: list[float]) -> float:
    return max(0.0, float(bbox[3]) - float(bbox[1]))


def _bbox_iou(a: list[float], b: list[float]) -> float:
    ix1 = max(float(a[0]), float(b[0]))
    iy1 = max(float(a[1]), float(b[1]))
    ix2 = min(float(a[2]), float(b[2]))
    iy2 = min(float(a[3]), float(b[3]))
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    intersection = iw * ih
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    union = area_a + area_b - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union
