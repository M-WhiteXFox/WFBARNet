from __future__ import annotations

from typing import Sequence

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

from src.court import opencv_court_homography_core as _court_core
from src.court.opencv_court_detector import CourtLinePrediction


class ManualCourtCalibrationService(QObject):
    """Manual four-point court calibration source compatible with court detection service calls."""

    resultReady = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self._latest_prediction: CourtLinePrediction | None = None

    def start(self) -> None:
        return

    def stop(self) -> None:
        return

    def reset(self) -> None:
        return

    def request_prediction(self) -> None:
        return

    def clear_pending(self) -> None:
        return

    def submit_frame(self, frame: object, frame_id: int, timestamp_ms: int) -> bool:
        return False

    def latest_prediction(self) -> CourtLinePrediction | None:
        return self._latest_prediction

    def latest_prediction_dict(self) -> dict | None:
        prediction = self.latest_prediction()
        return prediction.to_dict() if prediction is not None else None

    def clear_calibration(self) -> None:
        self._latest_prediction = None
        self.resultReady.emit(None)

    def set_calibration(
        self,
        corners: Sequence[Sequence[float]],
        *,
        source_size: tuple[int, int],
        frame_id: int = 0,
        timestamp_ms: int = 0,
    ) -> CourtLinePrediction:
        prediction = manual_court_prediction_from_corners(
            corners,
            source_size=source_size,
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
        )
        self._latest_prediction = prediction
        self.resultReady.emit(prediction.to_dict())
        return prediction


def manual_court_prediction_from_corners(
    corners: Sequence[Sequence[float]],
    *,
    source_size: tuple[int, int],
    frame_id: int = 0,
    timestamp_ms: int = 0,
) -> CourtLinePrediction:
    corner_array = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    if corner_array.shape != (4, 2):
        raise ValueError("Manual court calibration requires exactly four points.")
    if not _court_core.is_convex_quad(corner_array):
        raise ValueError("Manual court calibration points must form a convex quadrilateral.")

    court_to_image_h, image_to_court_h = _court_core.compute_homographies(corner_array)
    if court_to_image_h is None or image_to_court_h is None:
        raise ValueError("Manual court calibration points cannot produce a valid homography.")

    names_and_points = _court_core.template_keypoints_for_scheme("8")
    keypoint_names = [item[0] for item in names_and_points]
    template_points = np.asarray([item[1] for item in names_and_points], dtype=np.float32)
    keypoints = _court_core.project_points(template_points, court_to_image_h)
    projected_lines = _court_core.project_template_lines(court_to_image_h)
    width, height = int(source_size[0]), int(source_size[1])

    return CourtLinePrediction(
        frame_id=int(frame_id),
        timestamp_ms=max(0, int(timestamp_ms)),
        source_size=(width, height),
        valid=True,
        attempted=True,
        updated=True,
        update_type="manual calibration",
        status="manual calibration",
        confidence=1.0,
        candidate_confidence=1.0,
        reason="manual four-point calibration",
        scheme="manual",
        corners=_points_to_list(corner_array),
        keypoints=_keypoints_to_list(keypoint_names, keypoints),
        court_to_image_h=_matrix_to_list(court_to_image_h),
        image_to_court_h=_matrix_to_list(image_to_court_h),
        projected_lines={name: _points_to_list(points) for name, points in projected_lines.items()},
        metrics={
            "manual": True,
            "supported_keypoints": len(keypoint_names),
        },
        detect_ms=0.0,
        rejected_count=0,
    )


def create_manual_court_calibration_service() -> ManualCourtCalibrationService:
    service = ManualCourtCalibrationService()
    service.start()
    return service


def _points_to_list(points: object) -> list[list[float]]:
    array = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    return [[float(x), float(y)] for x, y in array]


def _keypoints_to_list(names: list[str], points: object) -> list[dict[str, object]]:
    point_list = _points_to_list(points)
    return [
        {
            "name": names[index] if index < len(names) else f"keypoint_{index}",
            "point": point,
        }
        for index, point in enumerate(point_list)
    ]


def _matrix_to_list(matrix: object) -> list[list[float]]:
    array = np.asarray(matrix, dtype=np.float64)
    if array.shape != (3, 3):
        return []
    return [[float(value) for value in row] for row in array.tolist()]
