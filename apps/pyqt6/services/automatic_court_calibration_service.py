from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

from apps.pyqt6.services.court_detection_service import CourtDetectionService
from apps.pyqt6.services.manual_court_calibration_service import manual_court_prediction_from_corners
from src.court import CourtPoseConfig
from src.court.opencv_court_detector import CourtLinePrediction


class AutomaticCourtCalibrationService(QObject):
    """Automatic court detection with a persistent manual correction override."""

    resultReady = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        config: CourtPoseConfig | None = None,
        *,
        automatic_service: Any | None = None,
    ) -> None:
        super().__init__()
        self._automatic = automatic_service or CourtDetectionService(config, backend="court_pose")
        self._automatic.resultReady.connect(self._on_automatic_result)
        self._automatic.failed.connect(self._on_automatic_failed)
        self._automatic_prediction: CourtLinePrediction | None = None
        self._manual_prediction: CourtLinePrediction | None = None

    def start(self) -> None:
        self._automatic.start()

    def stop(self) -> None:
        self._automatic.stop()

    def reset(self) -> None:
        if self.latest_prediction() is not None:
            self._automatic.clear_pending()
            return
        self._automatic.reset()

    def request_prediction(self) -> None:
        if self._manual_prediction is None and self._automatic_prediction is None:
            self._automatic.request_prediction()

    def clear_pending(self) -> None:
        self._automatic.clear_pending()

    def submit_frame(self, frame: np.ndarray, frame_id: int, timestamp_ms: int) -> bool:
        if self._manual_prediction is not None or self._automatic_prediction is not None:
            return False
        return bool(self._automatic.submit_frame(frame, frame_id, timestamp_ms))

    def latest_prediction(self) -> CourtLinePrediction | None:
        return self._manual_prediction or self._automatic_prediction

    def latest_prediction_dict(self) -> dict | None:
        prediction = self.latest_prediction()
        return prediction.to_dict() if prediction is not None else None

    def clear_calibration(self) -> None:
        self._manual_prediction = None
        self._automatic_prediction = None
        self._automatic.reset()
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
        self._manual_prediction = prediction
        self._automatic.clear_pending()
        self.resultReady.emit(prediction.to_dict())
        return prediction

    def _on_automatic_result(self, payload: object) -> None:
        if self._manual_prediction is not None or not isinstance(payload, dict):
            return
        prediction = self._automatic.latest_prediction()
        if not isinstance(prediction, CourtLinePrediction):
            return
        self._automatic_prediction = prediction if prediction.valid else None
        self.resultReady.emit(payload)

    def _on_automatic_failed(self, message: str) -> None:
        if self._manual_prediction is None:
            self.failed.emit(message)


def create_automatic_court_calibration_service(
    config: CourtPoseConfig | None = None,
) -> AutomaticCourtCalibrationService:
    service = AutomaticCourtCalibrationService(config)
    service.start()
    return service
