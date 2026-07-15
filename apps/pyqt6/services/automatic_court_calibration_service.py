from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

from apps.pyqt6.services.court_detection_service import CourtDetectionService
from apps.pyqt6.services.manual_court_calibration_service import manual_court_prediction_from_corners
from src.court import CourtKeyNetConfig, CourtLineBackend, create_court_line_detector
from src.court.batch_court import (
    BatchCourtPredictor,
    is_trusted_automatic_court_prediction,
)
from src.court.opencv_court_detector import CourtLinePrediction


_AUTOMATIC_BACKENDS: tuple[CourtLineBackend, ...] = (
    "courtkeynet",
    "shuttlecourt_seg",
    "monotrack",
    "opencv",
)


def _provisional_payload(payload: dict, prediction: CourtLinePrediction) -> dict:
    filtered = dict(payload)
    has_geometry = bool(
        len(prediction.corners) == 4
        and isinstance(prediction.projected_lines, dict)
        and prediction.projected_lines
    )
    metrics = prediction.metrics if isinstance(prediction.metrics, dict) else {}
    scan_exhausted = bool(metrics.get("bootstrap_exhausted"))
    if scan_exhausted and has_geometry:
        status = "automatic scan complete; editable draft did not pass white-line verification"
    elif scan_exhausted:
        status = "automatic scan complete; no usable court geometry found"
    elif prediction.scheme == "courtkeynet":
        components = metrics.get("components") if isinstance(metrics.get("components"), dict) else {}
        count = int(float(components.get("courtkeynet_confirmation_count", 0)))
        required = int(float(components.get("courtkeynet_confirmation_required", 0)))
        status = f"provisional CourtKeyNet court; confirmation {count}/{required}"
    else:
        status = "provisional automatic court; waiting for verified white-line evidence"
    filtered.update(
        {
            "valid": False,
            "provisional": has_geometry,
            "display_only": True,
            "updated": False,
            "update_type": "provisional automatic court",
            "status": status,
            "confidence": 0.0,
            "candidate_confidence": float(
                prediction.candidate_confidence
                if prediction.candidate_confidence is not None
                else prediction.confidence
            ),
        }
    )
    return filtered


def _create_automatic_detector(config: CourtKeyNetConfig | None) -> BatchCourtPredictor:
    if config is None:
        return BatchCourtPredictor(
            backends=_AUTOMATIC_BACKENDS,
            authoritative_backends=("courtkeynet",),
            fallback_confirm_frames=3,
            reset_unaccepted_detector_state=True,
            lock_confirmed_fallback_geometry=True,
            prediction_acceptor=lambda backend, prediction: is_trusted_automatic_court_prediction(
                prediction
            ),
        )

    def detector_factory(backend: CourtLineBackend):
        backend_config = config if backend == "courtkeynet" else None
        return create_court_line_detector(backend, config=backend_config)

    return BatchCourtPredictor(
        backends=_AUTOMATIC_BACKENDS,
        detector_factory=detector_factory,
        authoritative_backends=("courtkeynet",),
        fallback_confirm_frames=3,
        reset_unaccepted_detector_state=True,
        lock_confirmed_fallback_geometry=True,
        prediction_acceptor=lambda backend, prediction: is_trusted_automatic_court_prediction(
            prediction
        ),
    )


class AutomaticCourtCalibrationService(QObject):
    """Automatic court detection with a persistent manual correction override."""

    resultReady = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        config: CourtKeyNetConfig | None = None,
        *,
        automatic_service: Any | None = None,
    ) -> None:
        super().__init__()
        self._automatic = automatic_service or CourtDetectionService(
            detector_factory=lambda: _create_automatic_detector(config)
        )
        self._automatic.resultReady.connect(self._on_automatic_result)
        self._automatic.failed.connect(self._on_automatic_failed)
        self._automatic_prediction: CourtLinePrediction | None = None
        self._provisional_prediction: dict | None = None
        self._manual_prediction: CourtLinePrediction | None = None

    def start(self) -> None:
        self._automatic.start()

    def stop(self) -> None:
        self._automatic.stop()

    def reset(self) -> None:
        if self._manual_prediction is not None:
            self._automatic.clear_pending()
            return
        self._automatic_prediction = None
        self._provisional_prediction = None
        self._automatic.reset()

    def request_prediction(self) -> None:
        if self._manual_prediction is None:
            self._automatic.request_prediction()

    def clear_pending(self) -> None:
        self._automatic.clear_pending()

    def submit_frame(self, frame: np.ndarray, frame_id: int, timestamp_ms: int) -> bool:
        if self._manual_prediction is not None:
            return False
        return bool(self._automatic.submit_frame(frame, frame_id, timestamp_ms))

    def submit_bootstrap_frames(
        self,
        samples: Sequence[tuple[np.ndarray, int, int]],
    ) -> int:
        if self._manual_prediction is not None:
            return 0
        submit = getattr(self._automatic, "submit_bootstrap_frames", None)
        return int(submit(samples)) if callable(submit) else 0

    def latest_prediction(self) -> CourtLinePrediction | None:
        return self._manual_prediction or self._automatic_prediction

    def latest_prediction_dict(self) -> dict | None:
        prediction = self.latest_prediction()
        return prediction.to_dict() if prediction is not None else None

    def latest_display_prediction_dict(self) -> dict | None:
        prediction = self.latest_prediction()
        if prediction is not None:
            return prediction.to_dict()
        return dict(self._provisional_prediction) if self._provisional_prediction is not None else None

    def clear_calibration(self) -> None:
        self._manual_prediction = None
        self._automatic_prediction = None
        self._provisional_prediction = None
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
        self._provisional_prediction = None
        self._automatic.clear_pending()
        self.resultReady.emit(prediction.to_dict())
        return prediction

    def _on_automatic_result(self, payload: object) -> None:
        if self._manual_prediction is not None or not isinstance(payload, dict):
            return
        prediction = self._automatic.latest_prediction()
        if not isinstance(prediction, CourtLinePrediction):
            return
        if is_trusted_automatic_court_prediction(prediction):
            self._automatic_prediction = prediction
            self._provisional_prediction = None
            self.resultReady.emit(payload)
            return
        if self._automatic_prediction is not None:
            self.resultReady.emit(self._automatic_prediction.to_dict())
            return
        provisional = _provisional_payload(payload, prediction)
        self._provisional_prediction = provisional if provisional.get("provisional") else None
        self.resultReady.emit(provisional)

    def _on_automatic_failed(self, message: str) -> None:
        if self._manual_prediction is None:
            self.failed.emit(message)


def create_automatic_court_calibration_service(
    config: CourtKeyNetConfig | None = None,
) -> AutomaticCourtCalibrationService:
    service = AutomaticCourtCalibrationService(config)
    service.start()
    return service
