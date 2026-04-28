from __future__ import annotations

from dataclasses import dataclass
from threading import Condition, Lock
from time import monotonic

import numpy as np
from PyQt6.QtCore import QObject, QThread, pyqtSignal

from src.court.opencv_court_detector import (
    CourtLinePrediction,
    OpenCVCourtLineConfig,
    OpenCVCourtLineDetector,
)


@dataclass(slots=True)
class _PendingCourtFrame:
    frame: np.ndarray
    frame_id: int
    timestamp_ms: int
    generation: int
    force: bool = True


class OpenCVCourtDetectionWorker(QThread):
    resultReady = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        config: OpenCVCourtLineConfig | None = None,
        *,
        submit_interval_s: float = 0.75,
    ) -> None:
        super().__init__()
        self._config = config or OpenCVCourtLineConfig()
        self._submit_interval_s = max(0.1, float(submit_interval_s))
        self._condition = Condition()
        self._latest_lock = Lock()
        self._pending: _PendingCourtFrame | None = None
        self._latest_prediction: CourtLinePrediction | None = None
        self._stop_requested = False
        self._reset_requested = False
        self._last_accept_at = -1.0e9
        self._generation = 0
        self._prediction_requested = False

    def submit_frame(self, frame: np.ndarray, frame_id: int, timestamp_ms: int) -> bool:
        if frame is None or frame.ndim < 2:
            return False

        now = monotonic()
        with self._condition:
            if self._stop_requested or self._pending is not None or not self._prediction_requested:
                return False
            if now - self._last_accept_at < self._submit_interval_s:
                return False
            self._pending = _PendingCourtFrame(
                frame=frame.copy(),
                frame_id=int(frame_id),
                timestamp_ms=int(timestamp_ms),
                generation=self._generation,
                force=True,
            )
            self._prediction_requested = False
            self._last_accept_at = now
            self._condition.notify()
            return True

    def request_prediction(self) -> None:
        with self._condition:
            self._prediction_requested = True
            self._last_accept_at = -1.0e9
            self._condition.notify()

    def latest_prediction(self) -> CourtLinePrediction | None:
        with self._latest_lock:
            return self._latest_prediction

    def reset_detector(self) -> None:
        with self._condition:
            self._pending = None
            self._reset_requested = True
            self._last_accept_at = -1.0e9
            self._generation += 1
            self._prediction_requested = False
            self._condition.notify()

        with self._latest_lock:
            self._latest_prediction = None

    def clear_pending(self) -> None:
        with self._condition:
            self._pending = None
            self._last_accept_at = -1.0e9
            self._prediction_requested = False

    def request_stop(self) -> None:
        with self._condition:
            self._stop_requested = True
            self._condition.notify()

    def run(self) -> None:
        detector = OpenCVCourtLineDetector(self._config)

        while True:
            with self._condition:
                while self._pending is None and not self._stop_requested and not self._reset_requested:
                    self._condition.wait(timeout=0.2)

                if self._stop_requested:
                    return

                if self._reset_requested:
                    detector.reset()
                    self._reset_requested = False
                    if self._pending is None:
                        continue

                pending = self._pending
                self._pending = None

            if pending is None:
                continue

            try:
                prediction = detector.predict(
                    pending.frame,
                    pending.frame_id,
                    pending.timestamp_ms,
                    force=pending.force,
                )
            except Exception as exc:  # pragma: no cover - protects the UI worker loop.
                self.failed.emit(str(exc))
                continue

            with self._condition:
                if pending.generation != self._generation or self._reset_requested or self._stop_requested:
                    continue

            with self._latest_lock:
                self._latest_prediction = prediction
            self.resultReady.emit(prediction)


class CourtDetectionService(QObject):
    resultReady = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, config: OpenCVCourtLineConfig | None = None, *, submit_interval_s: float = 0.75) -> None:
        super().__init__()
        self._config = config or OpenCVCourtLineConfig()
        self._submit_interval_s = submit_interval_s
        self._worker: OpenCVCourtDetectionWorker | None = None

    def start(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        self._worker = OpenCVCourtDetectionWorker(self._config, submit_interval_s=self._submit_interval_s)
        self._worker.resultReady.connect(self._on_worker_result_ready)
        self._worker.failed.connect(self.failed.emit)
        self._worker.start()

    def stop(self) -> None:
        if self._worker is None:
            return
        self._worker.request_stop()
        self._worker.wait(5000)
        self._worker = None

    def reset(self) -> None:
        self.start()
        if self._worker is not None:
            self._worker.reset_detector()

    def request_prediction(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.request_prediction()

    def clear_pending(self) -> None:
        if self._worker is not None:
            self._worker.clear_pending()

    def submit_frame(self, frame: np.ndarray, frame_id: int, timestamp_ms: int) -> bool:
        if self._worker is None or not self._worker.isRunning():
            return False
        return self._worker.submit_frame(frame, frame_id, timestamp_ms)

    def latest_prediction(self) -> CourtLinePrediction | None:
        if self._worker is None:
            return None
        return self._worker.latest_prediction()

    def latest_prediction_dict(self) -> dict | None:
        prediction = self.latest_prediction()
        return prediction.to_dict() if prediction is not None else None

    def _on_worker_result_ready(self, prediction: object) -> None:
        if isinstance(prediction, CourtLinePrediction):
            self.resultReady.emit(prediction.to_dict())


def create_court_detection_service(config: OpenCVCourtLineConfig | None = None) -> CourtDetectionService:
    service = CourtDetectionService(config)
    service.start()
    return service
