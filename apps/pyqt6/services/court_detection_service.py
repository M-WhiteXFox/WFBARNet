from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from threading import Condition, Lock
from time import monotonic
from typing import Any, Callable, Sequence

import numpy as np
from PyQt6.QtCore import QObject, QThread, pyqtSignal

from src.court import (
    CourtLineBackend,
    CourtLineConfig,
    CourtLinePrediction,
    create_court_line_detector,
)


@dataclass(slots=True)
class _PendingCourtFrame:
    frame: np.ndarray
    frame_id: int
    timestamp_ms: int
    generation: int
    force: bool = True
    bootstrap: bool = False


@dataclass(slots=True)
class _CourtWorkerResult:
    prediction: object
    generation: int


@dataclass(slots=True)
class _CourtWorkerFailure:
    message: str
    generation: int


DetectorFactory = Callable[[], Any]


class CourtDetectionWorker(QThread):
    resultReady = pyqtSignal(object)
    failed = pyqtSignal(object)

    def __init__(
        self,
        config: CourtLineConfig | None = None,
        *,
        backend: CourtLineBackend = "shuttlecourt_seg",
        submit_interval_s: float = 0.75,
        detector_factory: DetectorFactory | None = None,
    ) -> None:
        super().__init__()
        self._backend = backend
        self._config = config
        self._submit_interval_s = max(0.1, float(submit_interval_s))
        self._detector_factory = detector_factory
        self._condition = Condition()
        self._latest_lock = Lock()
        self._pending: _PendingCourtFrame | None = None
        self._bootstrap_pending: deque[_PendingCourtFrame] = deque()
        self._latest_prediction: CourtLinePrediction | None = None
        self._stop_requested = False
        self._reset_requested = False
        self._last_accept_at = -1.0e9
        self._generation = 0
        self._prediction_requested = False
        self._force_next_prediction = False

    def submit_frame(self, frame: np.ndarray, frame_id: int, timestamp_ms: int) -> bool:
        if frame is None or frame.ndim < 2:
            return False

        now = monotonic()
        with self._condition:
            if (
                self._stop_requested
                or self._pending is not None
                or self._bootstrap_pending
                or not self._prediction_requested
            ):
                return False
            if now - self._last_accept_at < self._submit_interval_s:
                return False
            self._pending = _PendingCourtFrame(
                frame=frame.copy(),
                frame_id=int(frame_id),
                timestamp_ms=int(timestamp_ms),
                generation=self._generation,
                force=self._force_next_prediction,
            )
            self._force_next_prediction = False
            self._last_accept_at = now
            self._condition.notify()
            return True

    def submit_bootstrap_frames(
        self,
        samples: Sequence[tuple[np.ndarray, int, int]],
    ) -> int:
        pending: list[_PendingCourtFrame] = []
        with self._condition:
            if self._stop_requested or self._pending is not None or self._bootstrap_pending:
                return 0
            for frame, frame_id, timestamp_ms in samples:
                if frame is None or frame.ndim < 2:
                    continue
                pending.append(
                    _PendingCourtFrame(
                        frame=frame,
                        frame_id=int(frame_id),
                        timestamp_ms=int(timestamp_ms),
                        generation=self._generation,
                        force=True,
                        bootstrap=True,
                    )
                )
            if not pending:
                return 0
            self._bootstrap_pending.extend(pending)
            self._prediction_requested = True
            self._force_next_prediction = False
            self._last_accept_at = monotonic()
            self._condition.notify()
            return len(pending)

    def request_prediction(self) -> None:
        with self._condition:
            self._prediction_requested = True
            self._force_next_prediction = True
            self._last_accept_at = -1.0e9
            self._condition.notify()

    def latest_prediction(self) -> CourtLinePrediction | None:
        with self._latest_lock:
            return self._latest_prediction

    def reset_detector(self) -> None:
        with self._condition:
            self._pending = None
            self._bootstrap_pending.clear()
            self._reset_requested = True
            self._last_accept_at = -1.0e9
            self._generation += 1
            self._prediction_requested = False
            self._force_next_prediction = False
            self._condition.notify()

        with self._latest_lock:
            self._latest_prediction = None

    def clear_pending(self) -> None:
        with self._condition:
            self._pending = None
            self._bootstrap_pending.clear()
            self._last_accept_at = -1.0e9
            self._prediction_requested = False
            self._force_next_prediction = False
            self._generation += 1

    def request_stop(self) -> None:
        with self._condition:
            self._stop_requested = True
            self._condition.notify()

    def run(self) -> None:
        detector = (
            self._detector_factory()
            if self._detector_factory is not None
            else create_court_line_detector(self._backend, config=self._config)
        )

        while True:
            with self._condition:
                while (
                    self._pending is None
                    and not self._bootstrap_pending
                    and not self._stop_requested
                    and not self._reset_requested
                ):
                    self._condition.wait(timeout=0.2)

                if self._stop_requested:
                    return

                if self._reset_requested:
                    reset_detector = getattr(detector, "reset", None)
                    try:
                        if callable(reset_detector):
                            reset_detector()
                    except Exception as exc:  # pragma: no cover - backend-specific reset guard.
                        self.failed.emit(_CourtWorkerFailure(str(exc), self._generation))
                    finally:
                        self._reset_requested = False
                    if self._pending is None and not self._bootstrap_pending:
                        continue

                if self._bootstrap_pending:
                    pending = self._bootstrap_pending.popleft()
                else:
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
                if self.is_current_generation(pending.generation):
                    self.failed.emit(_CourtWorkerFailure(str(exc), pending.generation))
                continue

            with self._condition:
                if (
                    pending.generation != self._generation
                    or self._reset_requested
                    or self._stop_requested
                ):
                    continue
                if (
                    pending.bootstrap
                    and not bool(getattr(prediction, "valid", False))
                    and not self._bootstrap_pending
                    and isinstance(prediction, CourtLinePrediction)
                ):
                    metrics = dict(prediction.metrics) if isinstance(prediction.metrics, dict) else {}
                    metrics["bootstrap_exhausted"] = 1
                    prediction = replace(
                        prediction,
                        status="automatic court scan exhausted",
                        reason=f"automatic court scan exhausted: {prediction.reason}",
                        metrics=metrics,
                    )
                with self._latest_lock:
                    self._latest_prediction = prediction
                if pending.bootstrap and bool(getattr(prediction, "valid", False)):
                    self._bootstrap_pending.clear()
            self.resultReady.emit(_CourtWorkerResult(prediction, pending.generation))

    def is_current_generation(self, generation: int) -> bool:
        with self._condition:
            return bool(
                generation == self._generation
                and not self._reset_requested
                and not self._stop_requested
            )


class CourtDetectionService(QObject):
    resultReady = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        config: CourtLineConfig | None = None,
        *,
        backend: CourtLineBackend = "shuttlecourt_seg",
        submit_interval_s: float = 0.75,
        detector_factory: DetectorFactory | None = None,
    ) -> None:
        super().__init__()
        self._backend = backend
        self._config = config
        self._submit_interval_s = submit_interval_s
        self._detector_factory = detector_factory
        self._worker: CourtDetectionWorker | None = None

    def start(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        worker = CourtDetectionWorker(
            self._config,
            backend=self._backend,
            submit_interval_s=self._submit_interval_s,
            detector_factory=self._detector_factory,
        )
        self._worker = worker
        worker.resultReady.connect(
            lambda result, source=worker: self._on_worker_result_ready(source, result)
        )
        worker.failed.connect(
            lambda failure, source=worker: self._on_worker_failed(source, failure)
        )
        worker.start()

    def stop(self) -> None:
        if self._worker is None:
            return
        worker = self._worker
        worker.request_stop()
        if worker.wait(5000) and self._worker is worker:
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

    def submit_bootstrap_frames(
        self,
        samples: Sequence[tuple[np.ndarray, int, int]],
    ) -> int:
        if self._worker is None or not self._worker.isRunning():
            return 0
        return self._worker.submit_bootstrap_frames(samples)

    def latest_prediction(self) -> CourtLinePrediction | None:
        if self._worker is None:
            return None
        return self._worker.latest_prediction()

    def latest_prediction_dict(self) -> dict | None:
        prediction = self.latest_prediction()
        return prediction.to_dict() if prediction is not None else None

    def _on_worker_result_ready(self, worker: CourtDetectionWorker, result: object) -> None:
        if self._worker is not worker or not isinstance(result, _CourtWorkerResult):
            return
        if not worker.is_current_generation(result.generation):
            return
        if isinstance(result.prediction, CourtLinePrediction):
            self.resultReady.emit(result.prediction.to_dict())

    def _on_worker_failed(self, worker: CourtDetectionWorker, failure: object) -> None:
        if self._worker is not worker or not isinstance(failure, _CourtWorkerFailure):
            return
        if worker.is_current_generation(failure.generation):
            self.failed.emit(failure.message)


OpenCVCourtDetectionWorker = CourtDetectionWorker


def create_court_detection_service(
    config: CourtLineConfig | None = None,
    *,
    backend: CourtLineBackend = "shuttlecourt_seg",
) -> CourtDetectionService:
    service = CourtDetectionService(config, backend=backend)
    service.start()
    return service
