from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from src.court.court_line_detector import CourtLineBackend, CourtLineDetector, create_court_line_detector
from src.court.court_pose_detector import resolve_court_pose_weights
from src.court.opencv_court_detector import CourtLinePrediction
from src.court.shuttlecourt_seg_detector import resolve_shuttlecourt_weights


DetectorFactory = Callable[[CourtLineBackend], CourtLineDetector]


def default_batch_court_backends() -> tuple[CourtLineBackend, ...]:
    """Return the batch court fallback order available in the current workspace."""
    backends: list[CourtLineBackend] = []
    try:
        resolve_court_pose_weights("assets/weights/court_pose/best.pt")
    except FileNotFoundError:
        pass
    else:
        backends.append("court_pose")
    try:
        resolve_shuttlecourt_weights("weights/shttlecourtnet")
    except FileNotFoundError:
        pass
    else:
        backends.append("shuttlecourt_seg")
    backends.extend(["monotrack", "opencv"])
    return tuple(backends)


def _default_detector_factory(backend: CourtLineBackend) -> CourtLineDetector:
    return create_court_line_detector(backend=backend)


@dataclass
class BatchCourtPredictor:
    """Stateful court detector with backend fallback for offline batch analysis."""

    backends: tuple[CourtLineBackend, ...] = field(default_factory=default_batch_court_backends)
    detector_factory: DetectorFactory = _default_detector_factory
    active_backend: str = ""
    errors: list[str] = field(default_factory=list)
    _active_detector: CourtLineDetector | None = field(default=None, init=False, repr=False)

    def predict(
        self,
        frame: np.ndarray,
        frame_id: int,
        timestamp_ms: int,
        *,
        force: bool = False,
    ) -> CourtLinePrediction | None:
        if self._active_detector is not None:
            try:
                return self._active_detector.predict(frame, frame_id, timestamp_ms, force=force)
            except Exception as exc:
                self.errors.append(f"{self.active_backend}: {exc}")
                self._active_detector = None
                self.active_backend = ""

        return self._select_backend(frame, frame_id, timestamp_ms, force=force)

    def _select_backend(
        self,
        frame: np.ndarray,
        frame_id: int,
        timestamp_ms: int,
        *,
        force: bool,
    ) -> CourtLinePrediction | None:
        fallback_prediction: CourtLinePrediction | None = None
        fallback_detector: CourtLineDetector | None = None
        fallback_backend = ""

        for backend in self.backends:
            try:
                detector = self.detector_factory(backend)
                prediction = detector.predict(frame, frame_id, timestamp_ms, force=force)
            except Exception as exc:
                self.errors.append(f"{backend}: {exc}")
                continue

            if prediction.valid:
                self._active_detector = detector
                self.active_backend = backend
                return prediction
            if fallback_prediction is None:
                fallback_prediction = prediction
                fallback_detector = detector
                fallback_backend = backend

        self._active_detector = fallback_detector
        self.active_backend = fallback_backend
        return fallback_prediction
