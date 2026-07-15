from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

import numpy as np

from src.court.opencv_court_detector import (
    CourtLinePrediction,
    OpenCVCourtLineConfig,
    OpenCVCourtLineDetector,
)
from src.court.monotrack_court_detector import (
    MonoTrackCourtLineConfig,
    MonoTrackCourtLineDetector,
)
from src.court.shuttlecourt_seg_detector import (
    ShuttleCourtSegConfig,
    ShuttleCourtSegLineDetector,
)
from src.court.court_pose_detector import CourtPoseConfig, CourtPoseLineDetector
from src.court.courtkeynet_detector import CourtKeyNetConfig, CourtKeyNetLineDetector


CourtLineBackend = Literal["courtkeynet", "court_pose", "shuttlecourt_seg", "monotrack", "opencv"]
CourtLineConfig = (
    CourtKeyNetConfig
    | CourtPoseConfig
    | ShuttleCourtSegConfig
    | MonoTrackCourtLineConfig
    | OpenCVCourtLineConfig
)


@runtime_checkable
class CourtLineDetector(Protocol):
    def reset(self) -> None:
        ...

    def predict(
        self,
        frame: np.ndarray,
        frame_id: int,
        timestamp_ms: int,
        *,
        force: bool = False,
    ) -> CourtLinePrediction:
        ...

    def latest_prediction(self) -> CourtLinePrediction | None:
        ...


def create_court_line_detector(
    backend: CourtLineBackend = "shuttlecourt_seg",
    *,
    config: CourtLineConfig | None = None,
) -> CourtLineDetector:
    if backend == "courtkeynet":
        if config is not None and not isinstance(config, CourtKeyNetConfig):
            raise TypeError("CourtKeyNet detector requires CourtKeyNetConfig.")
        return CourtKeyNetLineDetector(config)
    if backend == "court_pose":
        if config is not None and not isinstance(config, CourtPoseConfig):
            raise TypeError("Court pose detector requires CourtPoseConfig.")
        return CourtPoseLineDetector(config)
    if backend == "shuttlecourt_seg":
        if config is not None and not isinstance(config, ShuttleCourtSegConfig):
            raise TypeError("ShuttleCourt segmentation detector requires ShuttleCourtSegConfig.")
        return ShuttleCourtSegLineDetector(config)
    if backend == "monotrack":
        if config is not None and not isinstance(config, MonoTrackCourtLineConfig):
            raise TypeError("MonoTrack court detector requires MonoTrackCourtLineConfig.")
        return MonoTrackCourtLineDetector(config)
    if backend == "opencv":
        if config is not None and not isinstance(config, OpenCVCourtLineConfig):
            raise TypeError("OpenCV court detector requires OpenCVCourtLineConfig.")
        return OpenCVCourtLineDetector(config)
    raise ValueError(f"Unsupported court line detector backend: {backend}")


def predict_court_lines(
    frame: np.ndarray,
    *,
    frame_id: int = 0,
    timestamp_ms: int = 0,
    detector: CourtLineDetector | None = None,
    backend: CourtLineBackend = "shuttlecourt_seg",
    config: CourtLineConfig | None = None,
    force: bool = True,
) -> CourtLinePrediction:
    engine = detector if detector is not None else create_court_line_detector(backend=backend, config=config)
    return engine.predict(frame, frame_id, timestamp_ms, force=force)
