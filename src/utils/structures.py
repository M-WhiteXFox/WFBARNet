from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class PersonPoseResult:
    person_id: int
    bbox: list[float]
    keypoints: list[list[float]]
    scores: list[float]
    person_score: float


@dataclass
class TrackResult:
    ball_xy: list[float]
    visible: int
    score: float
    heatmap_shape: list[int] = field(default_factory=list)


@dataclass
class FrameResult:
    frame_id: int
    pose: list[PersonPoseResult]
    track: TrackResult

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data
