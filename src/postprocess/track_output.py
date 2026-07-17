from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.utils.structures import TrackResult


def copy_track_result(track: TrackResult) -> TrackResult:
    """Copy a track result without sharing its mutable coordinate metadata."""

    return TrackResult(
        ball_xy=[float(value) for value in track.ball_xy[:2]],
        visible=int(bool(track.visible)),
        score=float(track.score),
        heatmap_shape=[int(value) for value in track.heatmap_shape],
    )


@dataclass(slots=True)
class TrackFrameOutput:
    """Keep detector measurements separate from optional trajectory estimates.

    Existing exporters and event consumers must use ``track`` (the measured
    result). Renderers may explicitly opt into ``render_track``.
    """

    frame_index: int
    measured_track: TrackResult
    measured_source: str
    estimated_track: TrackResult | None = None
    estimated_source: str | None = None
    candidate_rank: int = 0
    payload: Any = None
    debug_record: dict[str, object] | None = None

    @property
    def track(self) -> TrackResult:
        return self.measured_track

    @property
    def render_track(self) -> TrackResult:
        return self.estimated_track or self.measured_track
