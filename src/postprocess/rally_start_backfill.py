from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from src.postprocess.track_output import TrackFrameOutput, copy_track_result
from src.utils.structures import TrackResult


Point = tuple[float, float]
RALLY_START_BACKFILL_SOURCE = "rally_start_quadratic_backfill"


@dataclass(slots=True)
class RallyStartBackfillResult:
    estimated_tracks: dict[int, TrackResult]
    debug: dict[str, object]

    @property
    def points(self) -> dict[int, Point]:
        return {
            frame_index: (float(track.ball_xy[0]), float(track.ball_xy[1]))
            for frame_index, track in self.estimated_tracks.items()
        }


def fit_known_rally_start(
    tracks: Sequence[TrackResult],
    *,
    active_start: int,
    active_end: int,
    width: int,
    height: int,
    max_backfill_frames: int = 3,
    min_future_points: int = 4,
    max_future_points: int = 10,
    max_validation_error_px: float = 8.0,
) -> RallyStartBackfillResult:
    """Estimate only a short, externally known rally-start gap.

    The fit window is selected by one-step holdout error and never reads ground
    truth coordinates. Callers must provide a trusted lifecycle boundary.
    """

    if not tracks:
        return _empty("empty_track_sequence")
    start = int(active_start)
    end = int(active_end)
    if start < 0 or end < start or end >= len(tracks):
        raise ValueError(
            f"invalid active range [{start}, {end}] for {len(tracks)} tracks"
        )
    if width <= 0 or height <= 0:
        raise ValueError("frame width and height must be positive")

    first_visible = next(
        (
            frame_index
            for frame_index in range(start, end + 1)
            if _point(tracks[frame_index]) is not None
        ),
        None,
    )
    if first_visible is None:
        return _empty("no_visible_anchor")
    gap_length = first_visible - start
    if gap_length <= 0 or gap_length > max(0, int(max_backfill_frames)):
        return _empty("start_gap_out_of_range")

    future_tracks: list[TrackResult] = []
    future_points: list[Point] = []
    future_limit = min(end + 1, first_visible + max(1, int(max_future_points)))
    for frame_index in range(first_visible, future_limit):
        point = _point(tracks[frame_index])
        if point is None:
            break
        future_tracks.append(tracks[frame_index])
        future_points.append(point)
    required = max(4, int(min_future_points))
    if len(future_points) < required:
        return _empty("insufficient_future_points")

    options: list[tuple[float, int]] = []
    for count in range(required, len(future_points) + 1):
        train_t = np.arange(count - 1, dtype=np.float64)
        train = np.asarray(future_points[: count - 1], dtype=np.float64)
        coefficients = [np.polyfit(train_t, train[:, axis], 2) for axis in (0, 1)]
        prediction = np.asarray(
            [np.polyval(values, count - 1) for values in coefficients],
            dtype=np.float64,
        )
        error = float(
            np.linalg.norm(prediction - np.asarray(future_points[count - 1]))
        )
        options.append((error, count))
    validation_error, selected_count = min(options)
    scale = max(width / 1280.0, height / 720.0)
    if validation_error > float(max_validation_error_px) * scale:
        return _empty(
            "future_fit_validation_failed",
            validation_error_px=validation_error,
        )

    fit_t = np.arange(selected_count, dtype=np.float64)
    fit = np.asarray(future_points[:selected_count], dtype=np.float64)
    coefficients = [np.polyfit(fit_t, fit[:, axis], 2) for axis in (0, 1)]
    reference = future_tracks[0]
    estimated: dict[int, TrackResult] = {}
    for frame_index in range(start, first_visible):
        offset = frame_index - first_visible
        point = (
            float(np.polyval(coefficients[0], offset)),
            float(np.polyval(coefficients[1], offset)),
        )
        if (
            not all(math.isfinite(value) for value in point)
            or not (0.0 <= point[0] < width and 0.0 <= point[1] < height)
        ):
            return _empty("backfill_out_of_frame")
        estimated[frame_index] = TrackResult(
            ball_xy=[point[0], point[1]],
            visible=1,
            score=float(reference.score) * 0.5,
            heatmap_shape=[int(value) for value in reference.heatmap_shape],
        )
    return RallyStartBackfillResult(
        estimated_tracks=estimated,
        debug={
            "filled_frames": sorted(estimated),
            "reason": "quadratic_future_backfill",
            "first_visible_frame": first_visible,
            "fit_frames": selected_count,
            "validation_error_px": validation_error,
        },
    )


def apply_known_rally_start(
    outputs: Sequence[TrackFrameOutput],
    *,
    active_start: int,
    active_end: int,
    width: int,
    height: int,
) -> tuple[list[TrackFrameOutput], dict[str, object]]:
    """Attach lifecycle backfill to ``estimated_track`` only."""

    fit = fit_known_rally_start(
        [output.measured_track for output in outputs],
        active_start=active_start,
        active_end=active_end,
        width=width,
        height=height,
    )
    resolved: list[TrackFrameOutput] = []
    for output in outputs:
        estimate = fit.estimated_tracks.get(output.frame_index)
        resolved.append(
            TrackFrameOutput(
                frame_index=output.frame_index,
                measured_track=copy_track_result(output.measured_track),
                measured_source=output.measured_source,
                estimated_track=(
                    copy_track_result(estimate)
                    if estimate is not None
                    else (
                        copy_track_result(output.estimated_track)
                        if output.estimated_track is not None
                        else None
                    )
                ),
                estimated_source=(
                    RALLY_START_BACKFILL_SOURCE
                    if estimate is not None
                    else output.estimated_source
                ),
                candidate_rank=output.candidate_rank,
                payload=output.payload,
                debug_record=(
                    dict(output.debug_record)
                    if isinstance(output.debug_record, dict)
                    else None
                ),
            )
        )
    return resolved, dict(fit.debug)


def _point(track: TrackResult) -> Point | None:
    if not track.visible or len(track.ball_xy) < 2:
        return None
    x, y = float(track.ball_xy[0]), float(track.ball_xy[1])
    if not (math.isfinite(x) and math.isfinite(y)) or x < 0.0 or y < 0.0:
        return None
    return x, y


def _empty(reason: str, **details: object) -> RallyStartBackfillResult:
    return RallyStartBackfillResult(
        estimated_tracks={},
        debug={"filled_frames": [], "reason": reason, **details},
    )
