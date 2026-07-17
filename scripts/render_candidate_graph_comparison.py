from __future__ import annotations

import argparse
import csv
from collections import Counter, deque
from dataclasses import asdict
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Sequence

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.track_branch import TrackBranch
from src.postprocess.candidate_graph_track import CandidateGraphConfig, FixedLagCandidateGraph
from src.postprocess.fixed_lag_track import FixedLagTrackConfig, FixedLagTrackPostProcessor
from src.postprocess.track_filter import BallTrackFilter, BallTrackFilterConfig
from src.utils.structures import TrackResult
from src.utils.video import iter_video_frame_windows, probe_video


DEFAULT_VIDEO = Path(r"F:\DataSet\BD\New\19.mp4")
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "tracknetv3_candidate_graph"
DEFAULT_WEIGHT = ROOT / "assets" / "weights" / "track" / "model_best.pt"
Point = tuple[float, float]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render current fixed-lag postprocessing beside the candidate graph.",
    )
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "BD_New_19_current_vs_graph.mp4",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "BD_New_19_comparison_summary.json",
    )
    parser.add_argument(
        "--per-frame",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "BD_New_19_comparison_per_frame.csv",
    )
    parser.add_argument(
        "--poster",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "BD_New_19_largest_difference.jpg",
    )
    parser.add_argument(
        "--candidate-cache",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "BD_New_19_raw_candidates.json",
    )
    parser.add_argument("--weight", type=Path, default=DEFAULT_WEIGHT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--pane-width", type=int, default=960)
    parser.add_argument("--trail-frames", type=int, default=16)
    parser.add_argument("--force-inference", action="store_true")
    return parser.parse_args()


def _point(track: TrackResult | None, width: int, height: int) -> Point | None:
    if track is None or not track.visible or len(track.ball_xy) < 2:
        return None
    x, y = float(track.ball_xy[0]), float(track.ball_xy[1])
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    if not (0.0 <= x < width and 0.0 <= y < height):
        return None
    return x, y


def _track_to_dict(track: TrackResult) -> dict[str, Any]:
    return {
        "ball_xy": [float(value) for value in track.ball_xy[:2]],
        "visible": int(bool(track.visible)),
        "score": float(track.score),
        "heatmap_shape": [int(value) for value in track.heatmap_shape],
    }


def _track_from_dict(item: dict[str, Any]) -> TrackResult:
    return TrackResult(
        ball_xy=[float(value) for value in item.get("ball_xy", [-1.0, -1.0])],
        visible=int(bool(item.get("visible", 0))),
        score=float(item.get("score", 0.0)),
        heatmap_shape=[int(value) for value in item.get("heatmap_shape", [])],
    )


def _cache_identity(video: Path, weight: Path, metadata: Any) -> dict[str, Any]:
    video_stat = video.stat()
    weight_stat = weight.stat()
    return {
        "video": str(video.resolve()),
        "video_size": video_stat.st_size,
        "video_mtime_ns": video_stat.st_mtime_ns,
        "weight": str(weight.resolve()),
        "weight_size": weight_stat.st_size,
        "weight_mtime_ns": weight_stat.st_mtime_ns,
        "fps": float(metadata.fps),
        "width": int(metadata.width),
        "height": int(metadata.height),
        "frame_count": int(metadata.frame_count),
        "input_size": [512, 288],
        "score_threshold": 0.35,
        "max_candidates": 5,
        "candidate_score_threshold_ratio": 0.6,
    }


def _load_candidate_cache(
    path: Path,
    identity: dict[str, Any],
) -> tuple[list[list[TrackResult]], str] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("identity") != identity:
        return None
    candidates = [
        [_track_from_dict(item) for item in frame]
        for frame in payload.get("frames", [])
    ]
    if len(candidates) != int(identity["frame_count"]):
        return None
    return candidates, str(payload.get("backend", "unknown"))


def _write_candidate_cache(
    path: Path,
    identity: dict[str, Any],
    candidates: Sequence[Sequence[TrackResult]],
    backend: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "identity": identity,
        "backend": backend,
        "frames": [[_track_to_dict(track) for track in frame] for frame in candidates],
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _infer_candidates(
    video: Path,
    weight: Path,
    *,
    device: str,
    batch_size: int,
    expected_frames: int,
) -> tuple[list[list[TrackResult]], str]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    branch = TrackBranch(
        model_weight=str(weight),
        device=device,
        input_size=(512, 288),
        score_thr=0.35,
        max_candidates=5,
        candidate_score_thr_ratio=0.6,
    )
    candidates_by_frame: list[list[TrackResult]] = []
    batch: list[tuple[int, list[np.ndarray]]] = []

    def flush_batch() -> None:
        if not batch:
            return
        decoded = branch.infer_batch_candidate_results([windows for _, windows in batch])
        for (frame_id, _), candidates in zip(batch, decoded):
            if frame_id != len(candidates_by_frame):
                raise RuntimeError(
                    f"non-contiguous inference output: frame {frame_id}, expected {len(candidates_by_frame)}"
                )
            candidates_by_frame.append(list(candidates))
        print(
            f"inference {len(candidates_by_frame)}/{expected_frames or '?'} frames",
            flush=True,
        )
        batch.clear()

    for frame_id, _, windows in iter_video_frame_windows(str(video)):
        batch.append((frame_id, windows))
        if len(batch) >= batch_size:
            flush_batch()
    flush_batch()
    if expected_frames > 0 and len(candidates_by_frame) != expected_frames:
        raise RuntimeError(
            f"inference returned {len(candidates_by_frame)} frames, expected {expected_frames}"
        )
    return candidates_by_frame, branch.backend_name


def _run_current_method(
    frames: Sequence[Sequence[TrackResult]],
    *,
    fps: float,
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    track_filter = BallTrackFilter(
        BallTrackFilterConfig(fps=fps),
        debug_enabled=True,
    )
    fixed_lag = FixedLagTrackPostProcessor(
        FixedLagTrackConfig(fps=fps, delay_ms=300),
    )
    output: dict[int, dict[str, Any]] = {}

    def consume(lagged: Any) -> None:
        frame_id = int(lagged.payload["frame_id"])
        debug = dict(lagged.debug_record or {})
        output[frame_id] = {
            "track": lagged.track,
            "source": str(lagged.source),
            "action": str(debug.get("action", "")),
            "reason": str(debug.get("reason", "")),
        }

    frame_shape = (height, width, 3)
    for frame_id, candidates in enumerate(frames):
        causal = track_filter.update_candidates(
            candidates,
            dt=1.0 / fps,
            frame_shape=frame_shape,
        )
        debug = track_filter.last_debug_record() or {}
        lagged = fixed_lag.push(
            causal,
            candidates=candidates,
            debug_record=debug,
            payload={"frame_id": frame_id},
        )
        track_filter.debug_records.clear()
        if lagged is not None:
            consume(lagged)
    for lagged in fixed_lag.flush():
        consume(lagged)

    missing = [frame_id for frame_id in range(len(frames)) if frame_id not in output]
    if missing:
        raise RuntimeError(f"current postprocessor omitted frames: {missing[:8]}")
    return [output[frame_id] for frame_id in range(len(frames))]


def _run_candidate_graph(
    frames: Sequence[Sequence[TrackResult]],
    *,
    fps: float,
) -> tuple[list[dict[str, Any]], CandidateGraphConfig]:
    config = CandidateGraphConfig(
        fps=fps,
        delay_ms=300,
        beam_width=64,
        score_center=0.72,
    )
    decisions = FixedLagCandidateGraph(config).select_sequence(frames)
    if len(decisions) != len(frames):
        raise RuntimeError(
            f"candidate graph returned {len(decisions)} frames, expected {len(frames)}"
        )
    for frame_id, decision in enumerate(decisions):
        if decision.frame_index != frame_id:
            raise RuntimeError(
                f"candidate graph frame mismatch: {decision.frame_index} != {frame_id}"
            )
    return [
        {
            "track": decision.track,
            "source": decision.source,
            "candidate_rank": int(decision.candidate_rank),
        }
        for decision in decisions
    ], config


def _nearest_candidate(
    point: Point | None,
    candidates: Sequence[TrackResult],
    width: int,
    height: int,
) -> tuple[int, float] | None:
    if point is None:
        return None
    resolved = [
        (rank, candidate_point)
        for rank, candidate in enumerate(candidates, start=1)
        if (candidate_point := _point(candidate, width, height)) is not None
    ]
    if not resolved:
        return None
    rank, candidate_point = min(resolved, key=lambda item: math.dist(point, item[1]))
    return rank, math.dist(point, candidate_point)


def _comparison_rows(
    candidates_by_frame: Sequence[Sequence[TrackResult]],
    current: Sequence[dict[str, Any]],
    graph: Sequence[dict[str, Any]],
    *,
    fps: float,
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for frame_id, candidates in enumerate(candidates_by_frame):
        raw_points = [
            point
            for candidate in candidates
            if (point := _point(candidate, width, height)) is not None
        ]
        current_track = current[frame_id]["track"]
        graph_track = graph[frame_id]["track"]
        current_point = _point(current_track, width, height)
        graph_point = _point(graph_track, width, height)
        current_match = _nearest_candidate(current_point, candidates, width, height)
        graph_match = _nearest_candidate(graph_point, candidates, width, height)
        distance = (
            math.dist(current_point, graph_point)
            if current_point is not None and graph_point is not None
            else None
        )
        visibility_disagreement = (current_point is None) != (graph_point is None)
        rows.append(
            {
                "frame_id": frame_id,
                "time_seconds": frame_id / fps,
                "candidate_count": len(raw_points),
                "raw_top1": raw_points[0] if raw_points else None,
                "current_point": current_point,
                "current_score": float(current_track.score),
                "current_source": current[frame_id]["source"],
                "current_action": current[frame_id]["action"],
                "current_reason": current[frame_id]["reason"],
                "current_nearest_candidate_rank": current_match[0] if current_match else 0,
                "current_nearest_candidate_distance_px": current_match[1] if current_match else None,
                "graph_point": graph_point,
                "graph_score": float(graph_track.score),
                "graph_source": graph[frame_id]["source"],
                "graph_candidate_rank": graph[frame_id]["candidate_rank"],
                "graph_nearest_candidate_distance_px": graph_match[1] if graph_match else None,
                "method_distance_px": distance,
                "visibility_disagreement": visibility_disagreement,
                "material_difference": visibility_disagreement or (distance is not None and distance > 20.0),
            }
        )
    return rows


def _poster_frame(rows: Sequence[dict[str, Any]]) -> int:
    position_differences = [
        row for row in rows if row["method_distance_px"] is not None
    ]
    if position_differences:
        return int(
            max(
                position_differences,
                key=lambda row: (
                    float(row["method_distance_px"]),
                    -int(row["frame_id"]),
                ),
            )["frame_id"]
        )
    visibility_differences = [row for row in rows if row["visibility_disagreement"]]
    if visibility_differences:
        return int(visibility_differences[0]["frame_id"])
    return 0


def _scaled(point: Point | None, scale_x: float, scale_y: float) -> tuple[int, int] | None:
    if point is None:
        return None
    return int(round(point[0] * scale_x)), int(round(point[1] * scale_y))


def _draw_trail(
    image: np.ndarray,
    trail: deque[tuple[int, int] | None],
    color: tuple[int, int, int],
) -> None:
    points = list(trail)
    for index in range(1, len(points)):
        first, second = points[index - 1], points[index]
        if first is None or second is None:
            continue
        brightness = 0.30 + 0.70 * index / max(1, len(points) - 1)
        line_color = tuple(int(component * brightness) for component in color)
        cv2.line(image, first, second, line_color, 2, cv2.LINE_AA)


def _draw_candidates(
    image: np.ndarray,
    candidates: Sequence[TrackResult],
    *,
    width: int,
    height: int,
    scale_x: float,
    scale_y: float,
) -> None:
    for rank, candidate in enumerate(candidates, start=1):
        point = _scaled(_point(candidate, width, height), scale_x, scale_y)
        if point is None:
            continue
        cv2.circle(image, point, 5, (25, 25, 25), -1, cv2.LINE_AA)
        cv2.circle(image, point, 4, (65, 225, 255), 1, cv2.LINE_AA)
        cv2.putText(
            image,
            str(rank),
            (point[0] + 6, point[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (65, 225, 255),
            1,
            cv2.LINE_AA,
        )


def _fit_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)] + "..."


def _draw_panel(
    frame: np.ndarray,
    *,
    title: str,
    frame_id: int,
    fps: float,
    candidates: Sequence[TrackResult],
    point: tuple[int, int] | None,
    score: float,
    color: tuple[int, int, int],
    trail: deque[tuple[int, int] | None],
    status_detail: str,
    footer_detail: str,
    source_width: int,
    source_height: int,
    scale_x: float,
    scale_y: float,
    material_difference: bool,
) -> np.ndarray:
    panel = frame.copy()
    overlay = panel.copy()
    cv2.rectangle(overlay, (0, 0), (panel.shape[1], 60), (14, 17, 21), -1)
    cv2.rectangle(
        overlay,
        (0, panel.shape[0] - 48),
        (panel.shape[1], panel.shape[0]),
        (14, 17, 21),
        -1,
    )
    cv2.addWeighted(overlay, 0.80, panel, 0.20, 0.0, panel)
    _draw_candidates(
        panel,
        candidates,
        width=source_width,
        height=source_height,
        scale_x=scale_x,
        scale_y=scale_y,
    )
    _draw_trail(panel, trail, color)
    if point is not None:
        cv2.circle(panel, point, 6, (20, 20, 20), -1, cv2.LINE_AA)
        cv2.circle(panel, point, 6, color, 2, cv2.LINE_AA)
        cv2.circle(panel, point, 10, color, 2, cv2.LINE_AA)

    visible_text = "VISIBLE" if point is not None else "MISSING"
    visible_color = color if point is not None else (185, 185, 185)
    cv2.putText(panel, title, (18, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(
        panel,
        f"FRAME {frame_id:03d}   {frame_id / fps:05.2f}s",
        (18, 51),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (205, 211, 218),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        f"{visible_text}  score {score:.3f}  {_fit_text(status_detail, 55)}",
        (18, panel.shape[0] - 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        visible_color,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        _fit_text(footer_detail, 92),
        (18, panel.shape[0] - 9),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (205, 211, 218),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        f"RAW CANDIDATES {len(candidates)}",
        (panel.shape[1] - 190, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (65, 225, 255),
        1,
        cv2.LINE_AA,
    )
    if material_difference:
        cv2.rectangle(panel, (2, 2), (panel.shape[1] - 3, panel.shape[0] - 3), (45, 165, 255), 3)
    return panel


def _render(
    video: Path,
    output: Path,
    poster: Path,
    poster_frame: int,
    candidates_by_frame: Sequence[Sequence[TrackResult]],
    current: Sequence[dict[str, Any]],
    graph: Sequence[dict[str, Any]],
    rows: Sequence[dict[str, Any]],
    *,
    fps: float,
    source_width: int,
    source_height: int,
    pane_width: int,
    trail_frames: int,
) -> dict[str, Any]:
    if pane_width < 320:
        raise ValueError("pane_width must be at least 320")
    pane_height = int(round(source_height * pane_width / source_width))
    output_size = (pane_width * 2, pane_height)
    scale_x = pane_width / source_width
    scale_y = pane_height / source_height
    output.parent.mkdir(parents=True, exist_ok=True)
    poster.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        output_size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not create output video: {output}")
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        writer.release()
        raise RuntimeError(f"could not reopen source video: {video}")

    current_trail: deque[tuple[int, int] | None] = deque(maxlen=max(1, trail_frames))
    graph_trail: deque[tuple[int, int] | None] = deque(maxlen=max(1, trail_frames))
    written = 0
    try:
        while written < len(rows):
            ok, frame = capture.read()
            if not ok:
                break
            resized = cv2.resize(frame, (pane_width, pane_height), interpolation=cv2.INTER_AREA)
            row = rows[written]
            current_point = _scaled(row["current_point"], scale_x, scale_y)
            graph_point = _scaled(row["graph_point"], scale_x, scale_y)
            current_trail.append(current_point)
            graph_trail.append(graph_point)
            current_status = f"source {current[written]['source']}"
            current_footer = f"decision {current[written]['action']} / {current[written]['reason']}"
            graph_rank = int(graph[written]["candidate_rank"])
            graph_status = f"raw rank {graph_rank}" if graph_rank > 0 else "null state"
            graph_footer = (
                f"beam 64 / score center 0.72 / delay 9 frames / source {graph[written]['source']}"
            )
            left = _draw_panel(
                resized,
                title="CURRENT  Causal Filter + Fixed Lag 300ms",
                frame_id=written,
                fps=fps,
                candidates=candidates_by_frame[written],
                point=current_point,
                score=float(row["current_score"]),
                color=(75, 85, 245),
                trail=current_trail,
                status_detail=current_status,
                footer_detail=current_footer,
                source_width=source_width,
                source_height=source_height,
                scale_x=scale_x,
                scale_y=scale_y,
                material_difference=bool(row["material_difference"]),
            )
            right = _draw_panel(
                resized,
                title="NEW  Fixed-Lag Candidate Graph 300ms",
                frame_id=written,
                fps=fps,
                candidates=candidates_by_frame[written],
                point=graph_point,
                score=float(row["graph_score"]),
                color=(245, 210, 60),
                trail=graph_trail,
                status_detail=graph_status,
                footer_detail=graph_footer,
                source_width=source_width,
                source_height=source_height,
                scale_x=scale_x,
                scale_y=scale_y,
                material_difference=bool(row["material_difference"]),
            )
            composite = np.hstack((left, right))
            writer.write(composite)
            if written == poster_frame:
                if not cv2.imwrite(str(poster), composite):
                    raise RuntimeError(f"could not write poster: {poster}")
            written += 1
    finally:
        capture.release()
        writer.release()
    if written != len(rows):
        raise RuntimeError(f"rendered {written} frames, expected {len(rows)}")
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError("render output is empty")
    return {
        "frames": written,
        "fps": fps,
        "width": output_size[0],
        "height": output_size[1],
        "duration_seconds": written / fps,
    }


def _write_per_frame(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "frame_id",
        "time_seconds",
        "candidate_count",
        "raw_top1_x",
        "raw_top1_y",
        "current_visible",
        "current_x",
        "current_y",
        "current_score",
        "current_source",
        "current_action",
        "current_reason",
        "current_nearest_candidate_rank",
        "current_nearest_candidate_distance_px",
        "graph_visible",
        "graph_x",
        "graph_y",
        "graph_score",
        "graph_source",
        "graph_candidate_rank",
        "graph_nearest_candidate_distance_px",
        "method_distance_px",
        "visibility_disagreement",
        "material_difference",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            raw = row["raw_top1"]
            current = row["current_point"]
            graph = row["graph_point"]
            writer.writerow(
                {
                    "frame_id": row["frame_id"],
                    "time_seconds": row["time_seconds"],
                    "candidate_count": row["candidate_count"],
                    "raw_top1_x": raw[0] if raw else "",
                    "raw_top1_y": raw[1] if raw else "",
                    "current_visible": int(current is not None),
                    "current_x": current[0] if current else "",
                    "current_y": current[1] if current else "",
                    "current_score": row["current_score"],
                    "current_source": row["current_source"],
                    "current_action": row["current_action"],
                    "current_reason": row["current_reason"],
                    "current_nearest_candidate_rank": row["current_nearest_candidate_rank"],
                    "current_nearest_candidate_distance_px": row[
                        "current_nearest_candidate_distance_px"
                    ],
                    "graph_visible": int(graph is not None),
                    "graph_x": graph[0] if graph else "",
                    "graph_y": graph[1] if graph else "",
                    "graph_score": row["graph_score"],
                    "graph_source": row["graph_source"],
                    "graph_candidate_rank": row["graph_candidate_rank"],
                    "graph_nearest_candidate_distance_px": row[
                        "graph_nearest_candidate_distance_px"
                    ],
                    "method_distance_px": row["method_distance_px"],
                    "visibility_disagreement": int(row["visibility_disagreement"]),
                    "material_difference": int(row["material_difference"]),
                }
            )


def _method_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    distances = sorted(
        float(row["method_distance_px"])
        for row in rows
        if row["method_distance_px"] is not None
    )
    current_visible = sum(row["current_point"] is not None for row in rows)
    graph_visible = sum(row["graph_point"] is not None for row in rows)
    current_only = sum(
        row["current_point"] is not None and row["graph_point"] is None
        for row in rows
    )
    graph_only = sum(
        row["current_point"] is None and row["graph_point"] is not None
        for row in rows
    )
    return {
        "raw_top1_visible_frames": sum(row["raw_top1"] is not None for row in rows),
        "current_visible_frames": current_visible,
        "graph_visible_frames": graph_visible,
        "both_visible_frames": len(distances),
        "both_missing_frames": sum(
            row["current_point"] is None and row["graph_point"] is None
            for row in rows
        ),
        "current_only_visible_frames": current_only,
        "graph_only_visible_frames": graph_only,
        "visibility_disagreement_frames": current_only + graph_only,
        "material_difference_frames": sum(row["material_difference"] for row in rows),
        "both_visible_distance_px": {
            "mean": statistics.fmean(distances) if distances else None,
            "p50": distances[int(0.50 * (len(distances) - 1))] if distances else None,
            "p90": distances[int(0.90 * (len(distances) - 1))] if distances else None,
            "p95": distances[int(0.95 * (len(distances) - 1))] if distances else None,
            "max": max(distances) if distances else None,
        },
        "current_sources": dict(Counter(row["current_source"] for row in rows)),
        "current_actions": dict(Counter(row["current_action"] for row in rows)),
        "graph_sources": dict(Counter(row["graph_source"] for row in rows)),
        "graph_candidate_ranks": dict(
            Counter(str(row["graph_candidate_rank"]) for row in rows)
        ),
        "graph_measured_coordinate_mismatch": sum(
            row["graph_point"] is not None
            and (
                row["graph_nearest_candidate_distance_px"] is None
                or float(row["graph_nearest_candidate_distance_px"]) > 1e-9
            )
            for row in rows
        ),
    }


def _verify_video(path: Path, expected: dict[str, Any], sample_frames: Sequence[int]) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open rendered video: {path}")
    metadata = {
        "frames_reported": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    decoded = 0
    sample_stats: dict[str, dict[str, float]] = {}
    targets = set(int(value) for value in sample_frames)
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if decoded in targets:
            sample_stats[str(decoded)] = {
                "mean": float(frame.mean()),
                "stddev": float(frame.std()),
            }
        decoded += 1
    capture.release()
    metadata["frames_decoded"] = decoded
    metadata["sample_pixels"] = sample_stats
    if decoded != int(expected["frames"]):
        raise RuntimeError(f"decoded {decoded} frames, expected {expected['frames']}")
    for key in ("width", "height"):
        if metadata[key] != int(expected[key]):
            raise RuntimeError(f"rendered {key} mismatch: {metadata[key]} != {expected[key]}")
    if abs(metadata["fps"] - float(expected["fps"])) > 0.01:
        raise RuntimeError(f"rendered fps mismatch: {metadata['fps']} != {expected['fps']}")
    if len(sample_stats) != len(targets) or any(item["stddev"] <= 0.0 for item in sample_stats.values()):
        raise RuntimeError("rendered sample frame validation failed")
    return metadata


def main() -> None:
    args = _parse_args()
    started = time.perf_counter()
    video = args.video.resolve()
    weight = args.weight.resolve()
    if not video.is_file():
        raise FileNotFoundError(f"video not found: {video}")
    if not weight.is_file():
        raise FileNotFoundError(f"TrackNetV3 weight not found: {weight}")
    metadata = probe_video(str(video))
    identity = _cache_identity(video, weight, metadata)
    cached = None if args.force_inference else _load_candidate_cache(args.candidate_cache, identity)
    if cached is None:
        candidates_by_frame, backend = _infer_candidates(
            video,
            weight,
            device=args.device,
            batch_size=int(args.batch_size),
            expected_frames=metadata.frame_count,
        )
        _write_candidate_cache(args.candidate_cache, identity, candidates_by_frame, backend)
        cache_reused = False
    else:
        candidates_by_frame, backend = cached
        cache_reused = True
        print(f"reused candidate cache: {args.candidate_cache.resolve()}", flush=True)

    current = _run_current_method(
        candidates_by_frame,
        fps=metadata.fps,
        width=metadata.width,
        height=metadata.height,
    )
    graph, graph_config = _run_candidate_graph(candidates_by_frame, fps=metadata.fps)
    rows = _comparison_rows(
        candidates_by_frame,
        current,
        graph,
        fps=metadata.fps,
        width=metadata.width,
        height=metadata.height,
    )
    poster_frame = _poster_frame(rows)
    _write_per_frame(args.per_frame, rows)
    render_meta = _render(
        video,
        args.output,
        args.poster,
        poster_frame,
        candidates_by_frame,
        current,
        graph,
        rows,
        fps=metadata.fps,
        source_width=metadata.width,
        source_height=metadata.height,
        pane_width=int(args.pane_width),
        trail_frames=int(args.trail_frames),
    )
    verification = _verify_video(
        args.output,
        render_meta,
        sample_frames=(0, poster_frame, len(rows) // 2, len(rows) - 1),
    )
    summary = {
        "source": str(video),
        "output": str(args.output.resolve()),
        "poster": str(args.poster.resolve()),
        "per_frame": str(args.per_frame.resolve()),
        "candidate_cache": str(args.candidate_cache.resolve()),
        "candidate_cache_reused": cache_reused,
        "source_metadata": asdict(metadata),
        "render_metadata": render_meta,
        "backend": backend,
        "inference": {
            "weight": str(weight),
            "input_size": [512, 288],
            "score_threshold": 0.35,
            "max_candidates": 5,
            "candidate_score_threshold_ratio": 0.6,
        },
        "current_method": {
            "causal": asdict(BallTrackFilterConfig(fps=metadata.fps)),
            "fixed_lag": asdict(FixedLagTrackConfig(fps=metadata.fps, delay_ms=300)),
        },
        "candidate_graph": asdict(graph_config),
        "comparison": _method_summary(rows),
        "poster_frame": poster_frame,
        "poster_time_seconds": poster_frame / metadata.fps,
        "verification": verification,
        "elapsed_seconds": time.perf_counter() - started,
        "ground_truth_available": False,
        "interpretation_note": (
            "This artifact compares behavior on identical raw candidates. Without paired ground truth, "
            "visibility or disagreement counts do not determine which method is correct."
        ),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
