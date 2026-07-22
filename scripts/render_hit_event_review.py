from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import deque
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.postprocess.trajectory_events import (
    TrajectoryEventCandidateGenerator,
    TrajectoryEventDetectorConfig,
)


DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "hit_event_review"
USER_CONFIRMED_MISSED_FRAMES = (34, 60, 100, 117, 130, 255)
USER_REJECTED_RAW_HIT_FRAMES = (189, 241, 260, 340, 357)
MANUAL_CONFIRMED_BALL_POINTS = {255: [1038.0, 551.0]}
ORIGIN_COLORS = {
    "existing_confirmed": (55, 70, 245),
    "user_confirmed_missed": (40, 155, 255),
    "raw_kinematic_unconfirmed": (0, 220, 255),
    "model_candidate": (55, 70, 245),
    "evaluation_match": (70, 210, 105),
    "evaluation_false_positive": (55, 70, 245),
    "evaluation_false_negative": (40, 165, 255),
}
LANDING_COLOR = (75, 220, 95)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render detected badminton hits for manual review.")
    parser.add_argument("--video", type=Path, default=ROOT / "Dataset" / "10-1.mp4")
    parser.add_argument(
        "--frame-log",
        type=Path,
        default=ROOT / "outputs" / "pyqt_debug" / "10-1_20260717_220627_frame_log.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "10-1_hit_event_review.mp4",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "10-1_hit_event_manifest.csv",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "10-1_hit_event_review_summary.json",
    )
    parser.add_argument(
        "--contact-sheet",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "10-1_hit_event_contact_sheet.jpg",
    )
    parser.add_argument("--slowdown", type=int, default=2)
    parser.add_argument("--hold-seconds", type=float, default=0.8)
    parser.add_argument(
        "--high-recall",
        action="store_true",
        help="Review raw measured-trajectory hit candidates plus confirmed contact frames.",
    )
    parser.add_argument(
        "--include-landings",
        action="store_true",
        help="Include detected landing events alongside hit events.",
    )
    parser.add_argument(
        "--ground-truth-csv",
        type=Path,
        help="Evaluate detected hits against datatool Hit labels and render matches/errors.",
    )
    parser.add_argument(
        "--match-tolerance",
        type=int,
        default=3,
        help="Maximum frame offset used by the rendered one-to-one evaluation (default: 3).",
    )
    return parser.parse_args()


def _load_ground_truth(path: Path) -> dict[int, dict[str, float]]:
    hits: dict[int, dict[str, float]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"Frame", "Hit", "x", "y"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"ground-truth CSV is missing fields: {sorted(missing)}")
        for row in reader:
            if int(row["Hit"]) != 1:
                continue
            frame_id = int(row["Frame"])
            hits[frame_id] = {"x": float(row["x"]), "y": float(row["y"])}
    if not hits:
        raise ValueError(f"ground-truth CSV contains no Hit=1 rows: {path}")
    return hits


def _match_hit_frames(
    predicted_frames: list[int],
    ground_truth_frames: list[int],
    tolerance: int,
) -> list[tuple[int, int, int]]:
    """Maximize one-to-one matches, then minimize total absolute frame error."""
    if tolerance < 0:
        raise ValueError("match tolerance cannot be negative")
    predicted = tuple(sorted(set(int(frame) for frame in predicted_frames)))
    ground_truth = tuple(sorted(set(int(frame) for frame in ground_truth_frames)))

    @lru_cache(maxsize=None)
    def solve(
        predicted_index: int,
        ground_truth_index: int,
    ) -> tuple[int, int, tuple[tuple[int, int, int], ...]]:
        if predicted_index >= len(predicted) or ground_truth_index >= len(ground_truth):
            return 0, 0, ()
        candidates = [
            solve(predicted_index + 1, ground_truth_index),
            solve(predicted_index, ground_truth_index + 1),
        ]
        predicted_frame = predicted[predicted_index]
        ground_truth_frame = ground_truth[ground_truth_index]
        offset = predicted_frame - ground_truth_frame
        if abs(offset) <= tolerance:
            count, error, pairs = solve(predicted_index + 1, ground_truth_index + 1)
            candidates.append(
                (
                    count + 1,
                    error + abs(offset),
                    ((predicted_frame, ground_truth_frame, offset), *pairs),
                )
            )
        return min(candidates, key=lambda result: (-result[0], result[1], result[2]))

    return list(solve(0, 0)[2])


def _evaluation_metrics(
    predicted_frames: list[int],
    ground_truth_frames: list[int],
    tolerances: tuple[int, ...] = (0, 1, 2, 3, 5),
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for tolerance in tolerances:
        matches = _match_hit_frames(predicted_frames, ground_truth_frames, tolerance)
        true_positives = len(matches)
        false_positives = len(predicted_frames) - true_positives
        false_negatives = len(ground_truth_frames) - true_positives
        precision = true_positives / len(predicted_frames) if predicted_frames else 0.0
        recall = true_positives / len(ground_truth_frames) if ground_truth_frames else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if true_positives else 0.0
        absolute_errors = [abs(offset) for _, _, offset in matches]
        matched_predictions = {predicted for predicted, _, _ in matches}
        matched_ground_truth = {ground_truth for _, ground_truth, _ in matches}
        ordered_errors = sorted(absolute_errors)
        if ordered_errors:
            middle = len(ordered_errors) // 2
            median_error = (
                float(ordered_errors[middle])
                if len(ordered_errors) % 2
                else (ordered_errors[middle - 1] + ordered_errors[middle]) / 2.0
            )
        else:
            median_error = None
        results[str(tolerance)] = {
            "tolerance_frames": tolerance,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "matches": [
                {
                    "prediction_frame": predicted,
                    "ground_truth_frame": ground_truth,
                    "offset_frames": offset,
                }
                for predicted, ground_truth, offset in matches
            ],
            "unmatched_prediction_frames": [
                frame for frame in predicted_frames if frame not in matched_predictions
            ],
            "unmatched_ground_truth_frames": [
                frame for frame in ground_truth_frames if frame not in matched_ground_truth
            ],
            "mean_absolute_error_frames": (
                sum(absolute_errors) / len(absolute_errors) if absolute_errors else None
            ),
            "median_absolute_error_frames": median_error,
            "max_absolute_error_frames": max(absolute_errors) if absolute_errors else None,
        }
    return results


def _build_evaluation_events(
    detected_hits: list[dict[str, Any]],
    ground_truth_hits: dict[int, dict[str, float]],
    *,
    tolerance: int,
    source_width: int,
    source_height: int,
    fps: float,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    predicted_frames = [int(event["frame_id"]) for event in detected_hits]
    ground_truth_frames = sorted(ground_truth_hits)
    metrics = _evaluation_metrics(predicted_frames, ground_truth_frames)
    matches = _match_hit_frames(predicted_frames, ground_truth_frames, tolerance)
    prediction_events = {int(event["frame_id"]): event for event in detected_hits}
    matched_predictions = {predicted for predicted, _, _ in matches}
    matched_ground_truth = {ground_truth: (predicted, offset) for predicted, ground_truth, offset in matches}
    events: list[dict[str, Any]] = []

    for ground_truth_frame in ground_truth_frames:
        annotation = ground_truth_hits[ground_truth_frame]
        point = [annotation["x"] * source_width, annotation["y"] * source_height]
        matched = matched_ground_truth.get(ground_truth_frame)
        if matched is None:
            events.append(
                {
                    "event_type": "hit",
                    "frame_id": ground_truth_frame,
                    "timestamp_ms": int(round(ground_truth_frame * 1000.0 / fps)),
                    "ball_xy": point,
                    "rule": "ground_truth_only",
                    "confidence": 1.0,
                    "origin": "evaluation_false_negative",
                    "ground_truth_frame": ground_truth_frame,
                    "prediction_frame": None,
                    "kinematic_frame": None,
                    "frame_offset": None,
                    "verdict": "false_negative",
                    "match_tolerance": tolerance,
                }
            )
            continue
        predicted_frame, offset = matched
        prediction = prediction_events[predicted_frame]
        events.append(
            {
                **prediction,
                "frame_id": ground_truth_frame,
                "timestamp_ms": int(round(ground_truth_frame * 1000.0 / fps)),
                "ball_xy": point,
                "origin": "evaluation_match",
                "ground_truth_frame": ground_truth_frame,
                "prediction_frame": predicted_frame,
                "kinematic_frame": predicted_frame,
                "frame_offset": offset,
                "verdict": "true_positive",
                "match_tolerance": tolerance,
            }
        )

    for predicted_frame in sorted(set(predicted_frames).difference(matched_predictions)):
        prediction = prediction_events[predicted_frame]
        events.append(
            {
                **prediction,
                "origin": "evaluation_false_positive",
                "ground_truth_frame": None,
                "prediction_frame": predicted_frame,
                "kinematic_frame": predicted_frame,
                "frame_offset": None,
                "verdict": "false_positive",
                "match_tolerance": tolerance,
            }
        )
    events.sort(key=lambda event: (int(event["frame_id"]), str(event["origin"])))
    return events, metrics


def _load_log(path: Path) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    rows: dict[int, dict[str, Any]] = {}
    events: dict[tuple[str, int], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            row = json.loads(raw_line)
            frame_id = int(row["frame_id"])
            rows[frame_id] = row
            for field in ("trajectory_event", "hit_event", "landing_event"):
                raw_event = row.get(field)
                if not isinstance(raw_event, dict):
                    continue
                event_type = str(raw_event.get("event_type", ""))
                if event_type not in {"hit", "landing"}:
                    continue
                event_frame = int(raw_event["frame_id"])
                events[(event_type, event_frame)] = {
                    "event_type": event_type,
                    "frame_id": event_frame,
                    "timestamp_ms": int(raw_event.get("timestamp_ms", 0)),
                    "ball_xy": [
                        float(value) for value in raw_event.get("ball_xy", [-1.0, -1.0])[:2]
                    ],
                    "rule": str(raw_event.get("rule", "")),
                    "confidence": float(raw_event.get("confidence", 0.0)),
                    "origin": "model_candidate",
                }
    ordered_events = sorted(
        events.values(),
        key=lambda event: (int(event["frame_id"]), str(event["event_type"])),
    )
    if not ordered_events:
        raise RuntimeError(f"no hit or landing events found in {path}")
    return rows, ordered_events


def _measured_ball(row: dict[str, Any] | None) -> tuple[list[float], int, float]:
    ball = row.get("ball") if row is not None else None
    if not isinstance(ball, dict):
        return [-1.0, -1.0], 0, 0.0
    xy = ball.get("xy", [-1.0, -1.0])
    if not isinstance(xy, (list, tuple)) or len(xy) < 2:
        xy = [-1.0, -1.0]
    return [float(xy[0]), float(xy[1])], int(ball.get("visible", 0)), float(ball.get("score", 0.0))


def _build_high_recall_events(
    rows_by_frame: dict[int, dict[str, Any]],
    detected_events: list[dict[str, Any]],
    *,
    fps: float,
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    ordered_rows = [rows_by_frame[frame_id] for frame_id in sorted(rows_by_frame)]
    x: list[float] = []
    y: list[float] = []
    visibility: list[int] = []
    scores: list[float] = []
    for row in ordered_rows:
        point, visible, score = _measured_ball(row)
        x.append(point[0])
        y.append(point[1])
        visibility.append(visible)
        scores.append(score)

    generator = TrajectoryEventCandidateGenerator(
        TrajectoryEventDetectorConfig(fps=fps, min_speed_at_hit=6.0)
    )
    raw_candidates = [
        candidate
        for candidate in generator.generate(
            x,
            y,
            visibility,
            img_width=width,
            img_height=height,
            include_trajectory_end=False,
        )
        if candidate.get("event_type") == "hit"
    ]
    raw_events: list[dict[str, Any]] = []
    for candidate in raw_candidates:
        local_index = int(candidate["frame"])
        source_frame = int(ordered_rows[local_index]["frame_id"])
        raw_events.append(
            {
                "frame_id": source_frame,
                "event_type": "hit",
                "kinematic_frame": source_frame,
                "timestamp_ms": int(round(source_frame * 1000.0 / fps)),
                "ball_xy": [float(candidate["x"]), float(candidate["y"])],
                "rule": str(candidate["rule"]),
                "confidence": float(candidate["confidence"]),
                "track_score": scores[local_index],
                "kinematic_score": scores[local_index],
                "origin": "raw_kinematic_unconfirmed",
            }
        )

    confirmed: dict[int, tuple[str, dict[str, Any] | None]] = {
        int(event["frame_id"]): ("existing_confirmed", event) for event in detected_events
    }
    confirmed.update(
        {frame_id: ("user_confirmed_missed", None) for frame_id in USER_CONFIRMED_MISSED_FRAMES}
    )
    matched_raw_ids: set[int] = set()
    events: list[dict[str, Any]] = []
    for frame_id, (origin, detected) in sorted(confirmed.items()):
        point, visible, track_score = _measured_ball(rows_by_frame.get(frame_id))
        nearby = [item for item in raw_events if abs(int(item["frame_id"]) - frame_id) <= 2]
        if nearby:
            nearest = min(nearby, key=lambda item: abs(int(item["frame_id"]) - frame_id))
            matched_raw_ids.add(id(nearest))
            if not visible:
                point = list(nearest["ball_xy"])
            event = {
                **nearest,
                "frame_id": frame_id,
                "timestamp_ms": int(round(frame_id * 1000.0 / fps)),
                "ball_xy": point,
                "track_score": track_score,
                "origin": origin,
            }
        else:
            manual_point = MANUAL_CONFIRMED_BALL_POINTS.get(frame_id)
            if manual_point is not None:
                point = list(manual_point)
            event = {
                "frame_id": frame_id,
                "kinematic_frame": None,
                "timestamp_ms": int(round(frame_id * 1000.0 / fps)),
                "ball_xy": point,
                "rule": "manual_contact",
                "confidence": 1.0,
                "track_score": track_score,
                "kinematic_score": 0.0,
                "origin": origin,
                "ball_point_source": "manual_render_estimate" if manual_point is not None else "missing",
            }
        if detected is not None:
            event["confidence"] = float(detected["confidence"])
        events.append(event)
    events.extend(
        event
        for event in raw_events
        if id(event) not in matched_raw_ids
        and int(event["frame_id"]) not in USER_REJECTED_RAW_HIT_FRAMES
    )
    events.sort(key=lambda item: int(item["frame_id"]))
    return events


def _format_time(timestamp_ms: int) -> str:
    total_ms = max(0, int(timestamp_ms))
    minutes, remainder = divmod(total_ms, 60_000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def _scaled_point(
    point: list[float] | tuple[float, float],
    scale_x: float,
    scale_y: float,
) -> tuple[int, int] | None:
    if len(point) < 2:
        return None
    x, y = float(point[0]), float(point[1])
    if x < 0.0 or y < 0.0 or not np.isfinite([x, y]).all():
        return None
    return int(round(x * scale_x)), int(round(y * scale_y))


def _draw_trail(image: np.ndarray, trail: deque[tuple[int, int] | None]) -> None:
    points = list(trail)
    for index in range(1, len(points)):
        first = points[index - 1]
        second = points[index]
        if first is None or second is None:
            continue
        strength = 0.25 + 0.75 * index / max(1, len(points) - 1)
        color = (int(245 * strength), int(205 * strength), int(55 * strength))
        cv2.line(image, first, second, color, 2, cv2.LINE_AA)


def _draw_timeline(
    image: np.ndarray,
    events: list[dict[str, Any]],
    frame_id: int,
    total_frames: int,
    panel_top: int,
) -> None:
    left, right = 28, image.shape[1] - 28
    axis_y = panel_top + 92
    cv2.line(image, (left, axis_y), (right, axis_y), (105, 112, 122), 2, cv2.LINE_AA)
    for review_index, event in enumerate(events, start=1):
        event_x = left + int(round((right - left) * int(event["frame_id"]) / max(1, total_frames - 1)))
        color = _event_color(event)
        cv2.line(image, (event_x, axis_y - 12), (event_x, axis_y + 12), color, 3, cv2.LINE_AA)
        cv2.putText(
            image,
            f"{review_index:02d}",
            (event_x - 10, axis_y - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (225, 229, 235),
            1,
            cv2.LINE_AA,
        )
    current_x = left + int(round((right - left) * frame_id / max(1, total_frames - 1)))
    cv2.circle(image, (current_x, axis_y), 6, (80, 225, 255), -1, cv2.LINE_AA)


def _event_color(event: dict[str, Any]) -> tuple[int, int, int]:
    if event.get("event_type") == "landing":
        return LANDING_COLOR
    return ORIGIN_COLORS.get(str(event.get("origin", "model_candidate")), (55, 70, 245))


def _draw_frame(
    source: np.ndarray,
    row: dict[str, Any] | None,
    event: dict[str, Any] | None,
    near_event: tuple[int, dict[str, Any]] | None,
    events: list[dict[str, Any]],
    frame_id: int,
    total_frames: int,
    source_fps: float,
    trail: deque[tuple[int, int] | None],
    output_size: tuple[int, int],
) -> np.ndarray:
    output_width, output_height = output_size
    panel_height = 120
    image_height = output_height - panel_height
    resized = cv2.resize(source, (output_width, image_height), interpolation=cv2.INTER_AREA)
    image = cv2.copyMakeBorder(
        resized,
        0,
        panel_height,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(14, 17, 21),
    )
    scale_x = output_width / source.shape[1]
    scale_y = image_height / source.shape[0]
    current_point = None
    if row is not None:
        ball = row.get("ball")
        if isinstance(ball, dict) and int(ball.get("visible", 0)) == 1:
            current_point = _scaled_point(ball.get("xy", [-1.0, -1.0]), scale_x, scale_y)
    trail.append(current_point)
    _draw_trail(image, trail)
    if current_point is not None:
        cv2.circle(image, current_point, 7, (20, 20, 20), -1, cv2.LINE_AA)
        cv2.circle(image, current_point, 6, (245, 205, 55), -1, cv2.LINE_AA)
        cv2.circle(image, current_point, 10, (245, 205, 55), 2, cv2.LINE_AA)

    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (output_width, 58), (12, 15, 19), -1)
    cv2.addWeighted(overlay, 0.78, image, 0.22, 0.0, image)
    timestamp_ms = int(round(frame_id * 1000.0 / source_fps))
    evaluation_mode = any(str(event.get("origin", "")).startswith("evaluation_") for event in events)
    if evaluation_mode:
        header = "HIT EVENT EVALUATION  |  GREEN MATCH / RED FP / ORANGE FN"
    elif any(event.get("event_type") == "landing" for event in events):
        header = "TRAJECTORY EVENT REVIEW  |  HIT + LANDING"
    elif any(event.get("origin") == "raw_kinematic_unconfirmed" for event in events):
        header = "HIT EVENT REVIEW  |  HIGH RECALL"
    else:
        header = "HIT EVENT REVIEW  |  MODEL PREDICTIONS"
    cv2.putText(
        image,
        header,
        (22, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (242, 244, 247),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        f"SOURCE FRAME {frame_id:03d}    TIME {_format_time(timestamp_ms)}",
        (output_width - 430, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (210, 216, 224),
        1,
        cv2.LINE_AA,
    )

    panel_top = image_height
    if event is not None:
        review_index = next(index for index, item in enumerate(events, start=1) if item is event)
        event_type = str(event.get("event_type", "hit"))
        origin = str(event.get("origin", "model_candidate"))
        event_color = _event_color(event)
        kinematic_frame = event.get("kinematic_frame")
        kinematic_label = "---" if kinematic_frame is None else f"{int(kinematic_frame):03d}"
        event_point = _scaled_point(event["ball_xy"], scale_x, scale_y)
        if event_point is not None:
            cv2.circle(image, event_point, 19, event_color, 4, cv2.LINE_AA)
            cv2.line(
                image,
                (event_point[0] - 24, event_point[1]),
                (event_point[0] + 24, event_point[1]),
                event_color,
                3,
                cv2.LINE_AA,
            )
            if event_type == "landing":
                radius = 30
                diamond = np.asarray(
                    [
                        [event_point[0], event_point[1] - radius],
                        [event_point[0] + radius, event_point[1]],
                        [event_point[0], event_point[1] + radius],
                        [event_point[0] - radius, event_point[1]],
                    ],
                    dtype=np.int32,
                )
                cv2.polylines(image, [diamond], True, event_color, 4, cv2.LINE_AA)
            cv2.line(
                image,
                (event_point[0], event_point[1] - 24),
                (event_point[0], event_point[1] + 24),
                event_color,
                3,
                cv2.LINE_AA,
            )
        cv2.rectangle(image, (3, 3), (output_width - 4, image_height - 4), event_color, 5)
        if origin == "evaluation_match":
            event_label = "MATCHED HIT"
        elif origin == "evaluation_false_positive":
            event_label = "FALSE POSITIVE"
        elif origin == "evaluation_false_negative":
            event_label = "MISSED HIT"
        else:
            event_label = f"{event_type.upper()} CANDIDATE"
        cv2.putText(
            image,
            f"{event_label} {review_index:02d}",
            (22, 94),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.92,
            event_color,
            3,
            cv2.LINE_AA,
        )
        if origin.startswith("evaluation_"):
            prediction_frame = event.get("prediction_frame")
            prediction_label = "---" if prediction_frame is None else f"{int(prediction_frame):03d}"
            ground_truth_frame = event.get("ground_truth_frame")
            ground_truth_label = "---" if ground_truth_frame is None else f"{int(ground_truth_frame):03d}"
            offset = event.get("frame_offset")
            offset_label = "---" if offset is None else f"{int(offset):+d}"
            status = (
                f"{event_label}    GT {ground_truth_label}    PRED {prediction_label}    "
                f"OFFSET {offset_label} FRAME(S)    TOL +/-{int(event['match_tolerance'])}"
            )
        elif event_type == "landing":
            status = (
                f"LANDING {review_index:02d}/{len(events):02d}    FRAME {frame_id:03d}    "
                f"TIME {_format_time(event['timestamp_ms'])}    RULE {event['rule'].upper()}    "
                f"CONF {event['confidence']:.2f}"
            )
        else:
            status = (
                f"HIT {review_index:02d}/{len(events):02d}    "
                f"FRAME {frame_id:03d}    TIME {_format_time(event['timestamp_ms'])}    "
                f"KIN {kinematic_label}    {origin.upper()}"
            )
        color = event_color
    elif near_event is not None:
        review_index, nearby = near_event
        delta = int(nearby["frame_id"]) - frame_id
        status = f"CANDIDATE {review_index:02d} WINDOW    PREDICTED FRAME IN {delta:+d}"
        color = (50, 175, 255)
    else:
        hit_count = sum(event.get("event_type", "hit") == "hit" for event in events)
        landing_count = sum(event.get("event_type") == "landing" for event in events)
        status = f"HIT CANDIDATES: {hit_count}    LANDING CANDIDATES: {landing_count}"
        color = (215, 220, 228)
    cv2.putText(
        image,
        status,
        (22, panel_top + 37),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.61,
        color,
        2,
        cv2.LINE_AA,
    )
    legend = (
        "Green = matched GT frame; red = false positive; orange = missed GT hit."
        if evaluation_mode
        else "Red/orange = hit; yellow = raw hit candidate; green diamond = landing."
    )
    cv2.putText(
        image,
        legend,
        (22, panel_top + 67),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (185, 193, 203),
        1,
        cv2.LINE_AA,
    )
    _draw_timeline(image, events, frame_id, total_frames, panel_top)
    return image


def _write_manifest(path: Path, events: list[dict[str, Any]]) -> None:
    fields = [
        "review_index",
        "event_type",
        "event_frame",
        "event_time_seconds",
        "prediction_frame",
        "ground_truth_frame",
        "kinematic_frame",
        "frame_offset",
        "origin",
        "rule",
        "confidence",
        "track_score",
        "kinematic_score",
        "verdict",
        "corrected_frame",
        "notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for review_index, event in enumerate(events, start=1):
            kinematic_frame = event.get("kinematic_frame", event["frame_id"])
            origin = str(event.get("origin", ""))
            writer.writerow(
                {
                    "review_index": review_index,
                    "event_type": event.get("event_type", "hit"),
                    "event_frame": event["frame_id"],
                    "event_time_seconds": f"{event['timestamp_ms'] / 1000.0:.3f}",
                    "prediction_frame": event.get("prediction_frame", ""),
                    "ground_truth_frame": event.get("ground_truth_frame", ""),
                    "kinematic_frame": "" if kinematic_frame is None else kinematic_frame,
                    "frame_offset": event.get(
                        "frame_offset",
                        "" if kinematic_frame is None else int(event["frame_id"]) - int(kinematic_frame),
                    ),
                    "origin": event.get("origin", "existing_confirmed"),
                    "rule": event["rule"],
                    "confidence": f"{event['confidence']:.3f}",
                    "track_score": f"{float(event.get('track_score', 0.0)):.3f}",
                    "kinematic_score": f"{float(event.get('kinematic_score', 0.0)):.3f}",
                    "verdict": event.get(
                        "verdict",
                        "hit_confirmed"
                        if origin in {"existing_confirmed", "user_confirmed_missed"}
                        else "",
                    ),
                    "corrected_frame": "",
                    "notes": "",
                }
            )


def _render(args: argparse.Namespace) -> dict[str, Any]:
    if int(args.slowdown) < 1:
        raise ValueError("slowdown must be at least 1")
    if float(args.hold_seconds) < 0.0:
        raise ValueError("hold-seconds cannot be negative")
    if int(args.match_tolerance) < 0:
        raise ValueError("match-tolerance cannot be negative")
    rows, logged_events = _load_log(args.frame_log)
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {args.video}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if source_fps <= 0.0 or total_frames <= 0:
        capture.release()
        raise RuntimeError("video metadata is invalid")
    detected_hits = [event for event in logged_events if event.get("event_type") == "hit"]
    landing_events = [event for event in logged_events if event.get("event_type") == "landing"]
    evaluation = None
    if args.ground_truth_csv is not None:
        ground_truth_hits = _load_ground_truth(args.ground_truth_csv)
        if max(ground_truth_hits) >= total_frames:
            capture.release()
            raise ValueError("ground-truth CSV contains a frame outside the source video")
        events, evaluation = _build_evaluation_events(
            detected_hits,
            ground_truth_hits,
            tolerance=int(args.match_tolerance),
            source_width=source_width,
            source_height=source_height,
            fps=source_fps,
        )
    elif args.high_recall:
        events = _build_high_recall_events(
            rows,
            detected_hits,
            fps=source_fps,
            width=source_width,
            height=source_height,
        )
    else:
        events = detected_hits
    if args.include_landings and args.ground_truth_csv is None:
        events.extend(landing_events)
        events.sort(key=lambda event: (int(event["frame_id"]), str(event.get("event_type", "hit"))))
    for event in events:
        if "track_score" in event:
            continue
        _, _, score = _measured_ball(rows.get(int(event["frame_id"])))
        event["track_score"] = score
        event["kinematic_score"] = score
    output_size = (1280, 840)
    hold_frames = int(round(float(args.hold_seconds) * source_fps))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        source_fps,
        output_size,
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"could not create output video: {args.output}")

    events_by_frame = {int(event["frame_id"]): event for event in events}
    trail: deque[tuple[int, int] | None] = deque(maxlen=18)
    contact_frames: list[tuple[int, np.ndarray]] = []
    output_event_frames: list[int] = []
    written = 0
    frame_id = 0
    try:
        while True:
            ok, source = capture.read()
            if not ok:
                break
            event = events_by_frame.get(frame_id)
            near_event = None
            if event is None:
                nearby = [
                    (index, item)
                    for index, item in enumerate(events, start=1)
                    if abs(int(item["frame_id"]) - frame_id) <= 5
                ]
                if nearby:
                    near_event = min(nearby, key=lambda item: abs(int(item[1]["frame_id"]) - frame_id))
            rendered = _draw_frame(
                source,
                rows.get(frame_id),
                event,
                near_event,
                events,
                frame_id,
                total_frames,
                source_fps,
                trail,
                output_size,
            )
            if event is not None:
                output_event_frames.append(written)
                contact_frames.append((frame_id, rendered.copy()))
            copies = int(args.slowdown) + (hold_frames if event is not None else 0)
            for _ in range(copies):
                writer.write(rendered)
                written += 1
            frame_id += 1
    finally:
        capture.release()
        writer.release()
    if frame_id != total_frames:
        raise RuntimeError(f"decoded {frame_id} source frames, expected {total_frames}")
    _write_manifest(args.manifest, events)
    _write_contact_sheet(args.contact_sheet, contact_frames)
    summary = {
        "source_video": str(args.video.resolve()),
        "source_frame_log": str(args.frame_log.resolve()),
        "source": {
            "frames": total_frames,
            "fps": source_fps,
            "width": source_width,
            "height": source_height,
            "duration_seconds": total_frames / source_fps,
        },
        "output": str(args.output.resolve()),
        "manifest": str(args.manifest.resolve()),
        "contact_sheet": str(args.contact_sheet.resolve()),
        "events": [
            {"review_index": index, **event}
            for index, event in enumerate(events, start=1)
        ],
        "render": {
            "frames": written,
            "fps": source_fps,
            "width": output_size[0],
            "height": output_size[1],
            "duration_seconds": written / source_fps,
            "slowdown": int(args.slowdown),
            "hold_frames_per_event": hold_frames,
            "output_event_frames": output_event_frames,
            "mode": (
                "ground_truth_evaluation"
                if args.ground_truth_csv is not None
                else "high_recall"
                if args.high_recall
                else "detected_only"
            ),
        },
    }
    if evaluation is not None:
        summary["ground_truth_csv"] = str(args.ground_truth_csv.resolve())
        summary["evaluation"] = evaluation
        summary["primary_tolerance_frames"] = int(args.match_tolerance)
    return summary


def _write_contact_sheet(path: Path, frames: list[tuple[int, np.ndarray]]) -> None:
    if not frames:
        raise RuntimeError("no hit frames available for contact sheet")
    cell_width, cell_height = 480, 315
    columns = 3
    rows = (len(frames) + columns - 1) // columns
    sheet = np.full((rows * cell_height, columns * cell_width, 3), 18, dtype=np.uint8)
    for index, (frame_id, frame) in enumerate(frames):
        row, column = divmod(index, columns)
        resized = cv2.resize(frame[:720], (cell_width, 270), interpolation=cv2.INTER_AREA)
        top = row * cell_height
        left = column * cell_width
        sheet[top : top + 270, left : left + cell_width] = resized
        cv2.putText(
            sheet,
            f"CANDIDATE {index + 1:02d}  |  SOURCE FRAME {frame_id:03d}",
            (left + 14, top + 300),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (230, 233, 238),
            1,
            cv2.LINE_AA,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise RuntimeError(f"could not write contact sheet: {path}")


def _verify_video(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"could not reopen output video: {path}")
    decoded = 0
    sample_ids = {
        0,
        int(expected["frames"]) // 2,
        int(expected["frames"]) - 1,
        *[int(value) for value in expected["output_event_frames"]],
    }
    sample_stddev: dict[str, float] = {}
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if decoded in sample_ids:
            sample_stddev[str(decoded)] = float(frame.std())
        decoded += 1
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if decoded != int(expected["frames"]):
        raise RuntimeError(f"decoded {decoded} output frames, expected {expected['frames']}")
    if width != int(expected["width"]) or height != int(expected["height"]):
        raise RuntimeError("output video size mismatch")
    if abs(fps - float(expected["fps"])) > 0.01:
        raise RuntimeError("output video fps mismatch")
    if set(sample_stddev) != {str(value) for value in sample_ids}:
        raise RuntimeError("not all verification frames were decoded")
    if any(value <= 0.0 for value in sample_stddev.values()):
        raise RuntimeError("an output verification frame is blank")
    return {
        "frames_decoded": decoded,
        "fps": fps,
        "width": width,
        "height": height,
        "sample_stddev": sample_stddev,
    }


def main() -> None:
    args = _parse_args()
    summary = _render(args)
    summary["verification"] = _verify_video(args.output, summary["render"])
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
