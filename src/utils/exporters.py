from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TextIO

import numpy as np

from src.utils.structures import FrameResult, PersonPoseResult, TrackResult


def export_json(results: list[FrameResult], path: Path) -> None:
    path.write_text(json.dumps([item.to_dict() for item in results], ensure_ascii=False, indent=2), encoding="utf-8")


def export_csv(results: list[FrameResult], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["frame_id", "person_count", "ball_x", "ball_y", "ball_visible", "ball_score"],
        )
        writer.writeheader()
        for item in results:
            writer.writerow(
                {
                    "frame_id": item.frame_id,
                    "person_count": len(item.pose),
                    "ball_x": item.track.ball_xy[0],
                    "ball_y": item.track.ball_xy[1],
                    "ball_visible": item.track.visible,
                    "ball_score": item.track.score,
                }
            )


TRACK_DEBUG_FIELDS = [
    "frame_index",
    "action",
    "reason",
    "raw_candidate_count",
    "candidate_count",
    "court_filter_active",
    "court_filtered_count",
    "selected_candidate_index",
    "selected_candidate_rank",
    "static_filtered_count",
    "static_hotspot_count",
    "input_visible",
    "input_x",
    "input_y",
    "input_score",
    "output_visible",
    "output_x",
    "output_y",
    "output_score",
    "locked_before",
    "locked_after",
    "missed_before",
    "missed_after",
    "coast_before",
    "coast_after",
    "last_x_before",
    "last_y_before",
    "pred_x",
    "pred_y",
    "velocity_x_before",
    "velocity_y_before",
    "velocity_x_after",
    "velocity_y_after",
    "top_exit_remaining",
    "frame_w",
    "frame_h",
    "dt",
    "fixed_lag_frames",
    "fixed_lag_source",
    "source_frame_offset",
    "inpaint_mask",
    "candidates",
]


def export_track_debug_csv(records: list[dict[str, object]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=TRACK_DEBUG_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def frame_result_log_record(
    result: FrameResult,
    *,
    timestamp_ms: int | None = None,
    court_prediction: object | None = None,
    hit_event: object | None = None,
    trajectory_event: object | None = None,
    landing_event: object | None = None,
) -> dict[str, object]:
    return {
        "frame_id": int(result.frame_id),
        "timestamp_ms": int(timestamp_ms) if timestamp_ms is not None else None,
        "ball": _track_log_record(result.track),
        "pose": [_pose_log_record(person) for person in result.pose],
        "court": _court_log_record(court_prediction),
        "hit_event": _hit_event_log_record(hit_event),
        "trajectory_event": _trajectory_event_log_record(trajectory_event),
        "landing_event": _trajectory_event_log_record(landing_event),
    }


def write_frame_log_jsonl(log_file: TextIO | None, record: dict[str, object]) -> None:
    if log_file is None:
        return
    log_file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def _track_log_record(track: TrackResult) -> dict[str, object]:
    ball_xy = [float(value) for value in track.ball_xy[:2]] if len(track.ball_xy) >= 2 else [-1.0, -1.0]
    return {
        "xy": ball_xy,
        "visible": int(bool(track.visible)),
        "score": float(track.score),
    }


def _pose_log_record(person: PersonPoseResult) -> dict[str, object]:
    return {
        "person_id": int(person.person_id),
        "bbox": [float(value) for value in person.bbox[:4]],
        "person_score": float(person.person_score),
        "keypoints": [[float(point[0]), float(point[1])] for point in person.keypoints if len(point) >= 2],
        "keypoint_scores": [float(score) for score in person.scores],
    }


def _court_log_record(court_prediction: object | None) -> dict[str, object] | None:
    if court_prediction is None:
        return None
    payload = court_prediction if isinstance(court_prediction, dict) else None
    to_dict = getattr(court_prediction, "to_dict", None)
    if payload is None and callable(to_dict):
        value = to_dict()
        payload = value if isinstance(value, dict) else None
    if payload is None:
        return None
    return {
        "valid": bool(payload.get("valid", False)),
        "attempted": bool(payload.get("attempted", False)),
        "updated": bool(payload.get("updated", False)),
        "update_type": str(payload.get("update_type", "")),
        "status": str(payload.get("status", "")),
        "confidence": float(payload.get("confidence", 0.0) or 0.0),
        "candidate_confidence": _optional_float(payload.get("candidate_confidence")),
        "reason": str(payload.get("reason", "")),
        "scheme": str(payload.get("scheme", "")),
        "corners": _points_log_record(payload.get("corners")),
        "metrics": payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {},
        "detect_ms": float(payload.get("detect_ms", 0.0) or 0.0),
        "rejected_count": int(payload.get("rejected_count", 0) or 0),
    }


def _points_log_record(points: object) -> list[list[float]]:
    if not isinstance(points, (list, tuple)):
        return []
    output: list[list[float]] = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        output.append([float(point[0]), float(point[1])])
    return output


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _hit_event_log_record(hit_event: object | None) -> dict[str, object] | None:
    if not isinstance(hit_event, dict):
        return None
    ball_xy = hit_event.get("ball_xy", [-1.0, -1.0])
    if not isinstance(ball_xy, (list, tuple)) or len(ball_xy) < 2:
        ball_xy = [-1.0, -1.0]
    record: dict[str, object] = {
        "frame_id": int(hit_event.get("frame_id", -1)),
        "timestamp_ms": int(hit_event.get("timestamp_ms", 0)),
        "ball_xy": [float(ball_xy[0]), float(ball_xy[1])],
    }
    if "event_type" in hit_event:
        record["event_type"] = str(hit_event.get("event_type", ""))
    if "rule" in hit_event:
        record["rule"] = str(hit_event.get("rule", ""))
    if "confidence" in hit_event:
        record["confidence"] = float(hit_event.get("confidence", 0.0))
    if "all_rules" in hit_event:
        record["all_rules"] = [str(item) for item in _event_sequence(hit_event.get("all_rules"))]
    if "auxiliary_rules" in hit_event:
        record["auxiliary_rules"] = [str(item) for item in _event_sequence(hit_event.get("auxiliary_rules"))]
    if "features" in hit_event:
        record["features"] = hit_event.get("features", {})
    return record


def _trajectory_event_log_record(event: object | None) -> dict[str, object] | None:
    if not isinstance(event, dict):
        return None
    ball_xy = event.get("ball_xy", [-1.0, -1.0])
    if not isinstance(ball_xy, (list, tuple)) or len(ball_xy) < 2:
        ball_xy = [-1.0, -1.0]
    return {
        "event_type": str(event.get("event_type", "")),
        "frame_id": int(event.get("frame_id", -1)),
        "timestamp_ms": int(event.get("timestamp_ms", 0)),
        "ball_xy": [float(ball_xy[0]), float(ball_xy[1])],
        "rule": str(event.get("rule", "")),
        "confidence": float(event.get("confidence", 0.0)),
        "all_rules": [str(item) for item in _event_sequence(event.get("all_rules"))],
        "auxiliary_rules": [str(item) for item in _event_sequence(event.get("auxiliary_rules"))],
        "features": event.get("features", {}),
    }


def _event_sequence(value: object) -> list[object]:
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def export_npy(results: list[FrameResult], path: Path) -> None:
    max_persons = max((len(item.pose) for item in results), default=0)
    max_kpts = max((len(person.keypoints) for item in results for person in item.pose), default=0)
    frames = len(results)
    keypoints = np.zeros((frames, max_persons, max_kpts, 2), dtype=np.float32)
    keypoint_scores = np.zeros((frames, max_persons, max_kpts), dtype=np.float32)
    bboxes = np.zeros((frames, max_persons, 4), dtype=np.float32)
    person_scores = np.zeros((frames, max_persons), dtype=np.float32)
    ball_xy = np.zeros((frames, 2), dtype=np.float32)
    ball_visible = np.zeros((frames,), dtype=np.int32)
    ball_score = np.zeros((frames,), dtype=np.float32)

    for frame_idx, item in enumerate(results):
        ball_xy[frame_idx] = np.asarray(item.track.ball_xy, dtype=np.float32)
        ball_visible[frame_idx] = item.track.visible
        ball_score[frame_idx] = item.track.score
        for person_idx, person in enumerate(item.pose):
            bboxes[frame_idx, person_idx] = np.asarray(person.bbox, dtype=np.float32)
            person_scores[frame_idx, person_idx] = person.person_score
            for kp_idx, kp in enumerate(person.keypoints):
                keypoints[frame_idx, person_idx, kp_idx] = np.asarray(kp, dtype=np.float32)
                if kp_idx < len(person.scores):
                    keypoint_scores[frame_idx, person_idx, kp_idx] = person.scores[kp_idx]

    np.save(
        path,
        {
            "frame_ids": np.arange(frames, dtype=np.int32),
            "keypoints": keypoints,
            "keypoint_scores": keypoint_scores,
            "bboxes": bboxes,
            "person_scores": person_scores,
            "ball_xy": ball_xy,
            "ball_visible": ball_visible,
            "ball_score": ball_score,
        },
        allow_pickle=True,
    )
