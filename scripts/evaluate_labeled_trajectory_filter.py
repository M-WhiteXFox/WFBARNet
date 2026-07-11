from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.postprocess.track_filter import BallTrackFilter, BallTrackFilterConfig
from src.utils.structures import TrackResult


Point = tuple[float, float]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay cached TrackNet candidates against CVAT point annotations.",
    )
    parser.add_argument("datasets", nargs="+", help="Dataset stems, for example: 1 2")
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "Dataset")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "outputs" / "trajectory_filter_tuning" / "cache",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--distance-threshold", type=float, default=20.0)
    parser.add_argument(
        "--active-range",
        action="append",
        default=[],
        metavar="DATASET:START:END",
        help="Limit positive evaluation to the airborne/rally span; may be repeated.",
    )
    parser.add_argument("--person-confidence", type=float)
    parser.add_argument("--court-lateral-ratio", type=float)
    return parser.parse_args()


def _load_annotations(xml_path: Path) -> tuple[dict[int, Point], set[int]]:
    root = ET.parse(xml_path).getroot()
    visible: dict[int, Point] = {}
    annotated: set[int] = set()
    for element in root.findall(".//track/points"):
        frame = int(element.attrib["frame"])
        annotated.add(frame)
        if element.attrib.get("outside", "0") == "1":
            continue
        values = element.attrib.get("points", "").split(",")
        if len(values) != 2:
            continue
        visible[frame] = (float(values[0]), float(values[1]))
    if not visible:
        raise ValueError(f"No visible point annotations found in {xml_path}")
    return visible, annotated


def _load_cache(cache_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with cache_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _track_from_cache(item: dict[str, Any]) -> TrackResult:
    return TrackResult(
        ball_xy=[float(value) for value in item.get("ball_xy", [-1.0, -1.0])[:2]],
        visible=int(bool(item.get("visible", 0))),
        score=float(item.get("score", 0.0)),
        heatmap_shape=[int(value) for value in item.get("heatmap_shape", [])],
    )


def _point_from_track(track: TrackResult) -> Point | None:
    if not track.visible or len(track.ball_xy) < 2:
        return None
    return float(track.ball_xy[0]), float(track.ball_xy[1])


def _point_from_debug(debug: dict[str, Any], prefix: str) -> Point | None:
    if not bool(debug.get(f"{prefix}_visible", 0)):
        return None
    return float(debug[f"{prefix}_x"]), float(debug[f"{prefix}_y"])


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    alpha = position - lower
    return ordered[lower] * (1.0 - alpha) + ordered[upper] * alpha


def _metric_block(
    rows: Iterable[dict[str, Any]],
    *,
    point_key: str,
    threshold: float,
    active_start: int,
    active_end: int,
) -> dict[str, Any]:
    selected = list(rows)
    gt_positive = sum(
        bool(row["gt_visible"]) and active_start <= row["frame_id"] <= active_end
        for row in selected
    )
    pred_visible = sum(row[point_key] is not None for row in selected)
    correct = sum(
        active_start <= row["frame_id"] <= active_end
        and row[point_key] is not None
        and row["gt_point"] is not None
        and _distance(row[point_key], row["gt_point"]) <= threshold
        for row in selected
    )
    precision = correct / pred_visible if pred_visible else 0.0
    recall = correct / gt_positive if gt_positive else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "gt_positive": gt_positive,
        "pred_visible": pred_visible,
        "correct": correct,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pre_fp": sum(
            row["frame_id"] < active_start and row[point_key] is not None for row in selected
        ),
        "post_fp": sum(
            row["frame_id"] > active_end and row[point_key] is not None for row in selected
        ),
    }


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(value) for key, value in row.items()})


def _evaluate_dataset(
    name: str,
    *,
    dataset_dir: Path,
    cache_dir: Path,
    output_dir: Path,
    cache_meta: dict[str, Any],
    threshold: float,
    active_range: tuple[int, int] | None,
    person_confidence: float | None,
    court_lateral_ratio: float | None,
) -> dict[str, Any]:
    metadata = cache_meta["datasets"][name]
    fps = float(metadata["fps"])
    width = int(metadata["width"])
    height = int(metadata["height"])
    cache_rows = _load_cache(cache_dir / f"{name}_model_cache.jsonl")
    visible_gt, annotated_frames = _load_annotations(dataset_dir / f"{name}.xml")
    active_start, active_end = active_range or (min(visible_gt), max(visible_gt))
    if active_start > active_end:
        raise ValueError(f"Invalid active range for {name}: {active_start}>{active_end}")
    court_prediction = {
        "valid": True,
        "corners": cache_meta["court_corners"],
    }
    config = BallTrackFilterConfig(fps=fps)
    if person_confidence is not None:
        config.person_occlusion_accept_confidence = float(person_confidence)
    if court_lateral_ratio is not None:
        config.court_air_lateral_expansion_ratio = float(court_lateral_ratio)
    track_filter = BallTrackFilter(config, debug_enabled=True)

    analysis_rows: list[dict[str, Any]] = []
    per_frame_rows: list[dict[str, Any]] = []
    debug_rows: list[dict[str, Any]] = []
    for cache_row in cache_rows:
        frame_id = int(cache_row["frame_id"])
        candidates = [_track_from_cache(item) for item in cache_row.get("candidates", [])]
        person_bboxes = [tuple(float(value) for value in bbox) for bbox in cache_row.get("person_bboxes", [])]
        output = track_filter.update_candidates(
            candidates,
            dt=1.0 / fps,
            frame_shape=(height, width, 3),
            court_prediction=court_prediction,
            person_bboxes=person_bboxes,
        )
        debug = track_filter.last_debug_record() or {}
        gt_point = visible_gt.get(frame_id)
        input_point = _point_from_debug(debug, "input")
        output_point = _point_from_track(output)
        raw_errors = [
            _distance(_point_from_track(candidate), gt_point)
            for candidate in candidates
            if gt_point is not None and _point_from_track(candidate) is not None
        ]
        phase = "pre" if frame_id < active_start else "post" if frame_id > active_end else "active"
        analysis_row = {
            "frame_id": frame_id,
            "phase": phase,
            "gt_visible": gt_point is not None,
            "gt_point": gt_point,
            "input_point": input_point,
            "output_point": output_point,
            "raw_best_error_px": min(raw_errors) if raw_errors else None,
            "action": str(debug.get("action", "")),
            "reason": str(debug.get("reason", "")),
            "court_filtered_count": int(debug.get("court_filtered_count", 0)),
        }
        analysis_rows.append(analysis_row)
        per_frame_rows.append(
            {
                "frame_id": frame_id,
                "timestamp_s": frame_id / fps,
                "phase": phase,
                "gt_visible": gt_point is not None,
                "gt_x": gt_point[0] if gt_point else None,
                "gt_y": gt_point[1] if gt_point else None,
                "gt_annotation": "visible" if gt_point else "outside" if frame_id in annotated_frames else "unannotated",
                "raw_best_error_px": analysis_row["raw_best_error_px"],
                "input_visible": input_point is not None,
                "input_x": input_point[0] if input_point else -1.0,
                "input_y": input_point[1] if input_point else -1.0,
                "input_score": float(debug.get("input_score", 0.0)),
                "input_error_px": _distance(input_point, gt_point) if input_point and gt_point else None,
                "output_visible": output_point is not None,
                "output_x": output_point[0] if output_point else -1.0,
                "output_y": output_point[1] if output_point else -1.0,
                "output_score": float(output.score),
                "output_error_px": _distance(output_point, gt_point) if output_point and gt_point else None,
                "action": analysis_row["action"],
                "reason": analysis_row["reason"],
                "raw_candidate_count": len(candidates),
                "court_filtered_count": analysis_row["court_filtered_count"],
            }
        )
        debug_rows.append(debug)

    active_rows = [row for row in analysis_rows if active_start <= row["frame_id"] <= active_end]
    overlap_errors = [
        _distance(row["output_point"], row["gt_point"])
        for row in active_rows
        if row["output_point"] is not None and row["gt_point"] is not None
    ]
    actions = Counter(row["action"] for row in analysis_rows if row["action"])
    miss_reasons = Counter(
        row["reason"]
        for row in active_rows
        if row["gt_visible"]
        and (
            row["output_point"] is None
            or _distance(row["output_point"], row["gt_point"]) > threshold
        )
    )
    person_reject_rows = [
        row for row in active_rows if row["reason"] == "person_occlusion_candidate_high_score"
    ]
    court_removed_correct_frames = [
        row["frame_id"]
        for row in active_rows
        if row["court_filtered_count"] > 0
        and row["raw_best_error_px"] is not None
        and row["raw_best_error_px"] <= threshold
    ]
    threshold_label = f"{threshold:g}".replace(".", "p")
    jumps: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    for row in analysis_rows:
        if row["output_point"] is None:
            previous = None
            continue
        if previous is not None:
            jump = _distance(previous["output_point"], row["output_point"])
            if jump > 110.0:
                jumps.append(
                    {
                        "from": previous["frame_id"],
                        "to": row["frame_id"],
                        "jump_px": jump,
                        "phase": row["phase"],
                        "error_px": (
                            _distance(row["output_point"], row["gt_point"])
                            if row["gt_point"] is not None
                            else None
                        ),
                        "action": row["action"],
                        "reason": row["reason"],
                    }
                )
        previous = row

    summary = {
        "fps": fps,
        "frame_count": len(cache_rows),
        "active_start": active_start,
        "active_end": active_end,
        "gt_visible_active": sum(active_start <= frame_id <= active_end for frame_id in visible_gt),
        f"full_output_{threshold_label}": _metric_block(
            analysis_rows,
            point_key="output_point",
            threshold=threshold,
            active_start=active_start,
            active_end=active_end,
        ),
        f"active_output_{threshold_label}": _metric_block(
            active_rows,
            point_key="output_point",
            threshold=threshold,
            active_start=active_start,
            active_end=active_end,
        ),
        f"full_input_{threshold_label}": _metric_block(
            analysis_rows,
            point_key="input_point",
            threshold=threshold,
            active_start=active_start,
            active_end=active_end,
        ),
        f"active_input_{threshold_label}": _metric_block(
            active_rows,
            point_key="input_point",
            threshold=threshold,
            active_start=active_start,
            active_end=active_end,
        ),
        "output_overlap_error": {
            "count": len(overlap_errors),
            "median": median(overlap_errors) if overlap_errors else None,
            "p90": _percentile(overlap_errors, 0.90),
            "max": max(overlap_errors) if overlap_errors else None,
        },
        "person_occlusion_rejects_active": len(person_reject_rows),
        f"person_occlusion_rejects_correct_input_{threshold_label}": sum(
            row["input_point"] is not None
            and row["gt_point"] is not None
            and _distance(row["input_point"], row["gt_point"]) <= threshold
            for row in person_reject_rows
        ),
        "court_removed_correct_candidate_frames": len(court_removed_correct_frames),
        "court_removed_correct_frames": court_removed_correct_frames,
        "actions": dict(actions),
        "miss_reasons_active": dict(miss_reasons),
        "jumps_over_110px": jumps,
    }

    dataset_output = output_dir / name
    dataset_output.mkdir(parents=True, exist_ok=True)
    _write_csv(dataset_output / "per_frame.csv", per_frame_rows)
    _write_csv(dataset_output / "track_debug.csv", debug_rows)
    (dataset_output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = _parse_args()
    cache_meta = json.loads((args.cache_dir / "cache_meta.json").read_text(encoding="utf-8"))
    active_ranges: dict[str, tuple[int, int]] = {}
    for value in args.active_range:
        try:
            name, start, end = value.split(":", maxsplit=2)
            active_ranges[name] = (int(start), int(end))
        except ValueError as exc:
            raise ValueError(f"Invalid --active-range value: {value!r}") from exc
    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined = {
        name: _evaluate_dataset(
            name,
            dataset_dir=args.dataset_dir,
            cache_dir=args.cache_dir,
            output_dir=args.output_dir,
            cache_meta=cache_meta,
            threshold=float(args.distance_threshold),
            active_range=active_ranges.get(name),
            person_confidence=args.person_confidence,
            court_lateral_ratio=args.court_lateral_ratio,
        )
        for name in args.datasets
    }
    output_path = args.output_dir / "combined_summary.json"
    output_path.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
