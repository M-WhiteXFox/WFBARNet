from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_labeled_trajectory_filter import (
    _apply_config_overrides,
    _load_cache,
    _point_from_track,
    _track_from_cache,
)
from src.postprocess.fixed_lag_track import FixedLagTrackConfig, FixedLagTrackPostProcessor
from src.postprocess.track_filter import BallTrackFilter, BallTrackFilterConfig


Point = tuple[float, float]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay TrackNetV3 raw_data caches through the production postprocessor.",
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", action="append", choices=("raw_data", "raw_data2"))
    parser.add_argument("--threshold", action="append", type=float)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Temporarily override a BallTrackFilterConfig field; may be repeated.",
    )
    return parser.parse_args()


def _load_gt(path: Path, width: int, height: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for expected_frame, row in enumerate(csv.DictReader(handle)):
            frame_id = int(row["Frame"])
            if frame_id != expected_frame:
                raise ValueError(f"Non-contiguous frame id in {path}: {frame_id} != {expected_frame}")
            visible = int(row["Ball"]) == 1
            rows.append(
                {
                    "frame_id": frame_id,
                    "visible": visible,
                    "point": (
                        float(row["x"]) * width,
                        float(row["y"]) * height,
                    )
                    if visible
                    else None,
                }
            )
    return rows


def _distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _longest_true_run(values: Iterable[bool]) -> int:
    longest = 0
    current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def _metrics(rows: list[dict[str, Any]], point_key: str, threshold: float) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    errors: list[float] = []
    for row in rows:
        gt_point = row["gt_point"]
        point = row[point_key]
        if gt_point is None:
            counts["fp" if point is not None else "tn"] += 1
        elif point is None:
            counts["missing"] += 1
        else:
            error = _distance(point, gt_point)
            errors.append(error)
            counts["correct" if error <= threshold else "drift"] += 1
    gt_positive = counts["correct"] + counts["drift"] + counts["missing"]
    pred_positive = counts["correct"] + counts["drift"] + counts["fp"]
    precision = counts["correct"] / pred_positive if pred_positive else 0.0
    recall = counts["correct"] / gt_positive if gt_positive else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (counts["correct"] + counts["tn"]) / len(rows) if rows else 0.0
    return {
        "frames": len(rows),
        "gt_visible": gt_positive,
        "pred_visible": pred_positive,
        "correct": counts["correct"],
        "missing": counts["missing"],
        "drift": counts["drift"],
        "fp": counts["fp"],
        "tn": counts["tn"],
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "overlap_error_median": sorted(errors)[len(errors) // 2] if errors else None,
        "overlap_error_max": max(errors) if errors else None,
        "longest_missing_run": _longest_true_run(
            row["gt_point"] is not None and row[point_key] is None for row in rows
        ),
        "longest_failure_run": _longest_true_run(
            row["gt_point"] is not None
            and (
                row[point_key] is None
                or _distance(row[point_key], row["gt_point"]) > threshold
            )
            for row in rows
        ),
    }


def _candidate_oracle(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    visible_rows = [row for row in rows if row["gt_point"] is not None]
    correct = sum(
        any(_distance(point, row["gt_point"]) <= threshold for point in row["candidate_points"])
        for row in visible_rows
    )
    return {
        "gt_visible": len(visible_rows),
        "correct": correct,
        "recall": correct / len(visible_rows) if visible_rows else 0.0,
    }


def _evaluate_video(
    *,
    split: str,
    name: str,
    metadata: dict[str, Any],
    config_overrides: list[str],
) -> list[dict[str, Any]]:
    fps = float(metadata["fps"])
    width = int(metadata["width"])
    height = int(metadata["height"])
    cache_rows = _load_cache(Path(metadata["cache"]))
    gt_rows = _load_gt(Path(metadata["annotation"]), width, height)
    if len(cache_rows) != len(gt_rows):
        raise ValueError(f"Frame count mismatch for {split}/{name}: {len(cache_rows)} != {len(gt_rows)}")

    config = BallTrackFilterConfig(fps=fps)
    _apply_config_overrides(config, config_overrides)
    track_filter = BallTrackFilter(config, debug_enabled=True)
    fixed_lag = FixedLagTrackPostProcessor(FixedLagTrackConfig(fps=fps))
    causal_rows: list[dict[str, Any]] = []
    lagged_frames = []

    previous_gt: Point | None = None
    for cache_row, gt_row in zip(cache_rows, gt_rows):
        frame_id = int(cache_row["frame_id"])
        candidates = [_track_from_cache(item) for item in cache_row.get("candidates", [])]
        causal = track_filter.update_candidates(
            candidates,
            dt=1.0 / fps,
            frame_shape=(height, width, 3),
        )
        debug = track_filter.last_debug_record() or {}
        gt_point = gt_row["point"]
        gt_speed = (
            _distance(previous_gt, gt_point)
            if previous_gt is not None and gt_point is not None
            else None
        )
        previous_gt = gt_point
        row = {
            "split": split,
            "video": name,
            "frame_id": frame_id,
            "gt_point": gt_point,
            "gt_speed_px_per_frame": gt_speed,
            "candidate_points": [
                point for candidate in candidates if (point := _point_from_track(candidate)) is not None
            ],
            "causal_point": _point_from_track(causal),
            "causal_score": float(causal.score),
            "action": str(debug.get("action", "")),
            "reason": str(debug.get("reason", "")),
        }
        causal_rows.append(row)
        lagged = fixed_lag.push(
            causal,
            candidates=candidates,
            debug_record=debug,
            payload=frame_id,
        )
        track_filter.debug_records.clear()
        if lagged is not None:
            lagged_frames.append(lagged)
    lagged_frames.extend(fixed_lag.flush())
    if len(lagged_frames) != len(causal_rows):
        raise RuntimeError(f"Fixed-lag frame mismatch for {split}/{name}")
    for row, lagged in zip(causal_rows, lagged_frames):
        if int(lagged.payload) != int(row["frame_id"]):
            raise RuntimeError(f"Fixed-lag order mismatch for {split}/{name}")
        row["fixed_point"] = _point_from_track(lagged.track)
        row["fixed_source"] = lagged.source
    return causal_rows


def _write_per_frame(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "split",
        "video",
        "frame_id",
        "gt_visible",
        "gt_x",
        "gt_y",
        "gt_speed_px_per_frame",
        "causal_visible",
        "causal_x",
        "causal_y",
        "causal_score",
        "fixed_visible",
        "fixed_x",
        "fixed_y",
        "fixed_source",
        "action",
        "reason",
        "candidate_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            gt = row["gt_point"]
            causal = row["causal_point"]
            fixed = row["fixed_point"]
            writer.writerow(
                {
                    "split": row["split"],
                    "video": row["video"],
                    "frame_id": row["frame_id"],
                    "gt_visible": int(gt is not None),
                    "gt_x": gt[0] if gt else -1.0,
                    "gt_y": gt[1] if gt else -1.0,
                    "gt_speed_px_per_frame": row["gt_speed_px_per_frame"],
                    "causal_visible": int(causal is not None),
                    "causal_x": causal[0] if causal else -1.0,
                    "causal_y": causal[1] if causal else -1.0,
                    "causal_score": row["causal_score"],
                    "fixed_visible": int(fixed is not None),
                    "fixed_x": fixed[0] if fixed else -1.0,
                    "fixed_y": fixed[1] if fixed else -1.0,
                    "fixed_source": row["fixed_source"],
                    "action": row["action"],
                    "reason": row["reason"],
                    "candidate_count": len(row["candidate_points"]),
                }
            )


def main() -> None:
    args = _parse_args()
    metadata = json.loads((args.cache_dir / "cache_meta.json").read_text(encoding="utf-8"))
    splits = args.split or ["raw_data", "raw_data2"]
    thresholds = args.threshold or [10.0, 20.0]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "dataset_root": str(args.dataset_root.resolve()),
        "config_overrides": list(args.set),
        "thresholds": thresholds,
        "splits": {},
    }

    for split in splits:
        split_rows: list[dict[str, Any]] = []
        video_summaries: dict[str, Any] = {}
        split_meta = metadata["splits"].get(split, {})
        for name, video_meta in sorted(split_meta.items()):
            rows = _evaluate_video(
                split=split,
                name=name,
                metadata=video_meta,
                config_overrides=list(args.set),
            )
            split_rows.extend(rows)
            video_summaries[name] = {
                f"{threshold:g}px": {
                    "causal": _metrics(rows, "causal_point", threshold),
                    "fixed": _metrics(rows, "fixed_point", threshold),
                    "oracle": _candidate_oracle(rows, threshold),
                }
                for threshold in thresholds
            }
        speed_values = sorted(
            float(row["gt_speed_px_per_frame"])
            for row in split_rows
            if row["gt_speed_px_per_frame"] is not None
        )
        speed_p90 = speed_values[int(0.90 * (len(speed_values) - 1))] if speed_values else 0.0
        high_speed_rows = [
            row
            for row in split_rows
            if row["gt_speed_px_per_frame"] is not None
            and float(row["gt_speed_px_per_frame"]) >= speed_p90
        ]
        split_summary = {
            "videos": len(video_summaries),
            "frames": len(split_rows),
            "speed_p90_px_per_frame": speed_p90,
            "metrics": {
                f"{threshold:g}px": {
                    "causal": _metrics(split_rows, "causal_point", threshold),
                    "fixed": _metrics(split_rows, "fixed_point", threshold),
                    "oracle": _candidate_oracle(split_rows, threshold),
                    "high_speed_causal": _metrics(high_speed_rows, "causal_point", threshold),
                    "high_speed_fixed": _metrics(high_speed_rows, "fixed_point", threshold),
                }
                for threshold in thresholds
            },
            "fixed_sources": dict(Counter(row["fixed_source"] for row in split_rows)),
            "failure_reasons": dict(
                Counter(
                    row["reason"]
                    for row in split_rows
                    if row["gt_point"] is not None
                    and (
                        row["fixed_point"] is None
                        or _distance(row["fixed_point"], row["gt_point"]) > thresholds[0]
                    )
                )
            ),
            "per_video": video_summaries,
        }
        summary["splits"][split] = split_summary
        split_dir = args.output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        _write_per_frame(split_dir / "per_frame.csv", split_rows)

    output_path = args.output_dir / "summary.json"
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output_path.resolve())


if __name__ == "__main__":
    main()
