from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.postprocess.candidate_graph_track import (
    CandidateGraphConfig,
    CandidateGraphDecision,
    CandidateGraphRefinementConfig,
    refine_candidate_graph_sequence,
)
from scripts.evaluate_labeled_trajectory_filter import (
    _load_annotations,
    _load_cache,
    _point_from_track,
    _track_from_cache,
)
from scripts.evaluate_tracknetv3_raw_postprocess import _load_gt, _metrics
from src.postprocess.adaptive_track import (
    CANDIDATE_GRAPH_TRACK_ROUTE,
    CONTEXTUAL_TRACK_ROUTE,
    AdaptiveTrackPostProcessor,
)
from src.postprocess.fixed_lag_track import FixedLagTrackConfig
from src.postprocess.rally_start_backfill import fit_known_rally_start
from src.postprocess.track_filter import BallTrackFilter, BallTrackFilterConfig
from src.utils.structures import TrackResult


Point = tuple[float, float]
DEFAULT_OUTPUT = (
    ROOT / "outputs" / "tracknetv3_candidate_graph" / "refinement" / "summary.json"
)
LABELED_RANGES = {"1": (155, 315), "2": (43, 298), "10-1": (0, 374)}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate candidate-graph sequence refinement.")
    parser.add_argument(
        "--scope",
        action="append",
        choices=("raw_data", "raw_data2", "labeled", "bd19"),
        help="Evaluation scope; defaults to all scopes.",
    )
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--official-cache",
        type=Path,
        default=ROOT / "outputs" / "tracknetv3_raw_postprocess" / "cache",
    )
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "Dataset")
    parser.add_argument(
        "--bd19-cache",
        type=Path,
        default=ROOT / "outputs" / "tracknetv3_candidate_graph" / "BD_New_19_raw_candidates.json",
    )
    parser.add_argument(
        "--bd19-per-frame",
        type=Path,
        default=ROOT
        / "outputs"
        / "tracknetv3_candidate_graph"
        / "BD_New_19_comparison_per_frame.csv",
    )
    parser.add_argument(
        "--bd19-manifest",
        type=Path,
        default=ROOT
        / "outputs"
        / "tracknetv3_candidate_graph"
        / "BD_New_19_uncertain_frames_manifest.csv",
    )
    return parser.parse_args()


def _apply_overrides(config: CandidateGraphRefinementConfig, values: Sequence[str]) -> None:
    for value in values:
        try:
            name, raw = value.split("=", maxsplit=1)
        except ValueError as exc:
            raise ValueError(f"invalid --set value: {value!r}") from exc
        if not hasattr(config, name):
            raise ValueError(f"unknown refinement field: {name}")
        current = getattr(config, name)
        if isinstance(current, bool):
            normalized = raw.strip().lower()
            if normalized not in {"true", "false", "1", "0"}:
                raise ValueError(f"invalid boolean value for {name}: {raw!r}")
            parsed: bool | int | float = normalized in {"true", "1"}
        elif isinstance(current, int):
            parsed: int | float = int(raw)
        elif isinstance(current, float):
            parsed = float(raw)
        else:
            raise ValueError(f"unsupported refinement field: {name}")
        setattr(config, name, parsed)


def _point(decision: CandidateGraphDecision) -> Point | None:
    return _point_from_track(decision.track)


def _candidate_frames(cache_rows: Sequence[dict[str, Any]]) -> list[list[TrackResult]]:
    return [
        [_track_from_cache(item) for item in row.get("candidates", [])]
        for row in cache_rows
    ]


def _current_context(
    cache_rows: Sequence[dict[str, Any]],
    frames: Sequence[Sequence[TrackResult]],
    *,
    fps: float,
    width: int,
    height: int,
    court_prediction: dict[str, Any] | None = None,
) -> tuple[set[int], list[dict[str, Any]]]:
    track_filter = BallTrackFilter(BallTrackFilterConfig(fps=fps), debug_enabled=True)
    postprocessor = AdaptiveTrackPostProcessor(
        fps=fps,
        route=CONTEXTUAL_TRACK_ROUTE,
        reliable_context=True,
        track_filter=track_filter,
        fixed_lag_config=FixedLagTrackConfig(fps=fps, delay_ms=300),
    )
    vetoes: set[int] = set()
    output: dict[int, dict[str, Any]] = {}

    def consume(lagged: Any) -> None:
        debug = dict(lagged.debug_record or {})
        frame_id = int(lagged.payload)
        raw_count = int(debug.get("raw_candidate_count", 0))
        static_count = int(debug.get("static_filtered_count", 0))
        if raw_count > 0 and static_count >= raw_count:
            vetoes.add(frame_id)
        output[frame_id] = {
            "track": lagged.track,
            "source": str(lagged.measured_source),
            "action": str(debug.get("action", "")),
            "reason": str(debug.get("reason", "")),
        }

    for frame_id, (cache_row, candidates) in enumerate(zip(cache_rows, frames)):
        person_bboxes = [
            tuple(float(value) for value in bbox)
            for bbox in cache_row.get("person_bboxes", [])
        ]
        lagged_frames = postprocessor.push(
            candidates,
            dt=1.0 / fps,
            frame_shape=(height, width, 3),
            court_prediction=court_prediction,
            person_bboxes=person_bboxes,
            payload=frame_id,
        )
        for lagged in lagged_frames:
            consume(lagged)
    for lagged in postprocessor.flush():
        consume(lagged)
    if len(output) != len(frames):
        raise RuntimeError(f"current postprocessor returned {len(output)} of {len(frames)} frames")
    return vetoes, [output[frame_id] for frame_id in range(len(frames))]


def _run_methods(
    cache_rows: Sequence[dict[str, Any]],
    *,
    fps: float,
    width: int,
    height: int,
    refinement_config: CandidateGraphRefinementConfig,
    court_prediction: dict[str, Any] | None = None,
) -> tuple[
    list[list[TrackResult]],
    list[CandidateGraphDecision],
    list[CandidateGraphDecision],
    set[int],
    list[dict[str, Any]],
]:
    frames = _candidate_frames(cache_rows)
    graph_processor = AdaptiveTrackPostProcessor(
        fps=fps,
        route=CANDIDATE_GRAPH_TRACK_ROUTE,
        reliable_context=False,
        graph_config=CandidateGraphConfig(
            fps=fps,
            delay_ms=300,
            beam_width=64,
            score_center=0.72,
        ),
    )
    graph_outputs = []
    for frame_id, candidates in enumerate(frames):
        graph_outputs.extend(graph_processor.push(candidates, payload=frame_id))
    graph_outputs.extend(graph_processor.flush())
    graph = [
        CandidateGraphDecision(
            frame_index=output.frame_index,
            track=output.track,
            source=output.measured_source,
            candidate_rank=output.candidate_rank,
            payload=output.payload,
            debug_record=output.debug_record,
        )
        for output in graph_outputs
    ]
    static_vetoes, current = _current_context(
        cache_rows,
        frames,
        fps=fps,
        width=width,
        height=height,
        court_prediction=court_prediction,
    )
    refined = refine_candidate_graph_sequence(
        frames,
        graph,
        width=width,
        height=height,
        config=refinement_config,
        static_veto_frames=sorted(static_vetoes),
        current_proposals=[item["track"] for item in current],
        current_proposal_allowed=[
            item["source"] == "causal"
            and item["action"] in {"accept", "bootstrap_accept", "relock_accept"}
            for item in current
        ],
    )
    return frames, graph, refined, static_vetoes, current


def _changed_frames(
    graph: Sequence[CandidateGraphDecision],
    refined: Sequence[CandidateGraphDecision],
) -> list[int]:
    changed: list[int] = []
    for frame_id, (before, after) in enumerate(zip(graph, refined)):
        before_point = _point(before)
        after_point = _point(after)
        if (before_point is None) != (after_point is None):
            changed.append(frame_id)
        elif before_point is not None and after_point is not None and math.dist(before_point, after_point) > 1e-9:
            changed.append(frame_id)
    return changed


def _classification(gt: Point | None, point: Point | None, threshold: float) -> str:
    if gt is None:
        return "fp" if point is not None else "tn"
    if point is None:
        return "missing"
    return "correct" if math.dist(gt, point) <= threshold else "drift"


def _change_outcomes(
    rows: Sequence[dict[str, Any]],
    graph: Sequence[CandidateGraphDecision],
    refined: Sequence[CandidateGraphDecision],
    *,
    threshold: float,
) -> tuple[dict[str, dict[str, int]], list[dict[str, Any]]]:
    counters: dict[str, Counter[str]] = {}
    detail: list[dict[str, Any]] = []
    for frame_id in _changed_frames(graph, refined):
        source = refined[frame_id].source
        before = _classification(rows[frame_id]["gt_point"], _point(graph[frame_id]), threshold)
        after = _classification(rows[frame_id]["gt_point"], _point(refined[frame_id]), threshold)
        counters.setdefault(source, Counter())[f"{before}_to_{after}"] += 1
        detail.append(
            {
                "frame_id": frame_id,
                "source": source,
                "transition": f"{before}_to_{after}",
                "score": float(refined[frame_id].track.score),
                "candidate_rank": refined[frame_id].candidate_rank,
            }
        )
    return {source: dict(counter) for source, counter in counters.items()}, detail


def _official_scope(
    split: str,
    metadata: dict[str, Any],
    refinement_config: CandidateGraphRefinementConfig,
) -> dict[str, Any]:
    all_rows: list[dict[str, Any]] = []
    per_video: dict[str, Any] = {}
    source_counts: Counter[str] = Counter()
    aggregate_outcomes: dict[str, Counter[str]] = {}
    static_veto_count = 0
    for name, video_meta in sorted(metadata["splits"][split].items()):
        fps = float(video_meta["fps"])
        width = int(video_meta["width"])
        height = int(video_meta["height"])
        cache_rows = _load_cache(Path(video_meta["cache"]))
        gt_rows = _load_gt(Path(video_meta["annotation"]), width, height)
        if len(cache_rows) != len(gt_rows):
            raise ValueError(f"frame mismatch for {split}/{name}")
        frames, graph, refined, static_vetoes, current = _run_methods(
            cache_rows,
            fps=fps,
            width=width,
            height=height,
            refinement_config=refinement_config,
        )
        rows = [
            {
                "gt_point": gt_row["point"],
                "graph_point": _point(graph_decision),
                "refined_point": _point(refined_decision),
                "current_point": _point_from_track(current_decision["track"]),
                "context_hybrid_point": (
                    _point_from_track(current_decision["track"])
                    if _point_from_track(current_decision["track"]) is not None
                    else _point(graph_decision)
                ),
            }
            for gt_row, graph_decision, refined_decision, current_decision in zip(
                gt_rows, graph, refined, current
            )
        ]
        all_rows.extend(rows)
        changes = _changed_frames(graph, refined)
        source_counts.update(refined[frame_id].source for frame_id in changes)
        outcomes, change_detail = _change_outcomes(
            rows,
            graph,
            refined,
            threshold=20.0,
        )
        for source, transitions in outcomes.items():
            aggregate_outcomes.setdefault(source, Counter()).update(transitions)
        static_veto_count += len(static_vetoes)
        per_video[name] = {
            "frames": len(rows),
            "changes": len(changes),
            "change_detail_20px": change_detail,
            "10px": {
                "current": _metrics(rows, "current_point", 10.0),
                "graph": _metrics(rows, "graph_point", 10.0),
                "refined": _metrics(rows, "refined_point", 10.0),
                "context_hybrid": _metrics(rows, "context_hybrid_point", 10.0),
            },
            "20px": {
                "current": _metrics(rows, "current_point", 20.0),
                "graph": _metrics(rows, "graph_point", 20.0),
                "refined": _metrics(rows, "refined_point", 20.0),
                "context_hybrid": _metrics(rows, "context_hybrid_point", 20.0),
            },
        }
    return {
        "videos": len(per_video),
        "frames": len(all_rows),
        "static_veto_frames": static_veto_count,
        "change_sources": dict(source_counts),
        "change_outcomes_20px": {
            source: dict(counter) for source, counter in aggregate_outcomes.items()
        },
        "metrics": {
            "10px": {
                "current": _metrics(all_rows, "current_point", 10.0),
                "graph": _metrics(all_rows, "graph_point", 10.0),
                "refined": _metrics(all_rows, "refined_point", 10.0),
                "context_hybrid": _metrics(all_rows, "context_hybrid_point", 10.0),
            },
            "20px": {
                "current": _metrics(all_rows, "current_point", 20.0),
                "graph": _metrics(all_rows, "graph_point", 20.0),
                "refined": _metrics(all_rows, "refined_point", 20.0),
                "context_hybrid": _metrics(all_rows, "context_hybrid_point", 20.0),
            },
        },
        "per_video": per_video,
    }


def _labeled_metadata() -> tuple[dict[str, Any], dict[str, Any]]:
    first = json.loads(
        (ROOT / "outputs" / "trajectory_filter_tuning" / "cache" / "cache_meta.json").read_text(
            encoding="utf-8"
        )
    )
    third = json.loads(
        (
            ROOT
            / "outputs"
            / "trajectory_new_dataset_experiment"
            / "cache"
            / "cache_meta.json"
        ).read_text(encoding="utf-8")
    )
    datasets = {**first["datasets"], **third["datasets"]}
    return datasets, first


def _lifecycle_start_backfill(
    current: Sequence[dict[str, Any]],
    *,
    active_start: int,
    active_end: int,
    width: int,
    height: int,
) -> tuple[dict[int, Point], dict[str, Any]]:
    result = fit_known_rally_start(
        [row["track"] for row in current],
        active_start=active_start,
        active_end=active_end,
        width=width,
        height=height,
    )
    return result.points, result.debug


def _labeled_scope(
    dataset_dir: Path,
    refinement_config: CandidateGraphRefinementConfig,
) -> dict[str, Any]:
    datasets, court_meta = _labeled_metadata()
    court_prediction = {"valid": True, "corners": court_meta["court_corners"]}
    combined_rows: list[dict[str, Any]] = []
    per_dataset: dict[str, Any] = {}
    source_counts: Counter[str] = Counter()
    for name in ("1", "2", "10-1"):
        meta = datasets[name]
        fps = float(meta["fps"])
        width = int(meta["width"])
        height = int(meta["height"])
        cache_rows = _load_cache(Path(meta["cache"]))
        visible_gt, _ = _load_annotations(dataset_dir / f"{name}.xml")
        frames, graph, refined, static_vetoes, current = _run_methods(
            cache_rows,
            fps=fps,
            width=width,
            height=height,
            refinement_config=refinement_config,
            court_prediction=court_prediction,
        )
        start, end = LABELED_RANGES[name]
        lifecycle_backfill, lifecycle_debug = _lifecycle_start_backfill(
            current,
            active_start=start,
            active_end=end,
            width=width,
            height=height,
        )
        rows = [
            {
                "frame_id": frame_id,
                "gt_point": visible_gt.get(frame_id),
                "graph_point": _point(graph[frame_id]),
                "refined_point": _point(refined[frame_id]),
                "current_point": _point_from_track(current[frame_id]["track"]),
                "context_hybrid_point": (
                    _point_from_track(current[frame_id]["track"])
                    if _point_from_track(current[frame_id]["track"]) is not None
                    else _point(graph[frame_id])
                ),
                "context_lifecycle_point": lifecycle_backfill.get(
                    frame_id,
                    _point_from_track(current[frame_id]["track"]),
                ),
            }
            for frame_id in range(len(cache_rows))
            if start <= frame_id <= end
        ]
        combined_rows.extend(rows)
        failure_detail: list[dict[str, Any]] = []
        for row in rows:
            frame_id = int(row["frame_id"])
            gt = row["gt_point"]
            graph_point = row["graph_point"]
            if gt is None or (graph_point is not None and math.dist(gt, graph_point) <= 20.0):
                continue
            candidates = [
                (rank, candidate, math.dist(gt, point))
                for rank, candidate in enumerate(frames[frame_id], start=1)
                if (point := _point_from_track(candidate)) is not None
            ]
            best = min(candidates, key=lambda item: item[2]) if candidates else None
            failure_detail.append(
                {
                    "frame_id": frame_id,
                    "kind": "missing" if graph_point is None else "drift",
                    "best_candidate_rank": best[0] if best else 0,
                    "best_candidate_score": float(best[1].score) if best else 0.0,
                    "best_candidate_error_px": best[2] if best else None,
                    "correct_candidate_available": bool(best and best[2] <= 20.0),
                    "previous_graph_visible": frame_id > 0 and _point(graph[frame_id - 1]) is not None,
                    "next_graph_visible": frame_id + 1 < len(graph)
                    and _point(graph[frame_id + 1]) is not None,
                    "current_visible": _point_from_track(current[frame_id]["track"])
                    is not None,
                    "current_error_px": (
                        math.dist(gt, _point_from_track(current[frame_id]["track"]) or gt)
                        if _point_from_track(current[frame_id]["track"]) is not None
                        else None
                    ),
                    "current_source": current[frame_id]["source"],
                    "current_action": current[frame_id]["action"],
                    "current_reason": current[frame_id]["reason"],
                }
            )
        changes = _changed_frames(graph, refined)
        source_counts.update(refined[frame_id].source for frame_id in changes)
        per_dataset[name] = {
            "fps": fps,
            "active_start": start,
            "active_end": end,
            "static_veto_frames": sorted(static_vetoes),
            "changes_full_video": changes,
            "failures_20px": failure_detail,
            "lifecycle_start_backfill": lifecycle_debug,
            "15px": {
                "current": _metrics(rows, "current_point", 15.0),
                "graph": _metrics(rows, "graph_point", 15.0),
                "refined": _metrics(rows, "refined_point", 15.0),
                "context_hybrid": _metrics(rows, "context_hybrid_point", 15.0),
                "context_lifecycle": _metrics(
                    rows, "context_lifecycle_point", 15.0
                ),
            },
            "20px": {
                "current": _metrics(rows, "current_point", 20.0),
                "graph": _metrics(rows, "graph_point", 20.0),
                "refined": _metrics(rows, "refined_point", 20.0),
                "context_hybrid": _metrics(rows, "context_hybrid_point", 20.0),
                "context_lifecycle": _metrics(
                    rows, "context_lifecycle_point", 20.0
                ),
            },
            "pre_fp": {
                "graph": sum(_point(graph[index]) is not None for index in range(start)),
                "refined": sum(_point(refined[index]) is not None for index in range(start)),
            },
            "post_fp": {
                "graph": sum(_point(graph[index]) is not None for index in range(end + 1, len(graph))),
                "refined": sum(_point(refined[index]) is not None for index in range(end + 1, len(refined))),
            },
        }
    return {
        "datasets": per_dataset,
        "change_sources": dict(source_counts),
        "combined": {
            "15px": {
                "current": _metrics(combined_rows, "current_point", 15.0),
                "graph": _metrics(combined_rows, "graph_point", 15.0),
                "refined": _metrics(combined_rows, "refined_point", 15.0),
                "context_hybrid": _metrics(
                    combined_rows, "context_hybrid_point", 15.0
                ),
                "context_lifecycle": _metrics(
                    combined_rows, "context_lifecycle_point", 15.0
                ),
            },
            "20px": {
                "current": _metrics(combined_rows, "current_point", 20.0),
                "graph": _metrics(combined_rows, "graph_point", 20.0),
                "refined": _metrics(combined_rows, "refined_point", 20.0),
                "context_hybrid": _metrics(
                    combined_rows, "context_hybrid_point", 20.0
                ),
                "context_lifecycle": _metrics(
                    combined_rows, "context_lifecycle_point", 20.0
                ),
            },
        },
    }


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _csv_point(row: dict[str, str], prefix: str) -> Point | None:
    if row[f"{prefix}_visible"] != "1":
        return None
    return float(row[f"{prefix}_x"]), float(row[f"{prefix}_y"])


def _matches(point: Point | None, expected: Point | None, threshold: float = 20.0) -> bool:
    if point is None or expected is None:
        return point is None and expected is None
    return math.dist(point, expected) <= threshold


def _bd19_scope(
    cache_path: Path,
    per_frame_path: Path,
    manifest_path: Path,
    refinement_config: CandidateGraphRefinementConfig,
) -> dict[str, Any]:
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    identity = payload["identity"]
    cache_rows = [
        {"frame_id": frame_id, "candidates": candidates}
        for frame_id, candidates in enumerate(payload["frames"])
    ]
    _, graph, refined, static_vetoes, _ = _run_methods(
        cache_rows,
        fps=float(identity["fps"]),
        width=int(identity["width"]),
        height=int(identity["height"]),
        refinement_config=refinement_config,
    )
    comparison = {int(row["frame_id"]): row for row in _csv_rows(per_frame_path)}
    manifest = _csv_rows(manifest_path)
    detail: list[dict[str, Any]] = []
    for item in manifest:
        frame_id = int(item["frame_id"])
        verdict = item["verdict"]
        if verdict not in {"current", "graph"}:
            raise ValueError(f"unadjudicated BD19 frame: {frame_id}")
        expected = _csv_point(comparison[frame_id], verdict)
        base_point = _point(graph[frame_id])
        refined_point = _point(refined[frame_id])
        detail.append(
            {
                "review_index": int(item["review_index"]),
                "frame_id": frame_id,
                "verdict": verdict,
                "graph_correct": _matches(base_point, expected),
                "refined_correct": _matches(refined_point, expected),
                "refined_source": refined[frame_id].source,
            }
        )
    changes = _changed_frames(graph, refined)
    return {
        "adjudicated_frames": len(detail),
        "graph_correct": sum(row["graph_correct"] for row in detail),
        "refined_correct": sum(row["refined_correct"] for row in detail),
        "graph_preferred_preserved": sum(
            row["refined_correct"] for row in detail if row["verdict"] == "graph"
        ),
        "graph_preferred_total": sum(row["verdict"] == "graph" for row in detail),
        "current_preferred_recovered": sum(
            row["refined_correct"] for row in detail if row["verdict"] == "current"
        ),
        "current_preferred_total": sum(row["verdict"] == "current" for row in detail),
        "remaining_failures": [row["frame_id"] for row in detail if not row["refined_correct"]],
        "static_veto_frames": sorted(static_vetoes),
        "changes_full_video": changes,
        "change_sources": dict(Counter(refined[frame_id].source for frame_id in changes)),
        "adjudication": detail,
    }


def main() -> None:
    args = _parse_args()
    scopes = args.scope or ["raw_data", "raw_data2", "labeled", "bd19"]
    config = CandidateGraphRefinementConfig()
    _apply_overrides(config, args.set)
    summary: dict[str, Any] = {
        "graph_config": asdict(
            CandidateGraphConfig(delay_ms=300, beam_width=64, score_center=0.72)
        ),
        "refinement_config": asdict(config),
        "config_overrides": list(args.set),
        "scopes": scopes,
    }
    if "raw_data" in scopes or "raw_data2" in scopes:
        official_meta = json.loads(
            (args.official_cache / "cache_meta.json").read_text(encoding="utf-8")
        )
        summary["official"] = {}
        for split in ("raw_data", "raw_data2"):
            if split in scopes:
                summary["official"][split] = _official_scope(split, official_meta, config)
    if "labeled" in scopes:
        summary["labeled"] = _labeled_scope(args.dataset_dir, config)
    if "bd19" in scopes:
        summary["bd19"] = _bd19_scope(
            args.bd19_cache,
            args.bd19_per_frame,
            args.bd19_manifest,
            config,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
