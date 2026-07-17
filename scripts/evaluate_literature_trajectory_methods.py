from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_labeled_trajectory_filter import (  # noqa: E402
    _distance,
    _load_annotations,
    _load_cache,
    _point_from_track,
    _track_from_cache,
    _write_csv,
)
from src.postprocess.track_filter import BallTrackFilter, BallTrackFilterConfig  # noqa: E402


Point = tuple[float, float]
MEASURED_ACTIONS = {"accept", "bootstrap_accept", "relock_accept"}
DEFAULT_DELAY_MS = (0, 40, 80, 100, 160, 240, 400, 600, 800, 1000)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate literature-inspired offline trajectory repairs on cached TrackNet candidates.",
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
    )
    parser.add_argument(
        "--delay-ms",
        action="append",
        type=int,
        default=[],
        help="Fixed output delay to evaluate in milliseconds; may be repeated.",
    )
    return parser.parse_args()


def _point(track: Any) -> Point | None:
    value = _point_from_track(track)
    return None if value is None else (float(value[0]), float(value[1]))


def _same_point(a: Point | None, b: Point | None, tolerance: float = 1e-6) -> bool:
    if a is None or b is None:
        return a is b
    return _distance(a, b) <= tolerance


def _vector(a: Point, b: Point, frames: int = 1) -> Point:
    scale = 1.0 / max(1, frames)
    return ((b[0] - a[0]) * scale, (b[1] - a[1]) * scale)


def _add(point: Point, velocity: Point, frames: int = 1) -> Point:
    return (point[0] + velocity[0] * frames, point[1] + velocity[1] * frames)


def _cosine(a: Point, b: Point) -> float:
    denominator = math.hypot(*a) * math.hypot(*b)
    if denominator <= 1e-9:
        return 1.0
    return max(-1.0, min(1.0, (a[0] * b[0] + a[1] * b[1]) / denominator))


def _candidate_rows(cache_row: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for value in cache_row.get("candidates", []):
        track = _track_from_cache(value)
        point = _point(track)
        if point is not None:
            candidates.append({"point": point, "score": float(track.score)})
    return candidates


def replay_baseline(
    name: str,
    *,
    dataset_dir: Path,
    cache_dir: Path,
    cache_meta: dict[str, Any],
    active_range: tuple[int, int],
) -> list[dict[str, Any]]:
    metadata = cache_meta["datasets"][name]
    fps = float(metadata["fps"])
    width = int(metadata["width"])
    height = int(metadata["height"])
    visible_gt, annotated_frames = _load_annotations(dataset_dir / f"{name}.xml")
    track_filter = BallTrackFilter(BallTrackFilterConfig(fps=fps), debug_enabled=True)
    court_prediction = {"valid": True, "corners": cache_meta["court_corners"]}
    rows: list[dict[str, Any]] = []
    for cache_row in _load_cache(cache_dir / f"{name}_model_cache.jsonl"):
        frame_id = int(cache_row["frame_id"])
        candidates = [_track_from_cache(value) for value in cache_row.get("candidates", [])]
        output = track_filter.update_candidates(
            candidates,
            dt=1.0 / fps,
            frame_shape=(height, width, 3),
            court_prediction=court_prediction,
            person_bboxes=[
                tuple(float(component) for component in bbox)
                for bbox in cache_row.get("person_bboxes", [])
            ],
        )
        debug = track_filter.last_debug_record() or {}
        output_point = _point(output)
        rows.append(
            {
                "frame_id": frame_id,
                "gt_point": visible_gt.get(frame_id),
                "annotated": frame_id in annotated_frames,
                "point": output_point,
                "baseline_point": output_point,
                "score": float(output.score),
                "action": str(debug.get("action", "")),
                "reason": str(debug.get("reason", "")),
                "source": "baseline",
                "candidates": _candidate_rows(cache_row),
                "active": active_range[0] <= frame_id <= active_range[1],
                "active_start": active_range[0],
                "active_end": active_range[1],
            }
        )
    return rows


def _is_reliable_measurement(row: dict[str, Any], min_score: float = 0.50) -> bool:
    return (
        row["point"] is not None
        and row["action"] in MEASURED_ACTIONS
        and float(row["score"]) >= min_score
    )


def apply_hit_aware_short_gap(
    rows: list[dict[str, Any]],
    max_gap_frames: int = 2,
) -> list[dict[str, Any]]:
    repaired = copy.deepcopy(rows)
    gap_limit = max(0, int(max_gap_frames))
    index = 1
    while index < len(repaired) - 1:
        if repaired[index]["point"] is not None:
            index += 1
            continue
        start = index
        while index < len(repaired) and repaired[index]["point"] is None:
            index += 1
        end = index - 1
        gap_length = end - start + 1
        left_index = start - 1
        right_index = end + 1
        if gap_length > gap_limit or right_index >= len(repaired):
            continue
        left = repaired[left_index]
        right = repaired[right_index]
        low_motion_single_gap = (
            gap_length == 1
            and _is_reliable_measurement(left, 0.45)
            and _is_reliable_measurement(right, 0.45)
            and left["point"] is not None
            and right["point"] is not None
            and _distance(left["point"], right["point"]) <= 12.0
        )
        if not low_motion_single_gap and not (
            _is_reliable_measurement(left, 0.55)
            and _is_reliable_measurement(right, 0.55)
        ):
            continue
        if any("ground_bounce" in repaired[pos]["reason"] for pos in range(start, right_index + 1)):
            continue
        left_point = left["point"]
        right_point = right["point"]
        assert left_point is not None and right_point is not None
        crossing = _vector(left_point, right_point, gap_length + 1)
        if not low_motion_single_gap:
            if left_index < 1 or not _is_reliable_measurement(repaired[left_index - 1], 0.50):
                continue
            before = repaired[left_index - 1]["point"]
            assert before is not None
            incoming = _vector(before, left_point)
            if math.degrees(math.acos(_cosine(incoming, crossing))) >= 100.0:
                continue
        if math.hypot(*crossing) > 80.0:
            continue
        for offset, pos in enumerate(range(start, right_index), start=1):
            alpha = offset / (gap_length + 1)
            repaired[pos]["point"] = (
                left_point[0] * (1.0 - alpha) + right_point[0] * alpha,
                left_point[1] * (1.0 - alpha) + right_point[1] * alpha,
            )
            repaired[pos]["score"] = min(float(left["score"]), float(right["score"])) * 0.5
            repaired[pos]["source"] = "hit_aware_short_gap"
    return repaired


def apply_occlusion_relock(
    rows: list[dict[str, Any]],
    max_gap_frames: int = 3,
) -> list[dict[str, Any]]:
    repaired = copy.deepcopy(rows)
    gap_limit = max(0, int(max_gap_frames))
    index = 0
    while index < len(repaired):
        if repaired[index]["point"] is not None:
            index += 1
            continue
        start = index
        while index < len(repaired) and repaired[index]["point"] is None:
            index += 1
        end = index - 1
        if end - start + 1 > gap_limit or index >= len(repaired):
            continue
        context_start = max(0, start - 6)
        context_reasons = {row["reason"] for row in repaired[context_start:start]}
        if "person_occlusion_prediction" not in context_reasons:
            continue
        endpoint = repaired[index]
        if not _is_reliable_measurement(endpoint, 0.80):
            continue
        chain: list[dict[str, Any]] = []
        previous: Point | None = None
        valid = True
        for pos in range(start, end + 1):
            candidates = [value for value in repaired[pos]["candidates"] if value["score"] >= 0.80]
            if not candidates:
                valid = False
                break
            if previous is None:
                chosen = max(candidates, key=lambda value: value["score"])
            else:
                chosen = min(candidates, key=lambda value: _distance(previous, value["point"]))
                if _distance(previous, chosen["point"]) > 85.0:
                    valid = False
                    break
            chain.append(chosen)
            previous = chosen["point"]
        if not valid or len(chain) < 2 or previous is None or endpoint["point"] is None:
            continue
        if _distance(previous, endpoint["point"]) > 85.0:
            continue
        chain_points = [value["point"] for value in chain] + [endpoint["point"]]
        velocities = [_vector(chain_points[pos - 1], chain_points[pos]) for pos in range(1, len(chain_points))]
        if any(_cosine(velocities[pos - 1], velocities[pos]) < 0.50 for pos in range(1, len(velocities))):
            continue
        occlusion_positions = [
            pos
            for pos in range(context_start, start)
            if repaired[pos]["reason"] == "person_occlusion_prediction"
        ]
        if occlusion_positions:
            stale_start = occlusion_positions[0]
            stale_reasons = {
                "person_occlusion_prediction",
                "person_occlusion_motion_gate",
                "person_occlusion_candidate_high_score",
            }
            if all(repaired[pos]["reason"] in stale_reasons for pos in range(stale_start, start)):
                for pos in range(stale_start, start):
                    repaired[pos]["point"] = None
                    repaired[pos]["score"] = 0.0
                    repaired[pos]["source"] = "occlusion_stale_branch_removed"
        for pos, chosen in zip(range(start, end + 1), chain):
            repaired[pos]["point"] = chosen["point"]
            repaired[pos]["score"] = chosen["score"]
            repaired[pos]["source"] = "occlusion_model_reset"
    return repaired


def _last_two_reliable(rows: list[dict[str, Any]], before: int) -> tuple[int, Point, int, Point] | None:
    found: list[tuple[int, Point]] = []
    for pos in range(before - 1, max(-1, before - 8), -1):
        if _is_reliable_measurement(rows[pos], 0.50):
            point = rows[pos]["point"]
            assert point is not None
            found.append((pos, point))
            if len(found) == 2:
                return found[1][0], found[1][1], found[0][0], found[0][1]
    return None


def apply_fixed_lag_branch_recovery(
    rows: list[dict[str, Any]],
    max_future_frames: int = 5,
) -> list[dict[str, Any]]:
    repaired = copy.deepcopy(rows)
    future_limit = max(0, int(max_future_frames))
    if future_limit < 3:
        return repaired
    index = 2
    while index < len(repaired) - 1:
        current = repaired[index]
        history = _last_two_reliable(repaired, index)
        if current["point"] is None or history is None:
            index += 1
            continue
        first_index, first_point, last_index, last_point = history
        velocity = _vector(first_point, last_point, last_index - first_index)
        predicted = _add(last_point, velocity, index - last_index)
        if _distance(current["point"], predicted) <= 32.0:
            index += 1
            continue

        first_future = index + 1
        predicted_future = _add(last_point, velocity, first_future - last_index)
        future_candidates = [
            value
            for value in repaired[first_future]["candidates"]
            if value["score"] >= 0.40 and _distance(value["point"], predicted_future) <= 22.0
        ]
        if not future_candidates:
            index += 1
            continue
        chosen = min(future_candidates, key=lambda value: _distance(value["point"], predicted_future))
        if repaired[first_future]["point"] is None or _distance(repaired[first_future]["point"], chosen["point"]) <= 24.0:
            index += 1
            continue

        chain: list[tuple[int, dict[str, Any]]] = [(first_future, chosen)]
        previous_point = last_point
        previous_index = last_index
        chain_velocity = _vector(previous_point, chosen["point"], first_future - previous_index)
        rejoin_index: int | None = None
        for pos in range(
            first_future + 1,
            min(len(repaired), index + future_limit + 1),
        ):
            predicted_chain = _add(chain[-1][1]["point"], chain_velocity)
            baseline_point = repaired[pos]["point"]
            if baseline_point is not None and _distance(baseline_point, predicted_chain) <= 15.0:
                rejoin_index = pos
                break
            candidates = [value for value in repaired[pos]["candidates"] if value["score"] >= 0.35]
            if not candidates:
                break
            next_candidate = min(candidates, key=lambda value: _distance(value["point"], predicted_chain))
            if _distance(next_candidate["point"], predicted_chain) > 24.0:
                break
            new_velocity = _vector(chain[-1][1]["point"], next_candidate["point"])
            if (
                math.hypot(*chain_velocity) > 12.0
                and math.hypot(*new_velocity) > 12.0
                and _cosine(chain_velocity, new_velocity) < -0.20
            ):
                break
            chain.append((pos, next_candidate))
            chain_velocity = new_velocity
        if rejoin_index is None or len(chain) < 2:
            index += 1
            continue

        incoming = velocity
        recovered = _vector(last_point, chain[0][1]["point"], first_future - last_index)
        if math.degrees(math.acos(_cosine(incoming, recovered))) >= 100.0:
            index += 1
            continue

        alpha = (index - last_index) / (first_future - last_index)
        repaired[index]["point"] = (
            last_point[0] * (1.0 - alpha) + chain[0][1]["point"][0] * alpha,
            last_point[1] * (1.0 - alpha) + chain[0][1]["point"][1] * alpha,
        )
        repaired[index]["score"] = min(float(repaired[last_index]["score"]), chain[0][1]["score"]) * 0.5
        repaired[index]["source"] = "branch_outlier_interpolation"
        for pos, value in chain:
            repaired[pos]["point"] = value["point"]
            repaired[pos]["score"] = value["score"]
            repaired[pos]["source"] = "fixed_lag_branch_recovery"
        index = rejoin_index
    return repaired


def _metric(rows: list[dict[str, Any]], threshold: float, full_video: bool) -> dict[str, Any]:
    selected = rows if full_video else [row for row in rows if row["active"]]
    gt_positive = sum(row["gt_point"] is not None for row in selected if row["active"])
    pred_visible = sum(row["point"] is not None for row in selected)
    correct = sum(
        row["active"]
        and row["gt_point"] is not None
        and row["point"] is not None
        and _distance(row["point"], row["gt_point"]) <= threshold
        for row in selected
    )
    precision = correct / pred_visible if pred_visible else 0.0
    recall = correct / gt_positive if gt_positive else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    active_rows = [row for row in rows if row["active"]]
    true_negative = sum(row["gt_point"] is None and row["point"] is None for row in active_rows)
    accuracy = (correct + true_negative) / len(active_rows) if active_rows else 0.0
    return {
        "correct": correct,
        "gt_positive": gt_positive,
        "pred_visible": pred_visible,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "missing": sum(row["gt_point"] is not None and row["point"] is None for row in active_rows),
        "drift": sum(
            row["gt_point"] is not None
            and row["point"] is not None
            and _distance(row["point"], row["gt_point"]) > threshold
            for row in active_rows
        ),
        "severe_drift": sum(
            row["gt_point"] is not None
            and row["point"] is not None
            and _distance(row["point"], row["gt_point"]) > 50.0
            for row in active_rows
        ),
        "pre_fp": sum(row["frame_id"] < row["active_start"] and row["point"] is not None for row in rows),
        "post_fp": sum(row["frame_id"] > row["active_end"] and row["point"] is not None for row in rows),
    }


def _summarize_variant(rows: list[dict[str, Any]], threshold: float, paper_threshold: float) -> dict[str, Any]:
    active_rows = [row for row in rows if row["active"]]
    jumps = 0
    previous: Point | None = None
    for row in rows:
        if row["point"] is None:
            previous = None
            continue
        if previous is not None and _distance(previous, row["point"]) > 110.0:
            jumps += 1
        previous = row["point"]
    return {
        "active": _metric(rows, threshold, full_video=False),
        "full_video": _metric(rows, threshold, full_video=True),
        "paper_scaled_4px": _metric(rows, paper_threshold, full_video=False),
        "changed_frames": sum(not _same_point(row["point"], row["baseline_point"]) for row in rows),
        "changed_active_frames": sum(
            row["active"] and not _same_point(row["point"], row["baseline_point"]) for row in rows
        ),
        "jumps_over_110px": jumps,
        "sources": {
            source: sum(row["source"] == source for row in active_rows)
            for source in sorted({row["source"] for row in active_rows})
        },
    }


def _oracle(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    visible = [row for row in rows if row["active"] and row["gt_point"] is not None]
    raw = sum(
        any(_distance(candidate["point"], row["gt_point"]) <= threshold for candidate in row["candidates"])
        for row in visible
    )
    union = sum(
        (
            row["baseline_point"] is not None
            and _distance(row["baseline_point"], row["gt_point"]) <= threshold
        )
        or any(_distance(candidate["point"], row["gt_point"]) <= threshold for candidate in row["candidates"])
        for row in visible
    )

    def ideal(count: int) -> dict[str, float | int]:
        recall = count / len(visible)
        return {
            "correct": count,
            "recall": recall,
            "ideal_precision_1_f1": 2.0 * recall / (1.0 + recall),
        }

    return {"gt_positive": len(visible), "raw_candidate": ideal(raw), "raw_union_baseline": ideal(union)}


def _per_frame_rows(rows_by_variant: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    baseline = rows_by_variant["baseline"]
    output: list[dict[str, Any]] = []
    for index, row in enumerate(baseline):
        record: dict[str, Any] = {
            "frame_id": row["frame_id"],
            "active": row["active"],
            "gt_visible": row["gt_point"] is not None,
            "gt_x": row["gt_point"][0] if row["gt_point"] else None,
            "gt_y": row["gt_point"][1] if row["gt_point"] else None,
            "action": row["action"],
            "reason": row["reason"],
        }
        for name, variant_rows in rows_by_variant.items():
            value = variant_rows[index]
            point = value["point"]
            record[f"{name}_visible"] = point is not None
            record[f"{name}_x"] = point[0] if point else None
            record[f"{name}_y"] = point[1] if point else None
            record[f"{name}_source"] = value["source"]
            record[f"{name}_error_px"] = _distance(point, row["gt_point"]) if point and row["gt_point"] else None
        output.append(record)
    return output


def _percent(value: float) -> str:
    return f"{value * 100.0:.2f}%"


def _write_report(path: Path, summary: dict[str, Any], threshold: float) -> None:
    variants = summary["combined"]["variants"]
    macro_active = summary["combined"]["macro_active"]
    lines = [
        "# 文献方法轨迹后处理验证报告",
        "",
        "- 日期：2026-07-15",
        f"- 项目协议：原始 1920x1080 坐标，正确阈值 {threshold:g}px，只比较人工定义的有效飞行区间",
        "- 论文近似阈值：TrackNetV3 的 512x288 坐标 4px 等比例换算为原图 15px",
        "- 所有轨迹修复均不读取 XML 真值；真值只用于最终评分",
        "- 生产滤波器未修改，本报告为固定缓存上的离线实验",
        "",
        "## 结果",
        "",
        "| 方法 | Precision | Recall | 微平均 F1 | 宏平均 F1 | 正确 | 缺失 | 漂移 | 改动帧 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, value in variants.items():
        active = value["active"]
        lines.append(
            f"| {name} | {_percent(active['precision'])} | {_percent(active['recall'])} | "
            f"{_percent(active['f1'])} | {_percent(macro_active[name]['f1'])} | "
            f"{active['correct']} | {active['missing']} | "
            f"{active['drift']} | {value['changed_active_frames']} |"
        )
    lines.extend(
        [
            "",
            "### 分数据集与完整视频",
            "",
            "| 数据集 | 方法 | 有效 F1 | 完整视频 F1 | 正确 | 缺失 | 漂移 | 回合前/后输出 |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset_name, dataset_summary in summary["datasets"].items():
        for variant_name in ("baseline", "combined"):
            value = dataset_summary["variants"][variant_name]
            active = value["active"]
            full_video = value["full_video"]
            lines.append(
                f"| Dataset/{dataset_name} | {variant_name} | {_percent(active['f1'])} | "
                f"{_percent(full_video['f1'])} | {active['correct']} | {active['missing']} | "
                f"{active['drift']} | {full_video['pre_fp']} / {full_video['post_fp']} |"
            )
    paper_target = float(summary["paper_target_f1"])
    best_name = max(variants, key=lambda name: variants[name]["active"]["f1"])
    best_f1 = float(variants[best_name]["active"]["f1"])
    lines.extend(
        [
            "",
            "## 论文水平判断",
            "",
            f"TrackNetV3 论文报告 F1 为 {_percent(paper_target)}。本实验最佳方法为 `{best_name}`，"
            f"项目协议微平均 F1 {_percent(best_f1)}、宏平均 F1 {_percent(macro_active[best_name]['f1'])}，"
            f"微平均相差 {(paper_target - best_f1) * 100.0:.2f} 个百分点。",
            f"该方法在等比例 15px 阈值下 F1 为 {_percent(variants[best_name]['paper_scaled_4px']['f1'])}；"
            f"Accuracy/Precision/Recall 为 {_percent(variants[best_name]['active']['accuracy'])}/"
            f"{_percent(variants[best_name]['active']['precision'])}/"
            f"{_percent(variants[best_name]['active']['recall'])}。论文相应数值为 97.51%/97.79%/99.33%。",
            "两者数据集、分辨率和评估区间不同，因此只能判断当前验证集是否达到该数值，不能作为严格复现结论。",
            "",
            "## 候选上限",
            "",
            f"20px 下原始候选 oracle 为 {summary['combined']['oracle']['raw_candidate']['correct']}/"
            f"{summary['combined']['oracle']['gt_positive']}；原始候选与现有时序输出的并集 oracle 为 "
            f"{summary['combined']['oracle']['raw_union_baseline']['correct']}/"
            f"{summary['combined']['oracle']['gt_positive']}。候选选择本身无法补出没有峰值的帧，"
            "只有受约束的轨迹合成或重新运行/训练检测器才可能继续提高召回。",
            "",
            "## 方法对应关系",
            "",
            "- `hit_aware_short_gap`：对应 TrackNetV3 的缺失掩码修复，但仅修复双端可靠、最多 2 帧且不跨击球反转的缺口。",
            "- `occlusion_relock`：对应 Shishido 的击球附近模型切换与重新锁定，只在人物遮挡上下文和连续高分候选链成立时启用。",
            "- `fixed_lag_branch`：对应 Kopania 的预测门控和短时多假设思想，用未来 3 帧证据撤销单帧异常分支。",
            "- `combined`：依次执行短时分支恢复、短缺口修复和遮挡模型重置。",
            "",
            "## 本地文献",
            "",
            "- Chen & Wang (2023), *TrackNetV3: Enhancing ShuttleCock Tracking with Augmentations and Trajectory Rectification*，Zotero `WXASWY6D`，PDF `AHH4SR4I`。",
            "- Liu & Wang (2022), *MonoTrack: Shuttle Trajectory Reconstruction from Monocular Badminton Video*，Zotero `WSDQR63M`，PDF `GDQ49WKF`。",
            "- Kopania et al. (2022), *Automatic Shuttlecock Fall Detection System in or out of a Court in Badminton Games*，Zotero `GX3REWMU`，PDF `UB38DACL`。",
            "- Shishido et al. (2017), *Visual Tracking Method of a Quick and Anomalously Moving Badminton Shuttlecock*，Zotero `SDHNQ5GR`，PDF `H9JLX7SG`。",
            "",
            "## 限制",
            "",
            "当前只有两段、同分辨率且相近固定机位的数据。组合方法额外增加 3 帧固定滞后，且短缺口修复在 Dataset/1 回合前增加 1 个输出；在跨场馆数据复测前，不应直接替换实时生产滤波器。",
            "TrackNetV3 论文的 Accuracy 包含其测试集上的真负样本，本报告的 Accuracy 只在人工有效区间内按同类公式计算，因此不是严格同协议复现。",
            "",
            "逐帧改动见每个数据集的 `per_frame.csv`，精确指标见 `summary.json`。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    delay_values_ms = sorted(set(args.delay_ms or DEFAULT_DELAY_MS))
    if any(value < 0 or value > 1000 for value in delay_values_ms):
        raise ValueError("--delay-ms must be between 0 and 1000")
    active_ranges: dict[str, tuple[int, int]] = {}
    for value in args.active_range:
        name, start, end = value.split(":", maxsplit=2)
        active_ranges[name] = (int(start), int(end))
    cache_meta = json.loads((args.cache_dir / "cache_meta.json").read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_summaries: dict[str, Any] = {}
    combined_rows: dict[str, list[dict[str, Any]]] = {}
    for name in args.datasets:
        if name not in active_ranges:
            raise ValueError(f"Missing --active-range for dataset {name}")
        fps = float(cache_meta["datasets"][name]["fps"])
        baseline = replay_baseline(
            name,
            dataset_dir=args.dataset_dir,
            cache_dir=args.cache_dir,
            cache_meta=cache_meta,
            active_range=active_ranges[name],
        )
        fixed_lag = apply_fixed_lag_branch_recovery(baseline)
        short_gap = apply_hit_aware_short_gap(baseline)
        occlusion = apply_occlusion_relock(baseline)
        combined = apply_occlusion_relock(apply_hit_aware_short_gap(fixed_lag))
        variants = {
            "baseline": baseline,
            "hit_aware_short_gap": short_gap,
            "occlusion_relock": occlusion,
            "fixed_lag_branch": fixed_lag,
            "combined": combined,
        }
        delay_frames: dict[str, int] = {}
        for delay_ms in delay_values_ms:
            variant_name = f"delay_{delay_ms}ms"
            lag_frames = max(0, int(round(fps * delay_ms / 1000.0)))
            delay_frames[variant_name] = lag_frames
            if lag_frames == 0:
                variants[variant_name] = copy.deepcopy(baseline)
                continue
            delayed = apply_fixed_lag_branch_recovery(
                baseline,
                max_future_frames=lag_frames,
            )
            delayed = apply_hit_aware_short_gap(
                delayed,
                max_gap_frames=lag_frames,
            )
            variants[variant_name] = apply_occlusion_relock(
                delayed,
                max_gap_frames=lag_frames,
            )
        width = float(cache_meta["datasets"][name]["width"])
        paper_threshold = 4.0 * width / 512.0
        summary = {
            "active_range": active_ranges[name],
            "project_threshold_px": float(args.distance_threshold),
            "paper_scaled_threshold_px": paper_threshold,
            "delay_frames": delay_frames,
            "oracle": _oracle(baseline, float(args.distance_threshold)),
            "variants": {
                variant: _summarize_variant(rows, float(args.distance_threshold), paper_threshold)
                for variant, rows in variants.items()
            },
        }
        dataset_dir = args.output_dir / name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(dataset_dir / "per_frame.csv", _per_frame_rows(variants))
        (dataset_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        dataset_summaries[name] = summary
        for variant, rows in variants.items():
            combined_rows.setdefault(variant, []).extend(rows)

    combined_oracle_rows = combined_rows["baseline"]
    combined_summary = {
        "oracle": _oracle(combined_oracle_rows, float(args.distance_threshold)),
        "variants": {
            variant: _summarize_variant(
                rows,
                float(args.distance_threshold),
                4.0 * float(cache_meta["datasets"][args.datasets[0]]["width"]) / 512.0,
            )
            for variant, rows in combined_rows.items()
        },
        "macro_active": {
            variant: {
                metric: sum(
                    dataset_summaries[name]["variants"][variant]["active"][metric]
                    for name in args.datasets
                )
                / len(args.datasets)
                for metric in ("precision", "recall", "f1", "accuracy")
            }
            for variant in combined_rows
        },
    }
    result = {
        "paper_target_f1": 0.9856,
        "datasets": dataset_summaries,
        "combined": combined_summary,
    }
    output_path = args.output_dir / "combined_summary.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report(args.output_dir / "report.md", result, float(args.distance_threshold))
    print(output_path)


if __name__ == "__main__":
    main()
