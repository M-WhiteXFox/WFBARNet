from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.models.track_branch import TrackBranch
from src.utils.exporters import export_csv, export_json, export_npy
from src.utils.structures import FrameResult
from src.utils.visualize import draw_result


ProgressCallback = Callable[[dict[str, Any]], None]
StopCallback = Callable[[], bool]
TRACK_WEIGHT_PATH = PROJECT_ROOT / "assets" / "weights" / "track" / "model_best.pt"


@dataclass
class TrackTaskConfig:
    source: str
    output_dir: Path
    device: str = "auto"
    score_threshold: float = 0.50
    max_frames: int | None = None
    save_visualization: bool = True
    save_json: bool = True
    save_csv: bool = True
    save_npy: bool = True


@dataclass
class TrackTaskResult:
    summary: dict[str, Any]
    actions: list[dict[str, Any]]
    track_results: dict[int, dict[str, Any]]
    output_files: dict[str, str]
    last_frame: np.ndarray | None
    last_frame_idx: int
    total_frames: int
    fps: float
    stopped: bool
    progress: float


def _resolve_device(device: str) -> str:
    normalized = device.strip().lower()
    if normalized in {"", "auto"}:
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("当前环境不可用 CUDA，请切换到自动或 CPU。")
    if normalized == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None or not torch.backends.mps.is_available():
            raise RuntimeError("当前环境不可用 MPS，请切换到自动或 CPU。")
    return device


def _emit(progress_callback: ProgressCallback | None, payload: dict[str, Any]) -> None:
    if progress_callback is not None:
        progress_callback(payload)


def _build_summary(
    results: list[FrameResult],
    fps: float,
    output_dir: Path,
    duration_seconds: float,
) -> dict[str, Any]:
    total_frames = len(results)
    visible_results = [item for item in results if item.track.visible]
    all_scores = [item.track.score for item in results]
    visible_frames = len(visible_results)
    visibility_ratio = visible_frames / total_frames if total_frames else 0.0
    return {
        "model_name": "TrackNetV3",
        "total_frames": total_frames,
        "visible_frames": visible_frames,
        "visibility_ratio": visibility_ratio,
        "avg_confidence": sum(all_scores) / total_frames if total_frames else 0.0,
        "peak_confidence": max(all_scores, default=0.0),
        "inference_seconds": duration_seconds,
        "output_dir": str(output_dir),
        "current_action": "已输出轨迹结果" if total_frames else "暂无结果",
        "fps": fps,
    }


def _build_actions(results: list[FrameResult], fps: float) -> list[dict[str, Any]]:
    visible = [item for item in results if item.track.visible]
    if not visible:
        return []

    if len(visible) <= 8:
        selected = visible
    else:
        indexes = {round(i * (len(visible) - 1) / 7) for i in range(8)}
        selected = [visible[idx] for idx in sorted(indexes)]

    actions: list[dict[str, Any]] = []
    for idx, item in enumerate(selected, start=1):
        timestamp = item.frame_id / fps if fps > 0 else float(item.frame_id)
        x, y = item.track.ball_xy
        actions.append(
            {
                "id": idx,
                "frame_id": item.frame_id,
                "start_time": timestamp,
                "end_time": timestamp,
                "label": "关键轨迹点",
                "confidence": item.track.score,
                "detail": f"第 {item.frame_id} 帧，坐标 ({x:.0f}, {y:.0f})",
            }
        )
    return actions


def _export_results(config: TrackTaskConfig, results: list[FrameResult]) -> dict[str, str]:
    output_files: dict[str, str] = {}
    if config.save_json and results:
        path = config.output_dir / "track_results.json"
        export_json(results, path)
        output_files["JSON 结果"] = str(path)
    if config.save_csv and results:
        path = config.output_dir / "track_results.csv"
        export_csv(results, path)
        output_files["CSV 结果"] = str(path)
    if config.save_npy and results:
        path = config.output_dir / "track_results.npy"
        export_npy(results, path)
        output_files["NPY 结果"] = str(path)
    return output_files


def run_tracknet_task(
    config: TrackTaskConfig,
    progress_callback: ProgressCallback | None = None,
    stop_requested: StopCallback | None = None,
) -> TrackTaskResult:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    resolved_device = _resolve_device(config.device)

    _emit(progress_callback, {"type": "stage", "stage": "正在加载 TrackNetV3 模型权重", "progress": 0.03})
    branch = TrackBranch(
        model_weight=str(TRACK_WEIGHT_PATH),
        device=resolved_device,
        score_thr=config.score_threshold,
    )

    _emit(progress_callback, {"type": "stage", "stage": "正在打开视频并准备推理", "progress": 0.08})
    cap = cv2.VideoCapture(str(config.source))
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频文件：{config.source}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 25.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    expected_total = total_frames if total_frames > 0 else 0
    if config.max_frames is not None and config.max_frames > 0:
        expected_total = min(expected_total, config.max_frames) if expected_total else config.max_frames

    writer = None
    output_files: dict[str, str] = {}
    if config.save_visualization:
        vis_path = config.output_dir / "track_vis.mp4"
        writer = cv2.VideoWriter(
            str(vis_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (frame_width, frame_height),
        )
        output_files["可视化视频"] = str(vis_path)

    ok, first_frame = cap.read()
    if not ok:
        cap.release()
        if writer is not None:
            writer.release()
        raise RuntimeError("视频可以打开，但没有读取到任何帧。")

    ok, second_frame = cap.read()
    if not ok:
        second_frame = first_frame.copy()

    prev_frame = first_frame.copy()
    curr_frame = first_frame
    next_frame = second_frame
    frame_id = 0
    results: list[FrameResult] = []
    track_results: dict[int, dict[str, Any]] = {}
    last_frame: np.ndarray | None = None
    stopped = False
    start_time = time.perf_counter()

    while True:
        if stop_requested is not None and stop_requested():
            stopped = True
            break

        _, track = branch.infer([prev_frame, curr_frame, next_frame])
        frame_result = FrameResult(frame_id=frame_id, pose=[], track=track)
        results.append(frame_result)
        track_results[frame_id] = {
            "ball_xy": track.ball_xy,
            "visible": bool(track.visible),
            "score": track.score,
        }

        annotated_frame = draw_result(curr_frame, frame_result)
        last_frame = annotated_frame
        if writer is not None:
            writer.write(annotated_frame)

        processed = len(results)
        progress = processed / expected_total if expected_total else 0.0
        _emit(
            progress_callback,
            {
                "type": "frame",
                "stage": f"正在推理第 {processed} 帧",
                "frame_id": frame_id,
                "progress": min(progress, 0.98),
                "frame": annotated_frame,
                "track": track_results[frame_id],
            },
        )

        if config.max_frames is not None and config.max_frames > 0 and processed >= config.max_frames:
            break

        prev_frame = curr_frame
        curr_frame = next_frame
        ok, incoming = cap.read()
        if not ok:
            if frame_id > 0:
                frame_id += 1
                next_frame = curr_frame.copy()
                _, final_track = branch.infer([prev_frame, curr_frame, next_frame])
                final_result = FrameResult(frame_id=frame_id, pose=[], track=final_track)
                results.append(final_result)
                track_results[frame_id] = {
                    "ball_xy": final_track.ball_xy,
                    "visible": bool(final_track.visible),
                    "score": final_track.score,
                }
                last_frame = draw_result(curr_frame, final_result)
                if writer is not None and last_frame is not None:
                    writer.write(last_frame)
            break

        next_frame = incoming
        frame_id += 1

    cap.release()
    if writer is not None:
        writer.release()

    duration_seconds = time.perf_counter() - start_time
    output_files.update(_export_results(config, results))

    summary = _build_summary(results, fps, config.output_dir, duration_seconds)
    actions = _build_actions(results, fps)
    progress = len(results) / expected_total if expected_total else (1.0 if results else 0.0)
    if not stopped and results:
        progress = 1.0

    return TrackTaskResult(
        summary=summary,
        actions=actions,
        track_results=track_results,
        output_files=output_files,
        last_frame=last_frame,
        last_frame_idx=results[-1].frame_id if results else 0,
        total_frames=len(results),
        fps=fps,
        stopped=stopped,
        progress=min(progress, 1.0),
    )
