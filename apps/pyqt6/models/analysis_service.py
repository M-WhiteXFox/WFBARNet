from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from itertools import cycle
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from src.builders.bst_input_builder import BSTInputBuilder
from src.models.pose_branch import PoseBranch
from src.models.track_branch import TrackBranch
from src.postprocess.track_filter import BallTrackFilter
from src.preprocess.track import preprocess_track_batch
from src.utils.device import resolve_device
from src.utils.structures import FrameResult
from src.utils.video import iter_video_frame_windows

from apps.pyqt6.models.analysis_types import AnalysisAction, AnalysisResult


ProgressCallback = Callable[[dict[str, Any]], None]
StopCallback = Callable[[], bool]


@dataclass(slots=True)
class AnalysisConfig:
    source: str = ""
    output_dir: str = "outputs/pyqt6"
    device: str = "cpu"
    execution_mode: str = "serial"
    pose_backend: str = "mmpose"
    pose_config: str = "tools/mmpose/configs/rtmpose-s_8xb256-420e_coco-256x192.py"
    pose_weight: str = "assets/weights/pose/rtmpose-s_8xb256-420e_coco-256x192.pth"
    pose_bbox_mode: str = "whole_image"
    track_weight: str = "assets/weights/track/model_best.pt"
    save_json: bool = True
    save_csv: bool = True
    save_npy: bool = True
    save_vis: bool = False
    max_frames: int = 0
    pose_stride: int = 3  # 每隔 N 帧做一次 pose 推理，中间帧复用


class AnalysisService:
    def __init__(self, config_path: str | Path | None = None) -> None:
        self.project_root = Path(__file__).resolve().parents[3]
        self.config = self._load_config(config_path)
        self._pose_branch: PoseBranch | None = None
        self._track_branch: TrackBranch | None = None

    def _load_config(self, config_path: str | Path | None) -> AnalysisConfig:
        path = Path(config_path) if config_path is not None else self.project_root / "configs" / "default_infer.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            device = resolve_device(str(data.get("device", "auto")))
            return AnalysisConfig(
                source=str(data.get("source", "")),
                output_dir=str(data.get("output_dir", "outputs/run")),
                device=device,
                execution_mode=str(data.get("execution_mode", "serial")),
                pose_backend=str(data.get("pose_backend", "mmpose")),
                pose_config=str(data.get("pose_config", "")),
                pose_weight=str(data.get("pose_weight", "")),
                pose_bbox_mode=str(data.get("pose_bbox_mode", "whole_image")),
                track_weight=str(data.get("track_weight", "")),
                save_json=bool(data.get("save_json", True)),
                save_csv=bool(data.get("save_csv", True)),
                save_npy=bool(data.get("save_npy", True)),
                save_vis=bool(data.get("save_vis", False)),
                pose_stride=int(data.get("pose_stride", 3)),
            )
        return AnalysisConfig(device=resolve_device("auto"))

    def _resolve_path(self, raw_path: str | None) -> str | None:
        if not raw_path:
            return None
        path = Path(raw_path)
        if path.is_absolute():
            return str(path)
        candidate = self.project_root / path
        return str(candidate)

    def _build_pose_branch(self) -> PoseBranch:
        if self._pose_branch is None:
            self._pose_branch = PoseBranch(
                backend=self.config.pose_backend,
                device=self.config.device,
                model_config=self._resolve_path(self.config.pose_config),
                model_weight=self._resolve_path(self.config.pose_weight),
                bbox_mode=self.config.pose_bbox_mode,
            )
        return self._pose_branch

    def _build_track_branch(self) -> TrackBranch:
        if self._track_branch is None:
            self._track_branch = TrackBranch(
                model_weight=self._resolve_path(self.config.track_weight),
                device=self.config.device,
            )
        return self._track_branch

    def _probe_video(self, source: str) -> dict[str, Any]:
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            return {"fps": 0.0, "width": 0, "height": 0, "frame_count": 0}
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        return {"fps": fps, "width": width, "height": height, "frame_count": frame_count}

    def _emit(self, callback: ProgressCallback | None, payload: dict[str, Any]) -> None:
        if callback is not None:
            callback(payload)

    def _default_output_dir(self, source: Path) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base_dir = self._resolve_path(self.config.output_dir)
        if base_dir is None:
            base = self.project_root / "outputs" / "pyqt6"
        else:
            base = Path(base_dir)
        return base / source.stem / timestamp

    def analyze_video(
        self,
        video_path: str,
        progress_callback: ProgressCallback | None = None,
        stop_requested: StopCallback | None = None,
        output_dir: str | Path | None = None,
    ) -> AnalysisResult:
        source = Path(video_path)
        if not source.exists():
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        self._emit(progress_callback, {"type": "stage", "stage": "正在准备模型和配置...", "progress": 0.02})

        # 启用 cudnn benchmark（输入尺寸固定，可加速卷积）
        import torch
        if "cuda" in self.config.device:
            torch.backends.cudnn.benchmark = True

        pose_branch = self._build_pose_branch()
        track_branch = self._build_track_branch()

        probe = self._probe_video(str(source))
        fps = float(probe["fps"] or 0.0)
        if fps <= 0:
            fps = 25.0
        total_frames = int(probe["frame_count"] or 0)
        max_frames = self.config.max_frames if self.config.max_frames and self.config.max_frames > 0 else total_frames

        resolved_output_dir = Path(output_dir) if output_dir is not None else self._default_output_dir(source)
        resolved_output_dir.mkdir(parents=True, exist_ok=True)

        results: list[FrameResult] = []
        vis_writer: cv2.VideoWriter | None = None
        vis_path: Path | None = None
        if self.config.save_vis:
            vis_path = resolved_output_dir / "analysis_vis.mp4"
            w, h = int(probe["width"] or 0), int(probe["height"] or 0)
            if w > 0 and h > 0:
                vis_writer = cv2.VideoWriter(
                    str(vis_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
                )

        start_time = time.perf_counter()

        self._emit(progress_callback, {"type": "stage", "stage": "正在执行逐帧分析...", "progress": 0.08})

        BATCH = 16
        batch_frames: list[np.ndarray] = []
        batch_windows: list[list[np.ndarray]] = []
        frame_id = 0
        count_denom = max(max_frames, 1) if max_frames > 0 else max(total_frames, 1)
        pose_stride = max(1, self.config.pose_stride)
        last_pose: list = []
        last_progress_time = 0.0
        PROGRESS_INTERVAL = 0.3  # 进度回调最小间隔（秒）
        track_filter = BallTrackFilter(fps=fps)

        # 预处理线程池：在 GPU 推理当前 batch 时，CPU 预处理下一个 batch
        preprocess_executor = ThreadPoolExecutor(max_workers=1)
        pending_preprocess_future = None
        input_size = track_branch.input_size
        device = track_branch.device

        def _preprocess_batch(windows):
            """在后台线程中预处理一个 batch 的帧窗口为 tensor"""
            return preprocess_track_batch(windows, input_size, device)

        def _flush_batch() -> None:
            nonlocal last_pose, pending_preprocess_future, last_progress_time
            if not batch_frames:
                return

            # 如果有预处理好的 tensor，直接用；否则现场预处理
            if pending_preprocess_future is not None:
                tensor, metas = pending_preprocess_future.result()
                pending_preprocess_future = None
            else:
                tensor, metas = preprocess_track_batch(batch_windows, input_size, device)

            # Track 推理（FP16 已在 TrackBranch 内部处理）
            from src.postprocess.track import decode_track_heatmap_batch
            if track_branch._use_amp:
                tensor = tensor.half()
            with torch.no_grad():
                heatmaps = track_branch.model(tensor).float().detach().cpu().numpy()
            raw_track_results = decode_track_heatmap_batch(heatmaps, metas, track_branch.score_thr)
            track_results = [track_filter.update(track) for track in raw_track_results]

            batch_start = frame_id - len(batch_frames)
            for i, (frame, track) in enumerate(zip(batch_frames, track_results)):
                fid = batch_start + i
                if fid % pose_stride == 0:
                    last_pose = pose_branch.infer(frame)
                fr = FrameResult(frame_id=fid, pose=last_pose, track=track)
                results.append(fr)
                if vis_writer is not None:
                    from src.utils.visualize import draw_result
                    vis_writer.write(draw_result(frame, fr))

            # 节流进度回调：最多每 PROGRESS_INTERVAL 秒发一次
            now = time.perf_counter()
            if now - last_progress_time >= PROGRESS_INTERVAL:
                last_progress_time = now
                progress = min(frame_id / count_denom, 0.98)
                self._emit(
                    progress_callback,
                    {
                        "type": "frame",
                        "stage": f"正在分析第 {frame_id}/{count_denom} 帧",
                        "progress": progress,
                        "frame_id": frame_id - 1,
                        "track": {
                            "ball_xy": track_results[-1].ball_xy,
                            "visible": bool(track_results[-1].visible),
                            "score": float(track_results[-1].score),
                        },
                    },
                )
            batch_frames.clear()
            batch_windows.clear()

        try:
            stream_limit = max_frames if max_frames > 0 else None
            for fid, frame, window in iter_video_frame_windows(str(source), max_frames=stream_limit):
                if stop_requested is not None and stop_requested():
                    break

                batch_frames.append(frame)
                batch_windows.append(window)
                frame_id = fid + 1

                if len(batch_frames) >= BATCH:
                    # 提前提交预处理到后台线程（如果还没有 pending 的话）
                    if pending_preprocess_future is None:
                        pending_preprocess_future = preprocess_executor.submit(
                            _preprocess_batch, list(batch_windows)
                        )
                    _flush_batch()

            _flush_batch()
        finally:
            preprocess_executor.shutdown(wait=False)
            if vis_writer is not None:
                vis_writer.release()

        inference_seconds = time.perf_counter() - start_time
        if not results:
            raise RuntimeError("视频分析没有产生任何结果。")

        output_files = self._export_results(results, resolved_output_dir, save_json=self.config.save_json, save_csv=self.config.save_csv, save_npy=self.config.save_npy, save_vis=self.config.save_vis)
        bst_path = resolved_output_dir / "bst_input.npy"
        BSTInputBuilder(normalize=False).save(results, bst_path)
        output_files["BST 输入"] = str(bst_path)

        if vis_path is not None and vis_path.exists():
            output_files["可视化视频"] = str(vis_path)

        actions = self._build_actions(results, fps)
        avg_confidence = sum(item.track.score for item in results) / len(results)
        valid_pose_frames = sum(1 for item in results if item.pose)
        valid_track_frames = sum(1 for item in results if item.track.visible)
        summary_message = f"已完成对 {source.name} 的分析，生成 {len(actions)} 个动作片段"

        self._emit(progress_callback, {"type": "stage", "stage": "分析完成，正在整理结果...", "progress": 1.0})
        return AnalysisResult(
            status="success",
            actions=actions,
            total_frames=len(results),
            fps=fps,
            avg_confidence=avg_confidence,
            valid_pose_frames=valid_pose_frames,
            valid_track_frames=valid_track_frames,
            inference_seconds=inference_seconds,
            video_path=str(source),
            output_dir=str(resolved_output_dir),
            output_files=output_files,
            message=summary_message,
            payload={
                "probe": probe,
                "output_dir": str(resolved_output_dir),
            },
        )

    def _export_results(
        self,
        results: list[FrameResult],
        output_dir: Path,
        *,
        save_json: bool,
        save_csv: bool,
        save_npy: bool,
        save_vis: bool,
    ) -> dict[str, str]:
        from src.utils.exporters import export_csv, export_json, export_npy

        output_files: dict[str, str] = {}
        if save_json:
            json_path = output_dir / "analysis_results.json"
            export_json(results, json_path)
            output_files["JSON 结果"] = str(json_path)
        if save_csv:
            csv_path = output_dir / "analysis_results.csv"
            export_csv(results, csv_path)
            output_files["CSV 结果"] = str(csv_path)
        if save_npy:
            npy_path = output_dir / "analysis_results.npy"
            export_npy(results, npy_path)
            output_files["NPY 结果"] = str(npy_path)
        if save_vis:
            # 可视化视频在 analyze_video 中生成，先留占位，最终路径在调用方更新。
            pass
        return output_files

    def _build_actions(self, results: list[FrameResult], fps: float) -> list[AnalysisAction]:
        visible_segments: list[list[FrameResult]] = []
        current_segment: list[FrameResult] = []
        for item in results:
            if item.track.visible:
                current_segment.append(item)
            elif current_segment:
                visible_segments.append(current_segment)
                current_segment = []
        if current_segment:
            visible_segments.append(current_segment)

        if not visible_segments:
            best = max(results, key=lambda item: item.track.score)
            visible_segments = [[best]]

        action_names = cycle(["发球", "高远球", "杀球", "挑球", "吊球", "平抽球", "防守回球", "网前扑球"])
        actions: list[AnalysisAction] = []
        for segment in visible_segments[:8]:
            start = segment[0]
            end = segment[-1]
            confidence = sum(item.track.score for item in segment) / len(segment)
            ball_x = sum(item.track.ball_xy[0] for item in segment) / len(segment)
            ball_y = sum(item.track.ball_xy[1] for item in segment) / len(segment)
            start_time = start.frame_id / fps if fps > 0 else float(start.frame_id)
            end_time = end.frame_id / fps if fps > 0 else float(end.frame_id)
            label = next(action_names)
            actions.append(
                AnalysisAction(
                    time_range=f"{start_time:.1f}s - {end_time:.1f}s",
                    label=label,
                    confidence=float(confidence),
                    detail=f"球轨迹中心 ({ball_x:.0f}, {ball_y:.0f})，持续 {max(end_time - start_time, 0.0):.1f}s",
                    start_frame=start.frame_id,
                    end_frame=end.frame_id,
                    start_time=float(start_time),
                    end_time=float(end_time),
                )
            )
        return actions
