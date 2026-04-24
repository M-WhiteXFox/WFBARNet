# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import torch
from PyQt6.QtCore import QElapsedTimer, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QImage
from PyQt6.QtWidgets import QApplication, QFileDialog

from src.postprocess.track_filter import BallTrackFilter
from apps.pyqt6.utils.style import apply_theme, discover_themes
from apps.pyqt6.views.main_window_refined import MainWindow
from src.models.track_branch import TrackBranch
from src.utils.structures import FrameResult
from src.utils.visualize import draw_result


def frame_to_qimage(frame) -> QImage:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    bytes_per_line = rgb.strides[0]
    return QImage(
        rgb.data,
        width,
        height,
        bytes_per_line,
        QImage.Format.Format_RGB888,
    ).copy()


class VideoProbeWorker(QThread):
    finished = pyqtSignal(str, object)
    failed = pyqtSignal(str)

    def __init__(self, file_path: str, preview_ms: int = 0) -> None:
        super().__init__()
        self._file_path = file_path
        self._preview_ms = max(0, preview_ms)

    def run(self) -> None:
        cap = cv2.VideoCapture(self._file_path)
        if not cap.isOpened():
            self.failed.emit(f"无法打开视频: {self._file_path}")
            return

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            fps = 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration_ms = int(round((frame_count / fps) * 1000)) if frame_count > 0 else 0

        if self._preview_ms > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(self._preview_ms))

        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            self.failed.emit("视频已打开但无法读取预览帧")
            return

        position_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        if position_ms is None or position_ms <= 0:
            position_ms = float(self._preview_ms)

        payload = {
            "fps": fps,
            "width": width,
            "height": height,
            "frame_count": frame_count,
            "duration_ms": duration_ms,
            "position_ms": max(0, int(round(position_ms))),
            "image": frame_to_qimage(frame),
        }
        cap.release()
        self.finished.emit(self._file_path, payload)


class TrackNetPlaybackWorker(QThread):
    frameReady = pyqtSignal(object)
    playbackFinished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        video_path: str,
        track_branch: TrackBranch,
        *,
        start_ms: int = 0,
    ) -> None:
        super().__init__()
        self._video_path = video_path
        self._track_branch = track_branch
        self._start_ms = max(0, start_ms)
        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

    def _read_frame(
        self,
        cap: cv2.VideoCapture,
        fallback_index: int,
        fps: float,
    ) -> tuple[bool, Any, int]:
        ok, frame = cap.read()
        if not ok or frame is None:
            return False, None, 0
        position_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        if position_ms is None or position_ms <= 0:
            position_ms = (fallback_index * 1000.0) / fps if fps > 0 else 0.0
        return True, frame, int(round(position_ms))

    def _sleep_until(self, target_ms: int, clock: QElapsedTimer) -> bool:
        while not self._stop_requested:
            remaining = target_ms - clock.elapsed()
            if remaining <= 0:
                return True
            if remaining > 8:
                self.msleep(int(remaining - 4))
            else:
                self.usleep(max(500, int(remaining * 1000 / 2)))
        return False

    def run(self) -> None:
        cap = cv2.VideoCapture(self._video_path)
        if not cap.isOpened():
            self.failed.emit(f"无法打开视频: {self._video_path}")
            return

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            fps = 25.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration_ms = int(round((frame_count / fps) * 1000)) if frame_count > 0 else 0
        frame_interval_ms = int(round(1000.0 / fps)) if fps > 0 else 40

        if self._start_ms > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(self._start_ms))

        ok, current_frame, current_ms = self._read_frame(cap, 0, fps)
        if not ok:
            cap.release()
            self.failed.emit("无法读取第一帧视频")
            return

        ok, next_frame, next_ms = self._read_frame(cap, 1, fps)
        if not ok:
            next_frame = current_frame.copy()
            next_ms = current_ms + frame_interval_ms

        prev_frame = current_frame.copy()
        base_ms = current_ms
        processed_frames = 0
        visible_frames = 0
        score_sum = 0.0
        final_pass = False
        ema_infer_fps = 0.0
        track_filter = BallTrackFilter(fps=fps)

        clock = QElapsedTimer()
        clock.start()

        try:
            while not self._stop_requested:
                infer_start = perf_counter()
                _, raw_track = self._track_branch.infer([prev_frame, current_frame, next_frame])
                track = track_filter.update(raw_track)
                infer_elapsed = max(perf_counter() - infer_start, 1e-6)
                infer_fps = 1.0 / infer_elapsed
                ema_infer_fps = infer_fps if ema_infer_fps == 0.0 else (0.85 * ema_infer_fps + 0.15 * infer_fps)

                frame_id = int(round((current_ms / 1000.0) * fps)) if fps > 0 else processed_frames
                frame_result = FrameResult(frame_id=frame_id, pose=[], track=track)
                vis_frame = draw_result(current_frame, frame_result)
                cv2.putText(
                    vis_frame,
                    f"TrackNetV3 {ema_infer_fps:.1f} FPS",
                    (16, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

                target_ms = max(0, current_ms - base_ms)
                if not self._sleep_until(target_ms, clock):
                    break

                processed_frames += 1
                visible_frames += int(bool(track.visible))
                score_sum += float(track.score)
                avg_score = score_sum / max(processed_frames, 1)

                payload = {
                    "image": frame_to_qimage(vis_frame),
                    "frame_id": frame_id,
                    "position_ms": current_ms,
                    "duration_ms": duration_ms,
                    "progress": (current_ms / duration_ms) if duration_ms > 0 else 0.0,
                    "track": {
                        "ball_xy": list(track.ball_xy),
                        "visible": bool(track.visible),
                        "score": float(track.score),
                    },
                    "visible_frames": visible_frames,
                    "avg_score": avg_score,
                    "processed_frames": processed_frames,
                }
                self.frameReady.emit(payload)

                if final_pass:
                    break

                prev_frame = current_frame
                current_frame = next_frame
                current_ms = next_ms

                ok, incoming_frame, incoming_ms = self._read_frame(cap, processed_frames + 1, fps)
                if ok:
                    next_frame = incoming_frame
                    next_ms = incoming_ms
                else:
                    next_frame = current_frame.copy()
                    next_ms = current_ms + frame_interval_ms
                    final_pass = True
        finally:
            cap.release()

        self.playbackFinished.emit(
            {
                "stopped": self._stop_requested,
                "processed_frames": processed_frames,
                "visible_frames": visible_frames,
                "avg_score": (score_sum / processed_frames) if processed_frames else 0.0,
            }
        )


class MainController:
    """PyQt6 前端的多线程 TrackNetV3 预览控制器。"""

    def __init__(self, view: MainWindow) -> None:
        self.view = view
        self._theme_dirs = discover_themes()
        self._selected_video_path: str | None = None
        self._video_meta: dict[str, Any] = {}
        self._probe_worker: VideoProbeWorker | None = None
        self._playback_worker: TrackNetPlaybackWorker | None = None
        self._pending_seek_ms: int | None = None

        self._track_branch = self._build_track_branch()

        self._bind_events()
        self.view.populate_stylesheets(self._theme_dirs)
        self.view.video_timeline.set_interactive(True)
        self._reset_metrics()
        self._set_idle_state()
        self.view.append_log("[系统] 界面已就绪，请选择视频开始。")

    def _build_track_branch(self) -> TrackBranch:
        project_root = Path(__file__).resolve().parents[3]
        model_weight = project_root / "assets" / "weights" / "track" / "model_best.pt"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        branch = TrackBranch(
            model_weight=str(model_weight),
            device=device,
            input_size=(512, 288),
            score_thr=0.35,
        )
        self.view.append_log(f"[TrackNet] 模型已加载: {device}")
        return branch

    def _bind_events(self) -> None:
        self.view.btn_analyze.clicked.connect(self.handle_analyze)
        self.view.btn_reset.clicked.connect(self.handle_reset)
        self.view.video_player.selectRequested.connect(self.handle_upload)
        self.view.video_player.forceStopRequested.connect(self.handle_force_stop)
        self.view.video_timeline.seekRequested.connect(self.handle_seek)
        self.view._style_menu.triggered.connect(self._on_style_action_triggered)

    def _set_idle_state(self) -> None:
        has_video = self._selected_video_path is not None
        self.view.btn_analyze.setEnabled(has_video)
        self.view.btn_reset.setEnabled(True)
        self.view.video_player.btn_select_video.setEnabled(True)
        self.view.video_player.btn_force_stop.setEnabled(has_video)
        self.view.set_status_state("idle")

    def _set_running_state(self) -> None:
        self.view.btn_analyze.setEnabled(False)
        self.view.btn_reset.setEnabled(True)
        self.view.video_player.btn_select_video.setEnabled(False)
        self.view.video_player.btn_force_stop.setEnabled(True)
        self.view.set_status_state("loading")

    def _reset_metrics(self) -> None:
        self.view.reset_analysis()
        self.view.video_timeline.reset()

    def handle_upload(self) -> None:
        start_dir = str(Path(__file__).resolve().parents[3] / "videos")
        file_path, _ = QFileDialog.getOpenFileName(
            self.view,
            "选择视频",
            start_dir,
            "视频文件 (*.mp4 *.avi *.mov *.mkv *.wmv);;所有文件 (*)",
        )
        if not file_path:
            self.view.append_log("[信息] 视频选择已取消。")
            return

        self._stop_workers(clear_pending_seek=True)
        self._selected_video_path = file_path
        self._video_meta = {}
        self._reset_metrics()
        self.view.set_video_path(file_path)
        self.view.set_video_state("loaded")
        self.view.append_log(f"[信息] 正在加载预览: {Path(file_path).name}")
        self._start_probe(file_path, 0)

    def _start_probe(self, video_path: str, position_ms: int) -> None:
        if self._probe_worker is not None and self._probe_worker.isRunning():
            self._probe_worker.quit()
            self._probe_worker.wait(300)

        self._probe_worker = VideoProbeWorker(video_path, preview_ms=position_ms)
        self._probe_worker.finished.connect(self._on_probe_finished)
        self._probe_worker.failed.connect(self._on_probe_failed)
        self._probe_worker.start()

    def _on_probe_finished(self, file_path: str, payload: object) -> None:
        self._probe_worker = None
        if file_path != self._selected_video_path or not isinstance(payload, dict):
            return

        self._video_meta = payload
        self.view.show_video_frame(
            payload["image"],
            int(payload.get("position_ms", 0)),
            int(payload.get("duration_ms", 0)),
        )
        self.view.update_progress(0)
        self.view.set_video_state("loaded")
        self.view.append_log(
            f"[信息] 已加载 {Path(file_path).name} | "
            f"{payload.get('width', 0)} x {payload.get('height', 0)} | "
            f"FPS {float(payload.get('fps', 0.0)):.2f}"
        )
        self._set_idle_state()

    def _on_probe_failed(self, message: str) -> None:
        self._probe_worker = None
        self._selected_video_path = None
        self._video_meta = {}
        self.view.set_status_state("error")
        self.view.set_video_state("error")
        self.view.btn_analyze.setEnabled(False)
        self.view.video_player.btn_select_video.setEnabled(True)
        self.view.video_player.btn_force_stop.setEnabled(False)
        self.view.append_log(f"[错误] {message}")

    def handle_analyze(self) -> None:
        if not self._selected_video_path:
            self.view.append_log("[警告] 开始分析前请先选择视频。")
            return

        self._pending_seek_ms = None
        start_ms = int(self._video_meta.get("position_ms", 0)) if self._video_meta else 0
        self._start_playback(start_ms=start_ms)

    def _start_playback(self, *, start_ms: int = 0) -> None:
        if self._selected_video_path is None:
            return

        self._stop_workers(clear_pending_seek=False)
        self._set_running_state()
        self.view.video_player.play()
        self.view.append_log(
            f"[TrackNet] 开始播放: {Path(self._selected_video_path).name}"
        )

        self._playback_worker = TrackNetPlaybackWorker(
            self._selected_video_path,
            self._track_branch,
            start_ms=start_ms,
        )
        self._playback_worker.frameReady.connect(self._on_frame_ready)
        self._playback_worker.playbackFinished.connect(self._on_playback_finished)
        self._playback_worker.failed.connect(self._on_playback_failed)
        self._playback_worker.start()

    def _on_frame_ready(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return

        image = payload.get("image")
        if isinstance(image, QImage):
            self.view.show_video_frame(
                image,
                int(payload.get("position_ms", 0)),
                int(payload.get("duration_ms", 0)),
            )

        progress = max(0, min(int(float(payload.get("progress", 0.0)) * 100), 100))
        self.view.update_progress(progress)

        processed_frames = int(payload.get("processed_frames", 0))
        visible_frames = int(payload.get("visible_frames", 0))
        avg_score = float(payload.get("avg_score", 0.0))
        track = payload.get("track", {})
        current_score = float(track.get("score", 0.0)) if isinstance(track, dict) else 0.0

        self.view.lbl_total_actions.setText(str(processed_frames))
        self.view.lbl_valid_pose.setText(str(processed_frames))
        self.view.lbl_valid_track.setText(str(visible_frames))
        self.view.lbl_avg_conf.setText(f"{avg_score * 100:.1f}%")
        self.view.status_label.setText(
            f"系统状态：TrackNetV3 运行中 | Score {current_score:.2f}"
        )

    def _on_playback_finished(self, payload: object) -> None:
        self._playback_worker = None
        stopped = bool(payload.get("stopped")) if isinstance(payload, dict) else False

        if stopped:
            self.view.set_status_state("stopped")
            self.view.video_player.stop()
            self.view.append_log("[TrackNet] 播放已停止。")
        else:
            self.view.set_status_state("success")
            self.view.video_player.stop()
            self.view.update_progress(100)
            if isinstance(payload, dict):
                self.view.append_log(
                    f"[TrackNet] 已完成 | "
                    f"帧数 {int(payload.get('processed_frames', 0))} | "
                    f"可见 {int(payload.get('visible_frames', 0))} | "
                    f"平均 {float(payload.get('avg_score', 0.0)) * 100:.1f}%"
                )
            else:
                self.view.append_log("[TrackNet] 已完成。")

        self._set_idle_state()

        if self._pending_seek_ms is not None and self._selected_video_path is not None:
            pending_seek_ms = self._pending_seek_ms
            self._pending_seek_ms = None
            self._start_playback(start_ms=pending_seek_ms)

    def _on_playback_failed(self, message: str) -> None:
        self._playback_worker = None
        self.view.set_status_state("error")
        self.view.video_player.stop()
        self.view.append_log(f"[错误] {message}")
        self._set_idle_state()

    def handle_seek(self, position_ms: int) -> None:
        if self._selected_video_path is None:
            return

        self._video_meta["position_ms"] = position_ms
        if self._playback_worker is not None and self._playback_worker.isRunning():
            self._pending_seek_ms = position_ms
            self._playback_worker.request_stop()
            self.view.append_log(f"[信息] 正在跳转至 {position_ms / 1000:.2f}s")
            return

        self.view.append_log(f"[信息] 预览跳转至 {position_ms / 1000:.2f}s")
        self._start_probe(self._selected_video_path, position_ms)

    def handle_force_stop(self) -> None:
        if self._playback_worker is not None and self._playback_worker.isRunning():
            self.view.append_log("[信息] 正在停止 TrackNetV3 播放...")
            self._playback_worker.request_stop()
            return

        self.view.video_player.stop()
        self.view.set_status_state("stopped")
        self.view.append_log("[信息] 没有正在进行的播放任务。")

    def handle_reset(self) -> None:
        self._stop_workers(clear_pending_seek=True)
        self._selected_video_path = None
        self._video_meta = {}
        self.view.clear_video()
        self.view.log_console.clear()
        self._reset_metrics()
        self._set_idle_state()
        self.view.append_log("[系统] 工作区已重置。")

    def _stop_workers(self, *, clear_pending_seek: bool) -> None:
        if clear_pending_seek:
            self._pending_seek_ms = None

        if self._probe_worker is not None and self._probe_worker.isRunning():
            self._probe_worker.quit()
            self._probe_worker.wait(300)
        self._probe_worker = None

        if self._playback_worker is not None and self._playback_worker.isRunning():
            self._playback_worker.request_stop()
            self._playback_worker.wait(1000)
        self._playback_worker = None

    def _on_style_action_triggered(self, action) -> None:
        theme_name = str(action.data() or action.text()).strip()
        theme_label = action.text().strip() or theme_name.replace("_", " ").title()
        self.view.style_btn.setText(f"{theme_label}  ▾")
        self.handle_style_changed(theme_name)

    def handle_style_changed(self, theme_name: str) -> None:
        app = QApplication.instance()
        if app is None or not theme_name.strip():
            return

        theme_dir = next((d for d in self._theme_dirs if d.name == theme_name), None)
        if theme_dir is None:
            return

        def _apply() -> None:
            apply_theme(app, theme_dir)
            self.view.append_log(f"[主题] 已切换至 {theme_dir.name}")

        QTimer.singleShot(0, _apply)


MockController = MainController
