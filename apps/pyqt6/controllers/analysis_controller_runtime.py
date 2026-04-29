# -*- coding: utf-8 -*-
from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Any
import importlib.util

import cv2
import torch
from PyQt6.QtCore import QElapsedTimer, QSettings, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QImage
from PyQt6.QtWidgets import QApplication, QFileDialog

from apps.pyqt6.utils.style import apply_theme, discover_themes
from apps.pyqt6.utils.theme_transition import start_theme_ripple_transition
from apps.pyqt6.views.main_window_refined import MainWindow
from src.models.pose_branch import PoseBranch
from src.models.track_branch import TrackBranch
from src.postprocess.pose import CourtPoseTargetTracker
from src.postprocess.track_filter import BallTrackFilter
from src.utils.structures import FrameResult, TrackResult
from src.utils.visualize import TrackTrailRenderer


DISPLAY_FPS_LIMIT = 60.0
METRICS_UPDATE_INTERVAL_S = 0.25
PLAYBACK_LAG_TOLERANCE_FRAMES = 1.5
POSE_CANDIDATE_LIMIT = 12


@contextmanager
def quiet_opencv_camera_logs():
    previous_level = cv2.getLogLevel() if hasattr(cv2, "getLogLevel") else None
    stderr_fd = None
    saved_stderr_fd = None
    if not hasattr(cv2, "setLogLevel") or not hasattr(cv2, "getLogLevel"):
        previous_level = None
    try:
        if previous_level is not None:
            cv2.setLogLevel(0)
        try:
            sys.stderr.flush()
            saved_stderr_fd = os.dup(2)
            stderr_fd = os.open(os.devnull, os.O_WRONLY)
            os.dup2(stderr_fd, 2)
        except OSError:
            if saved_stderr_fd is not None:
                os.close(saved_stderr_fd)
                saved_stderr_fd = None
            if stderr_fd is not None:
                os.close(stderr_fd)
                stderr_fd = None
        yield
    finally:
        if saved_stderr_fd is not None:
            try:
                sys.stderr.flush()
                os.dup2(saved_stderr_fd, 2)
            finally:
                os.close(saved_stderr_fd)
        if stderr_fd is not None:
            os.close(stderr_fd)
        if previous_level is not None:
            cv2.setLogLevel(previous_level)


def open_camera_capture(
    camera_index: int,
    *,
    verify_frame: bool = False,
    quiet: bool = False,
) -> tuple[cv2.VideoCapture, str]:
    openers = [
        ("Auto", lambda: cv2.VideoCapture(camera_index)),
        ("MSMF", lambda: cv2.VideoCapture(camera_index + int(getattr(cv2, "CAP_MSMF", 0)))),
        ("DSHOW", lambda: cv2.VideoCapture(camera_index + int(getattr(cv2, "CAP_DSHOW", 0)))),
    ]

    def _open() -> tuple[cv2.VideoCapture, str]:
        for backend_name, opener in openers:
            cap = opener()
            if not cap.isOpened():
                cap.release()
                continue

            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if verify_frame:
                ok, frame = cap.read()
                if not ok or frame is None:
                    cap.release()
                    continue
            return cap, backend_name

        return cv2.VideoCapture(), ""

    if quiet:
        with quiet_opencv_camera_logs():
            return _open()
    return _open()


def frame_to_qimage(frame) -> QImage:
    if hasattr(QImage.Format, "Format_BGR888"):
        height, width = frame.shape[:2]
        bytes_per_line = frame.strides[0]
        image = QImage(
            frame.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_BGR888,
        )
        image._buffer = frame
        return image

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    bytes_per_line = rgb.strides[0]
    image = QImage(
        rgb.data,
        width,
        height,
        bytes_per_line,
        QImage.Format.Format_RGB888,
    )
    image._buffer = rgb
    return image


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
        pose_branch: PoseBranch,
        *,
        start_ms: int = 0,
        pose_stride: int = 3,
        track_enabled: bool = True,
        pose_enabled: bool = True,
        display_fps_limit: float = DISPLAY_FPS_LIMIT,
        court_service: Any | None = None,
    ) -> None:
        super().__init__()
        self._video_path = video_path
        self._track_branch = track_branch
        self._pose_branch = pose_branch
        self._start_ms = max(0, start_ms)
        self._pose_stride = max(1, pose_stride)
        self._track_enabled = track_enabled
        self._pose_enabled = pose_enabled
        self._display_fps_limit = max(1.0, float(display_fps_limit))
        self._court_service = court_service
        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

    def _pipeline_label(self) -> str:
        names = []
        if self._track_enabled:
            names.append("TrackNet")
        if self._pose_enabled:
            names.append("YOLO26s-Pose")
        return " + ".join(names) if names else "Preview"

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
        dropped_source_frames = 0
        visible_frames = 0
        pose_frames = 0
        score_sum = 0.0
        final_pass = False
        ema_infer_fps = 0.0
        last_pose = []
        track_filter = BallTrackFilter(fps=fps)
        pose_tracker = CourtPoseTargetTracker(max_missing_frames=int(round(fps * 0.8)))
        trail_renderer = TrackTrailRenderer(fps=fps, history_seconds=3.0)
        display_interval_ms = 1000.0 / self._display_fps_limit
        display_every_frame = fps <= self._display_fps_limit
        next_display_ms = float(base_ms)

        clock = QElapsedTimer()
        clock.start()

        try:
            while not self._stop_requested:
                loop_start = perf_counter()
                if self._track_enabled:
                    raw_track = self._track_branch.infer_result([prev_frame, current_frame, next_frame])
                    track = track_filter.update(raw_track, frame_shape=current_frame.shape)
                else:
                    track = TrackResult(ball_xy=[-1.0, -1.0], visible=0, score=0.0)

                frame_id = int(round((current_ms / 1000.0) * fps)) if fps > 0 else processed_frames
                court_prediction = None
                if self._court_service is not None:
                    self._court_service.submit_frame(current_frame, frame_id, current_ms)
                    court_prediction = self._court_service.latest_prediction()

                if self._pose_enabled:
                    detections = (
                        self._pose_branch.infer(current_frame)
                        if processed_frames % self._pose_stride == 0
                        else []
                    )
                    last_pose = pose_tracker.update(
                        detections,
                        court_prediction,
                        frame_shape=current_frame.shape,
                    )
                    pose_frames += int(bool(last_pose))
                else:
                    pose_tracker.reset()
                    last_pose = []

                infer_elapsed = max(perf_counter() - loop_start, 1e-6)
                infer_fps = 1.0 / infer_elapsed
                ema_infer_fps = infer_fps if ema_infer_fps == 0.0 else (0.85 * ema_infer_fps + 0.15 * infer_fps)

                should_emit = display_every_frame or final_pass or current_ms >= next_display_ms
                image = None
                if should_emit:
                    frame_result = FrameResult(frame_id=frame_id, pose=last_pose, track=track)
                    vis_frame = current_frame.copy()
                    trail_renderer.draw_on(vis_frame, frame_result, timestamp_ms=current_ms)
                    image = frame_to_qimage(vis_frame)

                target_ms = max(0, current_ms - base_ms)
                if not self._sleep_until(target_ms, clock):
                    break

                processed_frames += 1
                visible_frames += int(bool(track.visible))
                score_sum += float(track.score)
                avg_score = score_sum / max(processed_frames, 1)
                if should_emit:
                    payload = {
                        "image": image,
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
                        "pose_frames": pose_frames,
                        "person_count": len(last_pose),
                        "avg_score": avg_score,
                        "processed_frames": processed_frames,
                        "infer_fps": ema_infer_fps,
                        "court": court_prediction.to_dict() if court_prediction is not None else None,
                    }
                    self.frameReady.emit(payload)
                    if not display_every_frame:
                        while next_display_ms <= current_ms:
                            next_display_ms += display_interval_ms

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

                lag_ms = clock.elapsed() - max(0, current_ms - base_ms)
                max_lag_ms = frame_interval_ms * PLAYBACK_LAG_TOLERANCE_FRAMES
                while not final_pass and lag_ms > max_lag_ms:
                    prev_frame = current_frame
                    current_frame = next_frame
                    current_ms = next_ms
                    dropped_source_frames += 1

                    ok, incoming_frame, incoming_ms = self._read_frame(
                        cap,
                        processed_frames + dropped_source_frames + 1,
                        fps,
                    )
                    if ok:
                        next_frame = incoming_frame
                        next_ms = incoming_ms
                    else:
                        next_frame = current_frame.copy()
                        next_ms = current_ms + frame_interval_ms
                        final_pass = True
                        break
                    lag_ms = clock.elapsed() - max(0, current_ms - base_ms)
        finally:
            cap.release()

        self.playbackFinished.emit(
            {
                "stopped": self._stop_requested,
                "processed_frames": processed_frames,
                "dropped_source_frames": dropped_source_frames,
                "visible_frames": visible_frames,
                "pose_frames": pose_frames,
                "avg_score": (score_sum / processed_frames) if processed_frames else 0.0,
            }
        )


class CameraInferenceWorker(QThread):
    frameReady = pyqtSignal(object)
    inferFinished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        camera_index: int,
        track_branch: TrackBranch,
        pose_branch: PoseBranch,
        *,
        pose_stride: int = 3,
        track_enabled: bool = True,
        pose_enabled: bool = True,
        display_fps_limit: float = DISPLAY_FPS_LIMIT,
        court_service: Any | None = None,
    ) -> None:
        super().__init__()
        self._camera_index = camera_index
        self._track_branch = track_branch
        self._pose_branch = pose_branch
        self._pose_stride = max(1, pose_stride)
        self._track_enabled = track_enabled
        self._pose_enabled = pose_enabled
        self._display_fps_limit = max(1.0, float(display_fps_limit))
        self._court_service = court_service
        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

    def _pipeline_label(self) -> str:
        names = []
        if self._track_enabled:
            names.append("TrackNet")
        if self._pose_enabled:
            names.append("YOLO26s-Pose")
        return " + ".join(names) if names else "Preview"

    def run(self) -> None:
        cap, backend_name = open_camera_capture(self._camera_index, quiet=True)
        if not cap.isOpened():
            self.failed.emit(f"无法打开摄像头设备: {self._camera_index}")
            return

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            fps = 25.0

        ok, first_frame = cap.read()
        if not ok or first_frame is None:
            cap.release()
            self.failed.emit("摄像头已打开，但没有读取到画面")
            return

        ok, second_frame = cap.read()
        if not ok or second_frame is None:
            second_frame = first_frame.copy()

        prev_frame = first_frame.copy()
        current_frame = first_frame
        next_frame = second_frame
        processed_frames = 0
        visible_frames = 0
        pose_frames = 0
        score_sum = 0.0
        ema_infer_fps = 0.0
        last_pose = []
        track_filter = BallTrackFilter(fps=fps)
        pose_tracker = CourtPoseTargetTracker(max_missing_frames=int(round(fps * 0.8)))
        trail_renderer = TrackTrailRenderer(fps=fps, history_seconds=3.0)
        display_interval_ms = 1000.0 / self._display_fps_limit
        next_display_ms = 0.0
        clock = QElapsedTimer()
        clock.start()

        try:
            while not self._stop_requested:
                loop_start = perf_counter()
                if self._track_enabled:
                    raw_track = self._track_branch.infer_result([prev_frame, current_frame, next_frame])
                    track = track_filter.update(raw_track, frame_shape=current_frame.shape)
                else:
                    track = TrackResult(ball_xy=[-1.0, -1.0], visible=0, score=0.0)

                current_frame_id = processed_frames
                position_ms = clock.elapsed()
                court_prediction = None
                if self._court_service is not None:
                    self._court_service.submit_frame(current_frame, current_frame_id, position_ms)
                    court_prediction = self._court_service.latest_prediction()

                if self._pose_enabled:
                    detections = (
                        self._pose_branch.infer(current_frame)
                        if processed_frames % self._pose_stride == 0
                        else []
                    )
                    last_pose = pose_tracker.update(
                        detections,
                        court_prediction,
                        frame_shape=current_frame.shape,
                    )
                    pose_frames += int(bool(last_pose))
                else:
                    pose_tracker.reset()
                    last_pose = []

                infer_elapsed = max(perf_counter() - loop_start, 1e-6)
                infer_fps = 1.0 / infer_elapsed
                ema_infer_fps = infer_fps if ema_infer_fps == 0.0 else (0.85 * ema_infer_fps + 0.15 * infer_fps)

                processed_frames += 1
                visible_frames += int(bool(track.visible))
                score_sum += float(track.score)
                avg_score = score_sum / max(processed_frames, 1)
                if position_ms >= next_display_ms:
                    frame_result = FrameResult(frame_id=current_frame_id, pose=last_pose, track=track)
                    vis_frame = current_frame.copy()
                    trail_renderer.draw_on(vis_frame, frame_result, timestamp_ms=position_ms)
                    image = frame_to_qimage(vis_frame)

                    payload = {
                        "image": image,
                        "frame_id": current_frame_id,
                        "position_ms": position_ms,
                        "duration_ms": 0,
                        "progress": 0.0,
                        "track": {
                            "ball_xy": list(track.ball_xy),
                            "visible": bool(track.visible),
                            "score": float(track.score),
                        },
                        "visible_frames": visible_frames,
                        "pose_frames": pose_frames,
                        "person_count": len(last_pose),
                        "avg_score": avg_score,
                        "processed_frames": processed_frames,
                        "infer_fps": ema_infer_fps,
                        "court": court_prediction.to_dict() if court_prediction is not None else None,
                    }
                    self.frameReady.emit(payload)
                    while next_display_ms <= position_ms:
                        next_display_ms += display_interval_ms

                prev_frame = current_frame
                current_frame = next_frame
                ok, incoming_frame = cap.read()
                if not ok or incoming_frame is None:
                    break
                next_frame = incoming_frame
        finally:
            cap.release()

        self.inferFinished.emit(
            {
                "stopped": self._stop_requested,
                "processed_frames": processed_frames,
                "visible_frames": visible_frames,
                "pose_frames": pose_frames,
                "avg_score": (score_sum / processed_frames) if processed_frames else 0.0,
            }
        )


class MainController:
    """PyQt6 前端的多线程 TrackNetV3 预览控制器。"""

    def __init__(self, view: MainWindow, court_service: Any | None = None) -> None:
        self.view = view
        self._court_service = court_service
        self._project_root = Path(__file__).resolve().parents[3]
        self._settings = QSettings("WFBARNet", "PyQt6Runtime")
        self._default_pose_model_path = str(self._project_root / "assets" / "weights" / "pose" / "yolo26s-pose.pt")
        self._default_track_model_path = str(self._project_root / "assets" / "weights" / "track" / "model_best.pt")
        self._pose_model_path = self._load_model_path("pose_model_path", self._default_pose_model_path)
        self._track_model_path = self._load_model_path("track_model_path", self._default_track_model_path)
        self._pose_model_enabled = self._load_bool_setting("pose_model_enabled", True)
        self._track_model_enabled = self._load_bool_setting("track_model_enabled", True)
        self._theme_dirs = discover_themes()
        self._active_theme_name = self._resolve_initial_theme_name()
        self._selected_video_path: str | None = None
        self._video_meta: dict[str, Any] = {}
        self._input_mode = "video"
        self._camera_devices: list[tuple[int, str]] = []
        self._probe_worker: VideoProbeWorker | None = None
        self._playback_worker: TrackNetPlaybackWorker | None = None
        self._camera_worker: CameraInferenceWorker | None = None
        self._pending_seek_ms: int | None = None
        self._last_display_frame_time: float | None = None
        self._last_metrics_update_time: float | None = None
        self._display_fps_ema = 0.0
        self._last_court_log_frame = -1

        self.view.set_model_settings(self._pose_model_path, self._track_model_path)
        self.view.set_model_switches(self._pose_model_enabled, self._track_model_enabled)
        self._track_branch = self._build_track_branch()
        self._pose_branch = self._build_pose_branch()

        self._bind_court_service()
        self._bind_events()
        self.view.populate_stylesheets(self._theme_dirs, self._active_theme_name)
        self.view.video_timeline.set_interactive(True)
        self._refresh_camera_devices(log=False)
        self._reset_metrics()
        self._set_idle_state()
        self.view.append_log("[系统] 界面已就绪，请选择视频开始。")

    def _bind_court_service(self) -> None:
        if self._court_service is None:
            return
        self._court_service.resultReady.connect(self._on_court_prediction_ready)
        self._court_service.failed.connect(self._on_court_detection_failed)

    def _reset_court_detection(self, *, request_initial_prediction: bool = False) -> None:
        self._last_court_log_frame = -1
        if self._court_service is not None:
            self._court_service.reset()
            if request_initial_prediction:
                self._court_service.request_prediction()

    def _request_court_prediction(self) -> None:
        if self._court_service is None:
            self.view.append_log("[Court] OpenCV court line detector is unavailable.")
            return
        if not self._is_inference_running():
            self.view.append_log("[Court] 请先开始视频播放或摄像头推理，再重新预测球场线。")
            return
        self._last_court_log_frame = -1
        self._court_service.request_prediction()
        self.view.append_log("[Court] 已请求重新预测球场线，将使用下一帧画面。")

    def _on_court_prediction_ready(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        frame_id = int(payload.get("frame_id", -1))
        if frame_id == self._last_court_log_frame:
            return

        valid = bool(payload.get("valid"))
        updated = bool(payload.get("updated"))
        if valid and updated:
            self._last_court_log_frame = frame_id
            confidence = float(payload.get("confidence", 0.0))
            detect_ms = float(payload.get("detect_ms", 0.0))
            self.view.append_log(f"[Court] OpenCV court lines updated | frame {frame_id} | conf {confidence:.2f} | {detect_ms:.0f} ms")
        elif not valid and self._last_court_log_frame < 0:
            self._last_court_log_frame = frame_id
            self.view.append_log("[Court] OpenCV court line detector is running; waiting for a valid court.")

    def _on_court_detection_failed(self, message: str) -> None:
        self.view.append_log(f"[Court] OpenCV court line detector failed: {message}")

    def _load_model_path(self, key: str, default_path: str) -> str:
        raw_value = self._settings.value(key, default_path)
        value = str(raw_value or default_path).strip()
        if not value:
            return default_path

        resolved = self._resolve_model_path(value)
        if resolved.is_file():
            return str(resolved)
        return default_path

    def _load_bool_setting(self, key: str, default: bool) -> bool:
        raw_value = self._settings.value(key, default)
        if isinstance(raw_value, bool):
            return raw_value
        return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}

    def _resolve_model_path(self, raw_path: str) -> Path:
        path = Path(raw_path.strip()).expanduser()
        if path.is_absolute():
            return path
        return self._project_root / path

    def _resolve_initial_theme_name(self) -> str:
        if any(theme_dir.name == "office_light" for theme_dir in self._theme_dirs):
            return "office_light"
        return self._theme_dirs[0].name if self._theme_dirs else ""

    def _build_track_branch(self) -> TrackBranch:
        branch = self._create_track_branch(self._track_model_path)
        self.view.append_log(
            f"[TrackNet] 模型已加载: {branch.device} | 后端 {branch.backend_name} | {Path(self._track_model_path).name}"
        )
        return branch

    def _create_track_branch(self, model_weight: str) -> TrackBranch:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_path = Path(model_weight)
        if model_path.suffix.lower() == ".engine":
            if device != "cuda":
                raise RuntimeError(
                    "TensorRT INT8 engine 需要 CUDA 版 PyTorch；当前环境未检测到可用 CUDA。"
                )
            if importlib.util.find_spec("tensorrt") is None:
                raise RuntimeError(
                    "TensorRT INT8 engine 需要安装 tensorrt Python 包；当前环境未检测到 tensorrt。"
                )
        return TrackBranch(
            model_weight=str(model_weight),
            device=device,
            input_size=(512, 288),
            score_thr=0.35,
        )

    def _build_pose_branch(self) -> PoseBranch:
        branch = self._create_pose_branch(self._pose_model_path)
        self.view.append_log(f"[YOLO26s-Pose] 模型已加载: {branch.device} | {Path(self._pose_model_path).name}")
        return branch

    def _create_pose_branch(self, model_weight: str) -> PoseBranch:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return PoseBranch(
            backend="yolo26s-pose",
            model_weight=str(model_weight),
            device=device,
            conf_thr=0.35,
            max_persons=POSE_CANDIDATE_LIMIT,
        )

    def _bind_events(self) -> None:
        self.view.btn_analyze.clicked.connect(self.handle_analyze)
        self.view.btn_reset.clicked.connect(self.handle_reset)
        self.view.btn_preview_mode.clicked.connect(lambda: self.handle_input_mode("video"))
        self.view.btn_camera_mode.clicked.connect(lambda: self.handle_input_mode("camera"))
        self.view.btn_refresh_cameras.clicked.connect(lambda: self._refresh_camera_devices(log=True))
        self.view.camera_device_combo.currentIndexChanged.connect(lambda _index: self._set_idle_state())
        self.view.video_player.selectRequested.connect(self.handle_upload)
        self.view.video_player.forceStopRequested.connect(self.handle_force_stop)
        self.view.video_timeline.seekRequested.connect(self.handle_seek)
        self.view._style_menu.triggered.connect(self._on_style_action_triggered)
        self.view.poseModelBrowseRequested.connect(self.handle_browse_pose_model)
        self.view.trackModelBrowseRequested.connect(self.handle_browse_track_model)
        self.view.modelSettingsApplyRequested.connect(self.handle_model_settings_apply)
        self.view.modelSettingsDefaultsRequested.connect(self.handle_model_settings_defaults)
        self.view.modelSwitchesChanged.connect(self.handle_model_switches_changed)
        self.view.courtRedetectRequested.connect(self._request_court_prediction)

    def _set_idle_state(self) -> None:
        has_video = self._selected_video_path is not None
        has_camera = self.view.selected_camera_device() is not None
        self.view.btn_analyze.setEnabled(has_video if self._input_mode == "video" else has_camera)
        self.view.btn_reset.setEnabled(True)
        self.view.video_player.btn_select_video.setEnabled(self._input_mode == "video")
        self.view.btn_refresh_cameras.setEnabled(self._input_mode == "camera")
        self.view.camera_device_combo.setEnabled(self._input_mode == "camera")
        self.view.video_player.btn_force_stop.setEnabled(has_video if self._input_mode == "video" else False)
        self.view.btn_redetect_court.setEnabled(False)
        self.view.video_timeline.set_interactive(self._input_mode == "video")
        self.view.set_model_settings_enabled(True)
        self.view.set_status_state("idle")

    def _set_running_state(self) -> None:
        self.view.btn_analyze.setEnabled(False)
        self.view.btn_reset.setEnabled(True)
        self.view.video_player.btn_select_video.setEnabled(False)
        self.view.btn_refresh_cameras.setEnabled(False)
        self.view.camera_device_combo.setEnabled(False)
        self.view.video_player.btn_force_stop.setEnabled(True)
        self.view.btn_redetect_court.setEnabled(self._court_service is not None)
        self.view.video_timeline.set_interactive(False)
        self.view.set_model_settings_enabled(False)
        self.view.set_status_state("loading")

    def _reset_metrics(self) -> None:
        self.view.set_progress_busy(False)
        self.view.reset_analysis()
        self.view.video_timeline.reset()
        self._reset_display_fps()

    def _reset_display_fps(self) -> None:
        self._last_display_frame_time = None
        self._last_metrics_update_time = None
        self._display_fps_ema = 0.0
        self.view.lbl_realtime_fps.setText("0.0 FPS")

    def _update_display_fps(self) -> float:
        now = perf_counter()
        if self._last_display_frame_time is None:
            self._last_display_frame_time = now
            return self._display_fps_ema

        elapsed = max(now - self._last_display_frame_time, 1e-6)
        self._last_display_frame_time = now
        instant_fps = 1.0 / elapsed
        self._display_fps_ema = (
            instant_fps
            if self._display_fps_ema <= 0.0
            else (0.85 * self._display_fps_ema + 0.15 * instant_fps)
        )
        return self._display_fps_ema

    def _should_update_metrics_text(self) -> bool:
        now = perf_counter()
        if self._last_metrics_update_time is None:
            self._last_metrics_update_time = now
            return True
        if now - self._last_metrics_update_time < METRICS_UPDATE_INTERVAL_S:
            return False
        self._last_metrics_update_time = now
        return True

    def handle_input_mode(self, mode: str) -> None:
        if mode not in {"video", "camera"}:
            return

        self._stop_workers(clear_pending_seek=True)
        self._input_mode = mode
        self.view.set_input_mode(mode)
        self._reset_metrics()

        if mode == "camera":
            self.view.video_player.clear_video()
            self.view.video_player.set_live_source("摄像头实时推理")
            self.view.set_video_state("idle")
            self.view.append_log("[模式] 已切换到摄像头实时推理")
            if not self._camera_devices:
                self._refresh_camera_devices(log=True)
        else:
            self.view.append_log("[模式] 已切换到视频预览")
            if self._selected_video_path:
                self.view.set_video_path(self._selected_video_path)
                self.view.set_video_state("loaded")
                position_ms = int(self._video_meta.get("position_ms", 0)) if self._video_meta else 0
                self._start_probe(self._selected_video_path, position_ms)
            else:
                self.view.clear_video()

        self._set_idle_state()

    def _refresh_camera_devices(self, *, log: bool) -> None:
        devices: list[tuple[int, str]] = []
        for device_id in range(6):
            cap, backend_name = open_camera_capture(device_id, verify_frame=True, quiet=True)
            if cap.isOpened():
                devices.append((device_id, f"摄像头 {device_id} ({backend_name})"))
            cap.release()

        if not devices:
            devices = [(device_id, f"摄像头 {device_id} (手动)") for device_id in range(3)]

        self._camera_devices = devices
        self.view.set_camera_devices(devices)
        if log:
            if devices and "(手动)" not in devices[0][1]:
                labels = ", ".join(label for _, label in devices)
                self.view.append_log(f"[摄像头] 已发现设备: {labels}")
            else:
                self.view.append_log("[摄像头] 自动探测未读到画面，已提供 0/1/2 手动索引")
        self._set_idle_state()

    def _is_inference_running(self) -> bool:
        return bool(
            (self._playback_worker is not None and self._playback_worker.isRunning())
            or (self._camera_worker is not None and self._camera_worker.isRunning())
        )

    def _model_dialog_start_dir(self, current_path: str, default_path: str) -> str:
        current = self._resolve_model_path(current_path) if current_path.strip() else Path(default_path)
        if current.exists():
            return str(current.parent)
        if current.parent.exists():
            return str(current.parent)
        return str(Path(default_path).parent)

    def handle_browse_pose_model(self) -> None:
        pose_model_path, _track_model_path = self.view.model_settings()
        file_path, _ = QFileDialog.getOpenFileName(
            self.view,
            "选择骨骼模型",
            self._model_dialog_start_dir(pose_model_path, self._default_pose_model_path),
            "模型文件 (*.pt *.pth *.onnx *.engine *.ckpt);;所有文件 (*)",
        )
        if file_path:
            self.view.pose_model_edit.setText(file_path)

    def handle_browse_track_model(self) -> None:
        _pose_model_path, track_model_path = self.view.model_settings()
        file_path, _ = QFileDialog.getOpenFileName(
            self.view,
            "选择球轨迹模型",
            self._model_dialog_start_dir(track_model_path, self._default_track_model_path),
            "模型文件 (*.pt *.pth *.onnx *.engine *.ckpt);;所有文件 (*)",
        )
        if file_path:
            self.view.track_model_edit.setText(file_path)

    def handle_model_settings_defaults(self) -> None:
        self.view.set_model_settings(self._default_pose_model_path, self._default_track_model_path)
        self.view.set_model_switches(True, True)
        self.handle_model_switches_changed(True, True)
        self.handle_model_settings_apply(self._default_pose_model_path, self._default_track_model_path)

    def handle_model_switches_changed(self, pose_enabled: bool, track_enabled: bool) -> None:
        if self._is_inference_running():
            self.view.set_model_switches(self._pose_model_enabled, self._track_model_enabled)
            self.view.append_log("[设置] 请先停止当前推理，再切换模型开关。")
            return

        self._pose_model_enabled = pose_enabled
        self._track_model_enabled = track_enabled
        self._settings.setValue("pose_model_enabled", pose_enabled)
        self._settings.setValue("track_model_enabled", track_enabled)
        self._settings.sync()
        pose_text = "启用" if pose_enabled else "关闭"
        track_text = "启用" if track_enabled else "关闭"
        self.view.append_log(f"[设置] 模型开关已更新 | 骨骼 {pose_text} | 球轨迹 {track_text}")

    def handle_model_settings_apply(self, pose_model_path: str, track_model_path: str) -> None:
        if self._is_inference_running():
            self.view.append_log("[设置] 请先停止当前推理，再切换模型。")
            return

        pose_path = self._resolve_model_path(pose_model_path)
        track_path = self._resolve_model_path(track_model_path)
        missing_paths = [
            ("骨骼模型", pose_path),
            ("球轨迹模型", track_path),
        ]
        for label, path in missing_paths:
            if not path.is_file():
                self.view.set_status_state("error")
                self.view.append_log(f"[设置] {label}文件不存在: {path}")
                return

        pose_path_text = str(pose_path)
        track_path_text = str(track_path)
        requested_backend = "tensorrt" if track_path.suffix.lower() == ".engine" else "pytorch"
        if pose_path_text == self._pose_model_path and track_path_text == self._track_model_path:
            self.view.append_log(
                f"[设置] 模型路径未变化，当前球轨迹后端: {getattr(self._track_branch, 'backend_name', 'unknown')}"
            )
            return

        self.view.append_log(
            f"[设置] 正在应用模型 | 球轨迹: {track_path.name} | 目标后端 {requested_backend}"
        )
        self.view.set_model_settings_enabled(False)
        self.view.set_progress_busy(True, "正在加载模型")
        try:
            track_branch = self._create_track_branch(track_path_text)
            pose_branch = self._create_pose_branch(pose_path_text)
        except Exception as exc:
            self.view.set_progress_busy(False)
            self.view.set_model_settings_enabled(True)
            self.view.set_status_state("error")
            self.view.append_log(f"[设置] 模型加载失败: {exc}")
            self.view.append_log(
                f"[设置] 已保持当前模型 | 球轨迹: {Path(self._track_model_path).name} | "
                f"后端 {getattr(self._track_branch, 'backend_name', 'unknown')}"
            )
            self.view.set_model_settings(self._pose_model_path, self._track_model_path)
            return

        self._stop_workers(clear_pending_seek=True)
        self._track_branch = track_branch
        self._pose_branch = pose_branch
        self._pose_model_path = pose_path_text
        self._track_model_path = track_path_text
        self._settings.setValue("pose_model_path", pose_path_text)
        self._settings.setValue("track_model_path", track_path_text)
        self._settings.sync()
        self.view.set_model_settings(pose_path_text, track_path_text)
        self.view.set_progress_busy(False)
        self.view.append_log(f"[设置] 骨骼模型已应用: {pose_path.name}")
        self.view.append_log(f"[设置] 球轨迹模型已应用: {track_path.name} | 后端 {track_branch.backend_name}")
        self._set_idle_state()

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
        if self._input_mode == "camera":
            self._start_camera_inference()
            return

        if not self._selected_video_path:
            self.view.append_log("[警告] 开始分析前请先选择视频。")
            return

        self._pending_seek_ms = None
        start_ms = int(self._video_meta.get("position_ms", 0)) if self._video_meta else 0
        self._start_playback(start_ms=start_ms, request_court_prediction=True)

    def _start_camera_inference(self) -> None:
        camera_index = self.view.selected_camera_device()
        if camera_index is None:
            self.view.append_log("[警告] 请先选择可用摄像头设备。")
            return

        self._stop_workers(clear_pending_seek=True)
        self._reset_court_detection(request_initial_prediction=True)
        self._set_running_state()
        self.view.video_player.set_live_source(f"摄像头 {camera_index}")
        self.view.video_player.play()
        self.view.set_progress_busy(True, "实时推理中")
        self.view.append_log(
            f"[TrackNet] 开始摄像头实时推理: 摄像头 {camera_index} | "
            f"球轨迹 {'启用' if self._track_model_enabled else '关闭'} | "
            f"骨骼 {'启用' if self._pose_model_enabled else '关闭'}"
        )

        self._camera_worker = CameraInferenceWorker(
            camera_index,
            self._track_branch,
            self._pose_branch,
            pose_stride=3,
            track_enabled=self._track_model_enabled,
            pose_enabled=self._pose_model_enabled,
            court_service=self._court_service,
        )
        self._camera_worker.frameReady.connect(self._on_camera_frame_ready)
        self._camera_worker.inferFinished.connect(self._on_camera_finished)
        self._camera_worker.failed.connect(self._on_camera_failed)
        self._camera_worker.start()

    def _start_playback(self, *, start_ms: int = 0, request_court_prediction: bool = True) -> None:
        if self._selected_video_path is None:
            return

        self._stop_workers(clear_pending_seek=False)
        self._reset_court_detection(request_initial_prediction=request_court_prediction)
        self._set_running_state()
        self.view.video_player.play()
        self.view.append_log(
            f"[TrackNet] 开始播放: {Path(self._selected_video_path).name} | "
            f"球轨迹 {'启用' if self._track_model_enabled else '关闭'} | "
            f"骨骼 {'启用' if self._pose_model_enabled else '关闭'}"
        )

        self._playback_worker = TrackNetPlaybackWorker(
            self._selected_video_path,
            self._track_branch,
            self._pose_branch,
            start_ms=start_ms,
            pose_stride=3,
            track_enabled=self._track_model_enabled,
            pose_enabled=self._pose_model_enabled,
            court_service=self._court_service,
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
                payload.get("court"),
            )
            self._update_display_fps()

        progress = max(0, min(int(float(payload.get("progress", 0.0)) * 100), 100))
        self.view.update_progress(progress)
        if not self._should_update_metrics_text():
            return

        processed_frames = int(payload.get("processed_frames", 0))
        visible_frames = int(payload.get("visible_frames", 0))
        person_count = int(payload.get("person_count", 0))
        avg_score = float(payload.get("avg_score", 0.0))
        infer_fps = float(payload.get("infer_fps", 0.0))
        track = payload.get("track", {})
        current_score = float(track.get("score", 0.0)) if isinstance(track, dict) else 0.0
        court = payload.get("court", {})
        court_text = ""
        if isinstance(court, dict) and court.get("valid"):
            court_text = f" | Court {float(court.get('confidence', 0.0)):.2f}"

        self.view.lbl_valid_pose.setText(f"{infer_fps:.1f} FPS")
        self.view.lbl_valid_track.setText(str(visible_frames))
        self.view.lbl_avg_conf.setText(f"{avg_score * 100:.1f}%")
        self.view.lbl_realtime_fps.setText(f"{self._display_fps_ema:.1f} FPS")
        self.view.status_label.setText(
            f"系统状态：TrackNet + YOLO26s-Pose 运行中 | 人数 {person_count} | Score {current_score:.2f}{court_text}"
        )

    def _on_camera_frame_ready(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return

        image = payload.get("image")
        if isinstance(image, QImage):
            self.view.video_player.display_image(image, court=payload.get("court"))
            self._update_display_fps()
        if not self._should_update_metrics_text():
            return

        processed_frames = int(payload.get("processed_frames", 0))
        visible_frames = int(payload.get("visible_frames", 0))
        person_count = int(payload.get("person_count", 0))
        avg_score = float(payload.get("avg_score", 0.0))
        track = payload.get("track", {})
        current_score = float(track.get("score", 0.0)) if isinstance(track, dict) else 0.0
        infer_fps = float(payload.get("infer_fps", 0.0))
        court = payload.get("court", {})
        court_text = ""
        if isinstance(court, dict) and court.get("valid"):
            court_text = f" | Court {float(court.get('confidence', 0.0)):.2f}"

        self.view.lbl_valid_pose.setText(f"{infer_fps:.1f} FPS")
        self.view.lbl_valid_track.setText(str(visible_frames))
        self.view.lbl_avg_conf.setText(f"{avg_score * 100:.1f}%")
        self.view.lbl_realtime_fps.setText(f"{self._display_fps_ema:.1f} FPS")
        self.view.status_label.setText(
            f"系统状态：摄像头 TrackNet + YOLO26s-Pose 推理中 | 人数 {person_count} | Score {current_score:.2f}{court_text}"
        )

    def _on_camera_finished(self, payload: object) -> None:
        self._camera_worker = None
        self.view.set_progress_busy(False)
        stopped = bool(payload.get("stopped")) if isinstance(payload, dict) else False

        if stopped:
            self.view.set_status_state("stopped")
            self.view.video_player.stop()
            self.view.append_log("[TrackNet] 摄像头实时推理已停止。")
        else:
            self.view.set_status_state("success")
            self.view.video_player.stop()
            if isinstance(payload, dict):
                self.view.append_log(
                    f"[TrackNet] 摄像头实时推理结束 | "
                    f"帧数 {int(payload.get('processed_frames', 0))} | "
                    f"可见 {int(payload.get('visible_frames', 0))} | "
                    f"平均 {float(payload.get('avg_score', 0.0)) * 100:.1f}%"
                )
            else:
                self.view.append_log("[TrackNet] 摄像头实时推理结束。")

        self._set_idle_state()

    def _on_camera_failed(self, message: str) -> None:
        self._camera_worker = None
        self.view.set_progress_busy(False)
        self.view.set_status_state("error")
        self.view.video_player.stop()
        self.view.append_log(f"[错误] {message}")
        self._set_idle_state()

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
            self._start_playback(start_ms=pending_seek_ms, request_court_prediction=False)

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
        if self._camera_worker is not None and self._camera_worker.isRunning():
            self.view.append_log("[信息] 正在停止摄像头实时推理...")
            self._camera_worker.request_stop()
            return

        if self._playback_worker is not None and self._playback_worker.isRunning():
            self.view.append_log("[信息] 正在停止 TrackNetV3 播放...")
            self._playback_worker.request_stop()
            return

        self.view.video_player.stop()
        self.view.set_status_state("stopped")
        self.view.append_log("[信息] 没有正在进行的播放任务。")

    def handle_reset(self) -> None:
        self._stop_workers(clear_pending_seek=True)
        self._reset_court_detection()
        self._selected_video_path = None
        self._video_meta = {}
        self.view.clear_video()
        self.view.set_input_mode(self._input_mode)
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

        if self._camera_worker is not None and self._camera_worker.isRunning():
            self._camera_worker.request_stop()
            self._camera_worker.wait(1000)
        self._camera_worker = None
        if self._court_service is not None:
            self._court_service.clear_pending()
        self.view.set_progress_busy(False)

    def shutdown(self) -> None:
        self._stop_workers(clear_pending_seek=True)
        if self._court_service is not None:
            self._court_service.stop()

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
        if theme_dir.name == self._active_theme_name:
            return

        def _apply() -> None:
            apply_theme(app, theme_dir)
            self._active_theme_name = theme_dir.name
            self.view.append_log(f"[主题] 已切换至 {theme_dir.name}")

        QTimer.singleShot(
            0,
            lambda: start_theme_ripple_transition(
                self.view,
                _apply,
                origin_widget=self.view.style_btn,
            ),
        )


MockController = MainController
