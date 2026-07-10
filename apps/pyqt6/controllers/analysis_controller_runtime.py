# -*- coding: utf-8 -*-
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import traceback
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
from src.court.batch_court import BatchCourtPredictor
from src.models.bst_runtime import build_bst_model
from src.models.bst_stroke_runtime import BSTStrokeRecognizer
from src.models.pose_branch import PoseBranch
from src.models.track_branch import TrackBranch
from src.postprocess.player_distance import PlayerDistanceAccumulator
from src.postprocess.pose import CourtPoseTargetTracker
from src.postprocess.rally_stats import RallyStatsAccumulator
from src.postprocess.tracknet_v3_filter import create_tracknet_v3_ball_track_filter
from src.postprocess.trajectory_events import RealtimeTrajectoryEventDetector
from src.utils.exporters import TRACK_DEBUG_FIELDS, frame_result_log_record, write_frame_log_jsonl
from src.utils.report_generation import OpenAICompatibleReportApiClient, generate_report_from_rally_record
from src.utils.structures import FrameResult, PersonPoseResult, TrackResult
from src.utils.visualize import TrackTrailRenderer


DISPLAY_FPS_LIMIT = 60.0
METRICS_UPDATE_INTERVAL_S = 0.25
PLAYBACK_LAG_TOLERANCE_FRAMES = 1.5
POSE_CANDIDATE_LIMIT = 12
POSE_COURT_MARGIN_CM = 30.0
POSE_MAX_MISSING_SECONDS = 0.35
POSE_INFERENCE_STRIDE = 2
POSE_YOLO_IMGSZ = 960
POSE_CROP_IMGSZ = 640
POSE_CROP_PADDING = 0.30
DEFAULT_REPORT_API_ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
DEFAULT_REPORT_API_MODEL = "qwen-plus"
POSE_CROP_MIN_BOX_CONF = 0.45
POSE_MAX_CROPS = 8


def _flush_stderr_if_available() -> None:
    if sys.stderr is None:
        return
    try:
        sys.stderr.flush()
    except OSError:
        return


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
            _flush_stderr_if_available()
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
                _flush_stderr_if_available()
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


def open_track_debug_csv(path: str | None) -> tuple[object | None, csv.DictWriter | None]:
    if not path:
        return None, None
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    debug_file = output_path.open("w", newline="", encoding="utf-8-sig")
    writer = csv.DictWriter(debug_file, fieldnames=TRACK_DEBUG_FIELDS, extrasaction="ignore")
    writer.writeheader()
    return debug_file, writer


def write_track_debug_row(writer: csv.DictWriter | None, record: object) -> None:
    if writer is None or not isinstance(record, dict):
        return
    writer.writerow(record)


def frame_step_seconds(current_ms: int, previous_ms: int | None, fps: float) -> float:
    fallback = 1.0 / fps if fps > 0 else 1.0 / 25.0
    if previous_ms is None:
        return fallback
    elapsed = (int(current_ms) - int(previous_ms)) / 1000.0
    return elapsed if elapsed > 0.0 else fallback


def open_frame_log_jsonl(path: str | None) -> object | None:
    if not path:
        return None
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path.open("w", encoding="utf-8")


def project_poses_to_court(
    poses: list[PersonPoseResult],
    court_prediction: object | None,
) -> list[tuple[float, float]]:
    projected_by_player = project_player_points_to_court(poses, court_prediction)
    return [
        projected_by_player[index]
        for index in sorted(projected_by_player)
    ]


def project_player_points_to_court(
    poses: list[PersonPoseResult],
    court_prediction: object | None,
) -> dict[int, tuple[float, float]]:
    h = extract_image_to_court_h(court_prediction)
    if h is None:
        return {}

    projected: dict[int, tuple[float, float]] = {}
    for fallback_index, pose in enumerate(poses):
        anchor = pose_ground_anchor(pose)
        if anchor is None:
            continue
        court_xy = project_image_point(h, anchor)
        if court_xy is not None:
            person_index = _person_projection_index(pose, fallback_index)
            if person_index is not None:
                projected[person_index] = court_xy
    return projected


def project_ball_visual_intersection(
    track: TrackResult,
    court_prediction: object | None,
) -> tuple[float, float] | None:
    # This is the image ray's apparent intersection with the court plane,
    # not the shuttlecock's physical ground position while it is airborne.
    if not bool(getattr(track, "visible", 0)):
        return None
    h = extract_image_to_court_h(court_prediction)
    if h is None:
        return None
    ball_xy = getattr(track, "ball_xy", None)
    if not isinstance(ball_xy, (list, tuple)) or len(ball_xy) < 2:
        return None
    try:
        x = float(ball_xy[0])
        y = float(ball_xy[1])
    except (TypeError, ValueError):
        return None
    if not _is_finite(x) or not _is_finite(y):
        return None
    return project_image_point(h, (x, y))


def pose_person_bboxes(poses: list[PersonPoseResult]) -> list[tuple[float, float, float, float]]:
    bboxes: list[tuple[float, float, float, float]] = []
    for pose in poses:
        if len(pose.bbox) < 4:
            continue
        try:
            x1, y1, x2, y2 = [float(value) for value in pose.bbox[:4]]
        except (TypeError, ValueError):
            continue
        if not all(_is_finite(value) for value in (x1, y1, x2, y2)):
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        bboxes.append((x1, y1, x2, y2))
    return bboxes


def _person_projection_index(pose: PersonPoseResult, fallback_index: int) -> int | None:
    try:
        person_id = int(getattr(pose, "person_id", fallback_index))
    except (TypeError, ValueError):
        person_id = fallback_index
    if 0 <= person_id < 2:
        return person_id
    if 0 <= fallback_index < 2:
        return fallback_index
    return None


def pose_ground_anchor(pose: PersonPoseResult) -> tuple[float, float] | None:
    ankle_points = []
    for index in (15, 16):
        if index >= len(pose.keypoints):
            continue
        if index < len(pose.scores) and float(pose.scores[index]) < 0.20:
            continue
        point = pose.keypoints[index]
        if len(point) < 2:
            continue
        x, y = float(point[0]), float(point[1])
        if _is_finite(x) and _is_finite(y):
            ankle_points.append((x, y))
    if ankle_points:
        return (
            sum(point[0] for point in ankle_points) / len(ankle_points),
            sum(point[1] for point in ankle_points) / len(ankle_points),
        )

    if len(pose.bbox) >= 4:
        x1, _, x2, y2 = [float(value) for value in pose.bbox[:4]]
        if all(_is_finite(value) for value in (x1, x2, y2)) and x2 > x1:
            return (x1 + x2) * 0.5, y2
    return None


def extract_image_to_court_h(court_prediction: object | None) -> tuple[tuple[float, float, float], ...] | None:
    if court_prediction is None or not prediction_value(court_prediction, "valid", False):
        return None
    raw_h = prediction_value(court_prediction, "image_to_court_h", None)
    if raw_h is None:
        return None
    try:
        rows = tuple(tuple(float(value) for value in row) for row in raw_h)
    except (TypeError, ValueError):
        return None
    if len(rows) != 3 or any(len(row) != 3 for row in rows):
        return None
    if any(not _is_finite(value) for row in rows for value in row):
        return None
    return rows


def project_image_point(
    h: tuple[tuple[float, float, float], ...],
    point: tuple[float, float],
) -> tuple[float, float] | None:
    x, y = point
    u = h[0][0] * x + h[0][1] * y + h[0][2]
    v = h[1][0] * x + h[1][1] * y + h[1][2]
    w = h[2][0] * x + h[2][1] * y + h[2][2]
    if abs(w) < 1e-9:
        return None
    court_x = u / w
    court_y = v / w
    if not _is_finite(court_x) or not _is_finite(court_y):
        return None
    return float(court_x), float(court_y)


def prediction_value(prediction: object, key: str, default: object) -> object:
    if isinstance(prediction, dict):
        return prediction.get(key, default)
    return getattr(prediction, key, default)


def _is_finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


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
        debug_csv_path: str | None = None,
        frame_log_path: str | None = None,
        bst_model: Any | None = None,
        bst_device: str = "cpu",
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
        self._debug_csv_path = debug_csv_path
        self._frame_log_path = frame_log_path
        self._bst_model = bst_model
        self._bst_device = bst_device
        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

    def _pipeline_label(self) -> str:
        names = []
        if self._track_enabled:
            names.append("TrackNet")
        if self._pose_enabled:
            names.append("YOLO26s-Pose")
        if self._bst_model is not None and self._track_enabled and self._pose_enabled:
            names.append("BST")
        return " + ".join(names) if names else "Preview"

    def _create_bst_recognizer(self, width: int, height: int, fps: float) -> BSTStrokeRecognizer | None:
        if self._bst_model is None or not self._track_enabled or not self._pose_enabled:
            return None
        return BSTStrokeRecognizer(
            self._bst_model,
            self._bst_device,
            max(1, int(width)),
            max(1, int(height)),
            fps=fps,
        )

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
        try:
            self._run_impl()
        except Exception as exc:
            traceback.print_exc()
            self.failed.emit(f"播放推理失败: {exc}")

    def _run_impl(self) -> None:
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
        previous_track_ms: int | None = None
        track_filter = create_tracknet_v3_ball_track_filter(fps=fps, debug_enabled=self._debug_csv_path is not None)
        pose_tracker = CourtPoseTargetTracker(
            max_missing_frames=max(self._pose_stride, int(round(fps * POSE_MAX_MISSING_SECONDS))),
            court_margin=POSE_COURT_MARGIN_CM,
            detection_smoothing=0.78,
            velocity_smoothing=0.50,
            court_required=False,
            predict_missing_motion=True,
            motion_prediction_scale=0.55,
        )
        distance_accumulator = PlayerDistanceAccumulator()
        rally_stats = RallyStatsAccumulator(
            rally_id=self._video_path,
            rally_name=Path(self._video_path).name,
            fps=fps,
            freeze_after_rally_end=False,
        )
        trail_renderer = TrackTrailRenderer(fps=fps, history_seconds=0.5)
        event_detector = RealtimeTrajectoryEventDetector(fps=fps)
        frame_height, frame_width = current_frame.shape[:2]
        bst_recognizer = self._create_bst_recognizer(frame_width, frame_height, fps)
        pending_bst_predictions: list[dict[str, Any]] = []
        pending_bst_errors: list[str] = []
        display_interval_ms = 1000.0 / self._display_fps_limit
        display_every_frame = fps <= self._display_fps_limit
        next_display_ms = float(base_ms)

        clock = QElapsedTimer()
        clock.start()
        parallel_inference = (
            self._track_enabled
            and self._pose_enabled
            and getattr(self._track_branch, "backend_name", "") == "tensorrt"
        )
        debug_file, debug_writer = open_track_debug_csv(self._debug_csv_path)
        frame_log_file = open_frame_log_jsonl(self._frame_log_path)

        try:
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="wfb-playback-infer") as infer_executor:
                while not self._stop_requested:
                    loop_start = perf_counter()
                    frame_id = int(round((current_ms / 1000.0) * fps)) if fps > 0 else processed_frames
                    court_prediction = None
                    if self._court_service is not None:
                        self._court_service.submit_frame(current_frame, frame_id, current_ms)
                        court_prediction = self._court_service.latest_prediction()

                    pose_due = self._pose_enabled and processed_frames % self._pose_stride == 0
                    run_parallel = parallel_inference and pose_due
                    if run_parallel:
                        track_future = infer_executor.submit(
                            self._track_branch.infer_candidate_results,
                            [prev_frame, current_frame, next_frame],
                        )
                        pose_future = infer_executor.submit(
                            self._pose_branch.infer,
                            current_frame,
                            court_prediction=court_prediction,
                        )
                        candidates = track_future.result()
                        detections = pose_future.result()
                    else:
                        if self._track_enabled:
                            candidates = self._track_branch.infer_candidate_results([prev_frame, current_frame, next_frame])
                        else:
                            candidates = []
                        detections = (
                            self._pose_branch.infer(current_frame, court_prediction=court_prediction)
                            if pose_due
                            else []
                        )

                    if self._pose_enabled:
                        last_pose = pose_tracker.update(
                            detections,
                            court_prediction,
                            frame_shape=current_frame.shape,
                        )
                        pose_frames += int(bool(last_pose))
                    else:
                        pose_tracker.reset()
                        last_pose = []

                    if self._track_enabled:
                        track_dt = frame_step_seconds(current_ms, previous_track_ms, fps)
                        track = track_filter.update_candidates(
                            candidates,
                            dt=track_dt,
                            frame_shape=current_frame.shape,
                            court_prediction=court_prediction,
                            person_bboxes=pose_person_bboxes(last_pose),
                        )
                        previous_track_ms = current_ms
                    else:
                        track = TrackResult(ball_xy=[-1.0, -1.0], visible=0, score=0.0)

                    infer_elapsed = max(perf_counter() - loop_start, 1e-6)
                    infer_fps = 1.0 / infer_elapsed
                    ema_infer_fps = infer_fps if ema_infer_fps == 0.0 else (0.85 * ema_infer_fps + 0.15 * infer_fps)

                    frame_result = FrameResult(frame_id=frame_id, pose=last_pose, track=track)
                    trajectory_event = event_detector.update(
                        frame_result,
                        timestamp_ms=current_ms,
                        frame_shape=current_frame.shape,
                    )
                    hit_event = (
                        trajectory_event
                        if isinstance(trajectory_event, dict) and trajectory_event.get("event_type") == "hit"
                        else None
                    )
                    landing_event = (
                        trajectory_event
                        if isinstance(trajectory_event, dict) and trajectory_event.get("event_type") == "landing"
                        else None
                    )
                    should_emit = display_every_frame or final_pass or current_ms >= next_display_ms
                    image = None
                    if should_emit:
                        vis_frame = current_frame.copy()
                        trail_renderer.draw_on(
                            vis_frame,
                            frame_result,
                            timestamp_ms=current_ms,
                            trajectory_event=trajectory_event,
                        )
                        image = frame_to_qimage(vis_frame)
                    else:
                        trail_renderer.update_track_history(
                            frame_result,
                            timestamp_ms=current_ms,
                        )
                        trail_renderer.add_trajectory_event(trajectory_event)
                    write_frame_log_jsonl(
                        frame_log_file,
                        frame_result_log_record(
                            frame_result,
                            timestamp_ms=current_ms,
                            court_prediction=court_prediction,
                            hit_event=hit_event,
                            trajectory_event=trajectory_event,
                            landing_event=landing_event,
                        ),
                    )

                    if bst_recognizer is not None:
                        try:
                            bst_prediction = bst_recognizer.update(
                                frame_result,
                                hit_event=hit_event,
                                court_prediction=court_prediction,
                            )
                        except Exception as exc:
                            pending_bst_errors.append(str(exc))
                            bst_recognizer = None
                        else:
                            if bst_prediction is not None:
                                pending_bst_predictions.append(bst_prediction)

                    target_ms = max(0, current_ms - base_ms)
                    if not self._sleep_until(target_ms, clock):
                        break

                    processed_frames += 1
                    visible_frames += int(bool(track.visible))
                    score_sum += float(track.score)
                    avg_score = score_sum / max(processed_frames, 1)
                    track_debug = track_filter.last_debug_record()
                    player_points = project_player_points_to_court(last_pose, court_prediction)
                    player_projections = [player_points[index] for index in sorted(player_points)]
                    if player_points:
                        player_distances_m = distance_accumulator.update(player_points)
                    else:
                        distance_accumulator.reset_tracking_points()
                        player_distances_m = distance_accumulator.totals_m()
                    ball_visual_intersection = project_ball_visual_intersection(track, court_prediction)
                    rally_stats.update_frame(
                        timestamp_ms=current_ms,
                        player_points=player_points,
                        ball_visible=bool(track.visible),
                        ball_xy=track.ball_xy,
                        ball_court_xy=ball_visual_intersection,
                        ball_score=float(track.score),
                        court_valid=bool(prediction_value(court_prediction, "valid", False))
                        if court_prediction is not None
                        else False,
                    )
                    rally_stats.add_trajectory_event(trajectory_event)
                    for prediction in pending_bst_predictions:
                        rally_stats.add_bst_prediction(prediction)
                    rally_record = rally_stats.export_record()
                    write_track_debug_row(debug_writer, track_debug)
                    track_filter.debug_records.clear()
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
                            "ball_projection": ball_visual_intersection,
                            "ball_visual_intersection": ball_visual_intersection,
                            "player_projections": player_projections,
                            "player_distances_m": player_distances_m,
                            "rally_record": rally_record,
                            "track_debug": track_debug,
                            "trajectory_event": trajectory_event,
                            "landing_event": landing_event,
                            "bst_predictions": list(pending_bst_predictions),
                            "bst_errors": list(pending_bst_errors),
                        }
                        pending_bst_predictions.clear()
                        pending_bst_errors.clear()
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
            if debug_file is not None:
                debug_file.close()
            if frame_log_file is not None:
                frame_log_file.close()

        self.playbackFinished.emit(
            {
                "stopped": self._stop_requested,
                "processed_frames": processed_frames,
                "dropped_source_frames": dropped_source_frames,
                "visible_frames": visible_frames,
                "pose_frames": pose_frames,
                "avg_score": (score_sum / processed_frames) if processed_frames else 0.0,
                "player_distances_m": distance_accumulator.totals_m(),
                "rally_record": rally_stats.export_record(),
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
        debug_csv_path: str | None = None,
        frame_log_path: str | None = None,
        bst_model: Any | None = None,
        bst_device: str = "cpu",
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
        self._debug_csv_path = debug_csv_path
        self._frame_log_path = frame_log_path
        self._bst_model = bst_model
        self._bst_device = bst_device
        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

    def _pipeline_label(self) -> str:
        names = []
        if self._track_enabled:
            names.append("TrackNet")
        if self._pose_enabled:
            names.append("YOLO26s-Pose")
        if self._bst_model is not None and self._track_enabled and self._pose_enabled:
            names.append("BST")
        return " + ".join(names) if names else "Preview"

    def _create_bst_recognizer(self, width: int, height: int, fps: float) -> BSTStrokeRecognizer | None:
        if self._bst_model is None or not self._track_enabled or not self._pose_enabled:
            return None
        return BSTStrokeRecognizer(
            self._bst_model,
            self._bst_device,
            max(1, int(width)),
            max(1, int(height)),
            fps=fps,
        )

    def run(self) -> None:
        try:
            self._run_impl()
        except Exception as exc:
            traceback.print_exc()
            self.failed.emit(f"摄像头推理失败: {exc}")

    def _run_impl(self) -> None:
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
        previous_track_ms: int | None = None
        track_filter = create_tracknet_v3_ball_track_filter(fps=fps, debug_enabled=self._debug_csv_path is not None)
        pose_tracker = CourtPoseTargetTracker(
            max_missing_frames=max(self._pose_stride, int(round(fps * POSE_MAX_MISSING_SECONDS))),
            court_margin=POSE_COURT_MARGIN_CM,
            detection_smoothing=0.78,
            velocity_smoothing=0.50,
            court_required=False,
            predict_missing_motion=True,
            motion_prediction_scale=0.55,
        )
        distance_accumulator = PlayerDistanceAccumulator()
        rally_stats = RallyStatsAccumulator(
            rally_id=f"camera_{self._camera_index}",
            rally_name=f"摄像头 {self._camera_index}",
            fps=fps,
            freeze_after_rally_end=False,
        )
        trail_renderer = TrackTrailRenderer(fps=fps, history_seconds=0.5)
        event_detector = RealtimeTrajectoryEventDetector(fps=fps)
        frame_height, frame_width = first_frame.shape[:2]
        bst_recognizer = self._create_bst_recognizer(frame_width, frame_height, fps)
        pending_bst_predictions: list[dict[str, Any]] = []
        pending_bst_errors: list[str] = []
        display_interval_ms = 1000.0 / self._display_fps_limit
        next_display_ms = 0.0
        clock = QElapsedTimer()
        clock.start()
        parallel_inference = (
            self._track_enabled
            and self._pose_enabled
            and getattr(self._track_branch, "backend_name", "") == "tensorrt"
        )
        debug_file, debug_writer = open_track_debug_csv(self._debug_csv_path)
        frame_log_file = open_frame_log_jsonl(self._frame_log_path)

        try:
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="wfb-camera-infer") as infer_executor:
                while not self._stop_requested:
                    loop_start = perf_counter()
                    current_frame_id = processed_frames
                    position_ms = clock.elapsed()
                    court_prediction = None
                    if self._court_service is not None:
                        self._court_service.submit_frame(current_frame, current_frame_id, position_ms)
                        court_prediction = self._court_service.latest_prediction()

                    pose_due = self._pose_enabled and processed_frames % self._pose_stride == 0
                    run_parallel = parallel_inference and pose_due
                    if run_parallel:
                        track_future = infer_executor.submit(
                            self._track_branch.infer_candidate_results,
                            [prev_frame, current_frame, next_frame],
                        )
                        pose_future = infer_executor.submit(
                            self._pose_branch.infer,
                            current_frame,
                            court_prediction=court_prediction,
                        )
                        candidates = track_future.result()
                        detections = pose_future.result()
                    else:
                        if self._track_enabled:
                            candidates = self._track_branch.infer_candidate_results([prev_frame, current_frame, next_frame])
                        else:
                            candidates = []
                        detections = (
                            self._pose_branch.infer(current_frame, court_prediction=court_prediction)
                            if pose_due
                            else []
                        )

                    if self._pose_enabled:
                        last_pose = pose_tracker.update(
                            detections,
                            court_prediction,
                            frame_shape=current_frame.shape,
                        )
                        pose_frames += int(bool(last_pose))
                    else:
                        pose_tracker.reset()
                        last_pose = []

                    if self._track_enabled:
                        track_dt = frame_step_seconds(position_ms, previous_track_ms, fps)
                        track = track_filter.update_candidates(
                            candidates,
                            dt=track_dt,
                            frame_shape=current_frame.shape,
                            court_prediction=court_prediction,
                            person_bboxes=pose_person_bboxes(last_pose),
                        )
                        previous_track_ms = position_ms
                    else:
                        track = TrackResult(ball_xy=[-1.0, -1.0], visible=0, score=0.0)

                    infer_elapsed = max(perf_counter() - loop_start, 1e-6)
                    infer_fps = 1.0 / infer_elapsed
                    ema_infer_fps = infer_fps if ema_infer_fps == 0.0 else (0.85 * ema_infer_fps + 0.15 * infer_fps)

                    processed_frames += 1
                    visible_frames += int(bool(track.visible))
                    score_sum += float(track.score)
                    avg_score = score_sum / max(processed_frames, 1)
                    track_debug = track_filter.last_debug_record()
                    player_points = project_player_points_to_court(last_pose, court_prediction)
                    player_projections = [player_points[index] for index in sorted(player_points)]
                    if player_points:
                        player_distances_m = distance_accumulator.update(player_points)
                    else:
                        distance_accumulator.reset_tracking_points()
                        player_distances_m = distance_accumulator.totals_m()
                    ball_visual_intersection = project_ball_visual_intersection(track, court_prediction)
                    rally_stats.update_frame(
                        timestamp_ms=position_ms,
                        player_points=player_points,
                        ball_visible=bool(track.visible),
                        ball_xy=track.ball_xy,
                        ball_court_xy=ball_visual_intersection,
                        ball_score=float(track.score),
                        court_valid=bool(prediction_value(court_prediction, "valid", False))
                        if court_prediction is not None
                        else False,
                    )
                    write_track_debug_row(debug_writer, track_debug)
                    track_filter.debug_records.clear()
                    frame_result = FrameResult(frame_id=current_frame_id, pose=last_pose, track=track)
                    trajectory_event = event_detector.update(
                        frame_result,
                        timestamp_ms=position_ms,
                        frame_shape=current_frame.shape,
                    )
                    hit_event = (
                        trajectory_event
                        if isinstance(trajectory_event, dict) and trajectory_event.get("event_type") == "hit"
                        else None
                    )
                    landing_event = (
                        trajectory_event
                        if isinstance(trajectory_event, dict) and trajectory_event.get("event_type") == "landing"
                        else None
                    )
                    if position_ms >= next_display_ms:
                        vis_frame = current_frame.copy()
                        trail_renderer.draw_on(
                            vis_frame,
                            frame_result,
                            timestamp_ms=position_ms,
                            trajectory_event=trajectory_event,
                        )
                        image = frame_to_qimage(vis_frame)
                    else:
                        trail_renderer.update_track_history(
                            frame_result,
                            timestamp_ms=position_ms,
                        )
                        trail_renderer.add_trajectory_event(trajectory_event)
                    write_frame_log_jsonl(
                        frame_log_file,
                        frame_result_log_record(
                            frame_result,
                            timestamp_ms=position_ms,
                            court_prediction=court_prediction,
                            hit_event=hit_event,
                            trajectory_event=trajectory_event,
                            landing_event=landing_event,
                        ),
                    )

                    if bst_recognizer is not None:
                        try:
                            bst_prediction = bst_recognizer.update(
                                frame_result,
                                hit_event=hit_event,
                                court_prediction=court_prediction,
                            )
                        except Exception as exc:
                            pending_bst_errors.append(str(exc))
                            bst_recognizer = None
                        else:
                            if bst_prediction is not None:
                                pending_bst_predictions.append(bst_prediction)

                    rally_stats.add_trajectory_event(trajectory_event)
                    for prediction in pending_bst_predictions:
                        rally_stats.add_bst_prediction(prediction)
                    rally_record = rally_stats.export_record()

                    if position_ms >= next_display_ms:
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
                            "ball_projection": ball_visual_intersection,
                            "ball_visual_intersection": ball_visual_intersection,
                            "player_projections": player_projections,
                            "player_distances_m": player_distances_m,
                            "rally_record": rally_record,
                            "track_debug": track_debug,
                            "trajectory_event": trajectory_event,
                            "landing_event": landing_event,
                            "bst_predictions": list(pending_bst_predictions),
                            "bst_errors": list(pending_bst_errors),
                        }
                        pending_bst_predictions.clear()
                        pending_bst_errors.clear()
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
            if debug_file is not None:
                debug_file.close()
            if frame_log_file is not None:
                frame_log_file.close()

        self.inferFinished.emit(
            {
                "stopped": self._stop_requested,
                "processed_frames": processed_frames,
                "visible_frames": visible_frames,
                "pose_frames": pose_frames,
                "avg_score": (score_sum / processed_frames) if processed_frames else 0.0,
                "player_distances_m": distance_accumulator.totals_m(),
                "rally_record": rally_stats.export_record(),
            }
        )


class BatchInferenceWorker(QThread):
    progressChanged = pyqtSignal(object)
    rallyFinished = pyqtSignal(object)
    batchFinished = pyqtSignal(object)
    failed = pyqtSignal(str)

    VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}

    def __init__(
        self,
        folder_path: str,
        track_branch: TrackBranch,
        pose_branch: PoseBranch,
        *,
        pose_stride: int = 3,
        track_enabled: bool = True,
        pose_enabled: bool = True,
        bst_model: Any | None = None,
        bst_device: str = "cpu",
        court_backends: tuple[str, ...] | None = None,
    ) -> None:
        super().__init__()
        self._folder_path = folder_path
        self._track_branch = track_branch
        self._pose_branch = pose_branch
        self._pose_stride = max(1, pose_stride)
        self._track_enabled = track_enabled
        self._pose_enabled = pose_enabled
        self._bst_model = bst_model
        self._bst_device = bst_device
        self._court_backends = court_backends
        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

    def run(self) -> None:
        try:
            self._run_impl()
        except Exception as exc:
            traceback.print_exc()
            self.failed.emit(f"批量推理失败: {exc}")

    def _run_impl(self) -> None:
        folder = Path(self._folder_path)
        video_paths = self._video_paths(folder)
        if not video_paths:
            self.failed.emit(f"文件夹中没有可分析的视频: {folder}")
            return

        completed = 0
        failed_count = 0
        total = len(video_paths)
        for index, video_path in enumerate(video_paths):
            if self._stop_requested:
                break
            self.progressChanged.emit(
                {
                    "phase": "start_video",
                    "index": index,
                    "total": total,
                    "video_name": video_path.name,
                    "overall_progress": index / max(1, total),
                }
            )
            try:
                record = self._process_video(video_path, index=index, total=total)
            except Exception as exc:
                failed_count += 1
                record = {
                    "id": str(video_path),
                    "video_name": video_path.name,
                    "video_path": str(video_path),
                    "error": str(exc),
                    "summary": {},
                    "details": {"hits": [], "players": {}},
                }
            if record:
                completed += int(not record.get("error"))
                self.rallyFinished.emit(record)

        self.batchFinished.emit(
            {
                "stopped": self._stop_requested,
                "total": total,
                "completed": completed,
                "failed": failed_count,
            }
        )

    def _video_paths(self, folder: Path) -> list[Path]:
        if not folder.is_dir():
            return []
        return sorted(
            (
                path
                for path in folder.iterdir()
                if path.is_file() and path.suffix.lower() in self.VIDEO_SUFFIXES
            ),
            key=lambda path: path.name.lower(),
        )

    def _create_bst_recognizer(self, width: int, height: int, fps: float) -> BSTStrokeRecognizer | None:
        if self._bst_model is None or not self._track_enabled or not self._pose_enabled:
            return None
        return BSTStrokeRecognizer(
            self._bst_model,
            self._bst_device,
            max(1, int(width)),
            max(1, int(height)),
            fps=fps,
        )

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

    def _process_video(self, video_path: Path, *, index: int, total: int) -> dict[str, Any]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            fps = 25.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration_ms = int(round((frame_count / fps) * 1000)) if frame_count > 0 else 0
        frame_interval_ms = int(round(1000.0 / fps)) if fps > 0 else 40

        ok, current_frame, current_ms = self._read_frame(cap, 0, fps)
        if not ok:
            cap.release()
            raise RuntimeError("无法读取第一帧视频")

        ok, next_frame, next_ms = self._read_frame(cap, 1, fps)
        if not ok:
            next_frame = current_frame.copy()
            next_ms = current_ms + frame_interval_ms

        prev_frame = current_frame.copy()
        processed_frames = 0
        visible_frames = 0
        pose_frames = 0
        score_sum = 0.0
        final_pass = False
        last_pose = []
        previous_track_ms: int | None = None
        track_filter = create_tracknet_v3_ball_track_filter(fps=fps, debug_enabled=False)
        pose_tracker = CourtPoseTargetTracker(
            max_missing_frames=max(self._pose_stride, int(round(fps * POSE_MAX_MISSING_SECONDS))),
            court_margin=POSE_COURT_MARGIN_CM,
            detection_smoothing=0.78,
            velocity_smoothing=0.50,
            court_required=False,
            predict_missing_motion=True,
            motion_prediction_scale=0.55,
        )
        court_predictor = (
            BatchCourtPredictor(backends=tuple(self._court_backends))
            if self._court_backends
            else BatchCourtPredictor()
        )
        distance_accumulator = PlayerDistanceAccumulator()
        event_detector = RealtimeTrajectoryEventDetector(fps=fps)
        frame_height, frame_width = current_frame.shape[:2]
        bst_recognizer = self._create_bst_recognizer(frame_width, frame_height, fps)
        rally_stats = RallyStatsAccumulator(
            rally_id=str(video_path),
            rally_name=video_path.name,
            fps=fps,
            freeze_after_rally_end=False,
        )
        pending_bst_errors: list[str] = []
        parallel_inference = (
            self._track_enabled
            and self._pose_enabled
            and getattr(self._track_branch, "backend_name", "") == "tensorrt"
        )
        progress_every = max(1, int(round(fps)))

        try:
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="wfb-batch-infer") as infer_executor:
                while not self._stop_requested:
                    frame_id = int(round((current_ms / 1000.0) * fps)) if fps > 0 else processed_frames
                    court_prediction = court_predictor.predict(
                        current_frame,
                        frame_id,
                        current_ms,
                        force=processed_frames == 0,
                    )
                    pose_due = self._pose_enabled and processed_frames % self._pose_stride == 0
                    run_parallel = parallel_inference and pose_due
                    if run_parallel:
                        track_future = infer_executor.submit(
                            self._track_branch.infer_candidate_results,
                            [prev_frame, current_frame, next_frame],
                        )
                        pose_future = infer_executor.submit(
                            self._pose_branch.infer,
                            current_frame,
                            court_prediction=court_prediction,
                        )
                        candidates = track_future.result()
                        detections = pose_future.result()
                    else:
                        candidates = (
                            self._track_branch.infer_candidate_results([prev_frame, current_frame, next_frame])
                            if self._track_enabled
                            else []
                        )
                        detections = (
                            self._pose_branch.infer(current_frame, court_prediction=court_prediction)
                            if pose_due
                            else []
                        )

                    if self._pose_enabled:
                        last_pose = pose_tracker.update(
                            detections,
                            court_prediction,
                            frame_shape=current_frame.shape,
                        )
                        pose_frames += int(bool(last_pose))
                    else:
                        pose_tracker.reset()
                        last_pose = []

                    if self._track_enabled:
                        track_dt = frame_step_seconds(current_ms, previous_track_ms, fps)
                        track = track_filter.update_candidates(
                            candidates,
                            dt=track_dt,
                            frame_shape=current_frame.shape,
                            court_prediction=court_prediction,
                            person_bboxes=pose_person_bboxes(last_pose),
                        )
                        previous_track_ms = current_ms
                    else:
                        track = TrackResult(ball_xy=[-1.0, -1.0], visible=0, score=0.0)

                    processed_frames += 1
                    visible_frames += int(bool(track.visible))
                    score_sum += float(track.score)
                    player_points = project_player_points_to_court(last_pose, court_prediction)
                    if player_points:
                        distance_accumulator.update(player_points)
                    else:
                        distance_accumulator.reset_tracking_points()
                    ball_visual_intersection = project_ball_visual_intersection(track, court_prediction)
                    rally_stats.update_frame(
                        timestamp_ms=current_ms,
                        player_points=player_points,
                        ball_visible=bool(track.visible),
                        ball_xy=track.ball_xy,
                        ball_court_xy=ball_visual_intersection,
                        ball_score=float(track.score),
                        court_valid=bool(prediction_value(court_prediction, "valid", False)),
                    )

                    frame_result = FrameResult(frame_id=frame_id, pose=last_pose, track=track)
                    trajectory_event = event_detector.update(
                        frame_result,
                        timestamp_ms=current_ms,
                        frame_shape=current_frame.shape,
                    )
                    hit_event = (
                        trajectory_event
                        if isinstance(trajectory_event, dict) and trajectory_event.get("event_type") == "hit"
                        else None
                    )
                    rally_stats.add_trajectory_event(trajectory_event)

                    if bst_recognizer is not None:
                        try:
                            bst_prediction = bst_recognizer.update(
                                frame_result,
                                hit_event=hit_event,
                                court_prediction=court_prediction,
                            )
                        except Exception as exc:
                            pending_bst_errors.append(str(exc))
                            bst_recognizer = None
                        else:
                            if bst_prediction is not None:
                                rally_stats.add_bst_prediction(bst_prediction)

                    if processed_frames % progress_every == 0 or final_pass:
                        local_progress = (
                            min(1.0, processed_frames / frame_count)
                            if frame_count > 0
                            else 0.0
                        )
                        self.progressChanged.emit(
                            {
                                "phase": "video_progress",
                                "index": index,
                                "total": total,
                                "video_name": video_path.name,
                                "local_progress": local_progress,
                                "overall_progress": (index + local_progress) / max(1, total),
                                "processed_frames": processed_frames,
                            }
                        )

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

        record = rally_stats.export_record()
        summary = record["summary"]
        summary.update(
            {
                "video_name": video_path.name,
                "video_path": str(video_path),
                "duration_ms": max(int(summary.get("duration_ms", 0)), duration_ms),
                "processed_frames": processed_frames,
                "visible_frames": visible_frames,
                "pose_frames": pose_frames,
                "avg_score": (score_sum / processed_frames) if processed_frames else 0.0,
                "player_distances_m": distance_accumulator.totals_m(),
                "bst_errors": pending_bst_errors,
                "court_backend": court_predictor.active_backend or "none",
                "court_errors": court_predictor.errors[:8],
            }
        )
        record.update(
            {
                "id": str(video_path),
                "video_name": video_path.name,
                "video_path": str(video_path),
                "summary": summary,
            }
        )
        return record


class ReportGenerationWorker(QThread):
    reportFinished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        *,
        rally_record: dict[str, Any],
        athlete_name: str,
        video_name: str,
        output_html_path: Path,
        use_api: bool,
        endpoint: str,
        model: str,
        api_key: str,
    ) -> None:
        super().__init__()
        self._rally_record = rally_record
        self._athlete_name = athlete_name
        self._video_name = video_name
        self._output_html_path = output_html_path
        self._use_api = bool(use_api)
        self._endpoint = endpoint
        self._model = model
        self._api_key = api_key

    def run(self) -> None:
        api_error = ""
        try:
            api_client = None
            if self._use_api:
                api_client = OpenAICompatibleReportApiClient(
                    self._endpoint,
                    model=self._model,
                    api_key=self._api_key,
                    timeout_s=60.0,
                )
            try:
                response = generate_report_from_rally_record(
                    self._rally_record,
                    athlete_name=self._athlete_name,
                    user_player="bottom",
                    video_name=self._video_name,
                    api_client=api_client,
                    use_api=api_client is not None,
                    output_html_path=self._output_html_path,
                )
            except Exception as exc:
                if not self._use_api:
                    raise
                api_error = str(exc)
                response = generate_report_from_rally_record(
                    self._rally_record,
                    athlete_name=self._athlete_name,
                    user_player="bottom",
                    video_name=self._video_name,
                    output_html_path=self._output_html_path,
                )
        except Exception as exc:
            self.failed.emit(str(exc))
            return

        payload = response.to_dict()
        if api_error:
            payload["api_error"] = api_error
        payload["output_html_path"] = str(self._output_html_path)
        self.reportFinished.emit(payload)


class MainController:
    """PyQt6 前端的多线程 TrackNetV3 预览控制器。"""

    def __init__(self, view: MainWindow, court_service: Any | None = None) -> None:
        self.view = view
        self._court_service = court_service
        self._project_root = Path(__file__).resolve().parents[3]
        self._settings = QSettings("WFBARNet", "PyQt6Runtime")
        self._default_pose_model_path = str(self._project_root / "assets" / "weights" / "pose" / "yolo26s-pose.pt")
        self._default_track_model_path = str(self._project_root / "assets" / "weights" / "track" / "model_best.pt")
        self._default_bst_model_path = self._project_root / "assets" / "weights" / "bst" / "bst_CG_AP_JnB_bone_merged_10.pt"
        self._pose_model_path = self._load_model_path("pose_model_path", self._default_pose_model_path)
        self._track_model_path = self._load_model_path("track_model_path", self._default_track_model_path)
        self._pose_model_enabled = self._load_bool_setting("pose_model_enabled", True)
        self._track_model_enabled = self._load_bool_setting("track_model_enabled", True)
        self._debug_csv_enabled = self._load_bool_setting("debug_csv_enabled", False)
        self._report_api_enabled = self._load_bool_setting("report_api_enabled", False)
        self._report_api_provider = str(self._settings.value("report_api_provider", "bailian_openai") or "bailian_openai")
        self._report_api_endpoint = str(self._settings.value("report_api_endpoint", DEFAULT_REPORT_API_ENDPOINT) or DEFAULT_REPORT_API_ENDPOINT)
        self._report_api_model = str(self._settings.value("report_api_model", DEFAULT_REPORT_API_MODEL) or DEFAULT_REPORT_API_MODEL)
        self._report_api_key = str(self._settings.value("report_api_key", "") or "")
        self._theme_dirs = discover_themes()
        self._active_theme_name = self._resolve_initial_theme_name()
        self._selected_video_path: str | None = None
        self._selected_batch_folder: str | None = None
        self._video_meta: dict[str, Any] = {}
        self._input_mode = "video"
        self._camera_devices: list[tuple[int, str]] = []
        self._probe_worker: VideoProbeWorker | None = None
        self._playback_worker: TrackNetPlaybackWorker | None = None
        self._camera_worker: CameraInferenceWorker | None = None
        self._batch_worker: BatchInferenceWorker | None = None
        self._report_worker: ReportGenerationWorker | None = None
        self._batch_results: dict[str, dict[str, Any]] = {}
        self._batch_order: list[str] = []
        self._current_rally_record: dict[str, Any] | None = None
        self._live_rally_record: dict[str, Any] | None = None
        self._last_live_record_save_time = 0.0
        self._analysis_busy = False
        self._auto_report_after_stop = False
        self._manual_court_points: list[list[float]] = []
        self._manual_court_active = False
        self._pending_seek_ms: int | None = None
        self._last_display_frame_time: float | None = None
        self._last_metrics_update_time: float | None = None
        self._display_fps_ema = 0.0
        self._last_court_log_frame = -1
        self._last_track_debug_log_time = 0.0
        self._last_track_debug_log_key = ""

        self.view.set_model_settings(self._pose_model_path, self._track_model_path)
        self.view.set_model_switches(self._pose_model_enabled, self._track_model_enabled)
        self.view.set_debug_csv_enabled(self._debug_csv_enabled)
        self._log_missing_model_paths()
        self.view.set_report_api_settings(
            {
                "enabled": self._report_api_enabled,
                "provider": self._report_api_provider,
                "endpoint": self._report_api_endpoint,
                "model": self._report_api_model,
                "api_key": self._report_api_key,
            }
        )
        self._track_branch: TrackBranch | None = None
        self._pose_branch: PoseBranch | None = None
        self._bst_device = "cuda" if torch.cuda.is_available() else "cpu"
        self._bst_model: Any | None = None
        self._bst_model_checked = False

        self._bind_court_service()
        self._bind_events()
        self.view.populate_stylesheets(self._theme_dirs, self._active_theme_name)
        self.view.video_timeline.set_interactive(True)
        self._reset_metrics()
        self._set_idle_state()
        self.view.append_log("[系统] 界面已就绪，请选择视频开始；模型将在分析前按需加载。")

    def _log_missing_model_paths(self) -> None:
        checks = [
            ("骨骼模型", self._pose_model_path, self._pose_model_enabled),
            ("球轨迹模型", self._track_model_path, self._track_model_enabled),
        ]
        for label, raw_path, enabled in checks:
            if enabled and not self._resolve_model_path(raw_path).is_file():
                self.view.append_log(f"[设置] {label}文件暂未找到，可稍后在设置页选择: {self._resolve_model_path(raw_path)}")

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
        self.handle_manual_court_calibration()

    def handle_manual_court_calibration(self) -> None:
        if self._manual_court_active:
            self._finish_manual_court_calibration(cancelled=True)
            return
        if self._is_inference_running():
            self.view.append_log("[Court] 请先停止当前推理任务，再进行手动标定。")
            return
        if self._input_mode == "batch":
            self.view.append_log("[Court] 批量模式暂不支持逐视频手动标定。")
            return
        if self.view.video_player.source_size() is None:
            self.view.append_log("[Court] 请先选择视频或打开摄像头预览，再进行手动标定。")
            return
        self._manual_court_points = []
        self._manual_court_active = True
        self.view.set_manual_court_capture_enabled(True)
        self.view.append_log("[Court] 手动标定开始：请依次点击外框四角 TL, TR, BR, BL。")

    def handle_manual_court_point_selected(self, x: float, y: float) -> None:
        if not self._manual_court_active:
            return
        self._manual_court_points.append([float(x), float(y)])
        point_index = len(self._manual_court_points)
        self.view.append_log(f"[Court] 标定点 {point_index}/4: ({x:.1f}, {y:.1f})")
        if point_index < 4:
            return

        source_size = self.view.video_player.source_size()
        if source_size is None:
            self._finish_manual_court_calibration(cancelled=True)
            self.view.append_log("[Court] 标定失败：当前没有可用视频帧。")
            return
        try:
            set_calibration = getattr(self._court_service, "set_calibration", None)
            if not callable(set_calibration):
                raise RuntimeError("manual calibration service is not available")
            prediction = set_calibration(
                self._manual_court_points,
                source_size=source_size,
                frame_id=int(self._video_meta.get("frame_id", 0)),
                timestamp_ms=int(self._video_meta.get("position_ms", 0)),
            )
        except Exception as exc:
            self._finish_manual_court_calibration(cancelled=True)
            self.view.append_log(f"[Court] 手动标定失败: {exc}")
            return

        payload = prediction.to_dict() if hasattr(prediction, "to_dict") else prediction
        self.view.set_court_overlay(payload)
        self._finish_manual_court_calibration(cancelled=False)
        self.view.append_log("[Court] 手动标定完成，后续分析将使用该球场映射。")

    def handle_manual_court_corner_dragged(self, corner_index: int, x: float, y: float) -> None:
        if self._manual_court_active:
            return
        source_size = self.view.video_player.source_size()
        latest = self._latest_court_prediction_dict()
        corners = latest.get("corners") if isinstance(latest, dict) else None
        if source_size is None or not isinstance(corners, list) or len(corners) != 4:
            return
        if corner_index < 0 or corner_index >= len(corners):
            return

        updated_corners = [[float(point[0]), float(point[1])] for point in corners]
        updated_corners[corner_index] = [float(x), float(y)]
        try:
            set_calibration = getattr(self._court_service, "set_calibration", None)
            if not callable(set_calibration):
                return
            prediction = set_calibration(
                updated_corners,
                source_size=source_size,
                frame_id=int(self._video_meta.get("frame_id", 0)),
                timestamp_ms=int(self._video_meta.get("position_ms", 0)),
            )
        except Exception:
            return

        payload = prediction.to_dict() if hasattr(prediction, "to_dict") else prediction
        self.view.set_court_overlay(payload)

    def _finish_manual_court_calibration(self, *, cancelled: bool) -> None:
        self._manual_court_active = False
        self._manual_court_points = []
        self.view.set_manual_court_capture_enabled(False)
        if cancelled:
            self.view.append_log("[Court] 已取消手动标定。")

    def _clear_manual_court_calibration(self) -> None:
        self._finish_manual_court_calibration(cancelled=False)
        clear_calibration = getattr(self._court_service, "clear_calibration", None)
        if callable(clear_calibration):
            clear_calibration()
        self.view.set_court_overlay(None)

    def _latest_court_prediction_dict(self) -> dict[str, Any] | None:
        if self._court_service is None:
            return None
        latest_dict = getattr(self._court_service, "latest_prediction_dict", None)
        if callable(latest_dict):
            value = latest_dict()
            return value if isinstance(value, dict) else None
        latest = getattr(self._court_service, "latest_prediction", None)
        prediction = latest() if callable(latest) else None
        to_dict = getattr(prediction, "to_dict", None)
        value = to_dict() if callable(to_dict) else prediction
        return value if isinstance(value, dict) else None

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
            scheme = str(payload.get("scheme", "manual"))
            if scheme == "manual":
                self.view.append_log(f"[Court] manual calibration updated | frame {frame_id} | conf {confidence:.2f}")
            else:
                self.view.append_log(f"[Court] court updated | frame {frame_id} | conf {confidence:.2f} | {detect_ms:.0f} ms")
        elif not valid and self._last_court_log_frame < 0:
            self._last_court_log_frame = frame_id
            self.view.append_log("[Court] waiting for manual court calibration.")

    def _on_court_detection_failed(self, message: str) -> None:
        self.view.append_log(f"[Court] manual court calibration failed: {message}")

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

    def _make_track_debug_csv_path(self, stem: str) -> str | None:
        if not self._debug_csv_enabled:
            return None
        safe_stem = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in stem).strip("_")
        if not safe_stem:
            safe_stem = "track"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return str(self._project_root / "outputs" / "pyqt_debug" / f"{safe_stem}_{timestamp}_track_debug.csv")

    def _make_frame_log_jsonl_path(self, track_debug_csv_path: str | None) -> str | None:
        if not track_debug_csv_path:
            return None
        csv_path = Path(track_debug_csv_path)
        suffix = "_track_debug.csv"
        if csv_path.name.endswith(suffix):
            return str(csv_path.with_name(f"{csv_path.name[:-len(suffix)]}_frame_log.jsonl"))
        return str(csv_path.with_suffix(".jsonl"))

    def _runtime_data_dir(self) -> Path:
        return self._project_root / "outputs" / "runtime_data"

    def _write_json_file(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _save_live_rally_record(self, record: dict[str, Any] | None, *, force: bool = False) -> None:
        if record is None:
            return
        now = perf_counter()
        if not force and now - self._last_live_record_save_time < 1.0:
            return
        self._last_live_record_save_time = now
        payload = {
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "source": "live_analysis",
            "record_state": "running" if self._analysis_busy else "idle",
            "rally_record": record,
        }
        try:
            self._write_json_file(self._runtime_data_dir() / "live_rally_record.json", payload)
        except Exception as exc:
            self.view.append_log(f"[报告] 实时数据保存失败: {exc}")

    def _save_report_input_snapshot(
        self,
        record: dict[str, Any],
        *,
        trigger: str,
        use_api: bool,
    ) -> tuple[Path, Path]:
        saved_at = datetime.now()
        payload = {
            "saved_at": saved_at.isoformat(timespec="seconds"),
            "trigger": trigger,
            "report_api": {
                "enabled": bool(self._report_api_enabled),
                "used": bool(use_api),
                "provider": self._report_api_provider,
                "endpoint": self._report_api_endpoint,
                "model": self._report_api_model,
                "api_key_configured": bool(self._report_api_key),
            },
            "rally_record": record,
        }
        runtime_dir = self._runtime_data_dir()
        latest_path = runtime_dir / "latest_report_input.json"
        timestamp_path = runtime_dir / f"report_input_{saved_at.strftime('%Y%m%d_%H%M%S')}.json"
        dist_path = self._project_root / "assets" / "dist" / "report-data.json"
        for path in (latest_path, timestamp_path, dist_path):
            self._write_json_file(path, payload)
        self._save_live_rally_record(record, force=True)
        return latest_path, dist_path

    def _build_track_branch(self) -> TrackBranch:
        branch = self._create_track_branch(self._track_model_path)
        self.view.append_log(
            f"[TrackNet] 模型已加载: {branch.device} | 后端 {branch.backend_name} | {Path(self._track_model_path).name}"
        )
        return branch

    def _ensure_models_ready(self) -> bool:
        if not self._track_model_enabled and not self._pose_model_enabled:
            return True

        self.view.set_progress_busy(True, "正在加载模型")
        try:
            if self._track_model_enabled and self._track_branch is None:
                track_path = self._resolve_model_path(self._track_model_path)
                if not track_path.is_file():
                    raise FileNotFoundError(f"球轨迹模型文件不存在: {track_path}")
                self._track_model_path = str(track_path)
                self._track_branch = self._build_track_branch()

            if self._pose_model_enabled and self._pose_branch is None:
                pose_path = self._resolve_model_path(self._pose_model_path)
                if not pose_path.is_file():
                    raise FileNotFoundError(f"骨骼模型文件不存在: {pose_path}")
                self._pose_model_path = str(pose_path)
                self._pose_branch = self._build_pose_branch()

            if (
                self._track_model_enabled
                and self._pose_model_enabled
                and self._bst_model is None
                and not self._bst_model_checked
            ):
                self._bst_model_checked = True
                self._bst_model = self._build_bst_model()
        except Exception as exc:
            self.view.set_progress_busy(False)
            self.view.set_status_state("error")
            self.view.append_log(f"[模型] 加载失败: {exc}")
            return False

        self.view.set_progress_busy(False)
        return True

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

    def _build_bst_model(self) -> Any | None:
        if not self._default_bst_model_path.is_file():
            self.view.append_log(f"[BST] weight not found: {self._default_bst_model_path}")
            return None
        try:
            model = build_bst_model(self._default_bst_model_path)
            model.to(self._bst_device)
            model.eval()
        except Exception as exc:
            self.view.append_log(f"[BST] failed to load stroke model: {exc}")
            return None
        self.view.append_log(
            f"[BST] stroke model ready | {self._bst_device} | "
            f"seq_len {getattr(model, 'bst_seq_len', '?')} | classes {getattr(model, 'bst_n_classes', '?')}"
        )
        return model

    def _create_pose_branch(self, model_weight: str) -> PoseBranch:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return PoseBranch(
            backend="yolo26s-pose",
            model_weight=str(model_weight),
            device=device,
            conf_thr=0.35,
            max_persons=POSE_CANDIDATE_LIMIT,
            yolo_imgsz=POSE_YOLO_IMGSZ,
            yolo_crop_pose=True,
            yolo_crop_imgsz=POSE_CROP_IMGSZ,
            yolo_crop_padding=POSE_CROP_PADDING,
            yolo_crop_min_box_conf=POSE_CROP_MIN_BOX_CONF,
            yolo_max_pose_crops=POSE_MAX_CROPS,
            yolo_court_filter=True,
            yolo_court_required=False,
            yolo_court_margin=POSE_COURT_MARGIN_CM,
        )

    def _bind_events(self) -> None:
        self.view.btn_analyze.clicked.connect(self.handle_analyze)
        self.view.btn_reset.clicked.connect(self.handle_reset)
        self.view.btn_preview_mode.clicked.connect(lambda: self.handle_input_mode("video"))
        self.view.btn_camera_mode.clicked.connect(lambda: self.handle_input_mode("camera"))
        self.view.btn_batch_mode.clicked.connect(lambda: self.handle_input_mode("batch"))
        self.view.btn_refresh_cameras.clicked.connect(lambda: self._refresh_camera_devices(log=True))
        self.view.camera_device_combo.currentIndexChanged.connect(lambda _index: self._set_idle_state())
        self.view.video_player.selectRequested.connect(self.handle_upload)
        self.view.video_player.forceStopRequested.connect(self.handle_force_stop)
        self.view.video_timeline.seekRequested.connect(self.handle_seek)
        self.view.batchFolderBrowseRequested.connect(self.handle_browse_batch_folder)
        self.view.batchRallySelectionChanged.connect(self.handle_batch_rally_selection)
        self.view.batchExportRequested.connect(self.handle_export_batch_results)
        self.view.reportExportRequested.connect(self.handle_export_report)
        self.view.manualCourtCalibrationRequested.connect(self.handle_manual_court_calibration)
        self.view.video_player.framePointClicked.connect(self.handle_manual_court_point_selected)
        self.view.video_player.courtCornerDragged.connect(self.handle_manual_court_corner_dragged)
        self.view._style_menu.triggered.connect(self._on_style_action_triggered)
        self.view.poseModelBrowseRequested.connect(self.handle_browse_pose_model)
        self.view.trackModelBrowseRequested.connect(self.handle_browse_track_model)
        self.view.modelSettingsApplyRequested.connect(self.handle_model_settings_apply)
        self.view.modelSettingsDefaultsRequested.connect(self.handle_model_settings_defaults)
        self.view.modelSwitchesChanged.connect(self.handle_model_switches_changed)
        self.view.debugCsvChanged.connect(self.handle_debug_csv_changed)
        self.view.reportApiSettingsApplyRequested.connect(self.handle_report_api_settings_apply)

    def _set_idle_state(self) -> None:
        self._analysis_busy = False
        has_video = self._selected_video_path is not None
        has_camera = self.view.selected_camera_device() is not None
        has_batch_folder = self._selected_batch_folder is not None
        if self._input_mode == "camera":
            can_analyze = has_camera
        elif self._input_mode == "batch":
            can_analyze = has_batch_folder
        else:
            can_analyze = has_video
        self.view.btn_analyze.setEnabled(can_analyze)
        self.view.btn_reset.setEnabled(True)
        self.view.video_player.btn_select_video.setEnabled(self._input_mode == "video")
        self.view.btn_refresh_cameras.setEnabled(self._input_mode == "camera")
        self.view.camera_device_combo.setEnabled(self._input_mode == "camera")
        self.view.btn_select_batch_folder.setEnabled(self._input_mode == "batch")
        self.view.batch_video_combo.setEnabled(self._input_mode == "batch" and bool(self._batch_order))
        self.view.btn_export_batch.setEnabled(self._input_mode == "batch" and bool(self._batch_results))
        self.view.set_report_export_enabled(
            self._current_rally_record is not None
            and not self._analysis_busy
            and not self._is_report_generation_running()
        )
        self.view.video_player.btn_force_stop.setEnabled(has_video if self._input_mode == "video" else False)
        self.view.btn_redetect_court.setEnabled(self._input_mode != "batch")
        self.view.video_timeline.set_interactive(self._input_mode == "video")
        self.view.set_model_settings_enabled(True)
        self.view.set_status_state("idle")

    def _set_running_state(self) -> None:
        self._analysis_busy = True
        self.view.btn_analyze.setEnabled(False)
        self.view.btn_reset.setEnabled(True)
        self.view.video_player.btn_select_video.setEnabled(False)
        self.view.btn_refresh_cameras.setEnabled(False)
        self.view.camera_device_combo.setEnabled(False)
        self.view.btn_select_batch_folder.setEnabled(False)
        self.view.batch_video_combo.setEnabled(False)
        self.view.btn_export_batch.setEnabled(False)
        self.view.set_report_export_enabled(False)
        self.view.video_player.btn_force_stop.setEnabled(True)
        self.view.btn_redetect_court.setEnabled(False)
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
        if mode not in {"video", "camera", "batch"}:
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
        elif mode == "batch":
            self.view.clear_video()
            self.view.video_player.set_live_source("批量推理（无画面高速分析）")
            if self._selected_batch_folder:
                self.view.set_batch_folder_path(self._selected_batch_folder)
            self.view.set_batch_rally_options(self._batch_records_for_view(), self.view.selected_batch_rally_id())
            self.view.append_log("[模式] 已切换到批量推理")
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
            self._analysis_busy
            or
            (self._playback_worker is not None and self._playback_worker.isRunning())
            or (self._camera_worker is not None and self._camera_worker.isRunning())
            or (self._batch_worker is not None and self._batch_worker.isRunning())
        )

    def _is_report_generation_running(self) -> bool:
        return bool(self._report_worker is not None and self._report_worker.isRunning())

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
        self._pose_branch = None
        self._track_branch = None
        self._bst_model = None
        self._bst_model_checked = False
        self._settings.setValue("pose_model_enabled", pose_enabled)
        self._settings.setValue("track_model_enabled", track_enabled)
        self._settings.sync()
        pose_text = "启用" if pose_enabled else "关闭"
        track_text = "启用" if track_enabled else "关闭"
        self.view.append_log(f"[设置] 模型开关已更新 | 骨骼 {pose_text} | 球轨迹 {track_text}")

    def handle_debug_csv_changed(self, enabled: bool) -> None:
        if self._is_inference_running():
            self.view.set_debug_csv_enabled(self._debug_csv_enabled)
            self.view.append_log("[TrackDebug] Stop current inference before changing CSV debug output.")
            return

        self._debug_csv_enabled = bool(enabled)
        self._settings.setValue("debug_csv_enabled", self._debug_csv_enabled)
        self._settings.sync()
        state_text = "enabled" if self._debug_csv_enabled else "disabled"
        self.view.append_log(f"[TrackDebug] CSV debug output {state_text}.")

    def handle_report_api_settings_apply(self, settings: dict[str, object]) -> None:
        enabled = bool(settings.get("enabled"))
        endpoint = str(settings.get("endpoint") or DEFAULT_REPORT_API_ENDPOINT).strip()
        model = str(settings.get("model") or DEFAULT_REPORT_API_MODEL).strip()
        api_key = str(settings.get("api_key") or "").strip()
        provider = str(settings.get("provider") or "bailian_openai").strip()
        if enabled and (not endpoint or not model or not api_key):
            self.view.append_log("[报告] 启用 API 时需要填写 Endpoint、模型名和 API Key。")
            return

        self._report_api_enabled = enabled
        self._report_api_provider = provider
        self._report_api_endpoint = endpoint
        self._report_api_model = model
        self._report_api_key = api_key
        self._settings.setValue("report_api_enabled", self._report_api_enabled)
        self._settings.setValue("report_api_provider", self._report_api_provider)
        self._settings.setValue("report_api_endpoint", self._report_api_endpoint)
        self._settings.setValue("report_api_model", self._report_api_model)
        self._settings.setValue("report_api_key", self._report_api_key)
        self._settings.sync()
        state_text = "启用" if enabled else "关闭"
        self.view.append_log(f"[报告] 模型 API 已{state_text} | 平台 {provider} | 模型 {model}")

    def handle_model_settings_apply(self, pose_model_path: str, track_model_path: str) -> None:
        if self._is_inference_running():
            self.view.append_log("[设置] 请先停止当前推理，再切换模型。")
            return

        pose_path = self._resolve_model_path(pose_model_path)
        track_path = self._resolve_model_path(track_model_path)
        pose_path_text = str(pose_path)
        track_path_text = str(track_path)
        requested_backend = "tensorrt" if track_path.suffix.lower() == ".engine" else "pytorch"
        if pose_path_text == self._pose_model_path and track_path_text == self._track_model_path:
            self.view.append_log(
                f"[设置] 模型路径未变化，当前球轨迹后端: {getattr(self._track_branch, 'backend_name', 'not loaded')}"
            )
            return

        self.view.append_log(
            f"[设置] 已保存模型路径 | 球轨迹: {track_path.name} | 目标后端 {requested_backend}"
        )

        self._stop_workers(clear_pending_seek=True)
        self._track_branch = None
        self._pose_branch = None
        self._bst_model = None
        self._bst_model_checked = False
        self._pose_model_path = pose_path_text
        self._track_model_path = track_path_text
        self._settings.setValue("pose_model_path", pose_path_text)
        self._settings.setValue("track_model_path", track_path_text)
        self._settings.sync()
        self.view.set_model_settings(pose_path_text, track_path_text)
        if not pose_path.is_file():
            self.view.append_log(f"[设置] 骨骼模型路径暂不可用，开始分析前会再次校验: {pose_path}")
        if not track_path.is_file():
            self.view.append_log(f"[设置] 球轨迹模型路径暂不可用，开始分析前会再次校验: {track_path}")
        self.view.append_log("[设置] 模型缓存已清空，将在下次分析前加载。")
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
        if self._playback_worker is not None or self._camera_worker is not None:
            self.view.append_log("[信息] 正在等待上一项推理任务结束...")
            return
        self._selected_video_path = file_path
        self._video_meta = {}
        self._reset_metrics()
        self._set_current_rally_record(None)
        self._clear_manual_court_calibration()
        self.view.set_video_path(file_path)
        self.view.set_video_state("loaded")
        self.view.append_log(f"[信息] 正在加载预览: {Path(file_path).name}")
        self._start_probe(file_path, 0)

    def handle_browse_batch_folder(self) -> None:
        if self._is_inference_running():
            self.view.append_log("[批量推理] 请先停止当前推理任务，再选择文件夹。")
            return

        start_dir = str(Path(__file__).resolve().parents[3] / "videos")
        folder_path = QFileDialog.getExistingDirectory(
            self.view,
            "选择批量视频文件夹",
            self._selected_batch_folder or start_dir,
        )
        if not folder_path:
            self.view.append_log("[批量推理] 文件夹选择已取消。")
            return

        self._selected_batch_folder = folder_path
        self._batch_results.clear()
        self._batch_order.clear()
        self._clear_manual_court_calibration()
        options = self._batch_video_options(folder_path)
        self.view.set_batch_folder_path(folder_path)
        self.view.set_batch_rally_options(options)
        self.view.set_batch_export_enabled(False)
        self._set_current_rally_record(None)
        self.view.append_log(f"[批量推理] 已选择文件夹: {folder_path} | 视频 {len(options)} 个")
        self._set_idle_state()

    def handle_batch_rally_selection(self, rally_id: str) -> None:
        if not rally_id:
            self._set_current_rally_record(None)
            return
        record = self._batch_results.get(rally_id)
        if record is not None:
            self._set_current_rally_record(record)

    def _set_current_rally_record(self, record: object | None) -> None:
        current = record if self._is_reportable_rally_record(record) else None
        self._current_rally_record = current
        self._live_rally_record = current
        self._save_live_rally_record(current)
        self.view.set_rally_data(current)
        self.view.set_report_export_enabled(
            current is not None
            and not self._analysis_busy
            and not self._is_report_generation_running()
        )

    def _is_reportable_rally_record(self, record: object | None) -> bool:
        if not isinstance(record, dict) or record.get("error"):
            return False
        return isinstance(record.get("summary"), dict) and isinstance(record.get("details"), dict)

    def _batch_video_options(self, folder_path: str) -> list[dict[str, Any]]:
        folder = Path(folder_path)
        if not folder.is_dir():
            return []
        return [
            {
                "id": str(path),
                "video_name": path.name,
                "video_path": str(path),
            }
            for path in sorted(
                (
                    item
                    for item in folder.iterdir()
                    if item.is_file() and item.suffix.lower() in BatchInferenceWorker.VIDEO_SUFFIXES
                ),
                key=lambda item: item.name.lower(),
            )
        ]

    def _batch_records_for_view(self) -> list[dict[str, Any]]:
        if self._batch_order:
            return [self._batch_results[item_id] for item_id in self._batch_order if item_id in self._batch_results]
        if self._selected_batch_folder:
            return self._batch_video_options(self._selected_batch_folder)
        return []

    def _start_probe(self, video_path: str, position_ms: int) -> None:
        if self._probe_worker is not None and self._probe_worker.isRunning():
            self._probe_worker.quit()
            if not self._probe_worker.wait(300):
                self.view.append_log("[信息] 正在等待上一次视频预览任务结束...")
                return
            self._probe_worker = None

        self._probe_worker = VideoProbeWorker(video_path, preview_ms=position_ms)
        self._probe_worker.finished.connect(self._on_probe_finished)
        self._probe_worker.failed.connect(self._on_probe_failed)
        self._probe_worker.start()

    def _on_probe_finished(self, file_path: str, payload: object) -> None:
        worker = self._probe_worker
        try:
            if file_path != self._selected_video_path or not isinstance(payload, dict):
                return

            self._video_meta = payload
            self.view.show_video_frame(
                payload["image"],
                int(payload.get("position_ms", 0)),
                int(payload.get("duration_ms", 0)),
                self._latest_court_prediction_dict(),
            )
            self.view.update_progress(0)
            self.view.set_video_state("loaded")
            self.view.append_log(
                f"[信息] 已加载 {Path(file_path).name} | "
                f"{payload.get('width', 0)} x {payload.get('height', 0)} | "
                f"FPS {float(payload.get('fps', 0.0)):.2f}"
            )
            self._set_idle_state()
        finally:
            self._release_probe_worker(worker)

    def _on_probe_failed(self, message: str) -> None:
        worker = self._probe_worker
        try:
            self._selected_video_path = None
            self._video_meta = {}
            self.view.set_status_state("error")
            self.view.set_video_state("error")
            self.view.btn_analyze.setEnabled(False)
            self.view.video_player.btn_select_video.setEnabled(True)
            self.view.video_player.btn_force_stop.setEnabled(False)
            self.view.append_log(f"[错误] {message}")
        finally:
            self._release_probe_worker(worker)

    def handle_analyze(self) -> None:
        if self._input_mode == "camera":
            self._start_camera_inference()
            return
        if self._input_mode == "batch":
            self._start_batch_inference()
            return

        if not self._selected_video_path:
            self.view.append_log("[警告] 开始分析前请先选择视频。")
            return

        self._pending_seek_ms = None
        start_ms = int(self._video_meta.get("position_ms", 0)) if self._video_meta else 0
        self._start_playback(start_ms=start_ms, request_court_prediction=True)

    def _start_batch_inference(self) -> None:
        if not self._selected_batch_folder:
            self.view.append_log("[批量推理] 开始分析前请先选择视频文件夹。")
            return

        options = self._batch_video_options(self._selected_batch_folder)
        if not options:
            self.view.append_log("[批量推理] 该文件夹中没有可分析的视频。")
            return
        if not self._ensure_models_ready():
            self._set_idle_state()
            return

        self._stop_workers(clear_pending_seek=True)
        if (
            self._playback_worker is not None
            or self._camera_worker is not None
            or self._batch_worker is not None
        ):
            self.view.append_log("[信息] 正在等待上一项推理任务结束...")
            return

        self._batch_results.clear()
        self._batch_order.clear()
        self._reset_metrics()
        self._set_current_rally_record(None)
        self.view.set_batch_rally_options(options)
        self.view.set_batch_export_enabled(False)
        self.view.video_player.clear_video()
        self.view.video_player.set_live_source("批量推理（无画面高速分析）")
        self.view.set_progress_busy(False)
        self.view.update_progress(0)
        self._set_running_state()
        self.view.append_log(
            f"[批量推理] 开始分析 {len(options)} 个回合 | "
            f"球轨迹 {'启用' if self._track_model_enabled else '关闭'} | "
            f"骨骼 {'启用' if self._pose_model_enabled else '关闭'}"
        )

        self._batch_worker = BatchInferenceWorker(
            self._selected_batch_folder,
            self._track_branch,
            self._pose_branch,
            pose_stride=POSE_INFERENCE_STRIDE,
            track_enabled=self._track_model_enabled,
            pose_enabled=self._pose_model_enabled,
            bst_model=self._bst_model,
            bst_device=self._bst_device,
        )
        self._batch_worker.progressChanged.connect(self._on_batch_progress)
        self._batch_worker.rallyFinished.connect(self._on_batch_rally_finished)
        self._batch_worker.batchFinished.connect(self._on_batch_finished)
        self._batch_worker.failed.connect(self._on_batch_failed)
        self._batch_worker.finished.connect(lambda worker=self._batch_worker: self._release_batch_worker(worker))
        self._batch_worker.start()

    def _start_camera_inference(self) -> None:
        camera_index = self.view.selected_camera_device()
        if camera_index is None:
            self.view.append_log("[警告] 请先选择可用摄像头设备。")
            return
        if not self._ensure_models_ready():
            self._set_idle_state()
            return

        self._stop_workers(clear_pending_seek=True)
        if self._playback_worker is not None or self._camera_worker is not None:
            self.view.append_log("[信息] 正在等待上一项推理任务结束...")
            return
        self._reset_court_detection(request_initial_prediction=True)
        self._set_current_rally_record(None)
        self._set_running_state()
        self.view.video_player.set_live_source(f"摄像头 {camera_index}")
        self.view.video_player.play()
        debug_csv_path = self._make_track_debug_csv_path(f"camera_{camera_index}")
        frame_log_path = self._make_frame_log_jsonl_path(debug_csv_path)
        self.view.set_progress_busy(True, "实时推理中")
        self.view.append_log(
            f"[TrackNet] 开始摄像头实时推理: 摄像头 {camera_index} | "
            f"球轨迹 {'启用' if self._track_model_enabled else '关闭'} | "
            f"骨骼 {'启用' if self._pose_model_enabled else '关闭'}"
        )

        if debug_csv_path is not None:
            self.view.append_log(f"[TrackDebug] Writing CSV: {debug_csv_path}")
        if frame_log_path is not None:
            self.view.append_log(f"[FrameLog] Writing JSONL: {frame_log_path}")

        self._camera_worker = CameraInferenceWorker(
            camera_index,
            self._track_branch,
            self._pose_branch,
            pose_stride=POSE_INFERENCE_STRIDE,
            track_enabled=self._track_model_enabled,
            pose_enabled=self._pose_model_enabled,
            court_service=self._court_service,
            debug_csv_path=debug_csv_path,
            frame_log_path=frame_log_path,
            bst_model=self._bst_model,
            bst_device=self._bst_device,
        )
        self._camera_worker.frameReady.connect(self._on_camera_frame_ready)
        self._camera_worker.inferFinished.connect(self._on_camera_finished)
        self._camera_worker.failed.connect(self._on_camera_failed)
        self._camera_worker.finished.connect(lambda worker=self._camera_worker: self._release_camera_worker(worker))
        self._camera_worker.start()

    def _start_playback(self, *, start_ms: int = 0, request_court_prediction: bool = True) -> None:
        if self._selected_video_path is None:
            return
        if not self._ensure_models_ready():
            self._set_idle_state()
            return

        self._stop_workers(clear_pending_seek=False)
        if self._playback_worker is not None or self._camera_worker is not None:
            self.view.append_log("[信息] 正在等待上一项推理任务结束...")
            return
        self._reset_court_detection(request_initial_prediction=request_court_prediction)
        self._set_current_rally_record(None)
        self._set_running_state()
        self.view.video_player.play()
        debug_csv_path = self._make_track_debug_csv_path(Path(self._selected_video_path).stem)
        frame_log_path = self._make_frame_log_jsonl_path(debug_csv_path)
        self.view.append_log(
            f"[TrackNet] 开始播放: {Path(self._selected_video_path).name} | "
            f"球轨迹 {'启用' if self._track_model_enabled else '关闭'} | "
            f"骨骼 {'启用' if self._pose_model_enabled else '关闭'}"
        )
        if debug_csv_path is not None:
            self.view.append_log(f"[TrackDebug] Writing CSV: {debug_csv_path}")
        if frame_log_path is not None:
            self.view.append_log(f"[FrameLog] Writing JSONL: {frame_log_path}")

        self._playback_worker = TrackNetPlaybackWorker(
            self._selected_video_path,
            self._track_branch,
            self._pose_branch,
            start_ms=start_ms,
            pose_stride=POSE_INFERENCE_STRIDE,
            track_enabled=self._track_model_enabled,
            pose_enabled=self._pose_model_enabled,
            court_service=self._court_service,
            debug_csv_path=debug_csv_path,
            frame_log_path=frame_log_path,
            bst_model=self._bst_model,
            bst_device=self._bst_device,
        )
        self._playback_worker.frameReady.connect(self._on_frame_ready)
        self._playback_worker.playbackFinished.connect(self._on_playback_finished)
        self._playback_worker.failed.connect(self._on_playback_failed)
        self._playback_worker.finished.connect(
            lambda worker=self._playback_worker: self._release_playback_worker(worker)
        )
        self._playback_worker.start()

    def _on_frame_ready(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return

        image = payload.get("image")
        if isinstance(image, QImage):
            ball_visual_intersection = payload.get(
                "ball_visual_intersection",
                payload.get("ball_projection"),
            )
            self.view.show_video_frame(
                image,
                int(payload.get("position_ms", 0)),
                int(payload.get("duration_ms", 0)),
                payload.get("court"),
                ball_visual_intersection,
                payload.get("player_projections"),
            )
            self.view.set_player_distances(payload.get("player_distances_m"))
            self._update_display_fps()
        self._log_track_debug_event(payload.get("track_debug"))
        self._append_trajectory_event(payload.get("trajectory_event"))
        self._append_bst_predictions(payload)
        self._set_current_rally_record(payload.get("rally_record"))

        progress = max(0, min(int(float(payload.get("progress", 0.0)) * 100), 100))
        self.view.update_progress(progress)
        if not self._should_update_metrics_text():
            return

        person_count = int(payload.get("person_count", 0))
        infer_fps = float(payload.get("infer_fps", 0.0))
        track = payload.get("track", {})
        current_score = float(track.get("score", 0.0)) if isinstance(track, dict) else 0.0
        court = payload.get("court", {})
        court_text = ""
        if isinstance(court, dict) and court.get("valid"):
            court_text = f" | Court {float(court.get('confidence', 0.0)):.2f}"

        self.view.lbl_valid_pose.setText(f"{infer_fps:.1f} FPS")
        self.view.lbl_realtime_fps.setText(f"{self._display_fps_ema:.1f} FPS")
        self.view.status_label.setText(
            f"系统状态：TrackNet + YOLO26s-Pose 运行中 | 人数 {person_count} | Score {current_score:.2f}{court_text}"
        )

    def _on_camera_frame_ready(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return

        image = payload.get("image")
        if isinstance(image, QImage):
            ball_visual_intersection = payload.get(
                "ball_visual_intersection",
                payload.get("ball_projection"),
            )
            self.view.video_player.display_image(image, court=payload.get("court"))
            self.view.court_widget.set_ball_projection(ball_visual_intersection)
            self.view.court_widget.set_player_projections(payload.get("player_projections"))
            self.view.set_player_distances(payload.get("player_distances_m"))
            self._update_display_fps()
        self._log_track_debug_event(payload.get("track_debug"))
        self._append_trajectory_event(payload.get("trajectory_event"))
        self._append_bst_predictions(payload)
        self._set_current_rally_record(payload.get("rally_record"))
        if not self._should_update_metrics_text():
            return

        person_count = int(payload.get("person_count", 0))
        track = payload.get("track", {})
        current_score = float(track.get("score", 0.0)) if isinstance(track, dict) else 0.0
        infer_fps = float(payload.get("infer_fps", 0.0))
        court = payload.get("court", {})
        court_text = ""
        if isinstance(court, dict) and court.get("valid"):
            court_text = f" | Court {float(court.get('confidence', 0.0)):.2f}"

        self.view.lbl_valid_pose.setText(f"{infer_fps:.1f} FPS")
        self.view.lbl_realtime_fps.setText(f"{self._display_fps_ema:.1f} FPS")
        self.view.status_label.setText(
            f"系统状态：摄像头 TrackNet + YOLO26s-Pose 推理中 | 人数 {person_count} | Score {current_score:.2f}{court_text}"
        )

    def _on_batch_progress(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        progress = max(0, min(int(float(payload.get("overall_progress", 0.0)) * 100), 100))
        video_name = str(payload.get("video_name", ""))
        self.view.update_progress(progress)
        if payload.get("phase") == "start_video":
            index = int(payload.get("index", 0)) + 1
            total = int(payload.get("total", 0))
            self.view.append_log(f"[批量推理] {index}/{total} {video_name}")
        self.view.status_label.setText(f"系统状态：批量推理中 | {progress}% | {video_name}")

    def _on_batch_rally_finished(self, record: object) -> None:
        if not isinstance(record, dict):
            return
        rally_id = str(record.get("id", record.get("video_path", "")))
        if not rally_id:
            return
        if rally_id not in self._batch_order:
            self._batch_order.append(rally_id)
        self._batch_results[rally_id] = record
        selected_id = self.view.selected_batch_rally_id() or rally_id
        self.view.set_batch_rally_options(self._batch_records_for_view(), selected_id)
        if selected_id == rally_id or len(self._batch_order) == 1:
            self._set_current_rally_record(record)
        if record.get("error"):
            self.view.append_log(f"[批量推理] {record.get('video_name', rally_id)} 失败: {record.get('error')}")
            return

        summary = record.get("summary", {})
        hit_count = int(summary.get("rally_hit_count", 0)) if isinstance(summary, dict) else 0
        frame_count = int(summary.get("processed_frames", 0)) if isinstance(summary, dict) else 0
        self.view.append_log(
            f"[批量推理] 完成 {record.get('video_name', rally_id)} | 击球 {hit_count} 次 | 帧数 {frame_count}"
        )

    def _on_batch_finished(self, payload: object) -> None:
        self.view.set_progress_busy(False)
        stopped = bool(payload.get("stopped")) if isinstance(payload, dict) else False
        total = int(payload.get("total", 0)) if isinstance(payload, dict) else len(self._batch_order)
        completed = int(payload.get("completed", 0)) if isinstance(payload, dict) else len(self._batch_results)
        failed = int(payload.get("failed", 0)) if isinstance(payload, dict) else 0
        if stopped:
            self.view.set_status_state("stopped")
            self.view.append_log(f"[批量推理] 已停止 | 已完成 {completed}/{total}")
        else:
            self.view.set_status_state("success")
            self.view.update_progress(100)
            self.view.append_log(f"[批量推理] 全部完成 | 成功 {completed}/{total} | 失败 {failed}")
        self.view.set_batch_export_enabled(bool(self._batch_results))
        self._set_idle_state()

    def _on_batch_failed(self, message: str) -> None:
        self.view.set_progress_busy(False)
        self.view.set_status_state("error")
        self.view.append_log(f"[错误] {message}")
        self._set_idle_state()

    def _append_bst_predictions(self, payload: dict[str, Any]) -> None:
        errors = payload.get("bst_errors", [])
        if isinstance(errors, list):
            for error in errors:
                self.view.append_log(f"[BST] stroke inference disabled after error: {error}")

        predictions = payload.get("bst_predictions", [])
        if not isinstance(predictions, list):
            return
        for prediction in predictions:
            if not isinstance(prediction, dict):
                continue
            time_range = self._format_time_ms(int(prediction.get("timestamp_ms", 0)))
            label = str(prediction.get("pred_display_name", prediction.get("pred_name", "unknown")))
            confidence = float(prediction.get("confidence", 0.0))
            detail = self._format_bst_prediction_detail(prediction)
            self.view.add_action_row(time_range, label, confidence, detail)
            self.view.append_log(f"[BST] hit {time_range} -> {label} ({confidence * 100:.1f}%)")

    def _append_trajectory_event(self, event: object) -> None:
        if not isinstance(event, dict):
            return
        event_type = str(event.get("event_type", "unknown"))
        type_name = {
            "hit": "击球候选",
            "landing": "落地点",
            "out_of_frame": "出画",
        }.get(event_type, event_type)
        time_text = self._format_time_ms(int(event.get("timestamp_ms", 0)))
        rule = str(event.get("rule", ""))
        confidence = float(event.get("confidence", 0.0))
        ball_xy = event.get("ball_xy", [-1.0, -1.0])
        x, y = (-1.0, -1.0)
        if isinstance(ball_xy, (list, tuple)) and len(ball_xy) >= 2:
            x, y = float(ball_xy[0]), float(ball_xy[1])
        self.view.append_log(
            f"[Event] {type_name} {time_text} | frame {int(event.get('frame_id', -1))} | "
            f"({x:.1f},{y:.1f}) | {rule} {confidence * 100:.1f}%"
        )

    def _format_bst_prediction_detail(self, prediction: dict[str, Any]) -> str:
        top5 = prediction.get("top5_display", prediction.get("top5", []))
        top_items: list[str] = []
        if isinstance(top5, list):
            for item in top5[:3]:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("display_name", item.get("class_name", item.get("class_id", ""))))
                prob = float(item.get("probability", 0.0))
                top_items.append(f"{name} {prob * 100:.1f}%")
        h_text = "court H" if prediction.get("used_homography") else "fallback pos"
        top_text = ", ".join(top_items) if top_items else "n/a"
        return (
            f"frame {int(prediction.get('event_frame_id', -1))} | "
            f"clip {int(prediction.get('video_len', 0))}/{int(prediction.get('seq_len', 0))} | "
            f"{h_text} | failed {int(prediction.get('failed_frames', 0))} | top3 {top_text}"
        )

    def _format_time_ms(self, timestamp_ms: int) -> str:
        total_seconds = max(0.0, float(timestamp_ms) / 1000.0)
        minutes = int(total_seconds // 60)
        seconds = total_seconds - minutes * 60
        if minutes > 0:
            return f"{minutes:d}:{seconds:05.2f}"
        return f"{seconds:.2f}s"

    def _append_player_distance_summary(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        distances = payload.get("player_distances_m")
        if not isinstance(distances, dict):
            return
        top_distance = self._safe_distance_value(distances.get("top", 0.0))
        bottom_distance = self._safe_distance_value(distances.get("bottom", 0.0))
        self.view.set_player_distances(distances)
        self.view.append_log(
            f"[PoseDistance] 上方球员 {top_distance:.2f} m | 下方球员 {bottom_distance:.2f} m"
        )

    @staticmethod
    def _safe_distance_value(value: object) -> float:
        try:
            distance = float(value)
        except (TypeError, ValueError):
            return 0.0
        if distance != distance or distance in (float("inf"), float("-inf")):
            return 0.0
        return max(0.0, distance)

    def _log_track_debug_event(self, debug: object) -> None:
        if not isinstance(debug, dict):
            return
        action = str(debug.get("action", ""))
        if action in ("", "accept", "bootstrap_accept"):
            return

        frame = int(debug.get("frame_index", -1))
        reason = str(debug.get("reason", ""))
        key = f"{frame}:{action}:{reason}"
        now = perf_counter()
        if key == self._last_track_debug_log_key or now - self._last_track_debug_log_time < 0.35:
            return

        self._last_track_debug_log_key = key
        self._last_track_debug_log_time = now
        cand_count = int(debug.get("candidate_count", 0))
        selected = int(debug.get("selected_candidate_index", -1))
        score = float(debug.get("input_score", 0.0))
        pred_x = float(debug.get("pred_x", -1.0))
        pred_y = float(debug.get("pred_y", -1.0))
        out_visible = bool(debug.get("output_visible", 0))
        self.view.append_log(
            "[TrackDebug] "
            f"frame {frame} | {action}/{reason} | cand {cand_count} sel {selected} | "
            f"score {score:.2f} | pred ({pred_x:.1f},{pred_y:.1f}) | visible {int(out_visible)}"
        )

    def _release_probe_worker(self, worker: object) -> None:
        if worker is None:
            return
        if self._probe_worker is worker:
            if self._probe_worker.isRunning():
                QTimer.singleShot(10, lambda worker=worker: self._release_probe_worker(worker))
                return
            self._probe_worker = None

    def _release_camera_worker(self, worker: object) -> None:
        if self._camera_worker is worker:
            self._camera_worker = None

    def _release_playback_worker(self, worker: object) -> None:
        if self._playback_worker is worker:
            self._playback_worker = None

    def _release_batch_worker(self, worker: object) -> None:
        if self._batch_worker is worker:
            self._batch_worker = None

    def _on_camera_finished(self, payload: object) -> None:
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

        self._append_player_distance_summary(payload)
        if isinstance(payload, dict):
            self._set_current_rally_record(payload.get("rally_record"))
        self._set_idle_state()
        self._maybe_start_stopped_report(stopped)

    def _on_camera_failed(self, message: str) -> None:
        self.view.set_progress_busy(False)
        self.view.set_status_state("error")
        self.view.video_player.stop()
        self.view.append_log(f"[错误] {message}")
        self._set_idle_state()

    def _on_playback_finished(self, payload: object) -> None:
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

        self._append_player_distance_summary(payload)
        if isinstance(payload, dict):
            self._set_current_rally_record(payload.get("rally_record"))
        self._set_idle_state()
        self._maybe_start_stopped_report(stopped)

        if self._pending_seek_ms is not None and self._selected_video_path is not None:
            pending_seek_ms = self._pending_seek_ms
            self._pending_seek_ms = None
            self._start_playback(start_ms=pending_seek_ms, request_court_prediction=False)

    def _on_playback_failed(self, message: str) -> None:
        self.view.set_status_state("error")
        self.view.video_player.stop()
        self.view.append_log(f"[错误] {message}")
        self._set_idle_state()

    def handle_seek(self, position_ms: int) -> None:
        if self._selected_video_path is None:
            return

        self._video_meta["position_ms"] = position_ms
        if self._playback_worker is not None and self._playback_worker.isRunning():
            self._auto_report_after_stop = False
            self._pending_seek_ms = position_ms
            self._playback_worker.request_stop()
            self.view.append_log(f"[信息] 正在跳转至 {position_ms / 1000:.2f}s")
            return

        self.view.append_log(f"[信息] 预览跳转至 {position_ms / 1000:.2f}s")
        self._start_probe(self._selected_video_path, position_ms)

    def handle_force_stop(self) -> None:
        if self._camera_worker is not None and self._camera_worker.isRunning():
            self._auto_report_after_stop = True
            self.view.append_log("[信息] 正在停止摄像头实时推理...")
            self._camera_worker.request_stop()
            return

        if self._playback_worker is not None and self._playback_worker.isRunning():
            self._auto_report_after_stop = True
            self.view.append_log("[信息] 正在停止 TrackNetV3 播放...")
            self._playback_worker.request_stop()
            return

        if self._batch_worker is not None and self._batch_worker.isRunning():
            self._auto_report_after_stop = False
            self.view.append_log("[批量推理] 正在停止批量推理...")
            self._batch_worker.request_stop()
            return

        self.view.video_player.stop()
        self.view.set_status_state("stopped")
        self.view.append_log("[信息] 没有正在进行的播放任务。")

    def handle_reset(self) -> None:
        self._stop_workers(clear_pending_seek=True)
        self._reset_court_detection()
        self._selected_video_path = None
        self._selected_batch_folder = None
        self._batch_results.clear()
        self._batch_order.clear()
        self._video_meta = {}
        self._clear_manual_court_calibration()
        self.view.clear_video()
        self.view.set_input_mode(self._input_mode)
        self.view.set_batch_folder_path("")
        self.view.set_batch_rally_options([])
        self.view.set_batch_export_enabled(False)
        self._set_current_rally_record(None)
        self.view.log_console.clear()
        self._reset_metrics()
        self._set_idle_state()
        self.view.append_log("[系统] 工作区已重置。")

    def _stop_workers(self, *, clear_pending_seek: bool) -> None:
        if clear_pending_seek:
            self._pending_seek_ms = None
            self._auto_report_after_stop = False

        if self._probe_worker is not None and self._probe_worker.isRunning():
            self._probe_worker.quit()
            if self._probe_worker.wait(300):
                self._probe_worker = None
        elif self._probe_worker is not None:
            self._probe_worker = None

        if self._playback_worker is not None and self._playback_worker.isRunning():
            self._playback_worker.request_stop()
            if self._playback_worker.wait(1000):
                self._playback_worker = None
        elif self._playback_worker is not None:
            self._playback_worker = None

        if self._camera_worker is not None and self._camera_worker.isRunning():
            self._camera_worker.request_stop()
            if self._camera_worker.wait(1000):
                self._camera_worker = None
        elif self._camera_worker is not None:
            self._camera_worker = None

        if self._batch_worker is not None and self._batch_worker.isRunning():
            self._batch_worker.request_stop()
            if self._batch_worker.wait(1000):
                self._batch_worker = None
        elif self._batch_worker is not None:
            self._batch_worker = None
        if self._report_worker is not None and self._report_worker.isRunning():
            self._report_worker.quit()
            if self._report_worker.wait(1000):
                self._report_worker = None
        elif self._report_worker is not None:
            self._report_worker = None
        if self._court_service is not None:
            self._court_service.clear_pending()
        self.view.set_progress_busy(False)

    def shutdown(self) -> None:
        self._stop_workers(clear_pending_seek=True)
        if self._court_service is not None:
            self._court_service.stop()

    def handle_export_batch_results(self) -> None:
        if not self._batch_results:
            self.view.append_log("[批量推理] 暂无可导出的回合数据。")
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_path = str(self._project_root / "outputs" / f"batch_rally_data_{timestamp}.json")
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self.view,
            "导出批量回合数据",
            default_path,
            "JSON 数据 (*.json);;CSV 汇总 (*.csv)",
        )
        if not file_path:
            return

        output_path = Path(file_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix.lower() == ".csv" or "CSV" in selected_filter:
            if output_path.suffix.lower() != ".csv":
                output_path = output_path.with_suffix(".csv")
            self._write_batch_summary_csv(output_path)
        else:
            if output_path.suffix.lower() != ".json":
                output_path = output_path.with_suffix(".json")
            payload = {
                "exported_at": datetime.now().isoformat(timespec="seconds"),
                "source_folder": self._selected_batch_folder,
                "rallies": [
                    self._batch_results[item_id]
                    for item_id in self._batch_order
                    if item_id in self._batch_results
                ],
            }
            output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self.view.append_log(f"[批量推理] 已导出数据: {output_path}")

    def handle_export_report(self) -> None:
        record = self._current_rally_record or self._live_rally_record
        if record is None:
            self.view.append_log("[报告] 暂无可导出的回合报告，请先完成一次分析。")
            return
        trigger = "manual_running" if self._analysis_busy else "manual"
        self._start_report_generation(record, trigger=trigger)

    def _maybe_start_stopped_report(self, stopped: bool) -> None:
        if not stopped:
            self._auto_report_after_stop = False
            return
        if not self._auto_report_after_stop:
            return
        self._auto_report_after_stop = False
        record = self._current_rally_record or self._live_rally_record
        if record is None:
            self.view.append_log("[报告] 停止时没有可用统计快照，未调用报告 API。")
            return
        self._start_report_generation(record, trigger="stopped")

    def _start_report_generation(self, record: dict[str, Any], *, trigger: str) -> None:
        if self._report_worker is not None and self._report_worker.isRunning():
            self.view.append_log("[报告] 上一次报告生成仍在进行，请稍后。")
            return
        use_api = self._report_api_enabled and bool(
            self._report_api_endpoint and self._report_api_model and self._report_api_key
        )
        if self._report_api_enabled and not use_api:
            self.view.append_log("[报告] API 未完整配置，已使用本地报告生成。")
        if trigger == "stopped" and not use_api:
            self.view.append_log("[报告] 已停止分析；未启用完整 API 配置，本次将生成本地报告。")

        video_name = self._report_video_name(record)
        output_path = self._project_root / "assets" / "dist" / "index.html"
        try:
            latest_path, dist_data_path = self._save_report_input_snapshot(
                record,
                trigger=trigger,
                use_api=use_api,
            )
        except Exception as exc:
            self.view.append_log(f"[报告] 报告输入数据保存失败: {exc}")
            return
        self._report_worker = ReportGenerationWorker(
            rally_record=record,
            athlete_name=self._report_athlete_name(record),
            video_name=video_name,
            output_html_path=output_path,
            use_api=use_api,
            endpoint=self._report_api_endpoint,
            model=self._report_api_model,
            api_key=self._report_api_key,
        )
        self._report_worker.reportFinished.connect(self._on_report_generation_finished)
        self._report_worker.failed.connect(self._on_report_generation_failed)
        self._report_worker.finished.connect(lambda worker=self._report_worker: self._release_report_worker(worker))
        self.view.set_report_export_enabled(False)
        provider_text = "API" if use_api else "本地"
        if trigger == "stopped":
            action_text = "停止后自动生成报告"
        elif trigger == "manual_running":
            action_text = "生成当前实时报告"
        else:
            action_text = "生成报告"
        self.view.set_progress_busy(True, f"{action_text}中")
        self.view.append_log(f"[报告] 已保存输入快照: {latest_path}")
        self.view.append_log(f"[报告] 已同步 dist 数据: {dist_data_path}")
        self.view.append_log(f"[报告] {action_text} | provider {provider_text}")
        self._report_worker.start()

    def _on_report_generation_finished(self, payload: object) -> None:
        self.view.set_progress_busy(False)
        if not isinstance(payload, dict):
            self.view.append_log("[报告] 报告生成完成，但返回结果不可读。")
            self._set_idle_state()
            return
        output_path = Path(str(payload.get("output_html_path") or self._project_root / "assets" / "dist" / "index.html"))
        provider = str(payload.get("provider") or "unknown")
        api_error = str(payload.get("api_error") or "")
        if api_error:
            self.view.append_log(f"[报告] API 调用失败，已回退本地报告: {api_error}")
        self.view.append_log(f"[报告] 已写入 dist 报告文件: {output_path} | provider {provider}")
        self.view.set_report_preview_file(output_path)
        self._set_idle_state()

    def _on_report_generation_failed(self, message: str) -> None:
        self.view.set_progress_busy(False)
        self.view.append_log(f"[报告] 生成失败: {message}")
        self._set_idle_state()

    def _release_report_worker(self, worker: object) -> None:
        if self._report_worker is worker:
            self._report_worker = None

    def _report_video_name(self, record: dict[str, Any]) -> str:
        summary = record.get("summary", {})
        if not isinstance(summary, dict):
            summary = {}
        return str(
            record.get("video_name")
            or record.get("rally_name")
            or summary.get("video_name")
            or summary.get("rally_name")
            or "rally_report"
        )

    def _report_athlete_name(self, record: dict[str, Any]) -> str:
        summary = record.get("summary", {})
        players = summary.get("players", {}) if isinstance(summary, dict) else {}
        bottom = players.get("bottom", {}) if isinstance(players, dict) else {}
        label = bottom.get("label") if isinstance(bottom, dict) else None
        return str(label or "训练用户")

    def _write_batch_summary_csv(self, output_path: Path) -> None:
        fieldnames = [
            "video_name",
            "duration_s",
            "rally_hit_count",
            "top_distance_m",
            "bottom_distance_m",
            "top_avg_speed_mps",
            "bottom_avg_speed_mps",
            "top_max_speed_mps",
            "bottom_max_speed_mps",
            "top_stop_count",
            "bottom_stop_count",
            "top_start_count",
            "bottom_start_count",
            "top_front_hits",
            "top_mid_hits",
            "top_back_hits",
            "bottom_front_hits",
            "bottom_mid_hits",
            "bottom_back_hits",
            "ball_visible_rate",
            "pose_valid_rate",
            "court_valid_rate",
            "avg_ball_confidence",
        ]
        with output_path.open("w", newline="", encoding="utf-8-sig") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            for item_id in self._batch_order:
                record = self._batch_results.get(item_id)
                if not isinstance(record, dict):
                    continue
                writer.writerow(self._batch_summary_csv_row(record))

    def _batch_summary_csv_row(self, record: dict[str, Any]) -> dict[str, Any]:
        summary = record.get("summary", {})
        if not isinstance(summary, dict):
            summary = {}
        players = summary.get("players", {})
        if not isinstance(players, dict):
            players = {}
        top = players.get("top", {}) if isinstance(players.get("top", {}), dict) else {}
        bottom = players.get("bottom", {}) if isinstance(players.get("bottom", {}), dict) else {}
        reliability = summary.get("data_reliability", {})
        if not isinstance(reliability, dict):
            reliability = {}

        def zone_value(player: dict[str, Any], zone: str) -> int:
            zones = player.get("zone_hits", {})
            if not isinstance(zones, dict):
                return 0
            return int(zones.get(zone, 0))

        return {
            "video_name": record.get("video_name", summary.get("video_name", "")),
            "duration_s": self._safe_distance_value(summary.get("duration_s", 0.0)),
            "rally_hit_count": int(summary.get("rally_hit_count", 0)),
            "top_distance_m": self._safe_distance_value(top.get("distance_m", 0.0)),
            "bottom_distance_m": self._safe_distance_value(bottom.get("distance_m", 0.0)),
            "top_avg_speed_mps": self._safe_distance_value(top.get("avg_speed_mps", 0.0)),
            "bottom_avg_speed_mps": self._safe_distance_value(bottom.get("avg_speed_mps", 0.0)),
            "top_max_speed_mps": self._safe_distance_value(top.get("max_speed_mps", 0.0)),
            "bottom_max_speed_mps": self._safe_distance_value(bottom.get("max_speed_mps", 0.0)),
            "top_stop_count": int(top.get("stop_count", 0)),
            "bottom_stop_count": int(bottom.get("stop_count", 0)),
            "top_start_count": int(top.get("start_count", 0)),
            "bottom_start_count": int(bottom.get("start_count", 0)),
            "top_front_hits": zone_value(top, "front"),
            "top_mid_hits": zone_value(top, "mid"),
            "top_back_hits": zone_value(top, "back"),
            "bottom_front_hits": zone_value(bottom, "front"),
            "bottom_mid_hits": zone_value(bottom, "mid"),
            "bottom_back_hits": zone_value(bottom, "back"),
            "ball_visible_rate": self._safe_distance_value(reliability.get("ball_visible_rate", 0.0)),
            "pose_valid_rate": self._safe_distance_value(reliability.get("pose_valid_rate", 0.0)),
            "court_valid_rate": self._safe_distance_value(reliability.get("court_valid_rate", 0.0)),
            "avg_ball_confidence": self._safe_distance_value(reliability.get("avg_ball_confidence", 0.0)),
        }

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
