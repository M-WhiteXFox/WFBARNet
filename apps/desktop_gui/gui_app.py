from __future__ import annotations

import ctypes
import sys
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from typing import Any

import cv2
import dearpygui.dearpygui as dpg
import numpy as np
import torch

from apps.desktop_gui.controllers.tracknet_controller import TrackTaskConfig, run_tracknet_task
from apps.desktop_gui.state.gui_state import gui_state
from apps.desktop_gui.utils.theme import apply_global_theme


WINDOW_TITLE = "\u7fbd\u6bdb\u7403\u8f68\u8ff9\u5206\u6790\u5de5\u4f5c\u53f0"
PLACEHOLDER_VIDEO_SIZE = (960, 540)
VIDEO_PANEL_HEIGHT = 640
LOG_PANEL_HEIGHT = 220
EVENT_QUEUE: Queue[dict[str, Any]] = Queue()
STATUS_COLORS = {
    "idle": (148, 163, 184),
    "video_selected": (96, 165, 250),
    "running": (74, 222, 128),
    "completed": (56, 189, 248),
    "failed": (248, 113, 113),
    "stopped": (251, 191, 36),
}


def get_dpi_scale() -> float:
    if sys.platform != "win32":
        return 1.0
    try:
        user32 = ctypes.windll.user32
        dpi = user32.GetDpiForSystem()
        return max(1.0, float(dpi) / 96.0) if dpi else 1.0
    except Exception:
        return 1.0


def setup_fonts() -> None:
    scale = get_dpi_scale()
    font_size = max(16, round(16 * scale))
    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/msyh.ttf"),
        Path("C:/Windows/Fonts/simhei.ttf"),
    ]
    font_path = next((path for path in candidates if path.exists()), None)
    if font_path is None:
        return

    with dpg.font_registry():
        with dpg.font(str(font_path), font_size) as default_font:
            pass
        dpg.bind_font(default_font)


def available_devices() -> list[str]:
    devices = ["auto", "cpu"]
    if torch.cuda.is_available():
        devices.insert(1, "cuda:0")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and torch.backends.mps.is_available():
        devices.append("mps")
    return devices


def append_log(level: str, message: str) -> None:
    gui_state.logs.append(f"[{level}] {message}")
    _refresh_logs()


def _empty_texture_data() -> list[float]:
    width, height = PLACEHOLDER_VIDEO_SIZE
    texture = np.empty((height, width, 4), dtype=np.float32)
    texture[..., 0] = 0.08
    texture[..., 1] = 0.10
    texture[..., 2] = 0.12
    texture[..., 3] = 1.0
    return np.ascontiguousarray(texture).ravel().tolist()


def _frame_to_texture_data(frame: np.ndarray) -> list[float]:
    width, height = PLACEHOLDER_VIDEO_SIZE
    resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
    rgba = cv2.cvtColor(resized, cv2.COLOR_BGR2RGBA).astype(np.float32) / 255.0
    return np.ascontiguousarray(rgba).ravel().tolist()


def _update_preview(frame: np.ndarray | None) -> None:
    if not dpg.does_item_exist("video_texture"):
        return
    dpg.set_value("video_texture", _frame_to_texture_data(frame) if frame is not None else _empty_texture_data())


def _load_video_preview(video_path: str) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频文件：{video_path}")

    ok, frame = cap.read()
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    if not ok or frame is None:
        raise RuntimeError(f"视频已打开，但没有成功读取到首帧：{video_path}")

    gui_state.current_video_path = video_path
    gui_state.current_frame_image = frame
    gui_state.current_fps = fps
    gui_state.total_frames = total_frames
    gui_state.current_frame_idx = 0
    gui_state.summary.clear()
    gui_state.actions.clear()
    gui_state.track_results.clear()
    gui_state.result_files.clear()
    gui_state.status = "video_selected"
    gui_state.task_progress = 0.0
    gui_state.current_stage = "视频已加载，等待开始分析"
    gui_state.error_message = ""
    _update_preview(frame)


def _build_output_dir() -> Path:
    video_stem = Path(gui_state.current_video_path or "tracknet").stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(gui_state.output_dir) / f"{video_stem}_{timestamp}"


def _run_worker(config: TrackTaskConfig) -> None:
    try:
        result = run_tracknet_task(
            config,
            progress_callback=lambda payload: EVENT_QUEUE.put(payload),
            stop_requested=lambda: gui_state.stop_requested,
        )
        EVENT_QUEUE.put({"type": "complete", "payload": result})
    except Exception as exc:
        EVENT_QUEUE.put({"type": "error", "message": str(exc)})


def on_open_video_dialog() -> None:
    dpg.show_item("video_file_dialog")


def on_video_selected(_sender: str, app_data: dict[str, Any], _user_data: object) -> None:
    file_path = app_data.get("file_path_name", "")
    if not file_path:
        return
    try:
        _load_video_preview(file_path)
        append_log("INFO", f"已载入视频：{file_path}")
    except Exception as exc:
        gui_state.status = "failed"
        gui_state.current_stage = "视频加载失败"
        gui_state.error_message = str(exc)
        append_log("ERROR", str(exc))
        _update_preview(None)
    refresh_ui_from_state()
    _refresh_timeline()
    _refresh_output_files()


def on_device_changed(_sender: str, value: str) -> None:
    gui_state.device = value


def on_score_threshold_changed(_sender: str, value: float) -> None:
    gui_state.track_score_threshold = float(value)


def on_max_frames_changed(_sender: str, value: int) -> None:
    gui_state.max_frames = max(0, int(value))


def on_save_visualization_changed(_sender: str, value: bool) -> None:
    gui_state.save_visualization = bool(value)


def on_save_json_changed(_sender: str, value: bool) -> None:
    gui_state.save_json = bool(value)


def on_save_csv_changed(_sender: str, value: bool) -> None:
    gui_state.save_csv = bool(value)


def on_save_npy_changed(_sender: str, value: bool) -> None:
    gui_state.save_npy = bool(value)


def on_start() -> None:
    if gui_state.worker is not None and gui_state.worker.is_alive():
        append_log("WARNING", "当前任务仍在运行，请等待完成或先停止。")
        return
    if not gui_state.current_video_path:
        append_log("WARNING", "请先选择一个本地视频文件。")
        return

    gui_state.clear_results(keep_video=True)
    gui_state.stop_requested = False
    gui_state.status = "running"
    gui_state.current_stage = "正在启动 TrackNetV3 推理"
    gui_state.task_progress = 0.0
    gui_state.error_message = ""

    output_dir = _build_output_dir()
    config = TrackTaskConfig(
        source=gui_state.current_video_path,
        output_dir=output_dir,
        device=gui_state.device,
        score_threshold=gui_state.track_score_threshold,
        max_frames=gui_state.max_frames if gui_state.max_frames > 0 else None,
        save_visualization=gui_state.save_visualization,
        save_json=gui_state.save_json,
        save_csv=gui_state.save_csv,
        save_npy=gui_state.save_npy,
    )
    append_log("INFO", f"开始执行 TrackNetV3 推理，输出目录：{output_dir}")

    worker = Thread(target=_run_worker, args=(config,), daemon=True)
    gui_state.worker = worker
    worker.start()
    refresh_ui_from_state()


def on_stop() -> None:
    if gui_state.worker is None or not gui_state.worker.is_alive():
        append_log("WARNING", "当前没有正在运行的任务。")
        return
    gui_state.stop_requested = True
    gui_state.current_stage = "已请求停止，等待当前帧结束"
    append_log("WARNING", "已请求停止当前任务。")
    refresh_ui_from_state()


def on_reset() -> None:
    if gui_state.worker is not None and gui_state.worker.is_alive():
        gui_state.stop_requested = True
        append_log("WARNING", "任务仍在运行，已先请求停止。请稍候再次点击重置。")
        return
    gui_state.reset_runtime_fields()
    _update_preview(None)
    _refresh_timeline()
    _refresh_output_files()
    refresh_ui_from_state()
    _refresh_logs()


def create_sidebar_panel() -> None:
    with dpg.child_window(tag="sidebar_panel", border=True, height=-1):
        dpg.add_text("控制面板", color=[94, 234, 212])
        dpg.add_separator()
        dpg.add_spacer(height=6)

        with dpg.collapsing_header(label="视频输入", default_open=True):
            dpg.add_text("当前模型")
            dpg.add_text("TrackNetV3 羽毛球轨迹推理", color=[191, 219, 254])
            dpg.add_spacer(height=4)
            dpg.add_text("视频路径")
            with dpg.group(horizontal=True):
                dpg.add_input_text(
                    hint="尚未选择视频文件",
                    tag="input_video_path",
                    readonly=True,
                    width=-90,
                )
                browse_btn = dpg.add_button(label="浏览", width=84, callback=on_open_video_dialog)
                dpg.bind_item_theme(browse_btn, "theme_button_primary")

        with dpg.collapsing_header(label="推理配置", default_open=True):
            dpg.add_text("运行设备")
            dpg.add_combo(
                items=available_devices(),
                default_value=gui_state.device,
                tag="device_combo",
                width=-1,
                callback=on_device_changed,
            )
            dpg.add_spacer(height=4)
            dpg.add_text("轨迹置信度阈值")
            dpg.add_slider_float(
                default_value=gui_state.track_score_threshold,
                min_value=0.05,
                max_value=0.95,
                format="%.2f",
                width=-1,
                callback=on_score_threshold_changed,
            )
            dpg.add_spacer(height=4)
            dpg.add_text("最多处理帧数（0 表示全部）")
            dpg.add_input_int(
                default_value=gui_state.max_frames,
                min_value=0,
                min_clamped=True,
                step=100,
                width=-1,
                callback=on_max_frames_changed,
            )

        with dpg.collapsing_header(label="结果导出", default_open=True):
            dpg.add_checkbox(
                label="保存可视化视频",
                default_value=gui_state.save_visualization,
                callback=on_save_visualization_changed,
            )
            dpg.add_checkbox(label="保存 JSON", default_value=gui_state.save_json, callback=on_save_json_changed)
            dpg.add_checkbox(label="保存 CSV", default_value=gui_state.save_csv, callback=on_save_csv_changed)
            dpg.add_checkbox(label="保存 NPY", default_value=gui_state.save_npy, callback=on_save_npy_changed)

        dpg.add_spacer(height=10)
        dpg.add_separator()
        dpg.add_spacer(height=10)

        with dpg.group(horizontal=True):
            start_btn = dpg.add_button(label="开始分析", tag="btn_start", width=-180, callback=on_start)
            stop_btn = dpg.add_button(label="停止", tag="btn_stop", width=82, callback=on_stop)
            reset_btn = dpg.add_button(label="重置", tag="btn_reset", width=82, callback=on_reset)
            dpg.bind_item_theme(start_btn, "theme_button_start")
            dpg.bind_item_theme(stop_btn, "theme_button_stop")
            dpg.bind_item_theme(reset_btn, "theme_button_primary")

        dpg.add_spacer(height=12)
        dpg.add_text("状态：空闲", tag="text_status")
        dpg.add_text("阶段：等待载入视频", tag="text_stage", color=[180, 180, 180])
        dpg.add_progress_bar(tag="task_progress", default_value=0.0, width=-1)


def create_video_panel() -> None:
    width, height = PLACEHOLDER_VIDEO_SIZE
    with dpg.texture_registry(show=False):
        dpg.add_dynamic_texture(
            width,
            height,
            _empty_texture_data(),
            tag="video_texture",
            format=dpg.mvFormat_Float_rgba,
        )

    with dpg.child_window(tag="video_panel", border=True, height=VIDEO_PANEL_HEIGHT, no_scrollbar=True):
        with dpg.group(horizontal=True):
            dpg.add_text("视频预览", color=[94, 234, 212])
            dpg.add_text("帧率：0.0", tag="video_fps", color=[150, 150, 150])

        dpg.add_separator()
        dpg.add_spacer(height=10)
        with dpg.group(horizontal=True):
            dpg.add_spacer(width=8)
            dpg.add_image(
                "video_texture",
                tag="video_image_item",
                width=PLACEHOLDER_VIDEO_SIZE[0],
                height=PLACEHOLDER_VIDEO_SIZE[1],
            )
        dpg.add_spacer(height=10)
        dpg.add_text("帧号：0 / 0", tag="video_meta")


def _add_info_row(label: str, tag: str, default_value: str) -> None:
    with dpg.table_row():
        dpg.add_text(label, color=[180, 180, 180])
        dpg.add_text(default_value, tag=tag)


def create_info_panel() -> None:
    with dpg.child_window(tag="info_panel", border=True, height=-1):
        dpg.add_text("结果面板", color=[94, 234, 212])
        dpg.add_separator()
        dpg.add_spacer(height=5)

        with dpg.collapsing_header(label="当前帧信息", default_open=True):
            with dpg.table(header_row=False, borders_innerH=True, borders_outerH=True, borders_outerV=True):
                dpg.add_table_column(width_fixed=True, init_width_or_weight=110)
                dpg.add_table_column(width_stretch=True)
                _add_info_row("轨迹坐标", "info_ball_pos", "[-, -]")
                _add_info_row("轨迹分数", "info_ball_score", "0.00")
                _add_info_row("轨迹可见", "info_ball_visible", "否")
                _add_info_row("当前状态", "info_action", "暂无")

        dpg.add_spacer(height=8)
        with dpg.collapsing_header(label="推理摘要", default_open=True):
            with dpg.table(header_row=False, borders_innerH=True, borders_outerH=True, borders_outerV=True):
                dpg.add_table_column(width_fixed=True, init_width_or_weight=110)
                dpg.add_table_column(width_stretch=True)
                _add_info_row("已处理帧数", "summary_total_frames", "0")
                _add_info_row("可见轨迹帧", "summary_visible_frames", "0")
                _add_info_row("轨迹可见率", "summary_visibility", "0.0%")
                _add_info_row("平均置信度", "summary_avg_conf", "0.00")
                _add_info_row("最高置信度", "summary_peak_conf", "0.00")
                _add_info_row("输出目录", "summary_output_dir", "-")

        dpg.add_spacer(height=8)
        with dpg.collapsing_header(label="关键轨迹片段", default_open=True):
            with dpg.child_window(height=180, border=False, tag="timeline_panel"):
                dpg.add_text("暂无推理结果。", color=[150, 150, 150])

        dpg.add_spacer(height=8)
        with dpg.collapsing_header(label="输出文件", default_open=True):
            with dpg.child_window(height=160, border=False, tag="output_files_panel"):
                dpg.add_text("暂无导出文件。", color=[150, 150, 150])


def create_log_panel() -> None:
    with dpg.child_window(tag="log_panel", border=True, height=LOG_PANEL_HEIGHT):
        dpg.add_text("系统日志", color=[94, 234, 212])
        dpg.add_separator()
        with dpg.child_window(tag="log_scroll_area", width=-1, height=-1, border=False):
            dpg.add_text("[INFO] 图形界面已初始化。", color=(96, 165, 250))


def create_file_dialog() -> None:
    with dpg.file_dialog(
        directory_selector=False,
        show=False,
        callback=on_video_selected,
        tag="video_file_dialog",
        width=760,
        height=460,
        modal=True,
    ):
        dpg.add_file_extension("视频文件 (*.mp4 *.avi *.mov *.mkv){.mp4,.avi,.mov,.mkv}", color=(96, 165, 250, 255))
        dpg.add_file_extension(".*")


def _log_color(message: str) -> tuple[int, int, int]:
    if "[ERROR]" in message:
        return (248, 113, 113)
    if "[WARNING]" in message:
        return (251, 191, 36)
    if "[INFO]" in message:
        return (96, 165, 250)
    return (225, 230, 238)


def _refresh_logs() -> None:
    if not dpg.does_item_exist("log_scroll_area"):
        return
    dpg.delete_item("log_scroll_area", children_only=True)
    for line in gui_state.logs[-200:]:
        dpg.add_text(line, parent="log_scroll_area", color=_log_color(line))
    dpg.set_y_scroll("log_scroll_area", dpg.get_y_scroll_max("log_scroll_area"))


def _refresh_timeline() -> None:
    if not dpg.does_item_exist("timeline_panel"):
        return
    dpg.delete_item("timeline_panel", children_only=True)
    if not gui_state.actions:
        dpg.add_text("暂无关键轨迹片段。", parent="timeline_panel", color=[150, 150, 150])
        return
    for action in gui_state.actions:
        with dpg.group(parent="timeline_panel"):
            dpg.add_text(
                f"{action['label']}  |  第 {action['frame_id']} 帧  |  {action['start_time']:.2f} 秒",
                color=[191, 219, 254],
            )
            dpg.add_text(
                f"{action['detail']}  |  置信度 {action['confidence']:.2f}",
                color=[196, 181, 253],
            )
            dpg.add_separator()


def _refresh_output_files() -> None:
    if not dpg.does_item_exist("output_files_panel"):
        return
    dpg.delete_item("output_files_panel", children_only=True)
    if not gui_state.result_files:
        dpg.add_text("暂无导出文件。", parent="output_files_panel", color=[150, 150, 150])
        return
    for label, path in gui_state.result_files.items():
        with dpg.group(parent="output_files_panel"):
            dpg.add_text(label, color=[191, 219, 254])
            dpg.add_text(path, wrap=250, color=[196, 181, 253])
            dpg.add_spacer(height=6)


def drain_worker_events() -> None:
    while True:
        try:
            payload = EVENT_QUEUE.get_nowait()
        except Empty:
            break

        event_type = payload.get("type")
        if event_type == "stage":
            gui_state.current_stage = str(payload.get("stage", gui_state.current_stage))
            gui_state.task_progress = max(gui_state.task_progress, float(payload.get("progress", 0.0)))
        elif event_type == "frame":
            gui_state.current_stage = str(payload.get("stage", gui_state.current_stage))
            gui_state.current_frame_idx = int(payload.get("frame_id", gui_state.current_frame_idx))
            gui_state.task_progress = float(payload.get("progress", gui_state.task_progress))
            frame = payload.get("frame")
            if isinstance(frame, np.ndarray):
                gui_state.current_frame_image = frame
                _update_preview(frame)
            track = payload.get("track")
            if isinstance(track, dict):
                gui_state.track_results[gui_state.current_frame_idx] = track
        elif event_type == "complete":
            result = payload["payload"]
            gui_state.worker = None
            gui_state.summary = result.summary
            gui_state.actions = result.actions
            gui_state.track_results = result.track_results
            gui_state.result_files = result.output_files
            gui_state.current_fps = result.fps
            gui_state.total_frames = result.total_frames
            gui_state.current_frame_idx = result.last_frame_idx
            gui_state.task_progress = result.progress
            gui_state.status = "stopped" if result.stopped else "completed"
            gui_state.current_stage = "任务已停止，已保留当前结果" if result.stopped else "TrackNetV3 推理完成"
            if result.last_frame is not None:
                gui_state.current_frame_image = result.last_frame
                _update_preview(result.last_frame)
            _refresh_timeline()
            _refresh_output_files()
            append_log(
                "WARNING" if result.stopped else "INFO",
                "任务已停止，结果为截至当前帧的部分输出。" if result.stopped else "任务执行完成，结果已导出。",
            )
        elif event_type == "error":
            gui_state.worker = None
            gui_state.status = "failed"
            gui_state.current_stage = "推理执行失败"
            gui_state.error_message = str(payload.get("message", "未知错误"))
            append_log("ERROR", gui_state.error_message)


def refresh_ui_from_state() -> None:
    if dpg.does_item_exist("task_progress"):
        dpg.set_value("task_progress", gui_state.task_progress)

    if dpg.does_item_exist("text_status"):
        status_text = {
            "idle": "状态：空闲",
            "video_selected": "状态：已选择视频",
            "running": "状态：正在推理",
            "completed": "状态：已完成",
            "failed": "状态：失败",
            "stopped": "状态：已停止",
        }.get(gui_state.status, f"状态：{gui_state.status}")
        dpg.set_value("text_status", status_text)
        dpg.configure_item("text_status", color=STATUS_COLORS.get(gui_state.status, (225, 230, 238)))

    if dpg.does_item_exist("text_stage"):
        dpg.set_value("text_stage", f"阶段：{gui_state.current_stage or '-'}")

    if dpg.does_item_exist("input_video_path"):
        dpg.set_value("input_video_path", gui_state.current_video_path or "")

    if dpg.does_item_exist("video_meta"):
        dpg.set_value("video_meta", f"帧号：{gui_state.current_frame_idx} / {gui_state.total_frames}")

    if dpg.does_item_exist("video_fps"):
        dpg.set_value("video_fps", f"帧率：{gui_state.current_fps:.1f}")

    track = gui_state.track_results.get(gui_state.current_frame_idx, {})
    ball_xy = track.get("ball_xy", [-1, -1])
    visible = bool(track.get("visible", False))
    score = float(track.get("score", 0.0))

    if dpg.does_item_exist("info_ball_pos"):
        if ball_xy and len(ball_xy) >= 2 and visible:
            dpg.set_value("info_ball_pos", f"[{int(ball_xy[0])}, {int(ball_xy[1])}]")
        else:
            dpg.set_value("info_ball_pos", "[-, -]")

    if dpg.does_item_exist("info_ball_score"):
        dpg.set_value("info_ball_score", f"{score:.2f}")

    if dpg.does_item_exist("info_ball_visible"):
        dpg.set_value("info_ball_visible", "是" if visible else "否")

    if dpg.does_item_exist("info_action"):
        dpg.set_value("info_action", gui_state.current_stage or "暂无")

    summary = gui_state.summary
    if dpg.does_item_exist("summary_total_frames"):
        dpg.set_value("summary_total_frames", str(summary.get("total_frames", 0)))
    if dpg.does_item_exist("summary_visible_frames"):
        dpg.set_value("summary_visible_frames", str(summary.get("visible_frames", 0)))
    if dpg.does_item_exist("summary_visibility"):
        dpg.set_value("summary_visibility", f"{float(summary.get('visibility_ratio', 0.0)) * 100:.1f}%")
    if dpg.does_item_exist("summary_avg_conf"):
        dpg.set_value("summary_avg_conf", f"{float(summary.get('avg_confidence', 0.0)):.2f}")
    if dpg.does_item_exist("summary_peak_conf"):
        dpg.set_value("summary_peak_conf", f"{float(summary.get('peak_confidence', 0.0)):.2f}")
    if dpg.does_item_exist("summary_output_dir"):
        dpg.set_value("summary_output_dir", str(summary.get("output_dir", "-")))


def build_gui() -> None:
    dpg.create_context()
    apply_global_theme()
    setup_fonts()
    create_file_dialog()

    with dpg.window(tag="PrimaryWindow", no_scrollbar=True, no_scroll_with_mouse=True):
        with dpg.table(
            header_row=False,
            borders_innerH=False,
            borders_innerV=False,
            borders_outerH=False,
            borders_outerV=False,
        ):
            dpg.add_table_column(width_fixed=True, init_width_or_weight=330)
            dpg.add_table_column(width_stretch=True, init_width_or_weight=1.0)
            dpg.add_table_column(width_fixed=True, init_width_or_weight=320)

            with dpg.table_row():
                create_sidebar_panel()
                with dpg.child_window(tag="center_panel", border=False, height=-1, no_scrollbar=True):
                    create_video_panel()
                    dpg.add_spacer(height=6)
                    create_log_panel()
                create_info_panel()

    dpg.create_viewport(title=WINDOW_TITLE, width=1500, height=900)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("PrimaryWindow", True)

    _update_preview(gui_state.current_frame_image)
    _refresh_logs()
    _refresh_timeline()
    _refresh_output_files()
    refresh_ui_from_state()

    while dpg.is_dearpygui_running():
        drain_worker_events()
        refresh_ui_from_state()
        dpg.render_dearpygui_frame()

    dpg.destroy_context()


if __name__ == "__main__":
    build_gui()
