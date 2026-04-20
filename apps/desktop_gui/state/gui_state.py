from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import Thread
from typing import Any

import numpy as np


def _default_output_dir() -> str:
    project_root = Path(__file__).resolve().parents[3]
    return str(project_root / "outputs" / "desktop_gui")


@dataclass
class GUIState:
    status: str = "idle"
    task_progress: float = 0.0
    current_stage: str = "等待载入视频"

    current_video_path: str | None = None
    output_dir: str = field(default_factory=_default_output_dir)
    device: str = "auto"
    track_score_threshold: float = 0.50
    max_frames: int = 0
    save_visualization: bool = True
    save_json: bool = True
    save_csv: bool = True
    save_npy: bool = True

    current_frame_idx: int = 0
    total_frames: int = 0
    current_frame_image: np.ndarray | None = None
    current_fps: float = 0.0

    summary: dict[str, Any] = field(default_factory=dict)
    actions: list[dict[str, Any]] = field(default_factory=list)
    track_results: dict[int, dict[str, Any]] = field(default_factory=dict)
    result_files: dict[str, str] = field(default_factory=dict)

    error_message: str = ""
    logs: list[str] = field(default_factory=lambda: ["[INFO] 图形界面已初始化。"])
    stop_requested: bool = False
    worker: Thread | None = None

    def clear_results(self, keep_video: bool = True) -> None:
        self.status = "video_selected" if keep_video and self.current_video_path else "idle"
        self.task_progress = 0.0
        self.current_stage = "视频已加载，等待开始分析" if keep_video and self.current_video_path else "等待载入视频"
        if not keep_video:
            self.current_video_path = None
            self.current_frame_image = None
            self.current_fps = 0.0
            self.total_frames = 0
        self.current_frame_idx = 0
        self.summary.clear()
        self.actions.clear()
        self.track_results.clear()
        self.result_files.clear()
        self.error_message = ""
        self.stop_requested = False

    def reset_runtime_fields(self) -> None:
        self.current_video_path = None
        self.current_frame_image = None
        self.current_fps = 0.0
        self.total_frames = 0
        self.current_frame_idx = 0
        self.summary.clear()
        self.actions.clear()
        self.track_results.clear()
        self.result_files.clear()
        self.error_message = ""
        self.stop_requested = False
        self.worker = None
        self.status = "idle"
        self.task_progress = 0.0
        self.current_stage = "等待载入视频"
        self.logs = ["[INFO] 图形界面状态已重置。"]


gui_state = GUIState()
