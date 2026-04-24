# -*- coding: utf-8 -*-
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)


class VideoPlayerWidget(QFrame):
    """由外部帧驱动的纯显示视频预览组件。"""

    selectRequested = pyqtSignal()
    forceStopRequested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("videoPlayerCard")

        self._source_path = ""
        self._current_pixmap: QPixmap | None = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.btn_select_video = QPushButton("选择视频")
        self.btn_select_video.setObjectName("btnSelectVideo")

        self.path_edit = QLineEdit()
        self.path_edit.setObjectName("videoPathEdit")
        self.path_edit.setReadOnly(True)
        self.path_edit.setPlaceholderText("视频路径")
        self.path_edit.setClearButtonEnabled(False)

        self.btn_force_stop = QPushButton("停止")
        self.btn_force_stop.setObjectName("btnForceStop")

        self.preview_stack = QStackedWidget()
        self.preview_stack.setObjectName("videoStack")
        self.preview_stack.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        placeholder_page = QWidget()
        placeholder_layout = QVBoxLayout(placeholder_page)
        placeholder_layout.setContentsMargins(0, 0, 0, 0)

        placeholder_frame = QFrame()
        placeholder_frame.setObjectName("videoPlaceholderFrame")
        placeholder_frame_layout = QVBoxLayout(placeholder_frame)
        placeholder_frame_layout.setContentsMargins(24, 24, 24, 24)
        placeholder_frame_layout.setSpacing(10)

        placeholder_title = QLabel("预览")
        placeholder_title.setObjectName("videoPlaceholderTitle")
        placeholder_title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        placeholder_hint = QLabel("在此处选择视频以预览 TrackNetV3 帧。")
        placeholder_hint.setObjectName("videoPlaceholderHint")
        placeholder_hint.setWordWrap(True)
        placeholder_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)

        placeholder_frame_layout.addStretch(1)
        placeholder_frame_layout.addWidget(placeholder_title)
        placeholder_frame_layout.addWidget(placeholder_hint)
        placeholder_frame_layout.addStretch(1)
        placeholder_layout.addWidget(placeholder_frame)

        self.video_label = QLabel()
        self.video_label.setObjectName("videoLabel")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.video_label.setMinimumSize(320, 240)

        self.preview_stack.addWidget(placeholder_page)
        self.preview_stack.addWidget(self.video_label)
        self.preview_stack.setCurrentWidget(placeholder_page)

        outer.addWidget(self.preview_stack, stretch=1)

        self.btn_select_video.clicked.connect(self.selectRequested.emit)
        self.btn_force_stop.clicked.connect(self.forceStopRequested.emit)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._render_current_pixmap()

    def _render_current_pixmap(self) -> None:
        if self._current_pixmap is None:
            return
        label_size = self.video_label.size()
        if label_size.width() <= 0 or label_size.height() <= 0:
            label_size = self.preview_stack.size()
        if label_size.width() <= 0 or label_size.height() <= 0:
            QTimer.singleShot(10, self._render_current_pixmap)
            return
        scaled = self._current_pixmap.scaled(
            label_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(scaled)

    def _set_status(self, text: str, state: str) -> None:
        self.video_label.setProperty("state", state)
        self.video_label.setToolTip(text)
        self.style().unpolish(self.video_label)
        self.style().polish(self.video_label)
        self.video_label.update()

    def display_image(self, image: QImage) -> None:
        if image.isNull():
            return
        self._current_pixmap = QPixmap.fromImage(image)
        self.preview_stack.setCurrentWidget(self.video_label)
        QTimer.singleShot(0, self._render_current_pixmap)
        self._set_status("帧已就绪", "loaded")

    def set_video_path(self, path: str) -> None:
        self._source_path = path
        self.path_edit.setText(path)
        self.path_edit.setToolTip(path)

    def clear_video(self) -> None:
        self._source_path = ""
        self._current_pixmap = None
        self.path_edit.clear()
        self.path_edit.setToolTip("")
        self.video_label.clear()
        self._set_status("未加载视频", "idle")
        self.preview_stack.setCurrentWidget(self.preview_stack.widget(0))

    def play(self) -> None:
        self._set_status("播放中", "playing")

    def pause(self) -> None:
        if self._source_path:
            self._set_status("已暂停", "loaded")

    def stop(self) -> None:
        if self._source_path:
            self._set_status("已停止", "stopped")

    def set_video_state(self, state: str) -> None:
        mapping = {
            "idle": "未加载视频",
            "loaded": "就绪",
            "playing": "播放中",
            "stopped": "已停止",
            "error": "视频加载失败",
        }
        self._set_status(mapping.get(state, state), state)

    def current_path(self) -> str:
        return self._source_path


class VideoTimelineWidget(QFrame):
    """从控制器层手动控制的时间轴。"""

    seekRequested = pyqtSignal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("videoTimelineCard")
        self._dragging = False
        self._duration_ms = 0
        self._position_ms = 0

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(10)

        self.seek_slider = QSlider(Qt.Orientation.Horizontal)
        self.seek_slider.setObjectName("videoTimeline")
        self.seek_slider.setRange(0, 0)
        self.seek_slider.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )

        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setObjectName("timeLabel")
        self.time_label.setFixedWidth(110)
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self.seek_slider)
        layout.addWidget(self.time_label)

        self.seek_slider.sliderPressed.connect(self._on_slider_pressed)
        self.seek_slider.sliderReleased.connect(self._on_slider_released)
        self.seek_slider.sliderMoved.connect(self._on_slider_moved)

    @staticmethod
    def _format_time(ms: int) -> str:
        total_sec = max(0, ms) // 1000
        hours, remainder = divmod(total_sec, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

    def _refresh_label(self, position_ms: int | None = None) -> None:
        pos = self._position_ms if position_ms is None else position_ms
        self.time_label.setText(
            f"{self._format_time(pos)} / {self._format_time(self._duration_ms)}"
        )

    def _on_slider_pressed(self) -> None:
        self._dragging = True

    def _on_slider_released(self) -> None:
        self._dragging = False
        value = self.seek_slider.value()
        self._position_ms = value
        self._refresh_label()
        self.seekRequested.emit(value)

    def _on_slider_moved(self, value: int) -> None:
        self._refresh_label(value)

    def set_duration(self, duration_ms: int) -> None:
        self._duration_ms = max(0, duration_ms)
        self.seek_slider.setRange(0, self._duration_ms)
        self._refresh_label()

    def set_position(self, position_ms: int) -> None:
        self._position_ms = max(0, min(position_ms, self._duration_ms))
        if not self._dragging:
            self.seek_slider.setValue(self._position_ms)
            self._refresh_label()

    def set_interactive(self, enabled: bool) -> None:
        self.seek_slider.setEnabled(enabled)

    def reset(self) -> None:
        self._dragging = False
        self._duration_ms = 0
        self._position_ms = 0
        self.seek_slider.setRange(0, 0)
        self.seek_slider.setValue(0)
        self.time_label.setText("00:00 / 00:00")
