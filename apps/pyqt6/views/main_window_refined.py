from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QEasingCurve, QPoint, QPropertyAnimation, QRectF, QSize, Qt, pyqtProperty, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QPainter
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QCheckBox,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QComboBox,
    QSizePolicy,
    QPushButton,
    QProgressBar,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from apps.pyqt6.views.components.video_player_panel_runtime import (
    VideoPlayerWidget,
    VideoTimelineWidget,
)


class ToggleSwitch(QCheckBox):
    def __init__(self, tooltip: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._offset = 1.0
        self.setObjectName("modelSwitch")
        self.setFixedSize(46, 24)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(tooltip)
        self.setText("")
        self._animation = QPropertyAnimation(self, b"offset", self)
        self._animation.setDuration(140)
        self._animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.toggled.connect(self._animate_toggle)

    def get_offset(self) -> float:
        return self._offset

    def set_offset(self, value: float) -> None:
        self._offset = max(0.0, min(float(value), 1.0))
        self.update()

    offset = pyqtProperty(float, fget=get_offset, fset=set_offset)

    def sizeHint(self) -> QSize:
        return QSize(46, 24)

    def hitButton(self, pos: QPoint) -> bool:
        return self.rect().contains(pos)

    def setChecked(self, checked: bool) -> None:
        super().setChecked(checked)
        if not self._animation.state() == QPropertyAnimation.State.Running:
            self.set_offset(1.0 if checked else 0.0)

    def _animate_toggle(self, checked: bool) -> None:
        self._animation.stop()
        self._animation.setStartValue(self._offset)
        self._animation.setEndValue(1.0 if checked else 0.0)
        self._animation.start()

    def paintEvent(self, event) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        track_rect = QRectF(1, 2, 44, 20)
        off_color = QColor("#9CA3AF")
        on_color = QColor("#22C55E")
        track_color = QColor(
            int(off_color.red() + (on_color.red() - off_color.red()) * self._offset),
            int(off_color.green() + (on_color.green() - off_color.green()) * self._offset),
            int(off_color.blue() + (on_color.blue() - off_color.blue()) * self._offset),
        )
        if not self.isEnabled():
            track_color = QColor("#D1D5DB")
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(track_color)
        painter.drawRoundedRect(track_rect, 10, 10)

        knob_size = 18
        knob_x = 3 + (22 * self._offset)
        knob_rect = QRectF(knob_x, 3, knob_size, knob_size)
        painter.setBrush(QColor("#FFFFFF"))
        painter.drawEllipse(knob_rect)


class MainWindow(QMainWindow):
    """视图层：负责布局、控件实例化和基础状态展示。"""

    poseModelBrowseRequested = pyqtSignal()
    trackModelBrowseRequested = pyqtSignal()
    modelSettingsApplyRequested = pyqtSignal(str, str)
    modelSettingsDefaultsRequested = pyqtSignal()
    modelSwitchesChanged = pyqtSignal(bool, bool)
    courtRedetectRequested = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("羽毛球动作识别分析平台")
        self.resize(1360, 860)
        self.setMinimumSize(1200, 760)

        self.central_widget = QWidget()
        self.central_widget.setObjectName("appRoot")
        self.setCentralWidget(self.central_widget)

        self.root_layout = QVBoxLayout(self.central_widget)
        self.root_layout.setContentsMargins(20, 20, 20, 20)
        self.root_layout.setSpacing(16)

        self._build_header()
        self._build_body()

    def _build_header(self) -> None:
        self.header_card = QFrame()
        self.header_card.setObjectName("headerCard")
        header_layout = QHBoxLayout(self.header_card)
        header_layout.setContentsMargins(20, 18, 20, 18)
        header_layout.setSpacing(18)

        brand_col = QVBoxLayout()
        brand_col.setSpacing(6)

        self.title_label = QLabel("羽毛球动作分析平台")
        self.title_label.setObjectName("titleLabel")

        self.subtitle_label = QLabel(
            "YOLOv11 负责视觉理解，BST 负责动作时序识别。左侧预览，右侧看结果与日志。"
        )
        self.subtitle_label.setObjectName("subtitleLabel")
        self.subtitle_label.setWordWrap(True)

        chip_row = QHBoxLayout()
        chip_row.setSpacing(8)
        self.mode_chip = QLabel("实时分析界面")
        self.mode_chip.setObjectName("modeChip")
        self.pipeline_chip = QLabel("YOLOv11 · BST")
        self.pipeline_chip.setObjectName("pipelineChip")
        chip_row.addWidget(self.mode_chip)
        chip_row.addWidget(self.pipeline_chip)
        chip_row.addStretch(1)

        brand_col.addWidget(self.title_label)
        brand_col.addWidget(self.subtitle_label)
        brand_col.addLayout(chip_row)

        actions_col = QVBoxLayout()
        actions_col.setSpacing(10)

        button_row = QHBoxLayout()
        button_row.setSpacing(8)

        self.style_label = QLabel("主题")
        self.style_label.setObjectName("styleLabel")

        self.style_btn = QToolButton()
        self.style_btn.setObjectName("styleBtn")
        self.style_btn.setFixedHeight(32)
        self.style_btn.setMinimumWidth(140)
        self.style_btn.setMaximumWidth(180)
        self.style_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.style_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self._style_menu = QMenu(self.style_btn)
        self._style_menu.setObjectName("styleMenu")
        self.style_btn.setMenu(self._style_menu)

        self.btn_analyze = QPushButton("开始分析")
        self.btn_analyze.setObjectName("btnAnalyze")
        self.btn_reset = QPushButton("重置")
        self.btn_reset.setObjectName("btnReset")

        button_row.addWidget(self.style_label)
        button_row.addWidget(self.style_btn)
        button_row.addSpacing(4)
        button_row.addWidget(self.btn_analyze)
        button_row.addWidget(self.btn_reset)

        actions_col.addLayout(button_row)

        self.status_label = QLabel("系统状态：待机中")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setProperty("state", "idle")
        brand_col.addWidget(self.status_label)

        header_layout.addLayout(brand_col, stretch=1)
        header_layout.addLayout(actions_col, stretch=0)
        self.root_layout.addWidget(self.header_card)

    def populate_stylesheets(self, theme_dirs: list[Path], active_name: str = "office_light") -> None:
        self._style_menu.clear()
        for theme_dir in theme_dirs:
            display_name = theme_dir.name.replace("_", " ").title()
            action = QAction(display_name, self._style_menu)
            action.setData(theme_dir.name)
            self._style_menu.addAction(action)

        active = active_name if any(d.name == active_name for d in theme_dirs) else (theme_dirs[0].name if theme_dirs else "")
        active_label = active.replace("_", " ").title() if active else ""
        self.style_btn.setText(f"{active_label}  ▾")

    def _build_body(self) -> None:
        body_layout = QHBoxLayout()
        body_layout.setSpacing(16)

        self._build_preview_panel(body_layout)
        self._build_analytics_panel(body_layout)

        self.root_layout.addLayout(body_layout, stretch=1)

    def _build_preview_panel(self, body_layout: QHBoxLayout) -> None:
        preview_shell = QWidget()
        preview_shell.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        preview_shell_layout = QVBoxLayout(preview_shell)
        preview_shell_layout.setContentsMargins(0, 0, 0, 0)
        preview_shell_layout.setSpacing(0)

        preview_panel = QFrame()
        preview_panel.setObjectName("previewCard")
        preview_panel.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        preview_layout = QVBoxLayout(preview_panel)
        preview_layout.setContentsMargins(18, 18, 18, 18)
        preview_layout.setSpacing(14)

        preview_header = QHBoxLayout()
        preview_header.setSpacing(2)

        self.btn_preview_mode = QPushButton("视频预览")
        self.btn_preview_mode.setObjectName("btnPreviewMode")
        self.btn_preview_mode.setCheckable(True)
        self.btn_preview_mode.setChecked(True)

        self.btn_camera_mode = QPushButton("摄像头实时推理")
        self.btn_camera_mode.setObjectName("btnCameraMode")
        self.btn_camera_mode.setCheckable(True)

        preview_header.addWidget(self.btn_preview_mode)
        preview_header.addWidget(self.btn_camera_mode)
        preview_header.addStretch(1)

        self.progress_bar = QProgressBar(preview_panel)
        self.progress_bar.setObjectName("topProgress")
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFixedHeight(16)
        self.progress_bar.setVisible(False)

        self.video_player = VideoPlayerWidget()
        self.video_player.setMinimumHeight(360)

        court_controls = QFrame()
        court_controls.setObjectName("courtControlsBar")
        court_controls_layout = QHBoxLayout(court_controls)
        court_controls_layout.setContentsMargins(0, 0, 0, 0)
        court_controls_layout.setSpacing(8)
        court_controls_layout.addStretch(1)

        self.btn_redetect_court = QPushButton("重新预测球场线")
        self.btn_redetect_court.setObjectName("btnRedetectCourt")
        self.btn_redetect_court.setEnabled(False)
        self.btn_redetect_court.clicked.connect(self.courtRedetectRequested.emit)
        court_controls_layout.addWidget(self.btn_redetect_court)

        video_controls = QFrame()
        video_controls.setObjectName("videoControlsBar")
        controls_layout = QHBoxLayout(video_controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(8)
        controls_layout.addWidget(self.video_player.btn_select_video)
        controls_layout.addWidget(self.video_player.path_edit, stretch=1)
        self.camera_device_combo = QComboBox()
        self.camera_device_combo.setObjectName("cameraDeviceCombo")
        self.camera_device_combo.setMinimumWidth(220)
        self.camera_device_combo.setVisible(False)

        self.btn_refresh_cameras = QPushButton("刷新设备")
        self.btn_refresh_cameras.setObjectName("btnRefreshCameras")
        self.btn_refresh_cameras.setVisible(False)

        controls_layout.addWidget(self.camera_device_combo)
        controls_layout.addWidget(self.btn_refresh_cameras)
        controls_layout.addWidget(self.video_player.btn_force_stop)

        self.video_timeline = VideoTimelineWidget()
        timeline_bar = QWidget()
        timeline_bar.setObjectName("timelineBar")
        timeline_bar.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        timeline_bar_layout = QVBoxLayout(timeline_bar)
        timeline_bar_layout.setSpacing(0)
        timeline_bar_layout.addWidget(self.video_timeline)

        preview_layout.addWidget(court_controls, 0)
        preview_layout.addWidget(video_controls, 0)
        preview_layout.addWidget(self.video_player, 1)
        preview_layout.addWidget(timeline_bar, 0)

        preview_shell_layout.addLayout(preview_header)
        preview_shell_layout.addWidget(preview_panel, stretch=1)

        body_layout.addWidget(preview_shell, stretch=6)

    def _build_analytics_panel(self, body_layout: QHBoxLayout) -> None:
        analytics_panel = QFrame()
        analytics_panel.setObjectName("analyticsCard")
        analytics_layout = QVBoxLayout(analytics_panel)
        analytics_layout.setContentsMargins(18, 18, 18, 18)
        analytics_layout.setSpacing(14)

        metrics_grid = QGridLayout()
        metrics_grid.setHorizontalSpacing(12)
        metrics_grid.setVerticalSpacing(12)

        card1, self.lbl_realtime_fps = self._create_metric_card("实时帧数", "0.0 FPS")
        card2, self.lbl_avg_conf = self._create_metric_card("平均置信度", "0.0%")
        card3, self.lbl_valid_pose = self._create_metric_card("推理 FPS", "0.0 FPS")
        card4, self.lbl_valid_track = self._create_metric_card("有效轨迹帧数", "0")

        metrics_grid.addWidget(card1, 0, 0)
        metrics_grid.addWidget(card2, 0, 1)
        metrics_grid.addWidget(card3, 1, 0)
        metrics_grid.addWidget(card4, 1, 1)

        self.tabs = QTabWidget()
        self.tabs.setObjectName("mainTabs")

        self._build_overview_tab()
        self._build_pose_tab()
        self._build_settings_tab()
        self._build_log_tab()

        analytics_layout.addLayout(metrics_grid)
        analytics_layout.addWidget(self.tabs, stretch=1)
        body_layout.addWidget(analytics_panel, stretch=5)

    def _build_overview_tab(self) -> None:
        tab_overview = QWidget()
        overview_layout = QVBoxLayout(tab_overview)
        overview_layout.setContentsMargins(12, 12, 12, 12)
        overview_layout.setSpacing(12)

        section_header = QHBoxLayout()
        section_title = QLabel("动作时序识别结果")
        section_title.setObjectName("sectionTitle")
        section_note = QLabel("BST Model 输出")
        section_note.setObjectName("sectionNote")
        section_header.addWidget(section_title)
        section_header.addStretch(1)
        section_header.addWidget(section_note)

        self.table_actions = QTableWidget(0, 4)
        self.table_actions.setObjectName("actionTable")
        self.table_actions.setHorizontalHeaderLabels(["时间段", "动作类别", "置信度", "动作细节"])
        self.table_actions.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table_actions.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table_actions.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table_actions.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.table_actions.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table_actions.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table_actions.setAlternatingRowColors(True)
        self.table_actions.verticalHeader().setVisible(False)
        self.table_actions.setShowGrid(True)

        overview_layout.addLayout(section_header)
        overview_layout.addWidget(self.table_actions)
        self.tabs.addTab(tab_overview, "概览")

    def _build_pose_tab(self) -> None:
        tab_pose = QWidget()
        pose_layout = QVBoxLayout(tab_pose)
        pose_layout.setContentsMargins(12, 12, 12, 12)

        pose_frame = QFrame()
        pose_frame.setObjectName("emptyStateCard")
        pose_frame_layout = QVBoxLayout(pose_frame)
        pose_frame_layout.setContentsMargins(24, 24, 24, 24)
        pose_frame_layout.setSpacing(10)

        pose_title = QLabel("姿态与轨迹")
        pose_title.setObjectName("sectionTitle")
        pose_info = QLabel("后续可接入骨架点可视化、球轨迹图和更细粒度的动作回放。")
        pose_info.setObjectName("poseInfo")
        pose_info.setWordWrap(True)
        pose_info.setAlignment(Qt.AlignmentFlag.AlignCenter)

        pose_frame_layout.addWidget(pose_title, alignment=Qt.AlignmentFlag.AlignCenter)
        pose_frame_layout.addWidget(pose_info)
        pose_layout.addWidget(pose_frame)
        self.tabs.addTab(tab_pose, "姿态")

    def _build_log_tab(self) -> None:
        tab_logs = QWidget()
        logs_layout = QVBoxLayout(tab_logs)
        logs_layout.setContentsMargins(12, 12, 12, 12)

        self.log_console = QTextEdit()
        self.log_console.setObjectName("logConsole")
        self.log_console.setReadOnly(True)
        self.log_console.setPlaceholderText("系统日志")
        logs_layout.addWidget(self.log_console)
        self.tabs.addTab(tab_logs, "日志")

    def _build_settings_tab(self) -> None:
        tab_settings = QWidget()
        settings_layout = QVBoxLayout(tab_settings)
        settings_layout.setContentsMargins(12, 12, 12, 12)
        settings_layout.setSpacing(12)

        section_header = QHBoxLayout()
        section_title = QLabel("模型设置")
        section_title.setObjectName("sectionTitle")
        section_note = QLabel("应用后重新加载推理模型")
        section_note.setObjectName("sectionNote")
        section_header.addWidget(section_title)
        section_header.addStretch(1)
        section_header.addWidget(section_note)

        settings_frame = QFrame()
        settings_frame.setObjectName("emptyStateCard")
        settings_frame_layout = QGridLayout(settings_frame)
        settings_frame_layout.setContentsMargins(18, 18, 18, 18)
        settings_frame_layout.setHorizontalSpacing(10)
        settings_frame_layout.setVerticalSpacing(12)

        pose_label = QLabel("骨骼模型")
        pose_label.setObjectName("styleLabel")
        self.pose_model_enabled = ToggleSwitch("启用骨骼模型")
        self.pose_model_enabled.setChecked(True)
        self.pose_model_edit = QLineEdit()
        self.pose_model_edit.setObjectName("modelPathEdit")
        self.pose_model_edit.setPlaceholderText("选择骨骼/姿态模型权重文件")
        self.btn_browse_pose_model = QPushButton("浏览")
        self.btn_browse_pose_model.setObjectName("btnBrowsePoseModel")
        self.btn_browse_pose_model.clicked.connect(self.poseModelBrowseRequested.emit)

        track_label = QLabel("球轨迹模型")
        track_label.setObjectName("styleLabel")
        self.track_model_enabled = ToggleSwitch("启用球轨迹模型")
        self.track_model_enabled.setChecked(True)
        self.track_model_edit = QLineEdit()
        self.track_model_edit.setObjectName("modelPathEdit")
        self.track_model_edit.setPlaceholderText("选择球轨迹模型权重文件")
        self.btn_browse_track_model = QPushButton("浏览")
        self.btn_browse_track_model.setObjectName("btnBrowseTrackModel")
        self.btn_browse_track_model.clicked.connect(self.trackModelBrowseRequested.emit)

        settings_frame_layout.addWidget(pose_label, 0, 0)
        settings_frame_layout.addWidget(self.pose_model_enabled, 0, 1)
        settings_frame_layout.addWidget(self.pose_model_edit, 0, 2)
        settings_frame_layout.addWidget(self.btn_browse_pose_model, 0, 3)
        settings_frame_layout.addWidget(track_label, 1, 0)
        settings_frame_layout.addWidget(self.track_model_enabled, 1, 1)
        settings_frame_layout.addWidget(self.track_model_edit, 1, 2)
        settings_frame_layout.addWidget(self.btn_browse_track_model, 1, 3)
        settings_frame_layout.setColumnStretch(2, 1)

        self.pose_model_enabled.stateChanged.connect(self._emit_model_switches_changed)
        self.track_model_enabled.stateChanged.connect(self._emit_model_switches_changed)

        action_row = QHBoxLayout()
        action_row.addStretch(1)
        self.btn_model_defaults = QPushButton("恢复默认")
        self.btn_model_defaults.setObjectName("btnModelDefaults")
        self.btn_apply_model_settings = QPushButton("应用设置")
        self.btn_apply_model_settings.setObjectName("btnApplyModelSettings")
        self.btn_model_defaults.clicked.connect(self.modelSettingsDefaultsRequested.emit)
        self.btn_apply_model_settings.clicked.connect(self._emit_model_settings_apply)
        action_row.addWidget(self.btn_model_defaults)
        action_row.addWidget(self.btn_apply_model_settings)

        settings_layout.addLayout(section_header)
        settings_layout.addWidget(settings_frame)
        settings_layout.addLayout(action_row)
        settings_layout.addStretch(1)
        self.tabs.addTab(tab_settings, "设置")

    def _create_metric_card(self, title: str, value: str) -> tuple[QFrame, QLabel]:
        container = QFrame()
        container.setObjectName("metricCard")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(6)

        title_lbl = QLabel(title)
        title_lbl.setObjectName("metricTitle")
        value_lbl = QLabel(value)
        value_lbl.setObjectName("metricValue")

        layout.addWidget(title_lbl)
        layout.addWidget(value_lbl)
        return container, value_lbl

    def _refresh_widget(self, widget: QWidget) -> None:
        widget.style().unpolish(widget)
        widget.style().polish(widget)
        widget.update()

    def set_video_state(self, state: str) -> None:
        self.video_player.set_video_state(state)

    def set_video_path(self, path: str) -> None:
        self.video_player.set_video_path(path)

    def set_input_mode(self, mode: str) -> None:
        is_camera = mode == "camera"
        self.btn_preview_mode.setChecked(not is_camera)
        self.btn_camera_mode.setChecked(is_camera)
        self.video_player.btn_select_video.setVisible(not is_camera)
        self.video_player.path_edit.setVisible(not is_camera)
        self.camera_device_combo.setVisible(is_camera)
        self.btn_refresh_cameras.setVisible(is_camera)
        self.video_timeline.setVisible(not is_camera)
        self.btn_analyze.setText("开始推理" if is_camera else "开始分析")
        if is_camera:
            self.video_player.path_edit.clear()

    def set_camera_devices(self, devices: list[tuple[int, str]]) -> None:
        self.camera_device_combo.blockSignals(True)
        self.camera_device_combo.clear()
        for device_id, label in devices:
            self.camera_device_combo.addItem(label, device_id)
        self.camera_device_combo.blockSignals(False)

    def selected_camera_device(self) -> int | None:
        if self.camera_device_combo.count() <= 0:
            return None
        return int(self.camera_device_combo.currentData())

    def set_model_settings(self, pose_model_path: str, track_model_path: str) -> None:
        self.pose_model_edit.setText(pose_model_path)
        self.track_model_edit.setText(track_model_path)

    def model_settings(self) -> tuple[str, str]:
        return self.pose_model_edit.text().strip(), self.track_model_edit.text().strip()

    def set_model_switches(self, pose_enabled: bool, track_enabled: bool) -> None:
        self.pose_model_enabled.blockSignals(True)
        self.track_model_enabled.blockSignals(True)
        self.pose_model_enabled.setChecked(pose_enabled)
        self.track_model_enabled.setChecked(track_enabled)
        self.pose_model_enabled.blockSignals(False)
        self.track_model_enabled.blockSignals(False)

    def model_switches(self) -> tuple[bool, bool]:
        return self.pose_model_enabled.isChecked(), self.track_model_enabled.isChecked()

    def set_model_settings_enabled(self, enabled: bool) -> None:
        widgets = (
            self.pose_model_enabled,
            self.track_model_enabled,
            self.pose_model_edit,
            self.track_model_edit,
            self.btn_browse_pose_model,
            self.btn_browse_track_model,
            self.btn_model_defaults,
            self.btn_apply_model_settings,
        )
        for widget in widgets:
            widget.setEnabled(enabled)

    def _emit_model_settings_apply(self) -> None:
        pose_model_path, track_model_path = self.model_settings()
        self.modelSettingsApplyRequested.emit(pose_model_path, track_model_path)

    def _emit_model_switches_changed(self) -> None:
        pose_enabled, track_enabled = self.model_switches()
        self.modelSwitchesChanged.emit(pose_enabled, track_enabled)

    def show_video_frame(self, image, position_ms: int, duration_ms: int) -> None:
        self.video_player.display_image(image)
        self.video_timeline.set_duration(duration_ms)
        self.video_timeline.set_position(position_ms)

    def stop_video(self) -> None:
        self.video_player.stop()

    def clear_video(self) -> None:
        self.video_player.clear_video()
        self.video_timeline.reset()

    def set_status_state(self, state: str) -> None:
        self.status_label.setProperty("state", state)
        self._refresh_widget(self.status_label)

    def append_log(self, text: str) -> None:
        self.log_console.append(text)

    def update_progress(self, val: int) -> None:
        if self.progress_bar.minimum() != 0 or self.progress_bar.maximum() != 100:
            self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(val)

    def set_progress_busy(self, busy: bool, text: str = "") -> None:
        if busy:
            self.progress_bar.setRange(0, 0)
            if text:
                self.progress_bar.setFormat(text)
            return
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setFormat("%p%")

    def add_action_row(self, time_range: str, label: str, conf: float, detail: str) -> None:
        row = self.table_actions.rowCount()
        self.table_actions.insertRow(row)

        time_item = QTableWidgetItem(time_range)
        time_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.table_actions.setItem(row, 0, time_item)

        label_item = QTableWidgetItem(label)
        label_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.table_actions.setItem(row, 1, label_item)

        conf_item = QTableWidgetItem(f"{conf * 100:.1f}%")
        conf_item.setForeground(QColor("#22c55e"))
        conf_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.table_actions.setItem(row, 2, conf_item)

        self.table_actions.setItem(row, 3, QTableWidgetItem(detail))

    def reset_analysis(self) -> None:
        self.progress_bar.setValue(0)
        self.table_actions.setRowCount(0)
        self.lbl_realtime_fps.setText("0.0 FPS")
        self.lbl_avg_conf.setText("0.0%")
        self.lbl_valid_pose.setText("0.0 FPS")
        self.lbl_valid_track.setText("0")
