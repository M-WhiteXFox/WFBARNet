from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction, QColor
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMenu,
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


class MainWindow(QMainWindow):
    """视图层：负责布局、控件实例化和基础状态展示。"""

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
        preview_title = QLabel("视频预览")
        preview_title.setObjectName("sectionTitle")
        preview_header.addWidget(preview_title)
        preview_header.addStretch(1)

        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("topProgress")
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFixedHeight(16)

        self.video_player = VideoPlayerWidget()
        self.video_player.setMinimumHeight(360)

        video_controls = QFrame()
        video_controls.setObjectName("videoControlsBar")
        controls_layout = QHBoxLayout(video_controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(8)
        controls_layout.addWidget(self.video_player.btn_select_video)
        controls_layout.addWidget(self.video_player.path_edit, stretch=1)
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

        preview_layout.addLayout(preview_header)
        preview_layout.addWidget(self.progress_bar)
        preview_layout.addWidget(video_controls, 0)
        preview_layout.addWidget(self.video_player, 1)
        preview_layout.addWidget(timeline_bar, 0)

        body_layout.addWidget(preview_panel, stretch=6)

    def _build_analytics_panel(self, body_layout: QHBoxLayout) -> None:
        analytics_panel = QFrame()
        analytics_panel.setObjectName("analyticsCard")
        analytics_layout = QVBoxLayout(analytics_panel)
        analytics_layout.setContentsMargins(18, 18, 18, 18)
        analytics_layout.setSpacing(14)

        metrics_grid = QGridLayout()
        metrics_grid.setHorizontalSpacing(12)
        metrics_grid.setVerticalSpacing(12)

        card1, self.lbl_total_actions = self._create_metric_card("总识别动作数", "0")
        card2, self.lbl_avg_conf = self._create_metric_card("平均置信度", "0.0%")
        card3, self.lbl_valid_pose = self._create_metric_card("有效姿态帧数", "0")
        card4, self.lbl_valid_track = self._create_metric_card("有效轨迹帧数", "0")

        metrics_grid.addWidget(card1, 0, 0)
        metrics_grid.addWidget(card2, 0, 1)
        metrics_grid.addWidget(card3, 1, 0)
        metrics_grid.addWidget(card4, 1, 1)

        self.tabs = QTabWidget()
        self.tabs.setObjectName("mainTabs")

        self._build_overview_tab()
        self._build_pose_tab()
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
        self.progress_bar.setValue(val)

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
        self.lbl_total_actions.setText("0")
        self.lbl_avg_conf.setText("0.0%")
        self.lbl_valid_pose.setText("0")
        self.lbl_valid_track.setText("0")
