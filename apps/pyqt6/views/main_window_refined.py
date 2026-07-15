from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PyQt6.QtCore import QEasingCurve, QPoint, QPointF, QPropertyAnimation, QRect, QRectF, QSize, Qt, QUrl, pyqtProperty, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QDesktopServices, QLinearGradient, QPainter, QPalette, QPen, QTextCursor
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QFileDialog,
    QGridLayout,
    QGraphicsOpacityEffect,
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
    QSplitter,
    QTextBrowser,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

try:
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    from PyQt6.QtWebEngineCore import QWebEngineSettings
    WEBENGINE_IMPORT_ERROR = ""
except Exception as exc:  # Optional dependency; the tab keeps working with QTextBrowser.
    QWebEngineView = None
    QWebEngineSettings = None
    WEBENGINE_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

from apps.pyqt6.views.components.video_player_panel_runtime import (
    VideoPlayerWidget,
    VideoTimelineWidget,
)
from apps.pyqt6.views.heatmap_renderer import HeatmapRenderer, HeatmapRenderConfig
from src.utils.report_generation import copy_tabler_report_assets


class ToggleSwitch(QCheckBox):
    def __init__(self, tooltip: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._offset = 1.0
        self.setObjectName("modelSwitch")
        self.setFixedSize(46, 24)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(tooltip)
        self.setAccessibleName(tooltip)
        self.setAccessibleDescription(tooltip)
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
        palette = self.palette()
        off_color = palette.color(QPalette.ColorRole.Mid)
        on_color = palette.color(QPalette.ColorRole.Highlight)
        track_color = QColor(
            int(off_color.red() + (on_color.red() - off_color.red()) * self._offset),
            int(off_color.green() + (on_color.green() - off_color.green()) * self._offset),
            int(off_color.blue() + (on_color.blue() - off_color.blue()) * self._offset),
        )
        if not self.isEnabled():
            track_color = palette.color(QPalette.ColorRole.Midlight)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(track_color)
        painter.drawRoundedRect(track_rect, 10, 10)

        knob_size = 18
        knob_x = 3 + (22 * self._offset)
        knob_rect = QRectF(knob_x, 3, knob_size, knob_size)
        painter.setBrush(palette.color(QPalette.ColorRole.HighlightedText))
        painter.drawEllipse(knob_rect)


class StrokeBarChartWidget(QWidget):
    COLORS = (
        "#2563EB",
        "#DC2626",
        "#16A34A",
        "#F59E0B",
        "#7C3AED",
        "#0891B2",
        "#DB2777",
        "#65A30D",
        "#EA580C",
        "#4F46E5",
        "#0D9488",
        "#BE123C",
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._counts: dict[str, int] = {}
        self._colors_by_label: dict[str, QColor] = {}
        self.setObjectName("strokeBarChart")
        self.setAccessibleName("击球类型统计图")
        self.setMinimumSize(320, 260)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_counts(self, counts: dict[str, int]) -> None:
        cleaned: dict[str, int] = {}
        for label, count in counts.items():
            name = str(label).strip() or "未知"
            value = max(0, int(count))
            if value > 0:
                cleaned[name] = cleaned.get(name, 0) + value
                self._ensure_color(name)
        self._counts = cleaned
        self._update_accessible_summary()
        self.update()

    def increment(self, label: str, amount: int = 1) -> None:
        name = str(label).strip() or "未知"
        value = max(0, int(amount))
        if value <= 0:
            return
        self._ensure_color(name)
        self._counts[name] = self._counts.get(name, 0) + value
        self._update_accessible_summary()
        self.update()

    def clear_counts(self) -> None:
        self._counts.clear()
        self._update_accessible_summary()
        self.update()

    def total_count(self) -> int:
        return sum(self._counts.values())

    def counts(self) -> dict[str, int]:
        return dict(self._counts)

    def paintEvent(self, event) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        bounds = QRectF(self.rect()).adjusted(14, 14, -14, -14)
        total = self.total_count()
        if total <= 0:
            self._draw_empty_state(painter, bounds)
            return

        items = sorted(self._counts.items(), key=lambda item: (-item[1], item[0]))
        self._draw_bars(painter, bounds, items, total)

    def _draw_empty_state(self, painter: QPainter, bounds: QRectF) -> None:
        palette = self.palette()
        painter.setPen(palette.color(QPalette.ColorRole.PlaceholderText))
        painter.drawText(bounds, Qt.AlignmentFlag.AlignCenter, "暂无击球统计")

    def _draw_bars(
        self,
        painter: QPainter,
        bounds: QRectF,
        items: list[tuple[str, int]],
        total: int,
    ) -> None:
        palette = self.palette()
        metrics = painter.fontMetrics()
        text_color = palette.color(QPalette.ColorRole.Text)
        muted_color = palette.color(QPalette.ColorRole.PlaceholderText)
        track_color = palette.color(QPalette.ColorRole.Midlight)
        label_width = min(124.0, max(84.0, bounds.width() * 0.24))
        value_width = min(132.0, max(96.0, bounds.width() * 0.24))
        bar_left = bounds.left() + label_width + 12.0
        bar_width = max(48.0, bounds.width() - label_width - value_width - 24.0)
        max_count = max(count for _, count in items)
        row_height = min(42.0, max(28.0, bounds.height() / max(1, len(items))))
        content_height = row_height * len(items)
        y_start = bounds.top() + max(0.0, (bounds.height() - content_height) / 2.0)
        for index, (label, count) in enumerate(items):
            y = y_start + index * row_height
            if y + row_height > bounds.bottom() + 1:
                break
            label_text = metrics.elidedText(label, Qt.TextElideMode.ElideRight, int(label_width))
            painter.setPen(text_color)
            painter.drawText(
                QRectF(bounds.left(), y, label_width, row_height),
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                label_text,
            )
            track = QRectF(bar_left, y + row_height * 0.31, bar_width, max(8.0, row_height * 0.38))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(track_color)
            painter.drawRoundedRect(track, 4, 4)
            fill = QRectF(track)
            fill.setWidth(max(4.0, track.width() * count / max_count))
            painter.setBrush(self._ensure_color(label))
            painter.drawRoundedRect(fill, 4, 4)
            percent = count / max(1, total) * 100.0
            painter.setPen(text_color if count else muted_color)
            painter.drawText(
                QRectF(bar_left + bar_width + 12.0, y, value_width - 12.0, row_height),
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
                f"{count} 次  {percent:.1f}%",
            )

    def _update_accessible_summary(self) -> None:
        if not self._counts:
            self.setAccessibleDescription("暂无击球统计")
            return
        items = sorted(self._counts.items(), key=lambda item: (-item[1], item[0]))
        summary = "，".join(f"{label} {count} 次" for label, count in items)
        self.setAccessibleDescription(f"总击球 {self.total_count()} 次：{summary}")

    def _ensure_color(self, label: str) -> QColor:
        if label not in self._colors_by_label:
            color = QColor(self.COLORS[len(self._colors_by_label) % len(self.COLORS)])
            self._colors_by_label[label] = color
        return self._colors_by_label[label]


StrokePieChartWidget = StrokeBarChartWidget


class CourtHeatmapWidget(QWidget):
    COURT_LENGTH_MM = 13400.0
    COURT_WIDTH_MM = 6100.0
    COURT_LENGTH_CM = 1340.0
    COURT_WIDTH_CM = 610.0
    LINE_WIDTH_MM = 40.0
    SINGLES_SIDE_MARGIN_MM = 460.0
    DOUBLE_LONG_SERVICE_FROM_BACK_MM = 760.0
    SHORT_SERVICE_FROM_NET_MM = 1980.0
    HEATMAP_MAX_POINTS = 1600
    HEATMAP_REFRESH_INTERVAL = 12

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._ball_court_xy: tuple[float, float] | None = None
        self._player_court_points: list[tuple[float, float]] = []
        self._top_player_points: list[tuple[float, float]] = []
        self._bottom_player_points: list[tuple[float, float]] = []
        self._show_top_heatmap = True
        self._show_bottom_heatmap = True
        self._show_contours = True
        self._heatmap_opacity = 0.82
        self._top_color_mode = "blue"
        self._bottom_color_mode = "red"
        self._heatmap_config = HeatmapRenderConfig(
            sigma=24.0,
            alpha_power=0.62,
            min_alpha_threshold=0.02,
            max_alpha=205,
            normalization_percentile=99.0,
            temporal_decay_floor=0.50,
            color_gamma=0.70,
            contour_levels=(0.22, 0.36, 0.50, 0.64, 0.78),
            contour_alpha=72,
            contour_thickness=1,
            heatmap_opacity=self._heatmap_opacity,
            top_color_mode=self._top_color_mode,
            bottom_color_mode=self._bottom_color_mode,
            show_contours=self._show_contours,
        )
        self._heatmap_renderer = HeatmapRenderer(
            1,
            1,
            court_width=self.COURT_WIDTH_CM,
            court_height=self.COURT_LENGTH_CM,
            config=self._heatmap_config,
        )
        self._top_heatmap_pixmap = None
        self._bottom_heatmap_pixmap = None
        self._court_pixmap = None
        self._heatmap_update_count = 0
        self.setObjectName("badmintonCourtPreview")
        self.setMinimumSize(320, 220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def sizeHint(self) -> QSize:
        return QSize(320, 460)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._top_player_points or self._bottom_player_points:
            self.refresh_heatmap()

    def set_ball_projection(self, court_xy: tuple[float, float] | None) -> None:
        self._ball_court_xy = self._clean_ball_projection(court_xy)
        self.update()

    def set_player_projections(self, court_points: list[tuple[float, float]] | None) -> None:
        self._player_court_points = list(court_points or [])
        self._append_player_heatmap_points(self._player_court_points)
        self._heatmap_update_count += 1
        if self._should_refresh_heatmap():
            self.refresh_heatmap()
            self._heatmap_update_count = 0
        self.update()

    def set_court_pixmap(self, pixmap) -> None:
        self._court_pixmap = pixmap
        self.update()

    def set_top_player_points(self, points: list[tuple[float, float]] | None) -> None:
        self._top_player_points = self._filtered_court_points(points)
        self._trim_heatmap_points()
        self.refresh_heatmap()

    def set_bottom_player_points(self, points: list[tuple[float, float]] | None) -> None:
        self._bottom_player_points = self._filtered_court_points(points)
        self._trim_heatmap_points()
        self.refresh_heatmap()

    def set_show_top_heatmap(self, enabled: bool) -> None:
        self._show_top_heatmap = bool(enabled)
        self.update()

    def set_show_bottom_heatmap(self, enabled: bool) -> None:
        self._show_bottom_heatmap = bool(enabled)
        self.update()

    def set_show_contours(self, enabled: bool) -> None:
        self._show_contours = bool(enabled)
        self._heatmap_config.show_contours = self._show_contours
        self.refresh_heatmap()

    def set_heatmap_opacity(self, value: float) -> None:
        self._heatmap_opacity = max(0.0, min(float(value), 1.0))
        self._heatmap_config.heatmap_opacity = self._heatmap_opacity
        self.refresh_heatmap()

    def set_heatmap_parameters(self, **params: float | int | str | bool) -> None:
        for key, value in params.items():
            if not hasattr(self._heatmap_config, key):
                raise ValueError(f"Unsupported heatmap parameter: {key}")
            setattr(self._heatmap_config, key, value)
        self._top_color_mode = str(self._heatmap_config.top_color_mode)
        self._bottom_color_mode = str(self._heatmap_config.bottom_color_mode)
        self._show_contours = bool(self._heatmap_config.show_contours)
        self._heatmap_opacity = float(self._heatmap_config.heatmap_opacity)
        self.refresh_heatmap()

    def clear_player_heatmap(self) -> None:
        self._top_player_points.clear()
        self._bottom_player_points.clear()
        self._top_heatmap_pixmap = None
        self._bottom_heatmap_pixmap = None
        self._heatmap_update_count = 0
        self.update()

    def clear_heatmap(self) -> None:
        self.clear_player_heatmap()

    def refresh_heatmap(self) -> None:
        court = self._court_rect(self.rect().adjusted(8, 8, -8, -8))
        width = max(1, int(round(court.width())))
        height = max(1, int(round(court.height())))
        self._heatmap_renderer.set_size(width, height)
        self._top_heatmap_pixmap = self._heatmap_renderer.build_heatmap_pixmap(
            self._top_player_points,
            color_mode=self._top_color_mode,
            show_contours=self._show_contours,
            opacity=self._heatmap_opacity,
        )
        self._bottom_heatmap_pixmap = self._heatmap_renderer.build_heatmap_pixmap(
            self._bottom_player_points,
            color_mode=self._bottom_color_mode,
            show_contours=self._show_contours,
            opacity=self._heatmap_opacity,
        )
        self.update()

    def paintEvent(self, event) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        bounds = self.rect().adjusted(8, 8, -8, -8)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#0F5F3A"))
        painter.drawRoundedRect(QRectF(bounds), 8, 8)

        court = self._court_rect(bounds)
        line_width = max(2, round(self.LINE_WIDTH_MM * self._scale(court)))
        painter.setPen(QPen(QColor("#F8FAFC"), line_width))
        if self._court_pixmap is not None and not self._court_pixmap.isNull():
            painter.drawPixmap(court.toRect(), self._court_pixmap)
        else:
            painter.setBrush(QColor("#15803D"))
            painter.drawRect(court)

        self._draw_heatmap_layers(painter, court)
        painter.setPen(QPen(QColor("#FFFFFF"), line_width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.SquareCap))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        self._draw_court_lines(painter, court)
        self._draw_heatmap_legend(painter, bounds, court)
        self._draw_player_projections(painter, court)
        self._draw_ball_projection(painter, court)

    def _court_rect(self, bounds) -> QRectF:
        target_ratio = self.COURT_WIDTH_MM / self.COURT_LENGTH_MM
        width = float(bounds.width())
        height = width / target_ratio
        if height > bounds.height():
            height = float(bounds.height())
            width = height * target_ratio

        x = bounds.x() + (bounds.width() - width) / 2.0
        y = bounds.y() + (bounds.height() - height) / 2.0
        return QRectF(x, y, width, height)

    def _draw_court_lines(self, painter: QPainter, court: QRectF) -> None:
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(court)

        y_net = self._y(court, self.COURT_LENGTH_MM / 2.0)
        y_short_top = self._y(court, self.COURT_LENGTH_MM / 2.0 - self.SHORT_SERVICE_FROM_NET_MM)
        y_short_bottom = self._y(court, self.COURT_LENGTH_MM / 2.0 + self.SHORT_SERVICE_FROM_NET_MM)
        y_double_long_top = self._y(court, self.DOUBLE_LONG_SERVICE_FROM_BACK_MM)
        y_double_long_bottom = self._y(court, self.COURT_LENGTH_MM - self.DOUBLE_LONG_SERVICE_FROM_BACK_MM)

        x_single_left = self._x(court, self.SINGLES_SIDE_MARGIN_MM)
        x_single_right = self._x(court, self.COURT_WIDTH_MM - self.SINGLES_SIDE_MARGIN_MM)
        x_center = self._x(court, self.COURT_WIDTH_MM / 2.0)

        self._draw_line(painter, court.left(), y_net, court.right(), y_net)
        self._draw_line(painter, court.left(), y_double_long_top, court.right(), y_double_long_top)
        self._draw_line(painter, court.left(), y_double_long_bottom, court.right(), y_double_long_bottom)
        self._draw_line(painter, court.left(), y_short_top, court.right(), y_short_top)
        self._draw_line(painter, court.left(), y_short_bottom, court.right(), y_short_bottom)

        self._draw_line(painter, x_single_left, court.top(), x_single_left, court.bottom())
        self._draw_line(painter, x_single_right, court.top(), x_single_right, court.bottom())
        self._draw_line(painter, x_center, court.top(), x_center, y_short_top)
        self._draw_line(painter, x_center, y_short_bottom, x_center, court.bottom())

    def _draw_line(self, painter: QPainter, x1: float, y1: float, x2: float, y2: float) -> None:
        painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))

    def _append_player_heatmap_points(self, court_points: list[tuple[float, float]]) -> None:
        filtered = self._filtered_court_points(court_points)
        if filtered:
            self._top_player_points.append(filtered[0])
        if len(filtered) > 1:
            self._bottom_player_points.append(filtered[1])
        self._trim_heatmap_points()

    def _filtered_court_points(self, court_points: list[tuple[float, float]] | None) -> list[tuple[float, float]]:
        filtered: list[tuple[float, float]] = []
        for point in court_points or []:
            if point is None:
                continue
            try:
                if len(point) < 2:
                    continue
                court_x, court_y = float(point[0]), float(point[1])
            except (TypeError, ValueError, IndexError):
                continue
            if court_x != court_x or court_y != court_y:
                continue
            court_x = min(self.COURT_WIDTH_CM, max(0.0, court_x))
            court_y = min(self.COURT_LENGTH_CM, max(0.0, court_y))
            filtered.append((court_x, court_y))
        return filtered

    def _trim_heatmap_points(self) -> None:
        overflow = len(self._top_player_points) - self.HEATMAP_MAX_POINTS
        if overflow > 0:
            del self._top_player_points[:overflow]
        overflow = len(self._bottom_player_points) - self.HEATMAP_MAX_POINTS
        if overflow > 0:
            del self._bottom_player_points[:overflow]

    def _should_refresh_heatmap(self) -> bool:
        if self._top_heatmap_pixmap is None and self._top_player_points:
            return True
        if self._bottom_heatmap_pixmap is None and self._bottom_player_points:
            return True
        return self._heatmap_update_count >= self.HEATMAP_REFRESH_INTERVAL

    def _draw_heatmap_layers(self, painter: QPainter, court: QRectF) -> None:
        if self._show_top_heatmap and self._top_heatmap_pixmap is not None and not self._top_heatmap_pixmap.isNull():
            painter.drawPixmap(court.toRect(), self._top_heatmap_pixmap)
        if self._show_bottom_heatmap and self._bottom_heatmap_pixmap is not None and not self._bottom_heatmap_pixmap.isNull():
            painter.drawPixmap(court.toRect(), self._bottom_heatmap_pixmap)

    def _draw_heatmap_legend(self, painter: QPainter, bounds: QRectF, court: QRectF) -> None:
        if not self._top_player_points and not self._bottom_player_points:
            return
        legend_width = min(168.0, max(118.0, bounds.width() * 0.28))
        legend_height = 58.0
        x = min(bounds.right() - legend_width - 10.0, court.right() - legend_width - 8.0)
        y = court.top() + 10.0
        legend = QRectF(x, y, legend_width, legend_height)

        painter.save()
        palette = self.palette()
        border_color = QColor(palette.color(QPalette.ColorRole.Mid))
        border_color.setAlpha(150)
        surface_color = QColor(palette.color(QPalette.ColorRole.Base))
        surface_color.setAlpha(228)
        painter.setPen(QPen(border_color, 1))
        painter.setBrush(surface_color)
        painter.drawRoundedRect(legend, 7, 7)

        bar = QRectF(legend.left() + 12.0, legend.top() + 12.0, legend.width() - 24.0, 8.0)
        gradient = QLinearGradient(bar.left(), bar.top(), bar.right(), bar.top())
        gradient.setColorAt(0.0, QColor("#7DD3FC"))
        gradient.setColorAt(0.34, QColor("#2563EB"))
        gradient.setColorAt(0.66, QColor("#F97316"))
        gradient.setColorAt(1.0, QColor("#991B1B"))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(gradient)
        painter.drawRoundedRect(bar, 4, 4)

        painter.setPen(QPen(palette.color(QPalette.ColorRole.Text), 1))
        font = painter.font()
        font.setPointSize(max(7, font.pointSize() - 1))
        painter.setFont(font)
        top_text = f"上方 {len(self._top_player_points)}"
        bottom_text = f"下方 {len(self._bottom_player_points)}"
        painter.drawText(QRectF(legend.left() + 12.0, legend.top() + 26.0, legend.width() - 24.0, 14.0), Qt.AlignmentFlag.AlignLeft, top_text)
        painter.drawText(QRectF(legend.left() + 12.0, legend.top() + 40.0, legend.width() - 24.0, 14.0), Qt.AlignmentFlag.AlignLeft, bottom_text)
        painter.setPen(QPen(QColor("#0EA5E9"), 3))
        painter.drawLine(QPointF(legend.right() - 44.0, legend.top() + 33.0), QPointF(legend.right() - 22.0, legend.top() + 33.0))
        painter.setPen(QPen(QColor("#DC2626"), 3))
        painter.drawLine(QPointF(legend.right() - 44.0, legend.top() + 47.0), QPointF(legend.right() - 22.0, legend.top() + 47.0))
        painter.restore()

    def _draw_ball_projection(self, painter: QPainter, court: QRectF) -> None:
        if self._ball_court_xy is None:
            return

        court_x, court_y = self._ball_court_xy
        marker_x = court.left() + court.width() * (court_x / self.COURT_WIDTH_CM)
        marker_y = court.top() + court.height() * (court_y / self.COURT_LENGTH_CM)
        radius = max(6.0, min(court.width(), court.height()) * 0.022)
        center = QPointF(marker_x, marker_y)

        painter.save()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(249, 115, 22, 70))
        painter.drawEllipse(center, radius * 2.15, radius * 2.15)

        painter.setPen(QPen(QColor("#7C2D12"), max(2, int(radius * 0.36))))
        painter.setBrush(QColor("#FDBA74"))
        painter.drawEllipse(center, radius, radius)
        painter.setPen(QPen(QColor("#FFFFFF"), max(1, int(radius * 0.18))))
        painter.setBrush(QColor("#F97316"))
        painter.drawEllipse(center, radius * 0.62, radius * 0.62)

        cross_len = radius * 1.85
        painter.setPen(QPen(QColor("#FFFFFF"), max(1, int(radius * 0.18))))
        painter.drawLine(QPointF(marker_x - cross_len, marker_y), QPointF(marker_x + cross_len, marker_y))
        painter.drawLine(QPointF(marker_x, marker_y - cross_len), QPointF(marker_x, marker_y + cross_len))

        label = "球"
        font = painter.font()
        font.setPointSize(max(8, font.pointSize() - 1))
        font.setBold(True)
        painter.setFont(font)
        label_width = 26.0
        label_height = 18.0
        label_x = min(court.right() - label_width, max(court.left(), marker_x + radius + 5.0))
        label_y = min(court.bottom() - label_height, max(court.top(), marker_y - label_height * 0.5))
        label_rect = QRectF(label_x, label_y, label_width, label_height)
        painter.setPen(QPen(QColor(124, 45, 18, 185), 1))
        painter.setBrush(QColor(255, 247, 237, 228))
        painter.drawRoundedRect(label_rect, 5, 5)
        painter.setPen(QPen(QColor("#7C2D12"), 1))
        painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, label)
        painter.restore()

    def _draw_player_projections(self, painter: QPainter, court: QRectF) -> None:
        colors = (QColor("#38BDF8"), QColor("#FACC15"), QColor("#A78BFA"), QColor("#34D399"))
        radius = max(6.0, min(court.width(), court.height()) * 0.022)
        for index, (court_x, court_y) in enumerate(self._player_court_points):
            if court_x < 0.0 or court_x > self.COURT_WIDTH_CM or court_y < 0.0 or court_y > self.COURT_LENGTH_CM:
                continue
            marker_x = court.left() + court.width() * (court_x / self.COURT_WIDTH_CM)
            marker_y = court.top() + court.height() * (court_y / self.COURT_LENGTH_CM)
            color = colors[index % len(colors)]

            painter.setPen(QPen(QColor("#0F172A"), max(2, int(radius * 0.35))))
            painter.setBrush(color)
            painter.drawEllipse(QPointF(marker_x, marker_y), radius, radius)

            label_rect = QRectF(marker_x - radius, marker_y - radius, radius * 2.0, radius * 2.0)
            painter.setPen(QPen(QColor("#0F172A"), 1))
            painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, str(index + 1))

    def _clean_ball_projection(self, court_xy: object | None) -> tuple[float, float] | None:
        if not isinstance(court_xy, (list, tuple)) or len(court_xy) < 2:
            return None
        try:
            court_x = float(court_xy[0])
            court_y = float(court_xy[1])
        except (TypeError, ValueError):
            return None
        if court_x != court_x or court_y != court_y:
            return None
        court_x = min(self.COURT_WIDTH_CM, max(0.0, court_x))
        court_y = min(self.COURT_LENGTH_CM, max(0.0, court_y))
        return court_x, court_y

    def _scale(self, court: QRectF) -> float:
        return min(court.width() / self.COURT_WIDTH_MM, court.height() / self.COURT_LENGTH_MM)

    def _x(self, court: QRectF, mm: float) -> float:
        return court.left() + court.width() * (mm / self.COURT_WIDTH_MM)

    def _y(self, court: QRectF, mm: float) -> float:
        return court.top() + court.height() * (mm / self.COURT_LENGTH_MM)


BadmintonCourtWidget = CourtHeatmapWidget


class ReportPreviewWidget(QWidget):
    """Right-side report tab backed by the stable report template."""

    def __init__(self, project_root: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._project_root = project_root
        self._report_output_path = project_root / "assets" / "dist" / "index.html"
        self._template_path = project_root / "assets" / "report_template" / "report-template.html"
        self._current_report_path: Path | None = None
        self._theme_name = "office_light"

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        header = QHBoxLayout()
        header.setSpacing(8)
        title = QLabel("训练报告")
        title.setObjectName("sectionTitle")
        self.report_path_label = QLabel("")
        self.report_path_label.setObjectName("sectionNote")
        self.report_path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.backend_label = QLabel("")
        self.backend_label.setObjectName("sectionNote")

        self.btn_refresh_report = QPushButton("重新载入")
        self.btn_refresh_report.setObjectName("btnRefreshReport")
        self.btn_open_report = QPushButton("在浏览器中打开")
        self.btn_open_report.setObjectName("btnOpenReport")
        self.btn_refresh_report.clicked.connect(self.reload)
        self.btn_open_report.clicked.connect(self.open_external)

        header.addWidget(title)
        header.addWidget(self.report_path_label, stretch=1)
        header.addWidget(self.backend_label)
        header.addWidget(self.btn_refresh_report)
        header.addWidget(self.btn_open_report)
        layout.addLayout(header)

        if QWebEngineView is not None:
            self.viewer = QWebEngineView(self)
            self.viewer.setObjectName("reportWebView")
            self._configure_web_engine()
            self.viewer.loadFinished.connect(self._on_load_finished)
            self._uses_web_engine = True
        else:
            self.viewer = QTextBrowser(self)
            self.viewer.setObjectName("reportTextBrowser")
            self.viewer.setOpenExternalLinks(True)
            self._uses_web_engine = False
            self.backend_label.setText("简化预览")
            self.backend_label.setToolTip(
                WEBENGINE_IMPORT_ERROR or "未检测到 PyQt6.QtWebEngineWidgets，已降级为 QTextBrowser。"
            )

        if self._uses_web_engine:
            self.backend_label.setText("WebEngine")
            self.backend_label.setToolTip("当前使用 QWebEngineView 渲染 Tabler 报告。")
        self.backend_label.setVisible(False)

        layout.addWidget(self.viewer, stretch=1)
        self.load_template()

    def set_theme(self, theme_name: str) -> None:
        self._theme_name = str(theme_name or "office_light")
        is_dark = self._theme_name.endswith("_dark")
        if self._uses_web_engine:
            force_dark = getattr(QWebEngineSettings.WebAttribute, "ForceDarkModeEnabled", None)
            if force_dark is not None:
                self.viewer.settings().setAttribute(force_dark, is_dark)
            self.viewer.setStyleSheet(
                "background:#1D2129;" if is_dark else "background:#FFFFFF;"
            )
            self._apply_fallback_web_theme()
        else:
            self.viewer.setStyleSheet(
                "background:#1D2129;color:#E8EDF5;" if is_dark else ""
            )

    def _on_load_finished(self, _ok: bool) -> None:
        self._apply_fallback_web_theme()

    def _apply_fallback_web_theme(self) -> None:
        if not self._uses_web_engine:
            return
        is_dark = self._theme_name.endswith("_dark")
        if not is_dark:
            script = "(function(){const s=document.getElementById('wfbar-theme-override');if(s)s.remove();})();"
        else:
            css = (
                "html{color-scheme:dark}body,.page-wrapper{background:#1D2129!important;"
                "color:#E8EDF5!important}.card,.navbar,.page-header{background:#242933!important;"
                "border-color:#3A4050!important}.text-secondary,.text-muted{color:#A9B3C3!important}"
            )
            script = (
                "(function(){let s=document.getElementById('wfbar-theme-override');"
                "if(!s){s=document.createElement('style');s.id='wfbar-theme-override';"
                "document.head.appendChild(s);}s.textContent=" + repr(css) + ";"
                "})();"
            )
        self.viewer.page().runJavaScript(script)

    def load_template(self) -> None:
        path = self._report_output_path if self._report_output_path.exists() else self._template_path
        self.load_report_file(path)

    def load_report_file(self, path: Path | str) -> None:
        report_path = Path(path)
        self._current_report_path = report_path if report_path.exists() else None
        self._update_path_label(report_path)
        if not report_path.exists():
            self._set_fallback_html(f"报告模板不存在：{report_path}")
            return
        self._ensure_report_assets(report_path)

        if self._uses_web_engine:
            self.viewer.load(QUrl.fromLocalFile(str(report_path.resolve())))
        else:
            self.viewer.setHtml(self._fallback_preview_html(report_path))

    def load_report_html(self, html: str, base_path: Path | str | None = None) -> None:
        base = Path(base_path) if base_path else self._template_path.parent
        self._current_report_path = None
        self.report_path_label.setText("当前报告：内存预览")
        if self._uses_web_engine:
            self.viewer.setHtml(html, self._directory_url(base))
        else:
            self.viewer.setHtml(html)

    def reload(self) -> None:
        if self._current_report_path is not None:
            self.load_report_file(self._current_report_path)
        else:
            self.load_template()

    def open_external(self) -> None:
        if self._current_report_path is None or not self._current_report_path.exists():
            return
        self._ensure_report_assets(self._current_report_path)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._current_report_path.resolve())))

    def _configure_web_engine(self) -> None:
        if QWebEngineSettings is None:
            return
        settings = self.viewer.settings()
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalStorageEnabled, True)

    def _ensure_report_assets(self, report_path: Path) -> None:
        copy_tabler_report_assets(report_path.parent)

    @staticmethod
    def _directory_url(path: Path) -> QUrl:
        directory = path if path.is_dir() else path.parent
        value = str(directory.resolve()).replace("\\", "/")
        if not value.endswith("/"):
            value += "/"
        return QUrl.fromLocalFile(value)

    def _update_path_label(self, path: Path) -> None:
        if path == self._report_output_path:
            self.report_path_label.setText("最新训练报告")
        elif path == self._template_path:
            self.report_path_label.setText("报告模板预览")
        else:
            self.report_path_label.setText(f"当前报告：{path.name}")
        self.report_path_label.setToolTip(str(path))

    def _set_fallback_html(self, message: str) -> None:
        escaped = (
            message.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        self.viewer.setHtml(f"<h3>报告预览不可用</h3><p>{escaped}</p>")

    def _fallback_preview_html(self, report_path: Path) -> str:
        body = report_path.read_text(encoding="utf-8")
        reason = WEBENGINE_IMPORT_ERROR or "当前运行环境未启用 PyQt6-WebEngine。"
        escaped_reason = (
            reason.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        return (
            "<div style='padding:10px;margin-bottom:12px;border:1px solid #f59f00;"
            "background:#fff7e6;color:#5c3b00;font-family:Arial,sans-serif;'>"
            "<strong>当前为简化 HTML 预览，未使用浏览器内核渲染。</strong>"
            f"<br>原因：{escaped_reason}"
            "<br>请使用 WFBARNet 环境从源码重启应用，或重新打包包含 PyQt6-WebEngine 的程序。"
            "</div>"
            f"{body}"
        )


class MainWindow(QMainWindow):
    """视图层：负责布局、控件实例化和基础状态展示。"""

    poseModelBrowseRequested = pyqtSignal()
    trackModelBrowseRequested = pyqtSignal()
    modelSettingsApplyRequested = pyqtSignal(str, str)
    modelSettingsDefaultsRequested = pyqtSignal()
    reportApiSettingsApplyRequested = pyqtSignal(dict)
    modelSwitchesChanged = pyqtSignal(bool, bool)
    debugCsvChanged = pyqtSignal(bool)
    manualCourtCalibrationRequested = pyqtSignal()
    batchFolderBrowseRequested = pyqtSignal()
    batchRallySelectionChanged = pyqtSignal(str)
    batchExportRequested = pyqtSignal()
    reportExportRequested = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("羽毛球动作识别分析平台")
        self.resize(1360, 860)
        self.setMinimumSize(1024, 680)
        self._video_fullscreen = False
        self._video_fullscreen_previous_state = Qt.WindowState.WindowNoState
        self._normal_root_margins = (16, 16, 16, 16)
        self._analytics_opacity_effect: QGraphicsOpacityEffect | None = None
        self._analytics_panel_overlay = False

        self.central_widget = QWidget()
        self.central_widget.setObjectName("appRoot")
        self.setCentralWidget(self.central_widget)

        self.root_layout = QVBoxLayout(self.central_widget)
        self.root_layout.setContentsMargins(*self._normal_root_margins)
        self.root_layout.setSpacing(12)

        self._build_header()
        self._build_status_banner()
        self._build_body()

    def _build_header(self) -> None:
        self.header_card = QFrame()
        self.header_card.setObjectName("headerCard")
        header_layout = QHBoxLayout(self.header_card)
        header_layout.setContentsMargins(18, 12, 18, 12)
        header_layout.setSpacing(16)

        brand_col = QVBoxLayout()
        brand_col.setSpacing(4)

        self.title_label = QLabel("WFBARNet 羽毛球分析")
        self.title_label.setObjectName("titleLabel")

        brand_col.addWidget(self.title_label)

        self.status_label = QLabel("系统状态：待机中")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setProperty("state", "idle")
        self.status_label.setAccessibleName("系统状态")
        brand_col.addWidget(self.status_label)

        actions_col = QVBoxLayout()
        actions_col.setSpacing(8)

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

        header_layout.addLayout(brand_col, stretch=1)
        header_layout.addLayout(actions_col, stretch=0)
        self.root_layout.addWidget(self.header_card)

    def _build_status_banner(self) -> None:
        self.status_banner = QFrame()
        self.status_banner.setObjectName("statusBanner")
        self.status_banner.setProperty("state", "info")
        self.status_banner.setVisible(False)
        banner_layout = QHBoxLayout(self.status_banner)
        banner_layout.setContentsMargins(14, 8, 10, 8)
        banner_layout.setSpacing(10)

        self.status_banner_label = QLabel("")
        self.status_banner_label.setObjectName("statusBannerLabel")
        self.status_banner_label.setWordWrap(True)
        self.status_banner_label.setAccessibleName("操作提示")
        self.btn_notice_details = QPushButton("查看日志")
        self.btn_notice_details.setObjectName("btnNoticeDetails")
        self.btn_notice_details.clicked.connect(self.show_log_tab)
        self.btn_notice_dismiss = QPushButton("关闭")
        self.btn_notice_dismiss.setObjectName("btnNoticeDismiss")
        self.btn_notice_dismiss.clicked.connect(self.clear_status_notice)

        banner_layout.addWidget(self.status_banner_label, stretch=1)
        banner_layout.addWidget(self.btn_notice_details)
        banner_layout.addWidget(self.btn_notice_dismiss)
        self.root_layout.addWidget(self.status_banner)

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
        self.body_layout = QHBoxLayout()
        body_layout = self.body_layout
        body_layout.setSpacing(0)

        self.body_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.body_splitter.setObjectName("bodySplitter")
        self.body_splitter.setChildrenCollapsible(False)
        self.body_splitter.setHandleWidth(6)

        self._build_preview_panel(self.body_splitter)
        self._build_analytics_panel(self.body_splitter)
        self.body_splitter.setStretchFactor(0, 6)
        self.body_splitter.setStretchFactor(1, 5)
        self.body_splitter.setSizes([690, 570])

        body_layout.addWidget(self.body_splitter)
        self.root_layout.addLayout(body_layout, stretch=1)

    def _build_preview_panel(self, body_splitter: QSplitter) -> None:
        self.preview_shell = QWidget()
        preview_shell = self.preview_shell
        preview_shell.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        preview_shell_layout = QVBoxLayout(preview_shell)
        preview_shell_layout.setContentsMargins(0, 0, 0, 0)
        preview_shell_layout.setSpacing(0)

        self.preview_panel = QFrame()
        preview_panel = self.preview_panel
        preview_panel.setObjectName("previewCard")
        preview_panel.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        preview_layout = QVBoxLayout(preview_panel)
        preview_layout.setContentsMargins(18, 18, 18, 18)
        preview_layout.setSpacing(14)

        self.preview_mode_bar = QWidget()
        preview_header = QHBoxLayout(self.preview_mode_bar)
        preview_header.setContentsMargins(0, 0, 0, 0)
        preview_header.setSpacing(2)

        self.btn_preview_mode = QPushButton("视频预览")
        self.btn_preview_mode.setObjectName("btnPreviewMode")
        self.btn_preview_mode.setCheckable(True)
        self.btn_preview_mode.setChecked(True)

        self.btn_camera_mode = QPushButton("摄像头实时推理")
        self.btn_camera_mode.setObjectName("btnCameraMode")
        self.btn_camera_mode.setCheckable(True)

        self.btn_batch_mode = QPushButton("批量推理")
        self.btn_batch_mode.setObjectName("btnBatchMode")
        self.btn_batch_mode.setCheckable(True)

        preview_header.addWidget(self.btn_preview_mode)
        preview_header.addWidget(self.btn_camera_mode)
        preview_header.addWidget(self.btn_batch_mode)
        preview_header.addStretch(1)

        self.progress_notice = QFrame(preview_panel)
        self.progress_notice.setObjectName("analysisProgressNotice")
        self.progress_notice.setVisible(False)
        progress_notice_layout = QHBoxLayout(self.progress_notice)
        progress_notice_layout.setContentsMargins(14, 8, 14, 8)
        progress_notice_layout.setSpacing(10)

        self.progress_status_dot = QLabel(self.progress_notice)
        self.progress_status_dot.setObjectName("analysisProgressDot")
        self.progress_status_dot.setFixedSize(8, 8)
        self.progress_status_dot.setAccessibleName("分析任务正在进行")

        self.progress_label = QLabel("正在准备分析任务", self.progress_notice)
        self.progress_label.setObjectName("analysisProgressLabel")
        self.progress_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )

        self.progress_bar = QProgressBar(self.progress_notice)
        self.progress_bar.setObjectName("topProgress")
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedSize(132, 4)
        self.progress_bar.setVisible(False)
        self.progress_bar.setAccessibleName("分析任务进度")

        progress_notice_layout.addWidget(self.progress_status_dot, 0, Qt.AlignmentFlag.AlignVCenter)
        progress_notice_layout.addWidget(self.progress_label, 1)
        progress_notice_layout.addWidget(self.progress_bar, 0, Qt.AlignmentFlag.AlignVCenter)

        self.video_player = VideoPlayerWidget()
        self.video_player.setMinimumHeight(360)
        self.video_player.fullscreenRequested.connect(self.toggle_video_fullscreen)

        court_controls = QFrame()
        court_controls.setObjectName("courtControlsBar")
        court_controls_layout = QHBoxLayout(court_controls)
        court_controls_layout.setContentsMargins(0, 0, 0, 0)
        court_controls_layout.setSpacing(8)
        court_controls_layout.addStretch(1)

        self.btn_redetect_court = QPushButton("重新标注球场")
        self.btn_redetect_court.setObjectName("btnRedetectCourt")
        self.btn_redetect_court.setToolTip("自动标定完成后，可重新点击四角或直接拖动角点进行修正")
        self.btn_redetect_court.setEnabled(False)
        self.btn_redetect_court.clicked.connect(self.manualCourtCalibrationRequested.emit)
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

        self.btn_select_batch_folder = QPushButton("选择文件夹")
        self.btn_select_batch_folder.setObjectName("btnSelectBatchFolder")
        self.btn_select_batch_folder.setVisible(False)
        self.batch_folder_edit = QLineEdit()
        self.batch_folder_edit.setObjectName("videoPathEdit")
        self.batch_folder_edit.setReadOnly(True)
        self.batch_folder_edit.setPlaceholderText("批量视频文件夹")
        self.batch_folder_edit.setVisible(False)
        self.batch_video_combo = QComboBox()
        self.batch_video_combo.setObjectName("cameraDeviceCombo")
        self.batch_video_combo.setMinimumWidth(180)
        self.batch_video_combo.setVisible(False)
        self.btn_export_batch = QPushButton("导出数据")
        self.btn_export_batch.setObjectName("btnExportBatch")
        self.btn_export_batch.setEnabled(False)
        self.btn_export_batch.setVisible(False)
        self.btn_export_report = QPushButton("导出报告")
        self.btn_export_report.setObjectName("btnExportReport")
        self.btn_export_report.setEnabled(False)
        self.btn_export_report.setToolTip("将当前回合数据导出为 HTML 训练报告")

        controls_layout.addWidget(self.camera_device_combo)
        controls_layout.addWidget(self.btn_refresh_cameras)
        controls_layout.addWidget(self.btn_select_batch_folder)
        controls_layout.addWidget(self.batch_folder_edit, stretch=1)
        controls_layout.addWidget(self.batch_video_combo)
        controls_layout.addWidget(self.btn_export_batch)
        controls_layout.addWidget(self.btn_export_report)
        controls_layout.addWidget(self.video_player.btn_fullscreen)
        controls_layout.addWidget(self.video_player.btn_force_stop)
        self.btn_select_batch_folder.clicked.connect(self.batchFolderBrowseRequested.emit)
        self.batch_video_combo.currentIndexChanged.connect(self._emit_batch_rally_selection)
        self.btn_export_batch.clicked.connect(self.batchExportRequested.emit)
        self.btn_export_report.clicked.connect(self.reportExportRequested.emit)

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

        preview_layout.addWidget(self.progress_notice, 0)
        preview_layout.addWidget(court_controls, 0)
        preview_layout.addWidget(video_controls, 0)
        preview_layout.addWidget(self.video_player, 1)
        preview_layout.addWidget(timeline_bar, 0)

        preview_shell_layout.addWidget(self.preview_mode_bar, 0)
        preview_shell_layout.addWidget(preview_panel, stretch=1)

        body_splitter.addWidget(preview_shell)

    def _build_analytics_panel(self, body_splitter: QSplitter) -> None:
        self.analytics_panel = QFrame()
        analytics_panel = self.analytics_panel
        analytics_panel.setObjectName("analyticsCard")
        analytics_layout = QHBoxLayout(analytics_panel)
        analytics_layout.setContentsMargins(0, 0, 0, 0)
        analytics_layout.setSpacing(14)

        self.tabs = QTabWidget()
        self.tabs.setObjectName("mainTabs")
        self.tabs.setTabPosition(QTabWidget.TabPosition.North)
        self.tabs.setTabShape(QTabWidget.TabShape.Rounded)

        self._build_overview_tab()
        self._build_data_tab()
        self._build_report_tab()
        self._build_stats_tab()
        self._build_pose_tab()
        self._build_settings_tab()
        self._build_log_tab()

        analytics_layout.addWidget(self.tabs, stretch=1)
        body_splitter.addWidget(analytics_panel)

    def _build_overview_tab(self) -> None:
        tab_overview = QWidget()
        overview_layout = QVBoxLayout(tab_overview)
        overview_layout.setContentsMargins(12, 12, 12, 12)
        overview_layout.setSpacing(12)

        metrics_grid = QGridLayout()
        metrics_grid.setHorizontalSpacing(12)
        metrics_grid.setVerticalSpacing(12)
        card1, self.lbl_realtime_fps = self._create_metric_card("实时帧数", "0.0 FPS")
        card2, self.lbl_rally_state = self._create_rally_state_card()
        card3, self.lbl_valid_pose = self._create_metric_card("推理 FPS", "0.0 FPS")
        card4, self.lbl_ball_speed = self._create_metric_card("当前球速", "-- km/h")
        for index, card in enumerate((card1, card2, card3, card4)):
            card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            metrics_grid.addWidget(card, index // 2, index % 2)
        metrics_grid.setColumnStretch(0, 1)
        metrics_grid.setColumnStretch(1, 1)

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

        overview_layout.addLayout(metrics_grid)
        overview_layout.addLayout(section_header)
        overview_layout.addWidget(self.table_actions)
        self.tabs.addTab(tab_overview, "概览")

    def _build_data_tab(self) -> None:
        tab_data = QWidget()
        data_layout = QVBoxLayout(tab_data)
        data_layout.setContentsMargins(12, 12, 12, 12)
        data_layout.setSpacing(12)

        section_header = QHBoxLayout()
        section_title = QLabel("回合数据")
        section_title.setObjectName("sectionTitle")
        section_note = QLabel("实时推理 / 批量推理共用")
        section_note.setObjectName("sectionNote")
        section_header.addWidget(section_title)
        section_header.addStretch(1)
        section_header.addWidget(section_note)

        self.data_subtabs = QTabWidget()
        self.data_subtabs.setObjectName("secondaryTabs")

        summary_page = QWidget()
        summary_layout = QVBoxLayout(summary_page)
        summary_layout.setContentsMargins(0, 0, 0, 0)
        self.table_data_summary = QTableWidget(0, 4)
        self.table_data_summary.setObjectName("actionTable")
        self.table_data_summary.setHorizontalHeaderLabels(["指标", "上方球员", "下方球员", "全回合"])
        self.table_data_summary.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table_data_summary.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table_data_summary.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table_data_summary.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.table_data_summary.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table_data_summary.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table_data_summary.setAlternatingRowColors(True)
        self.table_data_summary.verticalHeader().setVisible(False)
        self.table_data_summary.setShowGrid(True)
        summary_layout.addWidget(self.table_data_summary)

        details_page = QWidget()
        details_layout = QVBoxLayout(details_page)
        details_layout.setContentsMargins(0, 0, 0, 0)
        self.table_data_details = QTableWidget(0, 7)
        self.table_data_details.setObjectName("actionTable")
        self.table_data_details.setHorizontalHeaderLabels(["时间", "类型", "球员", "区域", "动作", "置信度", "场地坐标"])
        self.table_data_details.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table_data_details.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table_data_details.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table_data_details.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.table_data_details.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.table_data_details.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        self.table_data_details.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeMode.ResizeToContents)
        self.table_data_details.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table_data_details.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table_data_details.setAlternatingRowColors(True)
        self.table_data_details.verticalHeader().setVisible(False)
        self.table_data_details.setShowGrid(True)
        details_layout.addWidget(self.table_data_details)

        self.data_subtabs.addTab(summary_page, "汇总")
        self.data_subtabs.addTab(details_page, "事件明细")
        data_layout.addLayout(section_header)
        data_layout.addWidget(self.data_subtabs, stretch=1)
        self.tabs.addTab(tab_data, "数据")

    def _build_report_tab(self) -> None:
        self.report_preview = ReportPreviewWidget(PROJECT_ROOT)
        self.tabs.addTab(self.report_preview, "报告")

    def _build_stats_tab(self) -> None:
        tab_stats = QWidget()
        stats_layout = QVBoxLayout(tab_stats)
        stats_layout.setContentsMargins(12, 12, 12, 12)
        stats_layout.setSpacing(12)

        section_header = QHBoxLayout()
        section_title = QLabel("击球类型统计")
        section_title.setObjectName("sectionTitle")
        section_note = QLabel("按 BST 输出累计")
        section_note.setObjectName("sectionNote")
        section_header.addWidget(section_title)
        section_header.addStretch(1)
        section_header.addWidget(section_note)

        stats_frame = QFrame()
        stats_frame.setObjectName("emptyStateCard")
        stats_frame_layout = QVBoxLayout(stats_frame)
        stats_frame_layout.setContentsMargins(18, 18, 18, 18)
        stats_frame_layout.setSpacing(12)

        self.lbl_stroke_total = QLabel("总击球 0 次")
        self.lbl_stroke_total.setObjectName("sectionNote")
        self.stroke_pie_chart = StrokePieChartWidget()

        stats_frame_layout.addWidget(self.lbl_stroke_total, alignment=Qt.AlignmentFlag.AlignRight)
        stats_frame_layout.addWidget(self.stroke_pie_chart, stretch=1)
        stats_layout.addLayout(section_header)
        stats_layout.addWidget(stats_frame, stretch=1)
        self.data_subtabs.addTab(tab_stats, "击球统计")

    def _build_pose_tab(self) -> None:
        tab_pose = QWidget()
        pose_layout = QVBoxLayout(tab_pose)
        pose_layout.setContentsMargins(12, 12, 12, 12)
        pose_layout.setSpacing(10)

        pose_frame = QFrame()
        pose_frame.setObjectName("emptyStateCard")
        pose_frame_layout = QVBoxLayout(pose_frame)
        pose_frame_layout.setContentsMargins(24, 24, 24, 24)
        pose_frame_layout.setSpacing(10)

        pose_title = QLabel("场上移动")
        pose_title.setObjectName("sectionTitle")
        header_row = QHBoxLayout()
        header_row.setSpacing(12)
        distance_layout = QVBoxLayout()
        distance_layout.setSpacing(4)
        self.lbl_top_player_distance = QLabel(self._format_player_distance("上方球员", 0.0))
        self.lbl_top_player_distance.setObjectName("distanceValue")
        self.lbl_top_player_distance.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.lbl_bottom_player_distance = QLabel(self._format_player_distance("下方球员", 0.0))
        self.lbl_bottom_player_distance.setObjectName("distanceValue")
        self.lbl_bottom_player_distance.setAlignment(Qt.AlignmentFlag.AlignRight)
        distance_layout.addWidget(self.lbl_top_player_distance, alignment=Qt.AlignmentFlag.AlignRight)
        distance_layout.addWidget(self.lbl_bottom_player_distance, alignment=Qt.AlignmentFlag.AlignRight)
        header_row.addWidget(pose_title, stretch=1, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        header_row.addLayout(distance_layout)
        self.court_widget = CourtHeatmapWidget()

        pose_frame_layout.addWidget(self.court_widget, stretch=1)
        pose_layout.addLayout(header_row)
        pose_layout.addWidget(pose_frame, stretch=1)
        self.data_subtabs.addTab(tab_pose, "场上移动")

    def _build_log_tab(self) -> None:
        tab_logs = QWidget()
        self.log_tab = tab_logs
        logs_layout = QVBoxLayout(tab_logs)
        logs_layout.setContentsMargins(12, 12, 12, 12)
        logs_layout.setSpacing(8)

        self._log_entries: list[str] = []
        log_toolbar = QHBoxLayout()
        log_toolbar.setSpacing(8)
        self.log_filter_combo = QComboBox()
        self.log_filter_combo.setObjectName("logFilterCombo")
        self.log_filter_combo.setAccessibleName("日志类型筛选")
        self.log_filter_combo.addItem("全部日志", "")
        self.log_filter_combo.addItem("错误", "error")
        self.log_filter_combo.addItem("警告", "warning")
        self.log_filter_combo.addItem("球场", "court")
        self.log_filter_combo.addItem("轨迹", "track")
        self.log_filter_combo.addItem("报告", "report")
        self.log_search_edit = QLineEdit()
        self.log_search_edit.setObjectName("logSearchEdit")
        self.log_search_edit.setPlaceholderText("搜索日志")
        self.log_search_edit.setClearButtonEnabled(True)
        self.log_search_edit.setAccessibleName("搜索日志")
        self.log_autoscroll = QCheckBox("自动滚动")
        self.log_autoscroll.setChecked(True)
        self.log_autoscroll.setAccessibleName("日志自动滚动")
        self.btn_copy_logs = QPushButton("复制")
        self.btn_copy_logs.setObjectName("btnCopyLogs")
        self.btn_clear_logs = QPushButton("清空")
        self.btn_clear_logs.setObjectName("btnClearLogs")
        self.btn_export_logs = QPushButton("导出")
        self.btn_export_logs.setObjectName("btnExportLogs")

        log_toolbar.addWidget(self.log_filter_combo)
        log_toolbar.addWidget(self.log_search_edit, stretch=1)
        log_toolbar.addWidget(self.log_autoscroll)
        log_toolbar.addWidget(self.btn_copy_logs)
        log_toolbar.addWidget(self.btn_export_logs)
        log_toolbar.addWidget(self.btn_clear_logs)

        self.log_console = QTextEdit()
        self.log_console.setObjectName("logConsole")
        self.log_console.setReadOnly(True)
        self.log_console.setPlaceholderText("系统日志")
        self.log_console.setAccessibleName("系统诊断日志")
        self.log_console.document().setMaximumBlockCount(2000)
        self.log_filter_combo.currentIndexChanged.connect(self._render_logs)
        self.log_search_edit.textChanged.connect(self._render_logs)
        self.btn_copy_logs.clicked.connect(self._copy_logs)
        self.btn_clear_logs.clicked.connect(self.clear_logs)
        self.btn_export_logs.clicked.connect(self._export_logs)
        logs_layout.addLayout(log_toolbar)
        logs_layout.addWidget(self.log_console)
        self.settings_subtabs.addTab(tab_logs, "诊断日志")

    def _build_settings_tab(self) -> None:
        tab_settings = QWidget()
        self.settings_container = tab_settings
        outer_layout = QVBoxLayout(tab_settings)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        self.settings_subtabs = QTabWidget()
        self.settings_subtabs.setObjectName("secondaryTabs")

        settings_page = QWidget()
        settings_layout = QVBoxLayout(settings_page)
        settings_layout.setContentsMargins(12, 12, 12, 12)
        settings_layout.setSpacing(12)

        section_header = QHBoxLayout()
        section_title = QLabel("模型设置")
        section_title.setObjectName("sectionTitle")
        section_note = QLabel("更改模型仅在停止分析时生效")
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
        self.pose_model_edit.setAccessibleName("骨骼模型文件路径")
        pose_label.setBuddy(self.pose_model_edit)
        self.btn_browse_pose_model = QPushButton("浏览")
        self.btn_browse_pose_model.setObjectName("btnBrowsePoseModel")
        self.btn_browse_pose_model.setAccessibleName("浏览骨骼模型文件")
        self.btn_browse_pose_model.clicked.connect(self.poseModelBrowseRequested.emit)

        track_label = QLabel("球轨迹模型")
        track_label.setObjectName("styleLabel")
        self.track_model_enabled = ToggleSwitch("启用球轨迹模型")
        self.track_model_enabled.setChecked(True)
        self.track_model_edit = QLineEdit()
        self.track_model_edit.setObjectName("modelPathEdit")
        self.track_model_edit.setPlaceholderText("选择球轨迹模型权重文件")
        self.track_model_edit.setAccessibleName("球轨迹模型文件路径")
        track_label.setBuddy(self.track_model_edit)
        self.btn_browse_track_model = QPushButton("浏览")
        self.btn_browse_track_model.setObjectName("btnBrowseTrackModel")
        self.btn_browse_track_model.setAccessibleName("浏览球轨迹模型文件")
        self.btn_browse_track_model.clicked.connect(self.trackModelBrowseRequested.emit)

        debug_label = QLabel("逐帧诊断")
        debug_label.setObjectName("styleLabel")
        self.debug_csv_enabled = ToggleSwitch("写入逐帧分析日志")
        self.debug_csv_enabled.setChecked(False)
        debug_note = QLabel("保存 TrackNet CSV 与包含球、姿态和击球事件的 JSONL。")
        debug_note.setObjectName("sectionNote")
        debug_note.setWordWrap(True)

        report_api_label = QLabel("报告 API")
        report_api_label.setObjectName("styleLabel")
        self.report_api_enabled = ToggleSwitch("启用报告模型 API")
        self.report_api_enabled.setChecked(False)
        self.report_api_enabled.toggled.connect(self._update_report_api_fields_enabled)
        self.report_api_provider_combo = QComboBox()
        self.report_api_provider_combo.setObjectName("modelPathEdit")
        self.report_api_provider_combo.setAccessibleName("报告 API 服务商")
        self.report_api_provider_combo.addItem("阿里百炼 OpenAI 兼容", "bailian_openai")
        self.report_api_endpoint_edit = QLineEdit()
        self.report_api_endpoint_edit.setObjectName("modelPathEdit")
        self.report_api_endpoint_edit.setAccessibleName("报告 API 地址")
        self.report_api_endpoint_edit.setPlaceholderText("https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
        self.report_api_model_edit = QLineEdit()
        self.report_api_model_edit.setObjectName("modelPathEdit")
        self.report_api_model_edit.setAccessibleName("报告 API 模型")
        self.report_api_model_edit.setPlaceholderText("qwen-plus")
        self.report_api_key_edit = QLineEdit()
        self.report_api_key_edit.setObjectName("modelPathEdit")
        self.report_api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.report_api_key_edit.setPlaceholderText("阿里百炼 API Key")
        self.report_api_key_edit.setAccessibleName("报告 API 密钥")
        self.btn_toggle_api_key = QPushButton("显示")
        self.btn_toggle_api_key.setObjectName("btnToggleApiKey")
        self.btn_toggle_api_key.setCheckable(True)
        self.btn_toggle_api_key.setAccessibleName("显示或隐藏 API 密钥")
        self.btn_toggle_api_key.toggled.connect(self._toggle_api_key_visibility)

        settings_frame_layout.addWidget(pose_label, 0, 0)
        settings_frame_layout.addWidget(self.pose_model_enabled, 0, 1)
        settings_frame_layout.addWidget(self.pose_model_edit, 0, 2)
        settings_frame_layout.addWidget(self.btn_browse_pose_model, 0, 3)
        settings_frame_layout.addWidget(track_label, 1, 0)
        settings_frame_layout.addWidget(self.track_model_enabled, 1, 1)
        settings_frame_layout.addWidget(self.track_model_edit, 1, 2)
        settings_frame_layout.addWidget(self.btn_browse_track_model, 1, 3)
        settings_frame_layout.addWidget(debug_label, 2, 0)
        settings_frame_layout.addWidget(self.debug_csv_enabled, 2, 1)
        settings_frame_layout.addWidget(debug_note, 2, 2, 1, 2)
        settings_frame_layout.addWidget(report_api_label, 3, 0)
        settings_frame_layout.addWidget(self.report_api_enabled, 3, 1)
        settings_frame_layout.addWidget(self.report_api_provider_combo, 3, 2, 1, 2)
        endpoint_label = QLabel("API 地址")
        endpoint_label.setBuddy(self.report_api_endpoint_edit)
        settings_frame_layout.addWidget(endpoint_label, 4, 0)
        settings_frame_layout.addWidget(self.report_api_endpoint_edit, 4, 1, 1, 3)
        report_model_label = QLabel("模型")
        report_model_label.setBuddy(self.report_api_model_edit)
        settings_frame_layout.addWidget(report_model_label, 5, 0)
        settings_frame_layout.addWidget(self.report_api_model_edit, 5, 1, 1, 3)
        api_key_label = QLabel("API 密钥")
        api_key_label.setBuddy(self.report_api_key_edit)
        settings_frame_layout.addWidget(api_key_label, 6, 0)
        settings_frame_layout.addWidget(self.report_api_key_edit, 6, 1, 1, 2)
        settings_frame_layout.addWidget(self.btn_toggle_api_key, 6, 3)
        settings_frame_layout.setColumnStretch(2, 1)

        self.pose_model_enabled.stateChanged.connect(self._emit_model_switches_changed)
        self.track_model_enabled.stateChanged.connect(self._emit_model_switches_changed)
        self.debug_csv_enabled.toggled.connect(self.debugCsvChanged.emit)

        action_row = QHBoxLayout()
        action_row.addStretch(1)
        self.btn_model_defaults = QPushButton("恢复默认")
        self.btn_model_defaults.setObjectName("btnModelDefaults")
        self.btn_apply_model_settings = QPushButton("应用模型设置")
        self.btn_apply_model_settings.setObjectName("btnApplyModelSettings")
        self.btn_apply_report_api_settings = QPushButton("保存报告设置")
        self.btn_apply_report_api_settings.setObjectName("btnApplyReportApiSettings")
        self.btn_model_defaults.clicked.connect(self.modelSettingsDefaultsRequested.emit)
        self.btn_apply_model_settings.clicked.connect(self._emit_model_settings_apply)
        self.btn_apply_report_api_settings.clicked.connect(self._emit_report_api_settings_apply)
        action_row.addWidget(self.btn_model_defaults)
        action_row.addWidget(self.btn_apply_report_api_settings)
        action_row.addWidget(self.btn_apply_model_settings)

        settings_layout.addLayout(section_header)
        settings_layout.addWidget(settings_frame)
        settings_layout.addLayout(action_row)
        settings_layout.addStretch(1)
        self.settings_subtabs.addTab(settings_page, "模型与服务")
        outer_layout.addWidget(self.settings_subtabs)
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

    def _create_rally_state_card(self) -> tuple[QFrame, QLabel]:
        container = QFrame()
        container.setObjectName("metricCard")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(6)

        title_lbl = QLabel("回合状态")
        title_lbl.setObjectName("metricTitle")
        value_lbl = QLabel("未开始")
        value_lbl.setObjectName("metricValue")

        layout.addWidget(title_lbl)
        layout.addWidget(value_lbl)
        return container, value_lbl

    def _refresh_widget(self, widget: QWidget) -> None:
        widget.style().unpolish(widget)
        widget.style().polish(widget)
        widget.update()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape and self._video_fullscreen:
            self.set_video_fullscreen(False)
            event.accept()
            return
        super().keyPressEvent(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._position_fullscreen_analytics_panel()

    def toggle_video_fullscreen(self) -> None:
        self.set_video_fullscreen(not self._video_fullscreen)

    def set_video_fullscreen(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled == self._video_fullscreen:
            self.video_player.set_fullscreen_mode(enabled)
            return

        if enabled:
            self._video_fullscreen_previous_state = self.windowState() & ~Qt.WindowState.WindowFullScreen

        self._video_fullscreen = enabled
        self.header_card.setVisible(not enabled)
        self.preview_mode_bar.setVisible(not enabled)

        margins = (6, 6, 6, 6) if enabled else self._normal_root_margins
        self.root_layout.setContentsMargins(*margins)
        self.root_layout.setSpacing(8 if enabled else 12)
        self._set_analytics_panel_opacity(0.46 if enabled else 1.0)
        self.video_player.set_fullscreen_mode(enabled)

        if enabled:
            self._float_analytics_panel()
            self.showFullScreen()
            self._position_fullscreen_analytics_panel()
        elif self._video_fullscreen_previous_state & Qt.WindowState.WindowMaximized:
            self._dock_analytics_panel()
            self.showMaximized()
        else:
            self._dock_analytics_panel()
            self.showNormal()

    def _float_analytics_panel(self) -> None:
        if self._analytics_panel_overlay:
            return
        self.analytics_panel.setParent(self.central_widget)
        self.analytics_panel.show()
        self.analytics_panel.raise_()
        self._analytics_panel_overlay = True

    def _dock_analytics_panel(self) -> None:
        if not self._analytics_panel_overlay:
            self.body_splitter.setStretchFactor(0, 6)
            self.body_splitter.setStretchFactor(1, 5)
            return
        self.body_splitter.insertWidget(1, self.analytics_panel)
        self.body_splitter.setStretchFactor(0, 6)
        self.body_splitter.setStretchFactor(1, 5)
        self.body_splitter.setSizes([690, 570])
        self.analytics_panel.show()
        self._analytics_panel_overlay = False

    def _position_fullscreen_analytics_panel(self) -> None:
        if not self._analytics_panel_overlay:
            return
        bounds = self.central_widget.rect().adjusted(8, 8, -8, -8)
        if bounds.width() <= 0 or bounds.height() <= 0:
            return
        panel_width = min(bounds.width(), max(420, int(bounds.width() * 0.46)))
        panel_height = min(bounds.height(), max(300, int(bounds.height() * 0.48)))
        self.analytics_panel.setGeometry(
            QRect(
                bounds.right() - panel_width + 1,
                bounds.bottom() - panel_height + 1,
                panel_width,
                panel_height,
            )
        )
        self.analytics_panel.raise_()

    def _set_analytics_panel_opacity(self, opacity: float) -> None:
        opacity = max(0.0, min(float(opacity), 1.0))
        if opacity >= 0.999:
            if self._analytics_opacity_effect is not None:
                self.analytics_panel.setGraphicsEffect(None)
                self._analytics_opacity_effect = None
            return
        if self._analytics_opacity_effect is None:
            self._analytics_opacity_effect = QGraphicsOpacityEffect(self.analytics_panel)
            self.analytics_panel.setGraphicsEffect(self._analytics_opacity_effect)
        self._analytics_opacity_effect.setOpacity(opacity)

    def set_video_state(self, state: str) -> None:
        self.video_player.set_video_state(state)

    def set_video_path(self, path: str) -> None:
        self.video_player.set_video_path(path)

    def set_input_mode(self, mode: str) -> None:
        is_camera = mode == "camera"
        is_batch = mode == "batch"
        self.btn_preview_mode.setChecked(not is_camera and not is_batch)
        self.btn_camera_mode.setChecked(is_camera)
        self.btn_batch_mode.setChecked(is_batch)
        self.video_player.btn_select_video.setVisible(not is_camera and not is_batch)
        self.video_player.path_edit.setVisible(not is_camera and not is_batch)
        self.camera_device_combo.setVisible(is_camera)
        self.btn_refresh_cameras.setVisible(is_camera)
        self.btn_select_batch_folder.setVisible(is_batch)
        self.batch_folder_edit.setVisible(is_batch)
        self.batch_video_combo.setVisible(is_batch)
        self.btn_export_batch.setVisible(is_batch)
        self.btn_export_report.setVisible(True)
        self.video_timeline.setVisible(not is_camera and not is_batch)
        self.btn_analyze.setText("开始推理" if is_camera else "开始分析")
        if is_batch:
            self.btn_analyze.setText("开始批量分析")
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

    def set_batch_folder_path(self, path: str) -> None:
        self.batch_folder_edit.setText(path)
        self.batch_folder_edit.setToolTip(path)

    def set_batch_rally_options(self, records: list[dict[str, object]], selected_id: str | None = None) -> None:
        current_id = selected_id or self.selected_batch_rally_id()
        self.batch_video_combo.blockSignals(True)
        self.batch_video_combo.clear()
        for record in records:
            rally_id = str(record.get("id", record.get("video_path", "")))
            label = str(record.get("video_name", rally_id))
            if not rally_id:
                continue
            self.batch_video_combo.addItem(label, rally_id)
        if current_id:
            index = self.batch_video_combo.findData(current_id)
            if index >= 0:
                self.batch_video_combo.setCurrentIndex(index)
        self.batch_video_combo.blockSignals(False)

    def selected_batch_rally_id(self) -> str:
        if self.batch_video_combo.count() <= 0:
            return ""
        return str(self.batch_video_combo.currentData() or "")

    def set_batch_export_enabled(self, enabled: bool) -> None:
        self.btn_export_batch.setEnabled(bool(enabled))

    def set_report_export_enabled(self, enabled: bool) -> None:
        self.btn_export_report.setEnabled(bool(enabled))

    def set_report_preview_file(self, path: str | Path) -> None:
        self.report_preview.load_report_file(Path(path))
        index = self.tabs.indexOf(self.report_preview)
        if index >= 0:
            self.tabs.setCurrentIndex(index)

    def set_report_preview_html(self, html: str, base_path: str | Path | None = None) -> None:
        self.report_preview.load_report_html(html, base_path=base_path)
        index = self.tabs.indexOf(self.report_preview)
        if index >= 0:
            self.tabs.setCurrentIndex(index)

    def _emit_batch_rally_selection(self, _index: int = -1) -> None:
        self.batchRallySelectionChanged.emit(self.selected_batch_rally_id())

    def set_model_settings(self, pose_model_path: str, track_model_path: str) -> None:
        self.pose_model_edit.setText(pose_model_path)
        self.track_model_edit.setText(track_model_path)
        self.pose_model_edit.setCursorPosition(0)
        self.track_model_edit.setCursorPosition(0)
        self.pose_model_edit.setToolTip(pose_model_path)
        self.track_model_edit.setToolTip(track_model_path)

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

    def set_debug_csv_enabled(self, enabled: bool) -> None:
        self.debug_csv_enabled.blockSignals(True)
        self.debug_csv_enabled.setChecked(enabled)
        self.debug_csv_enabled.blockSignals(False)

    def debug_csv_enabled_state(self) -> bool:
        return self.debug_csv_enabled.isChecked()

    def set_report_api_settings(self, settings: dict[str, object]) -> None:
        self.report_api_enabled.blockSignals(True)
        self.report_api_enabled.setChecked(bool(settings.get("enabled")))
        self.report_api_enabled.blockSignals(False)
        provider = str(settings.get("provider") or "bailian_openai")
        index = self.report_api_provider_combo.findData(provider)
        self.report_api_provider_combo.setCurrentIndex(index if index >= 0 else 0)
        self.report_api_endpoint_edit.setText(str(settings.get("endpoint") or ""))
        self.report_api_model_edit.setText(str(settings.get("model") or ""))
        self.report_api_key_edit.setText(str(settings.get("api_key") or ""))
        self.report_api_endpoint_edit.setCursorPosition(0)
        self.report_api_model_edit.setCursorPosition(0)
        self._update_report_api_fields_enabled(self.report_api_enabled.isChecked())

    def report_api_settings(self) -> dict[str, object]:
        return {
            "enabled": self.report_api_enabled.isChecked(),
            "provider": self.report_api_provider_combo.currentData(),
            "endpoint": self.report_api_endpoint_edit.text().strip(),
            "model": self.report_api_model_edit.text().strip(),
            "api_key": self.report_api_key_edit.text().strip(),
        }

    def set_model_settings_enabled(self, enabled: bool) -> None:
        widgets = (
            self.pose_model_enabled,
            self.track_model_enabled,
            self.debug_csv_enabled,
            self.pose_model_edit,
            self.track_model_edit,
            self.btn_browse_pose_model,
            self.btn_browse_track_model,
            self.btn_model_defaults,
            self.btn_apply_report_api_settings,
            self.btn_apply_model_settings,
            self.report_api_enabled,
            self.report_api_provider_combo,
            self.report_api_endpoint_edit,
            self.report_api_model_edit,
            self.report_api_key_edit,
            self.btn_toggle_api_key,
        )
        for widget in widgets:
            widget.setEnabled(enabled)
        self._update_report_api_fields_enabled(enabled and self.report_api_enabled.isChecked())

    def _emit_model_settings_apply(self) -> None:
        pose_model_path, track_model_path = self.model_settings()
        self.modelSettingsApplyRequested.emit(pose_model_path, track_model_path)

    def _emit_report_api_settings_apply(self) -> None:
        self.reportApiSettingsApplyRequested.emit(self.report_api_settings())

    def _emit_model_switches_changed(self) -> None:
        pose_enabled, track_enabled = self.model_switches()
        self.modelSwitchesChanged.emit(pose_enabled, track_enabled)

    def _toggle_api_key_visibility(self, visible: bool) -> None:
        self.report_api_key_edit.setEchoMode(
            QLineEdit.EchoMode.Normal if visible else QLineEdit.EchoMode.Password
        )
        self.btn_toggle_api_key.setText("隐藏" if visible else "显示")

    def _update_report_api_fields_enabled(self, enabled: bool) -> None:
        for widget in (
            self.report_api_provider_combo,
            self.report_api_endpoint_edit,
            self.report_api_model_edit,
            self.report_api_key_edit,
            self.btn_toggle_api_key,
        ):
            widget.setEnabled(bool(enabled))

    def show_video_frame(
        self,
        image,
        position_ms: int,
        duration_ms: int,
        court=None,
        ball_projection=None,
        player_projections=None,
    ) -> None:
        self.video_player.display_image(image, court=court)
        self.court_widget.set_ball_projection(ball_projection)
        self.court_widget.set_player_projections(player_projections)
        self.video_timeline.set_duration(duration_ms)
        self.video_timeline.set_position(position_ms)

    def set_manual_court_capture_enabled(self, enabled: bool) -> None:
        self.video_player.set_point_capture_enabled(enabled)
        self.btn_redetect_court.setText("取消重新标注" if enabled else "重新标注球场")

    def set_court_overlay(self, court: object | None) -> None:
        self.video_player.set_court_overlay(court)

    def stop_video(self) -> None:
        self.video_player.stop()

    def clear_video(self) -> None:
        self.video_player.clear_video()
        self.court_widget.set_ball_projection(None)
        self.court_widget.set_player_projections(None)
        self.court_widget.clear_player_heatmap()
        self.video_timeline.reset()

    def set_status_state(self, state: str) -> None:
        self.status_label.setProperty("state", state)
        self._refresh_widget(self.status_label)

    def set_system_status(self, text: str, state: str = "idle") -> None:
        message = str(text).strip() or "系统状态：待机中"
        self.status_label.setText(message)
        self.status_label.setAccessibleDescription(message)
        self.set_status_state(state)

    def show_status_notice(self, message: str, state: str = "info") -> None:
        text = str(message).strip()
        if not text:
            self.clear_status_notice()
            return
        self.status_banner_label.setText(text)
        self.status_banner_label.setAccessibleDescription(text)
        self.status_banner.setProperty("state", state)
        self._refresh_widget(self.status_banner)
        self.status_banner.setVisible(True)

    def clear_status_notice(self) -> None:
        self.status_banner.setVisible(False)
        self.status_banner_label.clear()

    def show_log_tab(self) -> None:
        settings_index = self.tabs.indexOf(self.settings_container)
        if settings_index >= 0:
            self.tabs.setCurrentIndex(settings_index)
        log_index = self.settings_subtabs.indexOf(self.log_tab)
        if log_index >= 0:
            self.settings_subtabs.setCurrentIndex(log_index)

    def set_active_theme(self, theme_name: str) -> None:
        self.report_preview.set_theme(theme_name)
        self.stroke_pie_chart.update()
        self.court_widget.update()

    def append_log(self, text: str) -> None:
        message = str(text)
        self._log_entries.append(message)
        if len(self._log_entries) > 2000:
            del self._log_entries[:-2000]
        if self._log_matches(message):
            self.log_console.append(message)
            if self.log_autoscroll.isChecked():
                self.log_console.moveCursor(QTextCursor.MoveOperation.End)

    def clear_logs(self) -> None:
        self._log_entries.clear()
        self.log_console.clear()

    def _log_matches(self, message: str) -> bool:
        category = str(self.log_filter_combo.currentData() or "").casefold()
        query = self.log_search_edit.text().strip().casefold()
        normalized = message.casefold()
        category_terms = {
            "error": ("错误", "失败", "error", "failed"),
            "warning": ("警告", "warning", "未找到", "缺少"),
            "court": ("court", "球场", "标定"),
            "track": ("track", "轨迹", "event"),
            "report": ("报告", "report"),
        }
        terms = category_terms.get(category, ())
        category_matches = not terms or any(term in normalized for term in terms)
        return category_matches and (not query or query in normalized)

    def _render_logs(self, *_args) -> None:
        visible_entries = [entry for entry in self._log_entries if self._log_matches(entry)]
        self.log_console.setPlainText("\n".join(visible_entries))
        if self.log_autoscroll.isChecked():
            self.log_console.moveCursor(QTextCursor.MoveOperation.End)

    def _copy_logs(self) -> None:
        QApplication.clipboard().setText(self.log_console.toPlainText())

    def _export_logs(self) -> None:
        file_path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "导出诊断日志",
            str(PROJECT_ROOT / "outputs" / "wfbar_runtime.log"),
            "日志文件 (*.log *.txt);;所有文件 (*)",
        )
        if not file_path:
            return
        try:
            Path(file_path).write_text(self.log_console.toPlainText(), encoding="utf-8")
        except OSError as exc:
            self.show_status_notice(f"日志导出失败：{exc}", "error")

    def update_progress(self, val: int) -> None:
        if self.progress_bar.minimum() != 0 or self.progress_bar.maximum() != 100:
            self.progress_bar.setRange(0, 100)
        value = max(0, min(int(val), 100))
        self.progress_bar.setValue(value)
        visible = 0 < value < 100
        if visible:
            progress_text = f"正在分析视频（{value}%）"
            self.progress_label.setText(progress_text)
            self.progress_bar.setAccessibleDescription(progress_text)
        self.progress_bar.setVisible(visible)
        self.progress_notice.setVisible(visible)

    def set_progress_busy(self, busy: bool, text: str = "") -> None:
        if busy:
            self.progress_bar.setRange(0, 0)
            progress_text = text.strip() or "正在准备分析任务"
            self.progress_label.setText(progress_text)
            self.progress_bar.setAccessibleDescription(progress_text)
            self.progress_bar.setVisible(True)
            self.progress_notice.setVisible(True)
            return
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setVisible(False)
        self.progress_notice.setVisible(False)

    def set_rally_data(self, record: object | None) -> None:
        if not isinstance(record, dict):
            self.table_data_summary.setRowCount(0)
            self.table_data_details.setRowCount(0)
            self._set_rally_state(None)
            self._set_ball_speed(None)
            return
        summary = record.get("summary", {})
        details = record.get("details", {})
        self._set_rally_state(summary.get("rally_state") if isinstance(summary, dict) else None)
        self._set_ball_speed(summary.get("ball_speed") if isinstance(summary, dict) else None)
        self._populate_rally_summary(summary if isinstance(summary, dict) else {})
        self._populate_rally_details(details if isinstance(details, dict) else {})

    def _populate_rally_summary(self, summary: dict[str, object]) -> None:
        players = summary.get("players", {})
        if not isinstance(players, dict):
            players = {}
        top = players.get("top", {}) if isinstance(players.get("top", {}), dict) else {}
        bottom = players.get("bottom", {}) if isinstance(players.get("bottom", {}), dict) else {}
        reliability = summary.get("data_reliability", {})
        if not isinstance(reliability, dict):
            reliability = {}
        ball_speed = summary.get("ball_speed", {})
        if not isinstance(ball_speed, dict):
            ball_speed = {}

        rows = [
            ("累计跑动距离", self._fmt_m(top.get("distance_m")), self._fmt_m(bottom.get("distance_m")), self._fmt_m(self._num(top.get("distance_m")) + self._num(bottom.get("distance_m")))),
            ("平均速度", self._fmt_speed(top.get("avg_speed_mps")), self._fmt_speed(bottom.get("avg_speed_mps")), ""),
            ("最大速度", self._fmt_speed(top.get("max_speed_mps")), self._fmt_speed(bottom.get("max_speed_mps")), ""),
            ("当前球速", "", "", self._fmt_ball_speed(ball_speed, "current_kmh")),
            ("平均球速", "", "", self._fmt_ball_speed(ball_speed, "avg_kmh")),
            ("最大球速", "", "", self._fmt_ball_speed(ball_speed, "max_kmh")),
            ("急停次数", str(int(self._num(top.get("stop_count")))), str(int(self._num(bottom.get("stop_count")))), ""),
            ("启动次数", str(int(self._num(top.get("start_count")))), str(int(self._num(bottom.get("start_count")))), ""),
            ("前场击球次数", str(self._zone_count(top, "front")), str(self._zone_count(bottom, "front")), ""),
            ("中场击球次数", str(self._zone_count(top, "mid")), str(self._zone_count(bottom, "mid")), ""),
            ("后场击球次数", str(self._zone_count(top, "back")), str(self._zone_count(bottom, "back")), ""),
            ("该回合击球次数", str(int(self._num(top.get("hit_count")))), str(int(self._num(bottom.get("hit_count")))), str(int(self._num(summary.get("rally_hit_count"))))),
            ("回合时长", "", "", self._fmt_seconds(summary.get("rally_duration_s", summary.get("duration_s")))),
            ("回合状态", "", "", str(summary.get("rally_state", "") or "")),
            ("平均击球间隔", "", "", self._fmt_seconds(self._num(summary.get("avg_hit_interval_ms")) / 1000.0)),
            ("高强度移动次数", str(int(self._num(top.get("high_intensity_count")))), str(int(self._num(bottom.get("high_intensity_count")))), str(int(self._num(summary.get("high_intensity_count"))))),
            ("最长连续移动", self._fmt_m(top.get("max_continuous_m")), self._fmt_m(bottom.get("max_continuous_m")), ""),
            ("被动击球次数", str(int(self._num(top.get("passive_hit_count")))), str(int(self._num(bottom.get("passive_hit_count")))), ""),
            ("运动强度评分", "", "", f"{self._num(summary.get('motion_intensity_score')):.1f}"),
            ("球可见率", "", "", self._fmt_percent(reliability.get("ball_visible_rate"))),
            ("姿态有效率", "", "", self._fmt_percent(reliability.get("pose_valid_rate"))),
            ("球场有效率", "", "", self._fmt_percent(reliability.get("court_valid_rate"))),
            ("平均球置信度", "", "", self._fmt_percent(reliability.get("avg_ball_confidence"))),
        ]
        self.table_data_summary.setRowCount(0)
        for row_values in rows:
            self._append_table_row(self.table_data_summary, row_values)

    def _populate_rally_details(self, details: dict[str, object]) -> None:
        hits = details.get("hits", [])
        if not isinstance(hits, list):
            hits = []
        self.table_data_details.setRowCount(0)
        for hit in hits:
            if not isinstance(hit, dict):
                continue
            confidence = self._num(hit.get("confidence")) or self._num(hit.get("event_confidence"))
            court_xy = hit.get("court_xy")
            coord_text = ""
            if isinstance(court_xy, (list, tuple)) and len(court_xy) >= 2:
                coord_text = f"{self._num(court_xy[0]):.1f}, {self._num(court_xy[1]):.1f} cm"
            self._append_table_row(
                self.table_data_details,
                (
                    self._fmt_time_ms(hit.get("timestamp_ms")),
                    "BST动作" if hit.get("source") == "bst" else "球轨候选",
                    str(hit.get("player_label", "")),
                    self._zone_label(str(hit.get("zone", ""))),
                    str(hit.get("stroke", "")),
                    self._fmt_percent(confidence),
                    coord_text,
                ),
            )

    def _append_table_row(self, table: QTableWidget, values: tuple[object, ...]) -> None:
        row = table.rowCount()
        table.insertRow(row)
        for column, value in enumerate(values):
            item = QTableWidgetItem(str(value))
            if column > 0:
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            table.setItem(row, column, item)

    def _set_rally_state(self, state: object | None) -> None:
        state_text = str(state).strip()
        mapping = {
            "回合中": "回合中",
            "active": "回合中",
            "ongoing": "回合中",
            "in_rally": "回合中",
            "rally_active": "回合中",
            "回合结束": "回合结束",
            "ended": "回合结束",
            "finished": "回合结束",
            "complete": "回合结束",
            "rally_ended": "回合结束",
        }
        self.lbl_rally_state.setText(mapping.get(state_text, "未开始"))

    def _set_ball_speed(self, speed: object | None) -> None:
        if (
            not isinstance(speed, dict)
            or int(self._num(speed.get("samples"))) <= 0
            or not bool(speed.get("current_valid"))
        ):
            self.lbl_ball_speed.setText("-- km/h")
            return
        self.lbl_ball_speed.setText(self._fmt_ball_speed(speed, "current_kmh"))

    def _zone_count(self, player: dict[str, object], zone: str) -> int:
        zones = player.get("zone_hits", {})
        if not isinstance(zones, dict):
            return 0
        return int(self._num(zones.get(zone)))

    @staticmethod
    def _zone_label(zone: str) -> str:
        return {"front": "前场", "mid": "中场", "back": "后场"}.get(zone, zone)

    @staticmethod
    def _num(value: object) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0
        if number != number or number in (float("inf"), float("-inf")):
            return 0.0
        return number

    def _fmt_m(self, value: object) -> str:
        return f"{self._num(value):.2f} m"

    def _fmt_speed(self, value: object) -> str:
        return f"{self._num(value):.2f} m/s"

    def _fmt_ball_speed(self, speed: dict[str, object], key: str) -> str:
        if int(self._num(speed.get("samples"))) <= 0:
            return "-- km/h"
        if key.startswith("current") and not bool(speed.get("current_valid")):
            return "-- km/h"
        return f"{self._num(speed.get(key)):.1f} km/h"

    def _fmt_percent(self, value: object) -> str:
        return f"{self._num(value) * 100:.1f}%"

    def _fmt_seconds(self, value: object) -> str:
        return f"{self._num(value):.2f} s"

    def _fmt_time_ms(self, value: object) -> str:
        total_seconds = max(0.0, self._num(value) / 1000.0)
        minutes = int(total_seconds // 60)
        seconds = total_seconds - minutes * 60
        if minutes > 0:
            return f"{minutes:d}:{seconds:05.2f}"
        return f"{seconds:.2f}s"

    def add_action_row(self, time_range: str, label: str, conf: float, detail: str) -> None:
        row = 0
        self.table_actions.insertRow(row)

        time_item = QTableWidgetItem(time_range)
        time_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.table_actions.setItem(row, 0, time_item)

        label_item = QTableWidgetItem(label)
        label_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.table_actions.setItem(row, 1, label_item)

        conf_item = QTableWidgetItem(f"{conf * 100:.1f}%")
        conf_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.table_actions.setItem(row, 2, conf_item)

        self.table_actions.setItem(row, 3, QTableWidgetItem(detail))
        self.table_actions.scrollToTop()
        self.stroke_pie_chart.increment(label)
        self._refresh_stroke_total()

    def reset_analysis(self) -> None:
        self.progress_bar.setValue(0)
        self.table_actions.setRowCount(0)
        self.stroke_pie_chart.clear_counts()
        self._refresh_stroke_total()
        self.court_widget.clear_player_heatmap()
        self.set_player_distances(None)
        self.lbl_realtime_fps.setText("0.0 FPS")
        self.lbl_rally_state.setText("未开始")
        self.lbl_valid_pose.setText("0.0 FPS")
        self.lbl_ball_speed.setText("-- km/h")
        self.set_rally_data(None)
        self.report_preview.load_template()

    def stroke_total_count(self) -> int:
        return self.stroke_pie_chart.total_count()

    def set_player_distances(self, distances_m: object | None) -> None:
        top_distance = 0.0
        bottom_distance = 0.0
        if isinstance(distances_m, dict):
            top_distance = self._safe_distance_m(distances_m.get("top", 0.0))
            bottom_distance = self._safe_distance_m(distances_m.get("bottom", 0.0))
        elif isinstance(distances_m, (list, tuple)):
            if len(distances_m) > 0:
                top_distance = self._safe_distance_m(distances_m[0])
            if len(distances_m) > 1:
                bottom_distance = self._safe_distance_m(distances_m[1])

        self.lbl_top_player_distance.setText(self._format_player_distance("上方球员", top_distance))
        self.lbl_bottom_player_distance.setText(self._format_player_distance("下方球员", bottom_distance))

    @staticmethod
    def _format_player_distance(label: str, distance_m: float) -> str:
        return f"{label}：{distance_m:.2f} 米"

    @staticmethod
    def _safe_distance_m(value: object) -> float:
        try:
            distance = float(value)
        except (TypeError, ValueError):
            return 0.0
        if distance != distance or distance in (float("inf"), float("-inf")):
            return 0.0
        return max(0.0, distance)

    def _refresh_stroke_total(self) -> None:
        total = self.stroke_total_count()
        self.lbl_stroke_total.setText(f"总击球 {total} 次")
