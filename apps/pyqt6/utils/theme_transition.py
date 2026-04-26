from __future__ import annotations

import math
from collections.abc import Callable

from PyQt6.QtCore import (
    QEasingCurve,
    QPoint,
    QPointF,
    QPropertyAnimation,
    QRect,
    QRectF,
    Qt,
    pyqtProperty,
)
from PyQt6.QtGui import QColor, QCursor, QPainter, QPainterPath, QPen, QPixmap
from PyQt6.QtWidgets import QWidget


class ThemeRippleOverlay(QWidget):
    def __init__(
        self,
        parent: QWidget,
        snapshot: QPixmap,
        origin: QPoint,
        duration_ms: int,
    ) -> None:
        super().__init__(parent)
        self._snapshot = snapshot
        self._origin = QPoint(origin)
        self._radius = 0.0
        self.setGeometry(parent.rect())
        self._max_radius = self._distance_to_cover(self.rect(), self._origin)

        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self._animation = self._build_animation(duration_ms)

    @staticmethod
    def _distance_to_cover(rect: QRect, origin: QPoint) -> float:
        corners = (
            rect.topLeft(),
            rect.topRight(),
            rect.bottomLeft(),
            rect.bottomRight(),
        )
        return (
            max(math.hypot(origin.x() - point.x(), origin.y() - point.y()) for point in corners)
            + 72.0
        )

    def _build_animation(self, duration_ms: int) -> QPropertyAnimation:
        animation = QPropertyAnimation(self, b"radius", self)
        animation.setDuration(max(160, duration_ms))
        animation.setStartValue(0.0)
        animation.setEndValue(self._max_radius)
        animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        animation.finished.connect(self.close)
        animation.finished.connect(self.deleteLater)
        return animation

    def get_radius(self) -> float:
        return self._radius

    def set_radius(self, value: float) -> None:
        self._radius = max(0.0, float(value))
        self.update()

    radius = pyqtProperty(float, fget=get_radius, fset=set_radius)

    def start(self) -> None:
        if self._snapshot.isNull() or self._max_radius <= 0:
            self.close()
            self.deleteLater()
            return
        self.show()
        self.raise_()
        self._animation.start()

    def paintEvent(self, event) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        edge_path = self._edge_path(self._radius)
        cover = QPainterPath()
        cover.addRect(QRectF(self.rect()))
        cover = cover.subtracted(edge_path)

        painter.setClipPath(cover)
        painter.drawPixmap(self.rect(), self._snapshot)
        painter.setClipping(False)

        self._draw_feathered_edge(painter)
        self._draw_wave_rings(painter)

    def _circle_rect(self, radius: float) -> QRectF:
        return QRectF(
            self._origin.x() - radius,
            self._origin.y() - radius,
            radius * 2,
            radius * 2,
        )

    def _circle_path(self, radius: float) -> QPainterPath:
        path = QPainterPath()
        path.addEllipse(self._circle_rect(radius))
        return path

    def _edge_path(self, radius: float) -> QPainterPath:
        radius = max(float(radius), 0.0)
        path = QPainterPath()
        if radius <= 0:
            path.addEllipse(self._circle_rect(0.0))
            return path

        progress = min(radius / self._max_radius, 1.0) if self._max_radius > 0 else 1.0
        amplitude = self._edge_amplitude(radius)
        segments = 144
        for index in range(segments):
            angle = (math.tau * index) / segments
            local_radius = max(radius + self._edge_offset(angle, progress, amplitude), 0.0)
            point = QPointF(
                self._origin.x() + math.cos(angle) * local_radius,
                self._origin.y() + math.sin(angle) * local_radius,
            )
            if index == 0:
                path.moveTo(point)
            else:
                path.lineTo(point)
        path.closeSubpath()
        return path

    def _edge_amplitude(self, radius: float) -> float:
        if radius <= 0 or self._max_radius <= 0:
            return 0.0

        progress = min(radius / self._max_radius, 1.0)
        envelope = math.sin(progress * math.pi) ** 0.65
        return min(42.0, max(9.0, radius * 0.066)) * envelope

    @staticmethod
    def _edge_offset(angle: float, progress: float, amplitude: float) -> float:
        ripple = math.sin(angle * 4.0 + progress * 9.0) * amplitude * 0.34
        ripple += math.sin(angle * 7.0 - progress * 6.0) * amplitude * 0.28
        ripple += math.sin(angle * 13.0 + progress * 14.0) * amplitude * 0.18
        ripple += math.sin(angle * 23.0 - progress * 18.0) * amplitude * 0.10
        ripple += math.sin(angle * 31.0 + progress * 11.0) * amplitude * 0.06
        return ripple

    def _draw_feathered_edge(self, painter: QPainter) -> None:
        if self._radius < 8.0:
            return

        feather_width = min(34.0, max(18.0, self._max_radius * 0.035))
        layers = 7
        step = feather_width / layers

        painter.save()
        for index in range(layers):
            outer = self._radius - index * step
            inner = max(self._radius - (index + 1) * step, 0.0)
            if outer <= 0:
                continue

            band = self._edge_path(outer).subtracted(self._edge_path(inner))
            opacity = 0.62 * ((layers - index) / layers) ** 1.5
            painter.setOpacity(opacity)
            painter.setClipPath(band)
            painter.drawPixmap(self.rect(), self._snapshot)

        painter.restore()
        painter.setClipping(False)

    def _draw_wave_rings(self, painter: QPainter) -> None:
        if self._radius < 8.0 or self._max_radius <= 0:
            return

        progress = min(self._radius / self._max_radius, 1.0)
        fade = (1.0 - progress) ** 0.65
        rings = (
            (self._radius, 92, 1.8),
            (self._radius - 18.0, 58, 1.2),
            (self._radius - 36.0, 32, 0.9),
        )
        for radius, alpha, width in rings:
            if radius <= 0:
                continue
            color = QColor(255, 255, 255, int(alpha * fade))
            painter.setPen(QPen(color, width))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(self._edge_path(radius))

        self._draw_edge_pattern(painter, fade)

    def _draw_edge_pattern(self, painter: QPainter, fade: float) -> None:
        if self._radius < 20.0:
            return

        painter.save()
        progress = min(self._radius / self._max_radius, 1.0)
        amplitude = self._edge_amplitude(self._radius)

        dashed_pen = QPen(QColor(255, 255, 255, int(52 * fade)), 1.15)
        dashed_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        dashed_pen.setDashPattern([2.6, 7.4])
        dashed_pen.setDashOffset(-progress * 24.0)
        painter.setPen(dashed_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        if self._radius > 12.0:
            painter.drawPath(self._edge_path(self._radius - 7.0))

        wavy = QPainterPath()
        segments = 96
        for index in range(segments + 1):
            angle = (math.tau * index) / segments
            ripple = math.sin(angle * 7.0 + progress * 8.0) * 3.2
            ripple += math.sin(angle * 13.0 - progress * 5.0) * 1.5
            ripple += self._edge_offset(angle, progress, amplitude * 0.75)
            radius = max(self._radius - 2.5 + ripple, 0.0)
            point = QPointF(
                self._origin.x() + math.cos(angle) * radius,
                self._origin.y() + math.sin(angle) * radius,
            )
            if index == 0:
                wavy.moveTo(point)
            else:
                wavy.lineTo(point)

        wave_pen = QPen(QColor(255, 255, 255, int(48 * fade)), 1.0)
        wave_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        wave_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(wave_pen)
        painter.drawPath(wavy)

        painter.setPen(Qt.PenStyle.NoPen)
        for index in range(40):
            angle = (math.tau * index) / 40.0 + progress * 0.9
            if (index + int(progress * 10)) % 3 == 0:
                continue
            radius = self._radius - 14.0
            radius += self._edge_offset(angle, progress, amplitude * 0.68)
            radius += math.sin(angle * 5.0) * 6.0
            if radius <= 0:
                continue
            size = 1.4 + ((index % 4) * 0.35)
            point = QPointF(
                self._origin.x() + math.cos(angle) * radius,
                self._origin.y() + math.sin(angle) * radius,
            )
            painter.setBrush(QColor(255, 255, 255, int(42 * fade)))
            painter.drawEllipse(point, size, size)

        painter.restore()


def start_theme_ripple_transition(
    root: QWidget,
    apply_change: Callable[[], None],
    *,
    origin_widget: QWidget | None = None,
    duration_ms: int = 1400,
) -> None:
    if not root.isVisible() or root.width() <= 0 or root.height() <= 0:
        apply_change()
        return

    previous_overlay = getattr(root, "_theme_ripple_overlay", None)
    if previous_overlay is not None:
        previous_overlay.hide()
        previous_overlay.deleteLater()
        setattr(root, "_theme_ripple_overlay", None)

    snapshot = root.grab()
    origin = _resolve_origin(root, origin_widget)
    overlay = ThemeRippleOverlay(root, snapshot, origin, duration_ms)
    setattr(root, "_theme_ripple_overlay", overlay)

    def _clear_overlay(_obj: object | None = None) -> None:
        if getattr(root, "_theme_ripple_overlay", None) is overlay:
            setattr(root, "_theme_ripple_overlay", None)

    overlay.destroyed.connect(_clear_overlay)
    overlay.show()

    apply_change()
    root.update()
    overlay.raise_()
    overlay.start()


def _resolve_origin(root: QWidget, origin_widget: QWidget | None) -> QPoint:
    origin = root.mapFromGlobal(QCursor.pos())
    if root.rect().contains(origin):
        return origin

    if origin_widget is not None:
        return origin_widget.mapTo(root, origin_widget.rect().center())

    rect = root.rect()
    return QPoint(
        max(rect.left(), min(origin.x(), rect.right())),
        max(rect.top(), min(origin.y(), rect.bottom())),
    )
