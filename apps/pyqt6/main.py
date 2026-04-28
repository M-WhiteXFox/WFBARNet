from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

QApplication.setHighDpiScaleFactorRoundingPolicy(
    Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
)

from apps.pyqt6.utils.style import load_stylesheet


def main() -> int:
    from apps.pyqt6.controllers.analysis_controller_runtime import MainController
    from apps.pyqt6.services.court_detection_service import create_court_detection_service
    from apps.pyqt6.views.main_window_refined import MainWindow

    app = QApplication(sys.argv)

    app.setEffectEnabled(Qt.UIEffect.UI_AnimateCombo, False)
    app.setEffectEnabled(Qt.UIEffect.UI_AnimateTooltip, False)

    # 设置应用图标
    icon_path = PROJECT_ROOT / "apps" / "pyqt6" / "resources" / "icons" / "app.ico"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    load_stylesheet(app, "office_light")

    window = MainWindow()
    court_service = create_court_detection_service()
    controller = MainController(window, court_service=court_service)
    app.aboutToQuit.connect(controller.shutdown)
    
    # 设置窗口图标
    if icon_path.exists():
        window.setWindowIcon(QIcon(str(icon_path)))

    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
