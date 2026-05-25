from __future__ import annotations

import ctypes
import sys
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication, QLabel, QProgressBar, QVBoxLayout, QWidget


APP_NAME = "WFBARNet"
APP_USER_MODEL_ID = "WFBARNet.Desktop"
WEBENGINE_IMPORT_ERROR: str | None = None


def runtime_root() -> Path:
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        return Path(bundle_root)
    return Path(__file__).resolve().parents[2]


def set_windows_app_user_model_id() -> None:
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_USER_MODEL_ID)
    except Exception:
        return


def preload_web_engine() -> None:
    """Import Qt WebEngine before QApplication is created when it is available."""
    global WEBENGINE_IMPORT_ERROR
    try:
        from PyQt6.QtWebEngineCore import QWebEngineSettings  # noqa: F401
        from PyQt6.QtWebEngineWidgets import QWebEngineView  # noqa: F401
    except Exception as exc:
        WEBENGINE_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


PROJECT_ROOT = runtime_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

QApplication.setHighDpiScaleFactorRoundingPolicy(
    Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
)

from apps.pyqt6.utils.style import load_stylesheet


class StartupSplash(QWidget):
    def __init__(self, icon: QIcon, app_name: str) -> None:
        super().__init__(None, Qt.WindowType.FramelessWindowHint | Qt.WindowType.SplashScreen)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        if not icon.isNull():
            self.setWindowIcon(icon)

        container = QWidget(self)
        container.setObjectName("startupSplashContainer")
        container.setFixedSize(420, 180)

        layout = QVBoxLayout(container)
        layout.setContentsMargins(32, 28, 32, 28)
        layout.setSpacing(14)

        title = QLabel(app_name)
        title.setObjectName("startupSplashTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._message = QLabel("正在启动...")
        self._message.setObjectName("startupSplashMessage")
        self._message.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setTextVisible(False)
        self._progress.setFixedHeight(8)

        layout.addStretch(1)
        layout.addWidget(title)
        layout.addWidget(self._message)
        layout.addWidget(self._progress)
        layout.addStretch(1)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(container)
        self.setFixedSize(container.size())
        screen = QApplication.primaryScreen()
        if screen is not None:
            rect = screen.availableGeometry()
            self.move(rect.center() - self.rect().center())
        self.setStyleSheet(
            """
            #startupSplashContainer {
                background: #f8fafc;
                border: 1px solid #d7dee8;
                border-radius: 8px;
            }
            #startupSplashTitle {
                color: #172033;
                font-size: 26px;
                font-weight: 700;
            }
            #startupSplashMessage {
                color: #5c667a;
                font-size: 13px;
            }
            QProgressBar {
                background: #e3e9f2;
                border: none;
                border-radius: 4px;
            }
            QProgressBar::chunk {
                background: #2563eb;
                border-radius: 4px;
            }
            """
        )

    def set_progress(self, value: int, message: str) -> None:
        self._progress.setValue(max(0, min(100, int(value))))
        self._message.setText(message)


def main() -> int:
    set_windows_app_user_model_id()
    preload_web_engine()
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(APP_NAME)

    app.setEffectEnabled(Qt.UIEffect.UI_AnimateCombo, False)
    app.setEffectEnabled(Qt.UIEffect.UI_AnimateTooltip, False)

    # 设置应用图标
    icon_path = PROJECT_ROOT / "apps" / "pyqt6" / "resources" / "icons" / "app.ico"
    icon = QIcon(str(icon_path)) if icon_path.exists() else QIcon()
    if not icon.isNull():
        app.setWindowIcon(icon)

    splash = StartupSplash(icon, APP_NAME)
    splash.show()
    app.processEvents()

    def update_startup(value: int, message: str) -> None:
        splash.set_progress(value, message)
        app.processEvents()

    update_startup(12, "正在加载界面样式...")
    load_stylesheet(app, "office_light")

    update_startup(32, "正在加载主窗口...")
    from apps.pyqt6.views.main_window_refined import MainWindow

    window = MainWindow()
    if not icon.isNull():
        window.setWindowIcon(icon)

    update_startup(56, "正在初始化手动标定服务...")
    from apps.pyqt6.services.manual_court_calibration_service import create_manual_court_calibration_service

    court_service = create_manual_court_calibration_service()

    update_startup(78, "正在连接控制器...")
    from apps.pyqt6.controllers.analysis_controller_runtime import MainController

    controller = MainController(window, court_service=court_service)
    app.aboutToQuit.connect(controller.shutdown)
    
    # 设置窗口图标
    update_startup(100, "启动完成")
    window.show()
    splash.close()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
