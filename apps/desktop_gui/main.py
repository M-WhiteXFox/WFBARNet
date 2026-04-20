from __future__ import annotations

import ctypes
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


def enable_high_dpi_awareness() -> None:
    if sys.platform != "win32":
        return

    user32 = ctypes.windll.user32
    try:
        user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        return
    except Exception:
        pass

    try:
        shcore = ctypes.windll.shcore
        shcore.SetProcessDpiAwareness(2)
        return
    except Exception:
        pass

    try:
        user32.SetProcessDPIAware()
    except Exception:
        pass


enable_high_dpi_awareness()

try:
    from apps.desktop_gui.gui_app import build_gui
except ModuleNotFoundError as exc:
    missing = exc.name or "未知依赖"
    raise SystemExit(
        f"缺少桌面前端依赖：{missing}。请先执行 `pip install -r apps/desktop_gui/requirements.txt` 再启动界面。"
    ) from exc


if __name__ == "__main__":
    build_gui()
