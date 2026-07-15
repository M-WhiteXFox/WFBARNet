from __future__ import annotations

import importlib.util
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

if importlib.util.find_spec("PyQt6") is None:
    raise unittest.SkipTest("PyQt6 is not installed in this test environment.")

from PyQt6.QtWidgets import QApplication

import apps.pyqt6.views.main_window_refined as main_window_module


class MainWindowGuiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])
        cls.web_view_patch = patch.object(main_window_module, "QWebEngineView", None)
        cls.web_settings_patch = patch.object(main_window_module, "QWebEngineSettings", None)
        cls.web_view_patch.start()
        cls.web_settings_patch.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.web_settings_patch.stop()
        cls.web_view_patch.stop()

    def setUp(self) -> None:
        self.window = main_window_module.MainWindow()

    def tearDown(self) -> None:
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()

    def test_navigation_is_consolidated_and_layout_is_resizable(self) -> None:
        self.assertEqual(
            [self.window.tabs.tabText(index) for index in range(self.window.tabs.count())],
            ["概览", "数据", "报告", "设置"],
        )
        self.assertEqual(
            [
                self.window.data_subtabs.tabText(index)
                for index in range(self.window.data_subtabs.count())
            ],
            ["汇总", "事件明细", "击球统计", "场上移动"],
        )
        self.assertEqual(
            [
                self.window.settings_subtabs.tabText(index)
                for index in range(self.window.settings_subtabs.count())
            ],
            ["模型与服务", "诊断日志"],
        )
        self.assertEqual(self.window.body_splitter.count(), 2)
        self.assertFalse(self.window.body_splitter.childrenCollapsible())
        self.assertEqual((self.window.minimumWidth(), self.window.minimumHeight()), (1024, 680))

    def test_key_controls_have_accessible_names(self) -> None:
        controls = (
            self.window.pose_model_enabled,
            self.window.track_model_enabled,
            self.window.debug_csv_enabled,
            self.window.report_api_enabled,
            self.window.pose_model_edit,
            self.window.track_model_edit,
            self.window.report_api_endpoint_edit,
            self.window.report_api_key_edit,
            self.window.video_player.path_edit,
            self.window.video_timeline.seek_slider,
        )
        self.assertTrue(all(control.accessibleName().strip() for control in controls))

    def test_fullscreen_overlay_can_redock_into_splitter(self) -> None:
        self.window._float_analytics_panel()
        self.assertTrue(self.window._analytics_panel_overlay)
        self.assertEqual(self.window.body_splitter.count(), 1)

        self.window._dock_analytics_panel()
        self.assertFalse(self.window._analytics_panel_overlay)
        self.assertEqual(self.window.body_splitter.count(), 2)
        self.assertIs(self.window.body_splitter.widget(1), self.window.analytics_panel)

    def test_progress_notice_and_log_tools_have_visible_states(self) -> None:
        self.window.set_progress_busy(True, "正在识别可信球场线...")
        self.assertFalse(self.window.progress_notice.isHidden())
        self.assertFalse(self.window.progress_bar.isHidden())
        self.assertEqual(self.window.progress_label.text(), "正在识别可信球场线...")
        self.assertEqual(self.window.progress_bar.accessibleName(), "分析任务进度")
        self.assertEqual((self.window.progress_bar.minimum(), self.window.progress_bar.maximum()), (0, 0))

        self.window.update_progress(42)
        self.assertFalse(self.window.progress_notice.isHidden())
        self.assertEqual(self.window.progress_bar.value(), 42)
        self.assertIn("42%", self.window.progress_label.text())

        self.window.set_progress_busy(False)
        self.assertTrue(self.window.progress_notice.isHidden())
        self.assertTrue(self.window.progress_bar.isHidden())

        self.window.show_status_notice("模型加载失败", "error")
        self.assertFalse(self.window.status_banner.isHidden())
        self.assertEqual(self.window.status_banner.property("state"), "error")
        self.window.clear_status_notice()
        self.assertTrue(self.window.status_banner.isHidden())

        self.window.append_log("[信息] ready")
        self.window.append_log("[错误] failed")
        self.window.log_filter_combo.setCurrentIndex(1)
        self.assertNotIn("ready", self.window.log_console.toPlainText())
        self.assertIn("failed", self.window.log_console.toPlainText())
        self.window.clear_logs()
        self.assertEqual(self.window.log_console.toPlainText(), "")


if __name__ == "__main__":
    unittest.main()
