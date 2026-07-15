from __future__ import annotations

import importlib.util
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

if importlib.util.find_spec("PyQt6") is None:
    raise unittest.SkipTest("PyQt6 is not installed in this test environment.")

from PyQt6.QtGui import QImage
from PyQt6.QtWidgets import QApplication

from apps.pyqt6.controllers.analysis_controller_runtime import MainController
from apps.pyqt6.views.components.video_player_panel_runtime import CourtLineOverlayWidget


class _Signal:
    def connect(self, callback) -> None:
        return


class _CourtService:
    def __init__(self, latest: dict | None = None) -> None:
        self.latest = latest
        self.clear_count = 0
        self.request_count = 0
        self.resultReady = _Signal()
        self.failed = _Signal()

    def latest_prediction_dict(self):
        return self.latest

    def latest_display_prediction_dict(self):
        return self.latest

    def clear_calibration(self) -> None:
        self.clear_count += 1
        self.latest = None

    def request_prediction(self) -> None:
        self.request_count += 1


class _VideoPlayer:
    def __init__(self) -> None:
        self.displayed_court = None

    def clear_video(self) -> None:
        return

    def set_live_source(self, label: str) -> None:
        return

    def display_image(self, image, *, court=None) -> None:
        self.displayed_court = court

    def stop(self) -> None:
        return


class _View:
    def __init__(self) -> None:
        self.video_player = _VideoPlayer()
        self.shown_court = None
        self.logs: list[str] = []
        self.progress_busy: tuple[bool, str] = (False, "")
        self.system_status: tuple[str, str] = ("", "")
        self.status_notices: list[tuple[str, str]] = []

    def show_video_frame(self, image, position_ms, duration_ms, court, *args) -> None:
        self.shown_court = court

    def set_player_distances(self, value) -> None:
        return

    def set_input_mode(self, mode: str) -> None:
        return

    def set_video_state(self, state: str) -> None:
        return

    def append_log(self, message: str) -> None:
        self.logs.append(message)

    def set_court_overlay(self, payload) -> None:
        self.shown_court = payload

    def set_manual_court_capture_enabled(self, enabled: bool) -> None:
        return

    def set_status_state(self, state: str) -> None:
        return

    def set_system_status(self, text: str, state: str) -> None:
        self.system_status = (state, text)

    def show_status_notice(self, message: str, state: str) -> None:
        self.status_notices.append((state, message))

    def update_progress(self, value: int) -> None:
        return

    def set_progress_busy(self, busy: bool, text: str = "") -> None:
        self.progress_busy = (busy, text)


def _bare_controller(service: _CourtService, view: _View | None = None) -> MainController:
    controller = MainController.__new__(MainController)
    controller.view = view or _View()
    controller._court_service = service
    controller._manual_court_active = False
    controller._manual_court_points = []
    controller._display_fps_ema = 0.0
    controller._last_court_log_frame = -1
    controller._pending_video_start_ms = None
    controller._court_bootstrap_exhausted = False
    controller._log_track_debug_event = lambda payload: None
    controller._append_trajectory_event = lambda payload: None
    controller._append_bst_predictions = lambda payload: None
    controller._set_current_rally_record = lambda payload: None
    controller._update_display_fps = lambda: 0.0
    controller._should_update_metrics_text = lambda: False
    return controller


class AnalysisControllerCourtRuntimeTest(unittest.TestCase):
    def test_status_helper_updates_text_and_semantic_state(self) -> None:
        controller = _bare_controller(_CourtService(None))

        controller._set_view_status("success", "系统状态：视频分析完成")

        self.assertEqual(
            controller.view.system_status,
            ("success", "系统状态：视频分析完成"),
        )

    def test_court_waiting_state_shows_busy_indicator(self) -> None:
        controller = _bare_controller(_CourtService(None))
        running_states: list[bool] = []
        controller._set_running_state = lambda: running_states.append(True)

        controller._set_court_waiting_state()

        self.assertEqual(running_states, [True])
        self.assertEqual(
            controller.view.progress_busy,
            (True, "正在识别可信球场线..."),
        )

    def test_video_analysis_waits_for_trusted_court_before_starting_playback(self) -> None:
        service = _CourtService({"valid": False, "provisional": True})
        controller = _bare_controller(service)
        controller._input_mode = "video"
        controller._selected_video_path = "match.mp4"
        controller._video_meta = {"position_ms": 1250}
        controller._pending_seek_ms = None
        controller._ensure_models_ready = lambda: True
        waiting_states: list[bool] = []
        controller._set_court_waiting_state = lambda: waiting_states.append(True)
        starts: list[dict] = []
        controller._start_playback = lambda **kwargs: starts.append(kwargs)

        controller.handle_analyze()

        self.assertEqual(starts, [])
        self.assertEqual(controller._pending_video_start_ms, 1250)
        self.assertEqual(waiting_states, [True])
        self.assertEqual(service.request_count, 1)

    def test_trusted_court_result_starts_waiting_video_once_without_reset(self) -> None:
        controller = _bare_controller(_CourtService(None))
        controller._pending_video_start_ms = 1250
        starts: list[dict] = []
        controller._start_playback = lambda **kwargs: starts.append(kwargs)
        payload = {
            "valid": True,
            "updated": True,
            "frame_id": 3,
            "confidence": 0.91,
            "detect_ms": 25.0,
            "scheme": "courtkeynet",
        }

        controller._on_court_prediction_ready(payload)
        controller._on_court_prediction_ready(payload)

        self.assertEqual(
            starts,
            [{"start_ms": 1250, "request_court_prediction": False}],
        )
        self.assertIsNone(controller._pending_video_start_ms)
        self.assertEqual(controller.view.progress_busy, (False, ""))

    def test_exhausted_untrusted_court_does_not_start_waiting_video(self) -> None:
        controller = _bare_controller(_CourtService(None))
        controller._pending_video_start_ms = 0
        idle_states: list[bool] = []
        controller._set_idle_state = lambda: idle_states.append(True)
        starts: list[dict] = []
        controller._start_playback = lambda **kwargs: starts.append(kwargs)
        payload = {
            "valid": False,
            "provisional": True,
            "frame_id": 6,
            "scheme": "courtkeynet",
            "candidate_confidence": 0.47,
            "metrics": {"bootstrap_exhausted": 1},
        }

        controller._on_court_prediction_ready(payload)

        self.assertEqual(starts, [])
        self.assertIsNone(controller._pending_video_start_ms)
        self.assertEqual(idle_states, [True])
        self.assertEqual(controller.view.progress_busy, (False, ""))
        self.assertTrue(any("未开始播放" in message for message in controller.view.logs))

    def test_start_after_empty_exhausted_scan_requires_manual_calibration(self) -> None:
        controller = _bare_controller(_CourtService(None))
        controller._on_court_prediction_ready(
            {
                "valid": False,
                "provisional": False,
                "frame_id": 6,
                "metrics": {"bootstrap_exhausted": 1},
            }
        )
        controller._input_mode = "video"
        controller._selected_video_path = "match.mp4"
        controller._video_meta = {"position_ms": 0}
        controller._pending_seek_ms = None
        controller._ensure_models_ready = lambda: True
        waiting_states: list[bool] = []
        controller._set_court_waiting_state = lambda: waiting_states.append(True)
        starts: list[dict] = []
        controller._start_playback = lambda **kwargs: starts.append(kwargs)

        controller.handle_analyze()

        self.assertEqual(starts, [])
        self.assertEqual(waiting_states, [])
        self.assertIsNone(controller._pending_video_start_ms)
        self.assertTrue(any("视频不会播放" in message for message in controller.view.logs))

    def test_frame_without_court_payload_reuses_latest_service_prediction(self) -> None:
        latest = {"valid": True, "scheme": "court_pose_white_line", "frame_id": 8}
        service = _CourtService(latest)
        controller = _bare_controller(service)
        image = QImage(2, 2, QImage.Format.Format_RGB32)

        controller._on_frame_ready({"image": image, "court": None})

        self.assertIs(controller.view.shown_court, latest)

    def test_camera_frame_without_court_payload_reuses_latest_service_prediction(self) -> None:
        latest = {"valid": True, "scheme": "court_pose_white_line", "frame_id": 8}
        service = _CourtService(latest)
        controller = _bare_controller(service)
        controller.view.court_widget = type(
            "CourtWidget",
            (),
            {
                "set_ball_projection": lambda self, value: None,
                "set_player_projections": lambda self, value: None,
            },
        )()
        image = QImage(2, 2, QImage.Format.Format_RGB32)

        controller._on_camera_frame_ready({"image": image, "court": None})

        self.assertIs(controller.view.video_player.displayed_court, latest)

    def test_seek_restart_requests_prediction_when_no_calibration_exists(self) -> None:
        controller = _bare_controller(_CourtService(None))
        controller._pending_seek_ms = 1250
        controller._selected_video_path = "match.mp4"
        controller._append_player_distance_summary = lambda payload: None
        controller._set_idle_state = lambda: None
        controller._maybe_start_stopped_report = lambda stopped: None
        starts: list[dict] = []
        controller._start_playback = lambda **kwargs: starts.append(kwargs)

        controller._on_playback_finished({"stopped": True})

        self.assertEqual(
            starts,
            [{"start_ms": 1250, "request_court_prediction": True}],
        )

    def test_input_mode_change_clears_calibration_from_previous_source(self) -> None:
        service = _CourtService({"valid": True, "scheme": "court_pose_white_line"})
        controller = _bare_controller(service)
        controller._input_mode = "video"
        controller._court_source_key = ("video", "match.mp4")
        controller._selected_video_path = "match.mp4"
        controller._selected_batch_folder = None
        controller._camera_devices = [(0, "Camera 0")]
        controller._stop_workers = lambda **kwargs: None
        controller._reset_metrics = lambda: None
        controller._set_idle_state = lambda: None

        controller.handle_input_mode("camera")

        self.assertEqual(service.clear_count, 1)

    def test_same_source_preserves_manual_calibration_but_camera_change_clears_it(self) -> None:
        service = _CourtService({"valid": True, "scheme": "manual"})
        controller = _bare_controller(service)
        controller._court_source_key = ("camera", "0")

        controller._activate_court_source(("camera", "0"))
        self.assertEqual(service.clear_count, 0)

        controller._activate_court_source(("camera", "1"))
        self.assertEqual(service.clear_count, 1)

    def test_provisional_prediction_is_shown_and_logged_as_unverified(self) -> None:
        service = _CourtService(None)
        controller = _bare_controller(service)
        payload = {
            "valid": False,
            "provisional": True,
            "frame_id": 5,
            "scheme": "court_pose_coarse",
            "candidate_confidence": 0.72,
            "corners": [[10.0, 10.0], [90.0, 10.0], [90.0, 70.0], [10.0, 70.0]],
            "projected_lines": {
                "doubles_outer": [[10.0, 10.0], [90.0, 10.0], [90.0, 70.0], [10.0, 70.0]]
            },
        }

        controller._on_court_prediction_ready(payload)

        self.assertIs(controller.view.shown_court, payload)
        self.assertIn("待确认自动标注", controller.view.logs[-1])

    def test_exhausted_bootstrap_logs_editable_draft_without_promising_upgrade(self) -> None:
        service = _CourtService(None)
        controller = _bare_controller(service)
        payload = {
            "valid": False,
            "provisional": True,
            "frame_id": 6,
            "scheme": "monotrack",
            "candidate_confidence": 0.80,
            "corners": [[10.0, 10.0], [90.0, 10.0], [90.0, 70.0], [10.0, 70.0]],
            "projected_lines": {
                "doubles_outer": [[10.0, 10.0], [90.0, 10.0], [90.0, 70.0], [10.0, 70.0]]
            },
            "metrics": {"bootstrap_exhausted": 1},
        }

        controller._on_court_prediction_ready(payload)

        self.assertIn("自动扫描完成", controller.view.logs[-1])
        self.assertIn("不会用于几何统计", controller.view.logs[-1])
        self.assertNotIn("会自动升级", controller.view.logs[-1])

    def test_overlay_exposes_provisional_corners_for_manual_dragging(self) -> None:
        app = QApplication.instance() or QApplication([])
        overlay = CourtLineOverlayWidget()
        payload = {
            "valid": False,
            "provisional": True,
            "source_size": [100, 80],
            "corners": [[10.0, 10.0], [90.0, 10.0], [90.0, 70.0], [10.0, 70.0]],
            "projected_lines": {
                "doubles_outer": [[10.0, 10.0], [90.0, 10.0], [90.0, 70.0], [10.0, 70.0]]
            },
        }

        overlay.set_court(payload)

        self.assertEqual(len(overlay.corners()), 4)
        self.assertIsNotNone(overlay._make_court_key(payload))
        app.processEvents()


if __name__ == "__main__":
    unittest.main()
