from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from apps.pyqt6.controllers.analysis_controller_runtime import (
    MainController,
    frame_step_seconds,
    track_state_gap_exceeded,
)


class _FakeTimeline:
    def set_interactive(self, value: bool) -> None:
        self.interactive = value


class _FakeVideoPlayer:
    def __init__(self) -> None:
        self.live_source = ""

    def clear_video(self) -> None:
        self.cleared = True

    def set_live_source(self, value: str) -> None:
        self.live_source = value


class _FakeView:
    def __init__(self) -> None:
        self.video_timeline = _FakeTimeline()
        self.video_player = _FakeVideoPlayer()
        self.logs: list[str] = []
        self.input_mode = ""
        self.video_state = ""
        self.court_overlay = None
        self.shown_frame = None

    def set_model_settings(self, pose_path: str, track_path: str) -> None:
        self.model_settings = (pose_path, track_path)

    def set_model_switches(self, pose_enabled: bool, track_enabled: bool) -> None:
        self.model_switches = (pose_enabled, track_enabled)

    def set_debug_csv_enabled(self, enabled: bool) -> None:
        self.debug_csv_enabled = enabled

    def set_report_api_settings(self, settings: dict[str, object]) -> None:
        self.report_api_settings = settings

    def populate_stylesheets(self, theme_dirs: list[object], active_theme_name: str) -> None:
        self.stylesheets = (theme_dirs, active_theme_name)

    def append_log(self, message: str) -> None:
        self.logs.append(message)

    def set_input_mode(self, mode: str) -> None:
        self.input_mode = mode

    def set_video_state(self, state: str) -> None:
        self.video_state = state

    def show_video_frame(self, *args, **kwargs) -> None:
        self.shown_frame = (args, kwargs)

    def update_progress(self, value: int) -> None:
        self.progress = value

    def set_court_overlay(self, payload: object) -> None:
        self.court_overlay = payload


class _FakeCourtService:
    def __init__(self) -> None:
        self.submissions: list[tuple[np.ndarray, int, int]] = []

    def submit_frame(self, frame: np.ndarray, frame_id: int, timestamp_ms: int) -> bool:
        self.submissions.append((frame, frame_id, timestamp_ms))
        return True


def _controller_init_patches():
    return (
        patch.object(MainController, "_bind_court_service", lambda self: None),
        patch.object(MainController, "_bind_events", lambda self: None),
        patch.object(MainController, "_log_missing_model_paths", lambda self: None),
        patch.object(MainController, "_reset_metrics", lambda self: None),
        patch.object(MainController, "_set_idle_state", lambda self: None),
    )


class CameraStartupTest(unittest.TestCase):
    def test_frame_step_seconds_uses_real_timestamp_gap(self) -> None:
        self.assertAlmostEqual(frame_step_seconds(166, 100, 60.0), 0.066)
        self.assertAlmostEqual(frame_step_seconds(100, None, 50.0), 0.02)
        self.assertAlmostEqual(frame_step_seconds(100, 100, 25.0), 0.04)

    def test_track_state_resets_only_after_long_gap(self) -> None:
        self.assertFalse(track_state_gap_exceeded(0.75))
        self.assertTrue(track_state_gap_exceeded(0.751))

    def test_controller_init_does_not_probe_cameras(self) -> None:
        view = _FakeView()
        patches = _controller_init_patches()
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            with patch.object(MainController, "_refresh_camera_devices") as refresh:
                MainController(view)

        refresh.assert_not_called()

    def test_camera_mode_probes_cameras_on_demand(self) -> None:
        view = _FakeView()
        patches = _controller_init_patches()
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            controller = MainController(view)

        with (
            patch.object(controller, "_stop_workers", lambda *, clear_pending_seek: None),
            patch.object(controller, "_reset_metrics", lambda: None),
            patch.object(controller, "_set_idle_state", lambda: None),
            patch.object(controller, "_refresh_camera_devices") as refresh,
        ):
            controller.handle_input_mode("camera")

        refresh.assert_called_once_with(log=True)

    def test_video_probe_submits_preview_frame_for_automatic_court_calibration(self) -> None:
        view = _FakeView()
        court_service = _FakeCourtService()
        controller = object.__new__(MainController)
        controller.view = view
        controller._court_service = court_service
        controller._selected_video_path = "sample.mp4"
        controller._probe_worker = None
        controller._video_meta = {}
        controller._latest_court_prediction_dict = lambda: None
        controller._reset_court_detection = lambda *, request_initial_prediction: setattr(
            controller,
            "court_prediction_requested",
            request_initial_prediction,
        )
        controller._set_idle_state = lambda: None
        controller._release_probe_worker = lambda worker: None
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        payload = {
            "fps": 25.0,
            "width": 64,
            "height": 48,
            "duration_ms": 1000,
            "position_ms": 40,
            "frame_id": 1,
            "image": object(),
            "court_frame": frame,
        }

        controller._on_probe_finished("sample.mp4", payload)

        self.assertTrue(controller.court_prediction_requested)
        self.assertEqual(len(court_service.submissions), 1)
        self.assertIs(court_service.submissions[0][0], frame)
        self.assertEqual(court_service.submissions[0][1:], (1, 40))
        self.assertNotIn("court_frame", controller._video_meta)

    def test_automatic_court_result_updates_preview_overlay(self) -> None:
        view = _FakeView()
        controller = object.__new__(MainController)
        controller.view = view
        controller._last_court_log_frame = -1
        payload = {
            "frame_id": 3,
            "valid": True,
            "updated": True,
            "confidence": 0.94,
            "detect_ms": 800.0,
            "scheme": "court_pose_white_line",
            "corners": [[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]],
        }

        controller._on_court_prediction_ready(payload)

        self.assertIs(view.court_overlay, payload)
        self.assertTrue(any("自动标定完成" in message for message in view.logs))


if __name__ == "__main__":
    unittest.main()
