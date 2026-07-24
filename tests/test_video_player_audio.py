from __future__ import annotations

import importlib.util
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

if importlib.util.find_spec("PyQt6") is None:
    raise unittest.SkipTest("PyQt6 is not installed in this test environment.")

from PyQt6.QtCore import QUrl
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtWidgets import QApplication

from apps.pyqt6.views.components.video_player_panel_runtime import VideoPlayerWidget


class _FakeMediaPlayer:
    def __init__(self) -> None:
        self._source = QUrl()
        self._position = 0
        self._state = QMediaPlayer.PlaybackState.StoppedState
        self.play_count = 0
        self.pause_count = 0
        self.stop_count = 0
        self.position_history: list[int] = []

    def source(self) -> QUrl:
        return self._source

    def setSource(self, source: QUrl) -> None:
        self._source = source

    def position(self) -> int:
        return self._position

    def setPosition(self, position_ms: int) -> None:
        self._position = int(position_ms)
        self.position_history.append(self._position)

    def playbackState(self) -> QMediaPlayer.PlaybackState:
        return self._state

    def play(self) -> None:
        self.play_count += 1
        self._state = QMediaPlayer.PlaybackState.PlayingState

    def pause(self) -> None:
        self.pause_count += 1
        self._state = QMediaPlayer.PlaybackState.PausedState

    def stop(self) -> None:
        self.stop_count += 1
        self._state = QMediaPlayer.PlaybackState.StoppedState


class VideoPlayerAudioTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.widget = VideoPlayerWidget()
        self.fake_player = _FakeMediaPlayer()

    def tearDown(self) -> None:
        self.widget.close()
        self.widget.deleteLater()
        self.app.processEvents()

    def test_local_video_playback_controls_audio_source_and_start_position(self) -> None:
        self.widget.set_video_path("D:/media/rally.mp4")
        self.widget._media_player = self.fake_player

        self.widget.play(start_ms=1250)

        self.assertEqual(
            self.fake_player.source().toLocalFile().replace("\\", "/"),
            "D:/media/rally.mp4",
        )
        self.assertEqual(self.fake_player.position_history, [1250])
        self.assertEqual(self.fake_player.play_count, 1)

        self.fake_player._position = 0
        pending_during_seek: list[int | None] = []
        set_position = self.fake_player.setPosition

        def track_pending_state(position_ms: int) -> None:
            pending_during_seek.append(self.widget._pending_audio_position_ms)
            set_position(position_ms)

        self.fake_player.setPosition = track_pending_state
        self.widget._on_audio_media_status_changed(QMediaPlayer.MediaStatus.BufferedMedia)
        self.assertEqual(self.fake_player.position_history, [1250, 1250])
        self.assertEqual(pending_during_seek, [None])
        self.assertIsNone(self.widget._pending_audio_position_ms)

        self.widget.pause()
        self.assertEqual(self.fake_player.pause_count, 1)
        self.widget.stop()
        self.assertEqual(self.fake_player.pause_count, 2)
        self.assertEqual(self.fake_player.stop_count, 0)

        self.widget.set_live_source("camera")
        self.assertIsNone(self.widget._media_player)
        self.assertEqual(self.widget._audio_source_path, "")
        self.widget.play()
        self.assertEqual(self.fake_player.play_count, 1)

    def test_audio_position_is_corrected_only_for_material_drift(self) -> None:
        self.widget.set_video_path("D:/media/rally.mp4")
        self.widget._media_player = self.fake_player
        self.widget.play(start_ms=100)

        self.widget.sync_audio_position(600)
        self.assertEqual(self.fake_player.position_history, [100, 600])

        self.widget.sync_audio_position(1000)
        self.assertEqual(self.fake_player.position_history, [100, 600])

        self.widget._last_audio_correction_at = 0.0
        self.widget.sync_audio_position(800)
        self.assertEqual(self.fake_player.position_history, [100, 600])

        self.widget.sync_audio_position(1000)
        self.assertEqual(self.fake_player.position_history, [100, 600, 1000])

        self.widget.pause()
        self.widget._last_audio_correction_at = 0.0
        self.widget.sync_audio_position(2000)
        self.assertEqual(self.fake_player.position_history, [100, 600, 1000])

    def test_lazy_media_backend_binds_audio_output_and_video_sink(self) -> None:
        player = self.widget._ensure_audio_player()

        self.assertIs(player.audioOutput(), self.widget._audio_output)
        self.assertIs(player.videoSink(), self.widget._audio_video_sink)
        self.assertFalse(self.widget._audio_output.isMuted())
        self.assertAlmostEqual(self.widget._audio_output.volume(), 1.0)


if __name__ == "__main__":
    unittest.main()
