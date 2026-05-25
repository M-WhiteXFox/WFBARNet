import unittest

from src.postprocess.rally_stats import RallyStatsAccumulator


class RallyStatsAccumulatorTest(unittest.TestCase):
    def test_accumulates_player_motion_and_start_stop_counts(self) -> None:
        stats = RallyStatsAccumulator(start_speed_mps=1.0, stop_speed_mps=0.2, min_step_cm=0.0)

        stats.update_frame(
            timestamp_ms=0,
            player_points={0: (100.0, 100.0), 1: (300.0, 1200.0)},
            ball_visible=True,
            ball_score=0.8,
            court_valid=True,
        )
        stats.update_frame(
            timestamp_ms=1000,
            player_points={0: (250.0, 100.0), 1: (300.0, 1200.0)},
            ball_visible=True,
            ball_score=0.6,
            court_valid=True,
        )
        stats.update_frame(
            timestamp_ms=2000,
            player_points={0: (250.0, 100.0), 1: (300.0, 1200.0)},
            ball_visible=False,
            ball_score=0.0,
            court_valid=False,
        )

        top = stats.summary()["players"]["top"]
        self.assertAlmostEqual(top["distance_m"], 1.5, places=6)
        self.assertAlmostEqual(top["avg_speed_mps"], 0.75, places=6)
        self.assertEqual(top["start_count"], 1)
        self.assertEqual(top["stop_count"], 1)

    def test_merges_trajectory_and_bst_hit_by_frame(self) -> None:
        stats = RallyStatsAccumulator()
        stats.update_frame(
            timestamp_ms=100,
            player_points={0: (305.0, 550.0)},
            ball_visible=True,
            ball_score=0.8,
            court_valid=True,
        )
        stats.add_trajectory_event(
            {
                "event_type": "hit",
                "frame_id": 12,
                "timestamp_ms": 480,
                "confidence": 0.7,
            },
            ball_court_xy=(300.0, 560.0),
        )
        stats.add_bst_prediction(
            {
                "event_frame_id": 12,
                "timestamp_ms": 480,
                "pred_display_name": "Top_杀球",
                "confidence": 0.9,
                "used_homography": True,
            }
        )
        stats.add_bst_prediction(
            {
                "event_frame_id": 12,
                "timestamp_ms": 480,
                "pred_display_name": "Top_杀球",
                "confidence": 0.9,
                "used_homography": True,
            }
        )

        summary = stats.summary()
        self.assertEqual(summary["rally_hit_count"], 1)
        self.assertEqual(summary["stroke_distribution"], {"杀球": 1})
        self.assertEqual(summary["players"]["top"]["hit_count"], 1)
        self.assertEqual(summary["players"]["top"]["zone_hits"]["front"], 1)

    def test_tracks_rally_state_transition(self) -> None:
        stats = RallyStatsAccumulator()
        stats.update_frame(
            timestamp_ms=0,
            player_points={0: (305.0, 550.0)},
            ball_visible=True,
            ball_score=0.8,
            court_valid=True,
        )
        stats.add_trajectory_event(
            {
                "event_type": "hit",
                "frame_id": 1,
                "timestamp_ms": 0,
                "confidence": 0.72,
            }
        )

        summary = stats.summary()
        self.assertEqual(summary["rally_state"], "回合中")
        self.assertEqual(summary["rally_start_ms"], 0)
        self.assertEqual(summary["landing_count"], 0)
        self.assertEqual(summary["out_of_frame_count"], 0)

        stats.add_trajectory_event(
            {
                "event_type": "landing",
                "frame_id": 12,
                "timestamp_ms": 480,
                "confidence": 0.72,
            }
        )

        summary = stats.summary()
        self.assertEqual(summary["rally_state"], "回合结束")
        self.assertEqual(summary["rally_end_ms"], 480)
        self.assertEqual(summary["landing_count"], 1)
        self.assertEqual(summary["out_of_frame_count"], 0)
        self.assertAlmostEqual(summary["rally_duration_s"], 0.48, places=6)

    def test_out_of_frame_tracks_loss_without_ending_rally(self) -> None:
        stats = RallyStatsAccumulator()
        stats.update_frame(
            timestamp_ms=0,
            player_points={0: (305.0, 550.0)},
            ball_visible=True,
            ball_score=0.8,
            court_valid=True,
        )
        stats.add_trajectory_event(
            {
                "event_type": "hit",
                "frame_id": 1,
                "timestamp_ms": 0,
                "confidence": 0.72,
            }
        )
        stats.update_frame(
            timestamp_ms=480,
            player_points={0: (305.0, 550.0)},
            ball_visible=False,
            ball_score=0.0,
            court_valid=True,
        )

        stats.add_trajectory_event(
            {
                "event_type": "out_of_frame",
                "frame_id": 12,
                "timestamp_ms": 480,
                "confidence": 0.45,
            }
        )

        summary = stats.summary()
        self.assertEqual(summary["rally_state"], "回合中")
        self.assertEqual(summary["rally_end_ms"], 480)
        self.assertEqual(summary["landing_count"], 0)
        self.assertEqual(summary["out_of_frame_count"], 1)

    def test_freezes_rally_stats_after_landing_end(self) -> None:
        stats = RallyStatsAccumulator()
        stats.add_trajectory_event(
            {
                "event_type": "hit",
                "frame_id": 1,
                "timestamp_ms": 100,
                "confidence": 0.72,
            }
        )
        stats.add_trajectory_event(
            {
                "event_type": "landing",
                "frame_id": 12,
                "timestamp_ms": 480,
                "confidence": 0.72,
            }
        )
        stats.update_frame(
            timestamp_ms=800,
            player_points={0: (305.0, 550.0)},
            ball_visible=True,
            ball_xy=(320.0, 580.0),
            ball_score=0.8,
            court_valid=True,
        )
        stats.add_trajectory_event(
            {
                "event_type": "hit",
                "frame_id": 20,
                "timestamp_ms": 800,
                "confidence": 0.8,
            }
        )
        stats.add_bst_prediction(
            {
                "event_frame_id": 20,
                "timestamp_ms": 800,
                "pred_display_name": "Top_杀球",
                "confidence": 0.9,
            }
        )

        summary = stats.summary()
        self.assertEqual(summary["rally_state"], "回合结束")
        self.assertEqual(summary["rally_end_ms"], 480)
        self.assertEqual(summary["rally_hit_count"], 1)
        self.assertEqual(summary["frame_count"], 0)

    def test_can_continue_collecting_after_landing_end(self) -> None:
        stats = RallyStatsAccumulator(freeze_after_rally_end=False)
        stats.add_trajectory_event(
            {
                "event_type": "hit",
                "frame_id": 1,
                "timestamp_ms": 100,
                "confidence": 0.72,
            }
        )
        stats.add_trajectory_event(
            {
                "event_type": "landing",
                "frame_id": 12,
                "timestamp_ms": 480,
                "confidence": 0.72,
            }
        )
        stats.update_frame(
            timestamp_ms=800,
            player_points={0: (305.0, 550.0)},
            ball_visible=True,
            ball_xy=(320.0, 580.0),
            ball_score=0.8,
            court_valid=True,
        )
        stats.add_trajectory_event(
            {
                "event_type": "hit",
                "frame_id": 20,
                "timestamp_ms": 800,
                "confidence": 0.8,
            }
        )
        stats.add_bst_prediction(
            {
                "event_frame_id": 20,
                "timestamp_ms": 800,
                "pred_display_name": "Top_杀球",
                "confidence": 0.9,
            }
        )

        summary = stats.summary()
        self.assertEqual(summary["rally_state"], "回合结束")
        self.assertEqual(summary["rally_end_ms"], 480)
        self.assertEqual(summary["rally_hit_count"], 2)
        self.assertEqual(summary["frame_count"], 1)
        self.assertEqual(summary["players"]["top"]["hit_count"], 1)

    def test_does_not_start_from_single_false_visible_ball(self) -> None:
        stats = RallyStatsAccumulator()
        stats.update_frame(
            timestamp_ms=0,
            player_points=None,
            ball_visible=True,
            ball_xy=(1716.0, 427.2),
            ball_score=0.36,
            court_valid=False,
        )
        stats.update_frame(
            timestamp_ms=1000,
            player_points=None,
            ball_visible=False,
            ball_xy=None,
            ball_score=0.0,
            court_valid=False,
        )

        summary = stats.summary()
        self.assertEqual(summary["rally_state"], "未开始")
        self.assertIsNone(summary["rally_start_ms"])
        self.assertIsNone(summary["rally_end_ms"])
        self.assertEqual(summary["rally_duration_s"], 0.0)

    def test_starts_from_stable_visible_ball_flight(self) -> None:
        stats = RallyStatsAccumulator(start_visible_frames=5, start_min_motion_px=30.0)
        samples = [
            (1600, (731.0, 5.4), 0.57),
            (1617, (728.8, 9.3), 0.68),
            (1650, (721.9, 27.6), 0.72),
            (1667, (718.9, 37.8), 0.70),
            (1700, (712.7, 61.0), 0.69),
        ]
        for timestamp_ms, ball_xy, score in samples:
            stats.update_frame(
                timestamp_ms=timestamp_ms,
                player_points=None,
                ball_visible=True,
                ball_xy=ball_xy,
                ball_score=score,
                court_valid=False,
            )

        summary = stats.summary()
        self.assertEqual(summary["rally_state"], "回合中")
        self.assertEqual(summary["rally_start_ms"], 1600)
        self.assertEqual(summary["rally_end_ms"], 1700)
        self.assertAlmostEqual(summary["rally_duration_s"], 0.1, places=6)

    def test_hit_event_starts_rally_when_stable_ball_unavailable(self) -> None:
        stats = RallyStatsAccumulator()
        stats.update_frame(
            timestamp_ms=1000,
            player_points=None,
            ball_visible=False,
            ball_xy=None,
            ball_score=0.0,
            court_valid=False,
        )
        stats.add_trajectory_event(
            {
                "event_type": "hit",
                "frame_id": 20,
                "timestamp_ms": 1200,
                "confidence": 0.85,
            }
        )

        summary = stats.summary()
        self.assertEqual(summary["rally_state"], "回合中")
        self.assertEqual(summary["rally_start_ms"], 1200)
        self.assertEqual(summary["rally_end_ms"], 1200)

    def test_uses_raw_bst_class_for_player_when_display_name_has_no_side(self) -> None:
        stats = RallyStatsAccumulator()
        stats.update_frame(
            timestamp_ms=100,
            player_points={1: (305.0, 790.0)},
            ball_visible=True,
            ball_score=0.8,
            court_valid=True,
        )
        stats.add_bst_prediction(
            {
                "event_frame_id": 20,
                "timestamp_ms": 800,
                "pred_name": "Bottom_杀球",
                "pred_display_name": "杀球",
                "confidence": 0.88,
                "used_homography": True,
            }
        )

        summary = stats.summary()
        hit = stats.details()["hits"][0]
        self.assertEqual(summary["players"]["bottom"]["hit_count"], 1)
        self.assertEqual(summary["players"]["bottom"]["zone_hits"]["front"], 1)
        self.assertEqual(hit["player"], "bottom")
        self.assertEqual(hit["stroke"], "杀球")


if __name__ == "__main__":
    unittest.main()
