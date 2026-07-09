from __future__ import annotations

import unittest

from apps.pyqt6.controllers.analysis_controller_runtime import project_ball_visual_intersection
from src.utils.structures import TrackResult


class BallVisualIntersectionTest(unittest.TestCase):
    def test_visible_ball_maps_image_point_to_court_plane(self) -> None:
        court_prediction = {
            "valid": True,
            "image_to_court_h": [
                [2.0, 0.0, 1.0],
                [0.0, 3.0, -5.0],
                [0.0, 0.0, 1.0],
            ],
        }
        track = TrackResult(ball_xy=[10.0, 20.0], visible=1, score=0.9)

        self.assertEqual(
            project_ball_visual_intersection(track, court_prediction),
            (21.0, 55.0),
        )

    def test_invisible_ball_has_no_visual_intersection(self) -> None:
        court_prediction = {
            "valid": True,
            "image_to_court_h": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        }
        track = TrackResult(ball_xy=[10.0, 20.0], visible=0, score=0.0)

        self.assertIsNone(project_ball_visual_intersection(track, court_prediction))

    def test_invalid_court_has_no_visual_intersection(self) -> None:
        track = TrackResult(ball_xy=[10.0, 20.0], visible=1, score=0.9)

        self.assertIsNone(
            project_ball_visual_intersection(
                track,
                {
                    "valid": False,
                    "image_to_court_h": [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                },
            )
        )


if __name__ == "__main__":
    unittest.main()
