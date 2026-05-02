import unittest

from src.postprocess.player_distance import PlayerDistanceAccumulator


class PlayerDistanceAccumulatorTest(unittest.TestCase):
    def test_accumulates_top_and_bottom_distances_in_meters(self) -> None:
        accumulator = PlayerDistanceAccumulator(min_step_cm=0.0, max_step_cm=100.0)
        accumulator.update({0: (10.0, 10.0), 1: (100.0, 100.0)})
        totals = accumulator.update({0: (13.0, 14.0), 1: (100.0, 110.0)})

        self.assertAlmostEqual(totals["top"], 0.05, places=6)
        self.assertAlmostEqual(totals["bottom"], 0.10, places=6)

    def test_ignores_obvious_projection_jumps(self) -> None:
        accumulator = PlayerDistanceAccumulator(min_step_cm=0.0, max_step_cm=50.0)
        accumulator.update({0: (10.0, 10.0)})
        totals = accumulator.update({0: (200.0, 200.0)})

        self.assertAlmostEqual(totals["top"], 0.0, places=6)

    def test_resets_tracking_point_when_player_is_missing(self) -> None:
        accumulator = PlayerDistanceAccumulator(min_step_cm=0.0, max_step_cm=100.0)
        accumulator.update({0: (10.0, 10.0)})
        accumulator.update({})
        totals = accumulator.update({0: (20.0, 10.0)})

        self.assertAlmostEqual(totals["top"], 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
