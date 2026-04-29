from __future__ import annotations

import unittest

import numpy as np

from src.postprocess.track import (
    decode_track_heatmap,
    decode_track_heatmap_batch,
    decode_track_heatmap_candidates,
)
from src.preprocess.track import TrackPreprocessMeta


class TrackPostprocessTest(unittest.TestCase):
    def test_decode_prefers_strong_peak_over_larger_low_confidence_blob(self) -> None:
        heatmaps = np.zeros((1, 2, 10, 10), dtype=np.float32)
        heatmaps[0, 1, 1:4, 1:4] = 0.60
        heatmaps[0, 1, 7, 8] = 0.95
        meta = TrackPreprocessMeta(
            orig_size=(20, 30),
            resized_size=(10, 10),
            scale_x=2.0,
            scale_y=3.0,
        )

        result = decode_track_heatmap(heatmaps, meta, score_thr=0.5)

        self.assertTrue(result.visible)
        self.assertAlmostEqual(result.ball_xy[0], 16.0)
        self.assertAlmostEqual(result.ball_xy[1], 21.0)
        self.assertAlmostEqual(result.score, 0.95, places=5)

    def test_decode_accepts_single_heatmap_plane(self) -> None:
        heatmap = np.zeros((10, 10), dtype=np.float32)
        heatmap[5, 4] = 0.90
        meta = TrackPreprocessMeta(
            orig_size=(20, 30),
            resized_size=(10, 10),
            scale_x=2.0,
            scale_y=3.0,
        )

        result = decode_track_heatmap(heatmap, meta, score_thr=0.5)

        self.assertTrue(result.visible)
        self.assertAlmostEqual(result.ball_xy[0], 8.0)
        self.assertAlmostEqual(result.ball_xy[1], 15.0)
        self.assertAlmostEqual(result.score, 0.90, places=5)

    def test_decode_candidates_returns_multiple_ranked_components(self) -> None:
        heatmap = np.zeros((10, 10), dtype=np.float32)
        heatmap[2, 3] = 0.80
        heatmap[7, 8] = 0.65
        meta = TrackPreprocessMeta(
            orig_size=(20, 30),
            resized_size=(10, 10),
            scale_x=2.0,
            scale_y=3.0,
        )

        results = decode_track_heatmap_candidates(
            heatmap,
            meta,
            score_thr=0.7,
            candidate_score_thr=0.5,
            max_candidates=4,
        )

        self.assertEqual(len(results), 2)
        self.assertAlmostEqual(results[0].ball_xy[0], 6.0)
        self.assertAlmostEqual(results[0].ball_xy[1], 6.0)
        self.assertAlmostEqual(results[1].ball_xy[0], 16.0)
        self.assertAlmostEqual(results[1].ball_xy[1], 21.0)
        self.assertGreater(results[0].score, results[1].score)

    def test_batch_decode_accepts_heatmap_planes(self) -> None:
        heatmaps = np.zeros((2, 10, 10), dtype=np.float32)
        heatmaps[0, 2, 3] = 0.80
        heatmaps[1, 6, 7] = 0.85
        metas = [
            TrackPreprocessMeta(orig_size=(20, 30), resized_size=(10, 10), scale_x=2.0, scale_y=3.0),
            TrackPreprocessMeta(orig_size=(10, 10), resized_size=(10, 10), scale_x=1.0, scale_y=1.0),
        ]

        results = decode_track_heatmap_batch(heatmaps, metas, score_thr=0.5)

        self.assertEqual(len(results), 2)
        self.assertAlmostEqual(results[0].ball_xy[0], 6.0)
        self.assertAlmostEqual(results[0].ball_xy[1], 6.0)
        self.assertAlmostEqual(results[1].ball_xy[0], 7.0)
        self.assertAlmostEqual(results[1].ball_xy[1], 6.0)


if __name__ == "__main__":
    unittest.main()
