from __future__ import annotations

import unittest

import numpy as np

from src.models.mmpose_backend import MMPoseInferenceItem
from src.models.yolo_pose_backend import (
    expanded_crop_rect,
    extract_image_to_court_h,
    filter_boxes_by_court,
    needs_crop_refine,
    translate_item,
)


class YoloPoseBackendHelpersTest(unittest.TestCase):
    def test_expanded_crop_rect_clamps_to_frame(self) -> None:
        rect = expanded_crop_rect([5.0, 10.0, 25.0, 70.0], (80, 100, 3), 0.5)

        self.assertEqual(rect, (0, 0, 48, 80))

    def test_translate_item_maps_crop_pose_to_original_image(self) -> None:
        item = MMPoseInferenceItem(
            bbox=[10.0, 20.0, 50.0, 80.0],
            keypoints=[[12.0, 24.0], [40.0, 70.0]],
            scores=[0.8, 0.9],
        )

        translated = translate_item(item, 100.0, 200.0)

        self.assertEqual(translated.bbox, [110.0, 220.0, 150.0, 280.0])
        self.assertEqual(translated.keypoints, [[112.0, 224.0], [140.0, 270.0]])
        self.assertEqual(translated.scores, [0.8, 0.9])

    def test_filter_boxes_by_court_keeps_only_footpoints_inside_margin(self) -> None:
        image_to_court_h = np.eye(3, dtype=np.float64)
        boxes = [
            [100.0, 100.0, 180.0, 300.0],
            [-120.0, 100.0, -80.0, 300.0],
            [250.0, 1300.0, 320.0, 1420.0],
        ]

        keep = filter_boxes_by_court(boxes, image_to_court_h, margin=30.0)

        self.assertEqual(keep, [0])

    def test_extract_image_to_court_h_requires_valid_prediction(self) -> None:
        valid = {
            "valid": True,
            "image_to_court_h": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        }

        self.assertIsNotNone(extract_image_to_court_h(valid))
        self.assertIsNone(extract_image_to_court_h({"valid": False, "image_to_court_h": valid["image_to_court_h"]}))

    def test_needs_crop_refine_only_for_low_quality_keypoints(self) -> None:
        self.assertFalse(needs_crop_refine([0.8] * 13 + [0.4] * 4, score_thr=0.65, min_strong_keypoints=10))
        self.assertTrue(needs_crop_refine([0.8] * 6 + [0.2] * 11, score_thr=0.65, min_strong_keypoints=10))
        self.assertTrue(needs_crop_refine([], score_thr=0.65, min_strong_keypoints=10))


if __name__ == "__main__":
    unittest.main()
