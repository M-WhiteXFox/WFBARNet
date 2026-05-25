from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import unittest

import cv2
import numpy as np

from src.court import (
    CourtLineDetector,
    MonoTrackCourtLineConfig,
    MonoTrackCourtLineDetector,
    ShuttleCourtSegConfig,
    ShuttleCourtSegLineDetector,
    create_court_line_detector,
    predict_court_lines,
)
from src.court import opencv_court_homography_core as court_core
from src.court.opencv_court_detector import (
    CourtLineOverlayRenderer,
    CourtLinePrediction,
    OpenCVCourtLineConfig,
    OpenCVCourtLineDetector,
    draw_court_prediction,
)


def _detection_from_corners(corners: list[list[float]], confidence: float = 0.0) -> court_core.CourtLineDetection:
    corner_array = np.asarray(corners, dtype=np.float32)
    court_to_image_h, image_to_court_h = court_core.compute_homographies(corner_array)
    if court_to_image_h is None or image_to_court_h is None:
        raise AssertionError("test corners should produce a valid homography")
    keypoint_names = ["outer_tl", "outer_tr", "outer_br", "outer_bl", "center_top", "center_bottom"]
    return court_core.CourtLineDetection(
        corners=corner_array,
        keypoints=corner_array.copy(),
        keypoint_names=keypoint_names,
        court_to_image_h=court_to_image_h,
        image_to_court_h=image_to_court_h,
        confidence=confidence,
        components={},
        line_count=24,
        merged_line_count=8,
        intersection_count=24,
        supported_keypoints=len(keypoint_names),
        avg_line_length=320.0,
        mask_support=0.5,
        green_side_support=0.7,
        snap_points=24,
        snap_mean_shift=6.0,
        scheme="6",
        reason="candidate",
        projected_lines=court_core.project_template_lines(court_to_image_h),
        debug_segments=[],
        debug_merged_lines=[],
    )


class _FakeMasks:
    def __init__(self, polygons: list[np.ndarray]) -> None:
        self.xy = polygons


class _FakeBoxes:
    def __init__(self, confidences: list[float], classes: list[int]) -> None:
        self.conf = np.asarray(confidences, dtype=np.float32)
        self.cls = np.asarray(classes, dtype=np.float32)


class _FakeSegResult:
    def __init__(self, polygons: list[np.ndarray], confidences: list[float]) -> None:
        self.masks = _FakeMasks(polygons)
        self.boxes = _FakeBoxes(confidences, [0 for _ in polygons])


class _FakeSegModel:
    def __init__(self, result: _FakeSegResult) -> None:
        self.result = result
        self.last_kwargs: dict | None = None

    def predict(self, frame: np.ndarray, **kwargs: object) -> list[_FakeSegResult]:
        self.last_kwargs = kwargs
        return [self.result]


class OpenCVCourtLineDetectorTest(unittest.TestCase):
    def test_detector_factory_returns_interface_compatible_detector(self) -> None:
        detector = create_court_line_detector()

        self.assertIsInstance(detector, CourtLineDetector)
        self.assertIsInstance(detector, ShuttleCourtSegLineDetector)

    def test_opencv_backend_still_returns_opencv_detector(self) -> None:
        detector = create_court_line_detector(backend="opencv")

        self.assertIsInstance(detector, CourtLineDetector)
        self.assertIsInstance(detector, OpenCVCourtLineDetector)

    def test_predict_court_lines_module_api(self) -> None:
        result = predict_court_lines(
            np.zeros((120, 160, 3), dtype=np.uint8),
            frame_id=2,
            timestamp_ms=80,
            backend="opencv",
        )

        self.assertEqual(result.frame_id, 2)
        self.assertEqual(result.timestamp_ms, 80)
        self.assertTrue(result.attempted)

    def test_shuttlecourt_segment_detector_builds_prediction_from_mask(self) -> None:
        frame = np.zeros((200, 300, 3), dtype=np.uint8)
        polygon = np.asarray(
            [
                [54.0, 36.0],
                [240.0, 32.0],
                [262.0, 164.0],
                [48.0, 172.0],
            ],
            dtype=np.float32,
        )
        model = _FakeSegModel(_FakeSegResult([polygon], [0.91]))
        detector = ShuttleCourtSegLineDetector(
            ShuttleCourtSegConfig(device="cpu", min_mask_area_ratio=0.001),
            model=model,
        )

        result = detector.predict(frame, frame_id=7, timestamp_ms=280, force=True)

        self.assertTrue(result.valid, result.to_dict())
        self.assertTrue(result.updated)
        self.assertEqual(result.scheme, "shuttlecourt_seg")
        self.assertEqual(len(result.corners), 4)
        self.assertIn("doubles_outer", result.projected_lines)
        self.assertEqual(result.metrics.get("components", {}).get("class_id"), 0.0)
        self.assertEqual(model.last_kwargs["imgsz"], 416)

    def test_shuttlecourt_segment_detector_prefers_main_court_candidate(self) -> None:
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        upper_large_candidate = np.asarray(
            [
                [12.0, 18.0],
                [628.0, 24.0],
                [620.0, 190.0],
                [20.0, 176.0],
            ],
            dtype=np.float32,
        )
        centered_main_court = np.asarray(
            [
                [92.0, 126.0],
                [548.0, 110.0],
                [576.0, 324.0],
                [66.0, 338.0],
            ],
            dtype=np.float32,
        )
        model = _FakeSegModel(_FakeSegResult([upper_large_candidate, centered_main_court], [0.88, 0.70]))
        detector = ShuttleCourtSegLineDetector(
            ShuttleCourtSegConfig(device="cpu", min_mask_area_ratio=0.001),
            model=model,
        )

        result = detector.predict(frame, frame_id=3, timestamp_ms=120, force=True)
        components = result.metrics.get("components", {})

        self.assertTrue(result.valid, result.to_dict())
        self.assertEqual(components.get("candidate_index"), 1.0)
        self.assertGreater(components.get("seg_center", 0.0), 0.95)
        self.assertAlmostEqual(result.corners[0][0], 92.0, delta=2.0)
        self.assertAlmostEqual(result.corners[0][1], 126.0, delta=2.0)

    def test_shuttlecourt_segment_detector_rejects_small_fragment_without_white_lines(self) -> None:
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        small_fragment = np.asarray(
            [
                [270.0, 200.0],
                [390.0, 202.0],
                [392.0, 286.0],
                [268.0, 284.0],
            ],
            dtype=np.float32,
        )
        model = _FakeSegModel(_FakeSegResult([small_fragment], [0.99]))
        detector = ShuttleCourtSegLineDetector(
            ShuttleCourtSegConfig(device="cpu", min_mask_area_ratio=0.001),
            model=model,
        )

        result = detector.predict(frame, frame_id=5, timestamp_ms=200, force=True)

        self.assertFalse(result.valid, result.to_dict())
        self.assertLess(result.candidate_confidence or 0.0, detector.config.medium_conf)
        self.assertIn("too small", result.reason)

    def test_shuttlecourt_segment_detector_ignores_fragment_when_full_court_exists(self) -> None:
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        small_fragment = np.asarray(
            [
                [270.0, 200.0],
                [390.0, 202.0],
                [392.0, 286.0],
                [268.0, 284.0],
            ],
            dtype=np.float32,
        )
        full_court = np.asarray(
            [
                [92.0, 126.0],
                [548.0, 110.0],
                [576.0, 324.0],
                [66.0, 338.0],
            ],
            dtype=np.float32,
        )
        model = _FakeSegModel(_FakeSegResult([small_fragment, full_court], [0.99, 0.88]))
        detector = ShuttleCourtSegLineDetector(
            ShuttleCourtSegConfig(device="cpu", min_mask_area_ratio=0.001),
            model=model,
        )

        result = detector.predict(frame, frame_id=6, timestamp_ms=240, force=True)
        components = result.metrics.get("components", {})

        self.assertTrue(result.valid, result.to_dict())
        self.assertEqual(components.get("candidate_index"), 1.0)
        self.assertGreater(components.get("seg_area_ratio", 0.0), 0.3)

    def test_shuttlecourt_segment_detector_refines_quad_to_white_lines(self) -> None:
        frame = np.full((360, 520, 3), (45, 120, 45), dtype=np.uint8)
        true_corners = np.asarray(
            [
                [130.0, 42.0],
                [390.0, 48.0],
                [448.0, 314.0],
                [80.0, 304.0],
            ],
            dtype=np.float32,
        )
        court_to_image_h, _ = court_core.compute_homographies(true_corners)
        if court_to_image_h is None:
            raise AssertionError("synthetic court should produce a valid homography")
        for name, points in court_core.project_template_lines(court_to_image_h).items():
            line_points = np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame, [line_points], name == "doubles_outer", (245, 245, 245), 5, lineType=cv2.LINE_AA)

        coarse_polygon = true_corners + np.asarray([0.0, 18.0], dtype=np.float32)
        model = _FakeSegModel(_FakeSegResult([coarse_polygon], [0.92]))
        detector = ShuttleCourtSegLineDetector(
            ShuttleCourtSegConfig(device="cpu", min_mask_area_ratio=0.001, snap_response_threshold=0.08),
            model=model,
        )

        result = detector.predict(frame, frame_id=4, timestamp_ms=160, force=True)
        refined = np.asarray(result.corners, dtype=np.float32)
        coarse_error = float(np.mean(np.linalg.norm(coarse_polygon - true_corners, axis=1)))
        refined_error = float(np.mean(np.linalg.norm(refined - true_corners, axis=1)))

        self.assertTrue(result.valid, result.to_dict())
        self.assertGreater(result.metrics.get("snap_points", 0), 10)
        self.assertLess(refined_error, coarse_error)

    def test_shuttlecourt_segment_detector_uses_white_lines_when_segmentation_includes_outside_area(self) -> None:
        frame = np.full((360, 520, 3), (45, 120, 45), dtype=np.uint8)
        true_corners = np.asarray(
            [
                [130.0, 42.0],
                [390.0, 48.0],
                [448.0, 314.0],
                [80.0, 304.0],
            ],
            dtype=np.float32,
        )
        court_to_image_h, _ = court_core.compute_homographies(true_corners)
        if court_to_image_h is None:
            raise AssertionError("synthetic court should produce a valid homography")
        for name, points in court_core.project_template_lines(court_to_image_h).items():
            line_points = np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame, [line_points], name == "doubles_outer", (245, 245, 245), 5, lineType=cv2.LINE_AA)

        oversized_seg = np.asarray(
            [
                [92.0, 18.0],
                [428.0, 20.0],
                [486.0, 340.0],
                [48.0, 334.0],
            ],
            dtype=np.float32,
        )
        model = _FakeSegModel(_FakeSegResult([oversized_seg], [0.94]))
        detector = ShuttleCourtSegLineDetector(
            ShuttleCourtSegConfig(
                device="cpu",
                min_mask_area_ratio=0.001,
                hough_threshold=20,
                snap_response_threshold=0.08,
            ),
            model=model,
        )

        result = detector.predict(frame, frame_id=8, timestamp_ms=320, force=True)
        detected = np.asarray(result.corners, dtype=np.float32)
        seg_error = float(np.mean(np.linalg.norm(oversized_seg - true_corners, axis=1)))
        detected_error = float(np.mean(np.linalg.norm(detected - true_corners, axis=1)))
        components = result.metrics.get("components", {})

        self.assertTrue(result.valid, result.to_dict())
        self.assertEqual(components.get("seg_line_fit"), 1.0)
        self.assertLess(detected_error, seg_error * 0.65)

    def test_blank_frame_returns_stable_payload(self) -> None:
        detector = OpenCVCourtLineDetector()

        result = detector.predict(np.zeros((120, 160, 3), dtype=np.uint8), frame_id=0, timestamp_ms=0)
        payload = result.to_dict()

        self.assertEqual(payload["frame_id"], 0)
        self.assertEqual(payload["timestamp_ms"], 0)
        self.assertEqual(payload["source_size"], [160, 120])
        self.assertTrue(payload["attempted"])
        self.assertFalse(payload["valid"])
        self.assertIn("court_to_image_h", payload)
        self.assertIn("image_to_court_h", payload)
        self.assertIn("projected_lines", payload)

    def test_monotrack_detector_finds_synthetic_court(self) -> None:
        frame = np.full((360, 520, 3), (45, 120, 45), dtype=np.uint8)
        corners = np.asarray(
            [
                [130.0, 40.0],
                [390.0, 48.0],
                [450.0, 315.0],
                [78.0, 305.0],
            ],
            dtype=np.float32,
        )
        court_to_image_h, _ = court_core.compute_homographies(corners)
        if court_to_image_h is None:
            raise AssertionError("synthetic court should produce a valid homography")
        for name, points in court_core.project_template_lines(court_to_image_h).items():
            line_points = np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)
            cv2_closed = name == "doubles_outer"

            cv2.polylines(frame, [line_points], cv2_closed, (245, 245, 245), 5, lineType=cv2.LINE_AA)

        detector = MonoTrackCourtLineDetector(
            MonoTrackCourtLineConfig(
                reliable_conf=0.05,
                medium_conf=0.03,
                hough_threshold=20,
                hough_min_line_length=40,
                max_lines_per_family=3,
                model_sample_step_px=24.0,
            )
        )
        result = detector.predict(frame, frame_id=0, timestamp_ms=0, force=True)

        self.assertTrue(result.valid, result.to_dict())
        self.assertEqual(result.scheme, "monotrack")
        self.assertIn("doubles_outer", result.projected_lines)

    def test_monotrack_detector_finds_real_video_frame(self) -> None:
        video_path = Path(__file__).resolve().parents[1] / "videos" / "set1" / "3fd67078ae9b133dc5bfbca410631643.mp4"
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise AssertionError(f"failed to open test video: {video_path}")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        mid_frame = max(0, frame_count // 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise AssertionError("failed to read middle frame from test video")

        detector = MonoTrackCourtLineDetector()
        result = detector.predict(frame, frame_id=mid_frame, timestamp_ms=0, force=True)

        self.assertTrue(result.valid, result.to_dict())
        self.assertEqual(result.scheme, "monotrack")
        self.assertGreater(result.confidence, 0.9)
        self.assertEqual(result.metrics.get("components", {}).get("monotrack_three_family"), 1.0)

    def test_cached_overlay_matches_direct_drawing(self) -> None:
        prediction = CourtLinePrediction(
            frame_id=1,
            timestamp_ms=40,
            source_size=(120, 80),
            valid=True,
            attempted=True,
            updated=True,
            update_type="reliable update",
            status="reliable update",
            confidence=0.9,
            candidate_confidence=0.9,
            reason="unit test",
            scheme="test",
            corners=[[20.0, 15.0], [100.0, 15.0], [100.0, 65.0], [20.0, 65.0]],
            keypoints=[],
            court_to_image_h=[],
            image_to_court_h=[],
            projected_lines={
                "doubles_outer": [[20.0, 15.0], [100.0, 15.0], [100.0, 65.0], [20.0, 65.0]],
                "center_line": [[60.0, 15.0], [60.0, 65.0]],
                "service_line": [[20.0, 40.0], [100.0, 40.0]],
            },
            metrics={},
            detect_ms=12.0,
            rejected_count=0,
        )
        frame = np.full((80, 120, 3), 64, dtype=np.uint8)

        direct = draw_court_prediction(frame, prediction)
        cached = CourtLineOverlayRenderer().draw(frame, prediction)

        self.assertLessEqual(np.abs(direct.astype(np.int16) - cached.astype(np.int16)).max(), 2)

    def test_skinny_false_quad_scores_below_medium_confidence(self) -> None:
        args = SimpleNamespace(**asdict(OpenCVCourtLineConfig()))
        detection = _detection_from_corners(
            [
                [630.8, -8.8],
                [668.4, -48.1],
                [1016.8, 528.6],
                [959.1, 531.3],
            ]
        )

        confidence, components, reason = court_core.score_court_detection(
            detection,
            previous=None,
            frame_shape=(576, 1280),
            args=args,
        )

        self.assertEqual(reason, "implausible court shape")
        self.assertLess(components["shape"], 0.55)
        self.assertLess(confidence, args.medium_conf)

    def test_medium_candidate_does_not_initialize_tracking(self) -> None:
        args = SimpleNamespace(**asdict(OpenCVCourtLineConfig()))
        state = court_core.TrackingState()
        detection = _detection_from_corners(
            [
                [250.0, 180.0],
                [1030.0, 180.0],
                [1130.0, 560.0],
                [150.0, 560.0],
            ],
            confidence=0.70,
        )

        court_core.update_tracking_state(state, detection, args, frame_id=10, timestamp=0.33)

        self.assertIsNone(state.current)
        self.assertEqual(state.last_update_type, "rejected")
        self.assertEqual(state.rejected_count, 1)

    def test_three_family_cross_lines_must_be_near_horizontal(self) -> None:
        args = SimpleNamespace(**asdict(OpenCVCourtLineConfig()))

        self.assertTrue(court_core.is_likely_transverse_family(0.0, args))
        self.assertTrue(court_core.is_likely_transverse_family(20.0, args))
        self.assertFalse(court_core.is_likely_transverse_family(60.0, args))
        self.assertFalse(court_core.is_likely_transverse_family(125.0, args))


if __name__ == "__main__":
    unittest.main()
