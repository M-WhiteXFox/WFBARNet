from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from src.court import (
    CourtKeyNetConfig,
    CourtKeyNetLineDetector,
    CourtPoseConfig,
    CourtPoseLineDetector,
    CourtLineDetector,
    MonoTrackCourtLineConfig,
    MonoTrackCourtLineDetector,
    ShuttleCourtSegConfig,
    ShuttleCourtSegLineDetector,
    create_court_line_detector,
    predict_court_lines,
)
from src.court import opencv_court_homography_core as court_core
from src.court import monotrack_court_detector as monotrack_module
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


class _FakePoseKeypoints:
    def __init__(self, points: np.ndarray, confidences: np.ndarray | None = None) -> None:
        self.xy = np.asarray([points], dtype=np.float32)
        self.conf = (
            np.asarray([confidences], dtype=np.float32)
            if confidences is not None
            else np.ones((1, len(points)), dtype=np.float32)
        )


class _FakePoseResult:
    def __init__(self, points: np.ndarray, confidence: float = 0.95) -> None:
        self.keypoints = _FakePoseKeypoints(points)
        self.boxes = _FakeBoxes([confidence], [0])


class _FakePoseModel:
    def __init__(self, points: np.ndarray) -> None:
        self.result = _FakePoseResult(points)
        self.last_kwargs: dict | None = None

    def predict(self, frame: np.ndarray, **kwargs: object) -> list[_FakePoseResult]:
        self.last_kwargs = kwargs
        return [self.result]


class OpenCVCourtLineDetectorTest(unittest.TestCase):
    def test_outer_support_samples_all_four_closed_edges(self) -> None:
        projected_lines = {
            "doubles_outer": np.asarray(
                [[10.0, 10.0], [90.0, 10.0], [90.0, 70.0], [10.0, 70.0]],
                dtype=np.float32,
            )
        }
        complete = np.zeros((82, 102), dtype=np.uint8)
        cv2.polylines(
            complete,
            [projected_lines["doubles_outer"].astype(np.int32).reshape(-1, 1, 2)],
            True,
            255,
            3,
        )

        complete_components = court_core.projected_outer_support_components(
            projected_lines,
            complete,
        )
        missing_left = complete.copy()
        cv2.rectangle(missing_left, (0, 0), (20, 81), 0, -1)
        missing_components = court_core.projected_outer_support_components(
            projected_lines,
            missing_left,
        )

        self.assertGreater(complete_components["outer_min_support"], 0.95)
        self.assertLess(missing_components["outer_left_support"], 0.10)
        self.assertEqual(
            missing_components["outer_min_support"],
            missing_components["outer_left_support"],
        )

    def test_detector_factory_returns_interface_compatible_detector(self) -> None:
        detector = create_court_line_detector()

        self.assertIsInstance(detector, CourtLineDetector)
        self.assertIsInstance(detector, ShuttleCourtSegLineDetector)

    def test_opencv_backend_still_returns_opencv_detector(self) -> None:
        detector = create_court_line_detector(backend="opencv")

        self.assertIsInstance(detector, CourtLineDetector)
        self.assertIsInstance(detector, OpenCVCourtLineDetector)

    def test_court_pose_backend_returns_pose_detector(self) -> None:
        detector = create_court_line_detector(
            backend="court_pose",
            config=CourtPoseConfig(device="cpu"),
        )

        self.assertIsInstance(detector, CourtLineDetector)
        self.assertIsInstance(detector, CourtPoseLineDetector)

    def test_courtkeynet_backend_returns_native_detector(self) -> None:
        config = CourtKeyNetConfig(device="cpu")

        detector = create_court_line_detector(backend="courtkeynet", config=config)

        self.assertIsInstance(detector, CourtLineDetector)
        self.assertIsInstance(detector, CourtKeyNetLineDetector)
        self.assertIs(detector.config, config)

    def test_courtkeynet_backend_rejects_wrong_config_type(self) -> None:
        with self.assertRaisesRegex(TypeError, "CourtKeyNet detector requires"):
            create_court_line_detector(
                backend="courtkeynet",
                config=CourtPoseConfig(device="cpu"),
            )

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
        self.assertGreater(
            result.metrics.get("components", {}).get("singles_min_support", 0.0),
            0.15,
        )
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

    def test_court_pose_detector_refines_coarse_surface_boundary_to_white_lines(self) -> None:
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

        coarse_corners = true_corners + np.asarray(
            [[-28.0, -20.0], [28.0, -20.0], [28.0, 20.0], [-28.0, 20.0]],
            dtype=np.float32,
        )
        model = _FakePoseModel(coarse_corners)
        detector = CourtPoseLineDetector(
            CourtPoseConfig(
                device="cpu",
                min_mask_area_ratio=0.001,
                hough_threshold=20,
                seg_line_min_area_ratio=0.30,
                reliable_conf=0.10,
                medium_conf=0.05,
                snap_response_threshold=0.08,
            ),
            model=model,
        )

        result = detector.predict(frame, frame_id=9, timestamp_ms=360, force=True)
        self.assertTrue(result.valid, result.to_dict())
        detected = np.asarray(result.corners, dtype=np.float32)
        coarse_error = float(np.mean(np.linalg.norm(coarse_corners - true_corners, axis=1)))
        detected_error = float(np.mean(np.linalg.norm(detected - true_corners, axis=1)))

        self.assertEqual(result.scheme, "court_pose_white_line", result.to_dict())
        self.assertEqual(result.metrics.get("components", {}).get("pose_white_line_refined"), 1.0)
        self.assertLess(detected_error, coarse_error * 0.65, result.to_dict())
        self.assertEqual(model.last_kwargs["imgsz"], 512)

    def test_court_pose_monotrack_requires_measured_white_line_evidence(self) -> None:
        frame = np.full((360, 520, 3), (45, 120, 45), dtype=np.uint8)
        pose_corners = np.asarray(
            [[130.0, 42.0], [390.0, 48.0], [448.0, 314.0], [80.0, 304.0]],
            dtype=np.float32,
        )
        monotrack = _detection_from_corners(pose_corners.tolist(), confidence=0.96)
        monotrack.scheme = "monotrack"
        monotrack.mask_support = 0.0
        monotrack.snap_points = 24
        detector = CourtPoseLineDetector(
            CourtPoseConfig(
                device="cpu",
                min_mask_area_ratio=0.001,
                reliable_conf=0.01,
                medium_conf=0.005,
            ),
            model=_FakePoseModel(pose_corners),
        )

        with patch(
            "src.court.court_pose_detector.detect_monotrack_court_lines",
            return_value=monotrack,
        ):
            result = detector.predict(frame, frame_id=0, timestamp_ms=0, force=True)

        components = result.metrics.get("components", {})
        self.assertTrue(result.valid, result.to_dict())
        self.assertEqual(result.scheme, "court_pose_coarse", result.to_dict())
        self.assertEqual(components.get("pose_monotrack_fused"), 1.0)
        self.assertEqual(components.get("pose_monotrack_white_line_evidence"), 0.0)

    def test_court_pose_rejects_single_corner_disaster_shift(self) -> None:
        frame = np.full((400, 600, 3), (45, 120, 45), dtype=np.uint8)
        pose_corners = np.asarray(
            [[120.0, 80.0], [480.0, 80.0], [520.0, 320.0], [80.0, 320.0]],
            dtype=np.float32,
        )
        shifted_corners = pose_corners.copy()
        shifted_corners[0] = [30.0, 30.0]
        shifted = _detection_from_corners(shifted_corners.tolist(), confidence=0.96)
        shifted.scheme = "court_pose_white_line"
        shifted.components.update(
            {
                "pose_monotrack_fused": 1.0,
                "pose_monotrack_white_line_evidence": 1.0,
                "outer_min_support": 0.99,
                "singles_min_support": 0.99,
            }
        )
        detector = CourtPoseLineDetector(
            CourtPoseConfig(
                device="cpu",
                min_mask_area_ratio=0.001,
                reliable_conf=0.01,
                medium_conf=0.005,
                max_pose_refine_shift_ratio=0.20,
                max_pose_refine_corner_shift_ratio=0.08,
            ),
            model=_FakePoseModel(pose_corners),
        )

        with patch.object(
            detector,
            "_detection_from_monotrack",
            return_value=(shifted, 4, 20.0),
        ):
            result = detector.predict(frame, frame_id=0, timestamp_ms=0, force=True)

        detected = np.asarray(result.corners, dtype=np.float32)
        components = result.metrics.get("components", {})
        self.assertTrue(result.valid, result.to_dict())
        self.assertEqual(result.scheme, "court_pose_coarse", result.to_dict())
        self.assertLess(float(np.max(np.linalg.norm(detected - pose_corners, axis=1))), 1.0)
        self.assertEqual(components.get("pose_refine_shift_rejected"), 1.0)
        self.assertNotEqual(components.get("outer_min_support"), 0.99)
        self.assertNotEqual(components.get("singles_min_support"), 0.99)

    def test_monotrack_applies_pose_roi_mask_before_hough(self) -> None:
        frame = np.zeros((80, 120, 3), dtype=np.uint8)
        roi_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        roi_mask[20:60, 30:90] = 255
        full_line_mask = np.full(frame.shape[:2], 255, dtype=np.uint8)
        captured: dict[str, np.ndarray] = {}

        def capture_mask(mask: np.ndarray, _args: SimpleNamespace) -> list[court_core.LineSegment]:
            captured["mask"] = mask.copy()
            return []

        args = SimpleNamespace(**asdict(MonoTrackCourtLineConfig(detect_max_width=960)))
        with patch.object(monotrack_module, "create_monotrack_line_mask", return_value=full_line_mask), patch.object(
            monotrack_module,
            "detect_monotrack_hough_segments",
            side_effect=capture_mask,
        ):
            result = monotrack_module.detect_monotrack_court_lines(
                frame,
                previous=None,
                args=args,
                roi_mask=roi_mask,
            )

        self.assertIsNone(result)
        self.assertEqual(cv2.countNonZero(captured["mask"]), cv2.countNonZero(roi_mask))
        self.assertEqual(int(captured["mask"][10, 10]), 0)
        self.assertEqual(int(captured["mask"][40, 60]), 255)

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
        self.assertGreater(
            result.metrics.get("components", {}).get("singles_min_support", 0.0),
            0.15,
        )

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

    def test_consistent_coarse_candidates_initialize_after_confirmation(self) -> None:
        args = SimpleNamespace(
            **asdict(
                CourtPoseConfig(
                    coarse_startup_confirm_frames=3,
                    coarse_startup_max_corner_shift_ratio=0.03,
                )
            )
        )
        state = court_core.TrackingState()
        base_corners = np.asarray(
            [[250.0, 180.0], [1030.0, 180.0], [1130.0, 560.0], [150.0, 560.0]],
            dtype=np.float32,
        )

        for index, offset in enumerate((0.0, 3.0, -2.0)):
            candidate = _detection_from_corners((base_corners + [offset, 0.0]).tolist(), confidence=0.60)
            candidate.scheme = "court_pose_coarse"
            court_core.update_tracking_state(state, candidate, args, frame_id=index, timestamp=index / 25.0)
            if index < 2:
                self.assertIsNone(state.current)

        self.assertIsNotNone(state.current)
        self.assertEqual(state.last_update_type, "medium startup")
        self.assertEqual(state.current.scheme, "court_pose_coarse")

    def test_inconsistent_coarse_candidate_restarts_confirmation(self) -> None:
        args = SimpleNamespace(
            **asdict(
                CourtPoseConfig(
                    coarse_startup_confirm_frames=3,
                    coarse_startup_max_corner_shift_ratio=0.02,
                )
            )
        )
        state = court_core.TrackingState()
        first = _detection_from_corners(
            [[250.0, 180.0], [1030.0, 180.0], [1130.0, 560.0], [150.0, 560.0]],
            confidence=0.60,
        )
        first.scheme = "court_pose_coarse"
        jump = _detection_from_corners(
            [[350.0, 180.0], [1130.0, 180.0], [1230.0, 560.0], [250.0, 560.0]],
            confidence=0.60,
        )
        jump.scheme = "court_pose_coarse"

        court_core.update_tracking_state(state, first, args, frame_id=0, timestamp=0.0)
        court_core.update_tracking_state(state, first, args, frame_id=1, timestamp=0.04)
        court_core.update_tracking_state(state, jump, args, frame_id=2, timestamp=0.08)

        self.assertIsNone(state.current)
        self.assertEqual(state.startup_candidate_count, 1)
        self.assertEqual(state.last_update_type, "coarse confirmation 1/3")

    def test_precise_court_pose_state_rejects_coarse_quality_downgrade(self) -> None:
        precise = _detection_from_corners(
            [[120.0, 70.0], [480.0, 70.0], [520.0, 330.0], [80.0, 330.0]],
            confidence=0.96,
        )
        precise.scheme = "court_pose_white_line"
        coarse = _detection_from_corners(
            [[145.0, 55.0], [455.0, 55.0], [490.0, 345.0], [110.0, 345.0]],
            confidence=0.58,
        )
        coarse.scheme = "court_pose_coarse"
        args = SimpleNamespace(**asdict(CourtPoseConfig()))
        state = court_core.TrackingState(current=precise)

        court_core.update_tracking_state(state, coarse, args, frame_id=12, timestamp=0.48)

        self.assertIs(state.current, precise)
        self.assertEqual(state.last_update_type, "quality downgrade rejected")
        self.assertEqual(state.rejected_count, 1)

    def test_court_pose_white_line_upgrade_replaces_coarse_state_without_blending(self) -> None:
        coarse = _detection_from_corners(
            [[150.0, 55.0], [450.0, 55.0], [485.0, 345.0], [115.0, 345.0]],
            confidence=0.58,
        )
        coarse.scheme = "court_pose_coarse"
        precise = _detection_from_corners(
            [[120.0, 70.0], [480.0, 70.0], [520.0, 330.0], [80.0, 330.0]],
            confidence=0.96,
        )
        precise.scheme = "court_pose_white_line"
        args = SimpleNamespace(**asdict(CourtPoseConfig()))
        state = court_core.TrackingState(current=coarse)

        court_core.update_tracking_state(state, precise, args, frame_id=12, timestamp=0.48)

        self.assertIsNotNone(state.current)
        np.testing.assert_allclose(state.current.corners, precise.corners, atol=1.0e-5)
        self.assertEqual(state.current.scheme, "court_pose_white_line")

    def test_court_pose_coarse_state_uses_short_redetection_interval(self) -> None:
        coarse = _detection_from_corners(
            [[120.0, 70.0], [480.0, 70.0], [520.0, 330.0], [80.0, 330.0]],
            confidence=0.58,
        )
        coarse.scheme = "court_pose_coarse"
        state = court_core.TrackingState(
            current=coarse,
            last_attempt_frame=0,
            last_attempt_time=0.0,
        )
        args = SimpleNamespace(**asdict(CourtPoseConfig()))

        self.assertFalse(court_core.should_redetect(state, frame_id=30, timestamp=0.50, args=args))
        self.assertTrue(court_core.should_redetect(state, frame_id=45, timestamp=0.76, args=args))

    def test_three_family_cross_lines_must_be_near_horizontal(self) -> None:
        args = SimpleNamespace(**asdict(OpenCVCourtLineConfig()))

        self.assertTrue(court_core.is_likely_transverse_family(0.0, args))
        self.assertTrue(court_core.is_likely_transverse_family(20.0, args))
        self.assertFalse(court_core.is_likely_transverse_family(60.0, args))
        self.assertFalse(court_core.is_likely_transverse_family(125.0, args))


if __name__ == "__main__":
    unittest.main()
