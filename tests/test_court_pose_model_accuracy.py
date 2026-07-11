from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
import statistics
import sys
import time
import unittest

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_WEIGHTS = REPO_ROOT / "assets" / "weights" / "court_pose" / "CourtPose.pt"
DEFAULT_VIDEO = REPO_ROOT / "videos" / "MVI_0211.MP4"
DEFAULT_LOG_GLOB = "MVI_0211_*_frame_log.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "court_pose_accuracy"
CORNER_NAMES = ("top_left", "top_right", "bottom_right", "bottom_left")


@dataclass(frozen=True)
class EvaluationConfig:
    weights: Path
    video: Path
    frame_logs: tuple[Path, ...]
    output_dir: Path
    samples: int = 48
    imgsz: int = 512
    confidence: float = 0.25
    device: str = "0"
    preview_count: int = 12
    refine_white_lines: bool = False


def _order_quad_tl_tr_br_bl(corners: np.ndarray) -> np.ndarray:
    points = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    by_y = points[np.argsort(points[:, 1])]
    top = by_y[:2][np.argsort(by_y[:2, 0])]
    bottom = by_y[2:][np.argsort(by_y[2:, 0])]
    return np.asarray([top[0], top[1], bottom[1], bottom[0]], dtype=np.float32)


def _first_manual_corners(log_path: Path) -> np.ndarray | None:
    with log_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            record = json.loads(raw_line)
            court = record.get("court") or {}
            corners = court.get("corners")
            if court.get("valid") and court.get("scheme") == "manual" and isinstance(corners, list) and len(corners) == 4:
                return _order_quad_tl_tr_br_bl(np.asarray(corners, dtype=np.float32))
    return None


def load_manual_reference(frame_logs: tuple[Path, ...]) -> tuple[np.ndarray, dict[str, object]]:
    calibrations: list[np.ndarray] = []
    used_logs: list[str] = []
    for log_path in frame_logs:
        corners = _first_manual_corners(log_path)
        if corners is None:
            continue
        calibrations.append(corners)
        used_logs.append(str(log_path.resolve()))
    if not calibrations:
        raise ValueError("no valid manual court calibration was found in the supplied frame logs")

    stacked = np.stack(calibrations).astype(np.float32)
    reference = np.median(stacked, axis=0).astype(np.float32)
    calibration_errors = np.linalg.norm(stacked - reference[None, :, :], axis=2)
    metadata: dict[str, object] = {
        "calibration_count": len(calibrations),
        "used_logs": used_logs,
        "reference_corners_tl_tr_br_bl": reference.tolist(),
        "calibration_mean_deviation_px": float(calibration_errors.mean()),
        "calibration_p90_deviation_px": float(np.percentile(calibration_errors, 90)),
        "per_corner_calibration_mean_deviation_px": {
            name: float(calibration_errors[:, index].mean()) for index, name in enumerate(CORNER_NAMES)
        },
    }
    return reference, metadata


def _polygon_iou(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float32).reshape(-1, 2)
    second = np.asarray(second, dtype=np.float32).reshape(-1, 2)
    first_area = abs(float(cv2.contourArea(first)))
    second_area = abs(float(cv2.contourArea(second)))
    if first_area <= 1.0 or second_area <= 1.0:
        return 0.0
    try:
        intersection_area, _ = cv2.intersectConvexConvex(first, second)
    except cv2.error:
        return 0.0
    union_area = first_area + second_area - float(intersection_area)
    return float(intersection_area) / union_area if union_area > 0.0 else 0.0


def _extract_best_prediction(result: object) -> tuple[np.ndarray, float] | None:
    keypoints = getattr(result, "keypoints", None)
    boxes = getattr(result, "boxes", None)
    if keypoints is None or boxes is None or len(keypoints.xy) == 0 or len(boxes.conf) == 0:
        return None
    best_index = int(boxes.conf.argmax().item())
    corners = keypoints.xy[best_index].detach().cpu().numpy().astype(np.float32)
    if corners.shape != (4, 2) or not np.isfinite(corners).all():
        return None
    return corners, float(boxes.conf[best_index].item())


def _draw_preview(records: list[dict[str, object]], reference: np.ndarray, output_path: Path, count: int) -> None:
    selected = sorted(records, key=lambda item: float(item["mean_error_px"]), reverse=True)[:count]
    frames = [item for item in selected if isinstance(item.get("frame"), np.ndarray)]
    if not frames:
        return

    tile_width, tile_height, columns = 640, 360, 3
    rows = int(np.ceil(len(frames) / columns))
    canvas = np.full((rows * tile_height, columns * tile_width, 3), 32, dtype=np.uint8)
    for index, record in enumerate(frames):
        frame = np.asarray(record["frame"])
        tile = cv2.resize(frame, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
        scale = np.asarray([tile_width / frame.shape[1], tile_height / frame.shape[0]], dtype=np.float32)
        ground_truth = reference * scale
        prediction = np.asarray(record["prediction"], dtype=np.float32) * scale
        cv2.polylines(tile, [np.rint(ground_truth).astype(np.int32)], True, (0, 220, 0), 3, cv2.LINE_AA)
        cv2.polylines(tile, [np.rint(prediction).astype(np.int32)], True, (0, 0, 255), 3, cv2.LINE_AA)
        for point in ground_truth:
            cv2.circle(tile, tuple(np.rint(point).astype(int)), 5, (0, 220, 0), -1, cv2.LINE_AA)
        for point in prediction:
            cv2.drawMarker(
                tile,
                tuple(np.rint(point).astype(int)),
                (0, 0, 255),
                cv2.MARKER_TILTED_CROSS,
                13,
                2,
                cv2.LINE_AA,
            )
        label = (
            f"frame={record['frame_index']} mean={float(record['mean_error_px']):.1f}px "
            f"IoU={float(record['polygon_iou']):.3f} conf={float(record['confidence']):.3f}"
        )
        cv2.putText(tile, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        row, column = divmod(index, columns)
        canvas[row * tile_height : (row + 1) * tile_height, column * tile_width : (column + 1) * tile_width] = tile
    cv2.imwrite(str(output_path), canvas)


def _percentile(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(values, percentile)) if values.size else float("nan")


def evaluate_court_pose(config: EvaluationConfig) -> dict[str, object]:
    if config.samples < 2:
        raise ValueError("samples must be at least 2")
    if not config.weights.is_file():
        raise FileNotFoundError(config.weights)
    if not config.video.is_file():
        raise FileNotFoundError(config.video)

    reference, reference_metadata = load_manual_reference(config.frame_logs)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(REPO_ROOT / ".ultralytics"))
    detector = None
    model = None
    if config.refine_white_lines:
        from src.court.court_pose_detector import CourtPoseConfig, CourtPoseLineDetector

        detector = CourtPoseLineDetector(
            CourtPoseConfig(
                weights=str(config.weights),
                imgsz=config.imgsz,
                conf=config.confidence,
                device=config.device,
            )
        )
    else:
        from ultralytics import YOLO

        model = YOLO(str(config.weights.resolve()))

    capture = cv2.VideoCapture(str(config.video))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {config.video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if width <= 0 or height <= 0 or fps <= 0.0 or frame_count <= 0:
        capture.release()
        raise RuntimeError("video metadata is incomplete")

    sample_indices = np.unique(np.linspace(0, frame_count - 1, config.samples, dtype=np.int64))
    records: list[dict[str, object]] = []
    missed_indices: list[int] = []
    wall_times_ms: list[float] = []
    inference_times_ms: list[float] = []
    algorithm_times_ms: list[float] = []

    try:
        for frame_index in sample_indices.tolist():
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = capture.read()
            if not ok or frame is None:
                missed_indices.append(int(frame_index))
                continue
            started = time.perf_counter()
            if detector is not None:
                refined = detector.predict(
                    frame,
                    frame_id=int(frame_index),
                    timestamp_ms=int(round(frame_index / fps * 1000.0)),
                    force=True,
                )
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                wall_times_ms.append(elapsed_ms)
                algorithm_times_ms.append(float(refined.detect_ms))
                if not refined.valid or len(refined.corners) != 4:
                    missed_indices.append(int(frame_index))
                    continue
                prediction = np.asarray(refined.corners, dtype=np.float32)
                confidence = float(refined.confidence)
                prediction_scheme = refined.scheme
            else:
                if model is None:
                    raise AssertionError("raw YOLO model was not initialized")
                result = model.predict(
                    frame,
                    imgsz=config.imgsz,
                    conf=config.confidence,
                    device=config.device,
                    verbose=False,
                )[0]
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                wall_times_ms.append(elapsed_ms)
                algorithm_times_ms.append(elapsed_ms)
                speed = getattr(result, "speed", {}) or {}
                if "inference" in speed:
                    inference_times_ms.append(float(speed["inference"]))
                extracted = _extract_best_prediction(result)
                if extracted is None:
                    missed_indices.append(int(frame_index))
                    continue
                prediction, confidence = extracted
                prediction_scheme = "raw_yolo_pose"
            corner_errors = np.linalg.norm(prediction - reference, axis=1)
            records.append(
                {
                    "frame_index": int(frame_index),
                    "timestamp_ms": float(frame_index / fps * 1000.0),
                    "confidence": confidence,
                    "scheme": prediction_scheme,
                    "prediction": prediction,
                    "corner_errors_px": corner_errors,
                    "mean_error_px": float(corner_errors.mean()),
                    "polygon_iou": _polygon_iou(prediction, reference),
                    "frame": frame,
                }
            )
    finally:
        capture.release()

    if not records:
        raise AssertionError("the model produced no valid four-corner predictions")

    all_corner_errors = np.concatenate([np.asarray(item["corner_errors_px"]) for item in records])
    per_frame_errors = np.asarray([item["mean_error_px"] for item in records], dtype=np.float32)
    polygon_ious = np.asarray([item["polygon_iou"] for item in records], dtype=np.float32)
    confidences = np.asarray([item["confidence"] for item in records], dtype=np.float32)
    predictions = np.stack([np.asarray(item["prediction"], dtype=np.float32) for item in records])
    median_prediction = np.median(predictions, axis=0)
    temporal_deviations = np.linalg.norm(predictions - median_prediction[None, :, :], axis=2)
    mean_prediction = predictions.mean(axis=0)
    systematic_biases = np.linalg.norm(mean_prediction - reference, axis=1)
    signed_offsets = mean_prediction - reference
    image_diagonal = float(np.hypot(width, height))
    per_corner_errors = np.stack([np.asarray(item["corner_errors_px"]) for item in records])

    metrics: dict[str, object] = {
        "requested_samples": int(len(sample_indices)),
        "detected_samples": len(records),
        "missed_samples": len(missed_indices),
        "detection_rate": len(records) / len(sample_indices),
        "mean_corner_error_px": float(all_corner_errors.mean()),
        "median_corner_error_px": float(np.median(all_corner_errors)),
        "p90_corner_error_px": _percentile(all_corner_errors, 90),
        "p95_corner_error_px": _percentile(all_corner_errors, 95),
        "mean_corner_error_ratio_of_image_diagonal": float(all_corner_errors.mean() / image_diagonal),
        "median_corner_error_ratio_of_image_diagonal": float(np.median(all_corner_errors) / image_diagonal),
        "p90_frame_mean_error_px": _percentile(per_frame_errors, 90),
        "mean_polygon_iou": float(polygon_ious.mean()),
        "median_polygon_iou": float(np.median(polygon_ious)),
        "p10_polygon_iou": _percentile(polygon_ious, 10),
        "mean_confidence": float(confidences.mean()),
        "min_confidence": float(confidences.min()),
        "mean_temporal_deviation_px": float(temporal_deviations.mean()),
        "p90_temporal_deviation_px": _percentile(temporal_deviations, 90),
        "p90_temporal_deviation_ratio_of_image_diagonal": float(
            np.percentile(temporal_deviations, 90) / image_diagonal
        ),
        "mean_systematic_bias_px": float(systematic_biases.mean()),
        "systematic_bias_to_temporal_deviation_ratio": float(
            systematic_biases.mean() / max(float(temporal_deviations.mean()), 1e-6)
        ),
        "mean_prediction_corners_tl_tr_br_bl": mean_prediction.tolist(),
        "per_corner_mean_error_px": {
            name: float(per_corner_errors[:, index].mean()) for index, name in enumerate(CORNER_NAMES)
        },
        "per_corner_mean_signed_offset_px": {
            name: {"dx": float(signed_offsets[index, 0]), "dy": float(signed_offsets[index, 1])}
            for index, name in enumerate(CORNER_NAMES)
        },
        "per_corner_temporal_deviation_px": {
            name: float(temporal_deviations[:, index].mean()) for index, name in enumerate(CORNER_NAMES)
        },
        "mean_model_inference_ms": statistics.fmean(inference_times_ms) if inference_times_ms else None,
        "mean_algorithm_time_ms": statistics.fmean(algorithm_times_ms) if algorithm_times_ms else None,
        "median_wall_time_ms": statistics.median(wall_times_ms) if wall_times_ms else None,
    }
    report: dict[str, object] = {
        "model": str(config.weights.resolve()),
        "video": str(config.video.resolve()),
        "video_width": width,
        "video_height": height,
        "video_fps": fps,
        "video_frame_count": frame_count,
        "inference_imgsz": config.imgsz,
        "confidence_threshold": config.confidence,
        "device": config.device,
        "mode": "pose_plus_white_line_refinement" if config.refine_white_lines else "raw_yolo_pose",
        "manual_reference": reference_metadata,
        "metrics": metrics,
        "missed_frame_indices": missed_indices,
    }

    serializable_records: list[dict[str, object]] = []
    for item in records:
        serializable_records.append(
            {
                "frame_index": item["frame_index"],
                "timestamp_ms": item["timestamp_ms"],
                "confidence": item["confidence"],
                "scheme": item["scheme"],
                "prediction_tl_tr_br_bl": np.asarray(item["prediction"]).tolist(),
                "corner_errors_px": np.asarray(item["corner_errors_px"]).tolist(),
                "mean_error_px": item["mean_error_px"],
                "polygon_iou": item["polygon_iou"],
            }
        )
    (config.output_dir / "court_pose_accuracy.json").write_text(
        json.dumps({**report, "samples": serializable_records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (config.output_dir / "court_pose_accuracy.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame_index",
                "timestamp_ms",
                "confidence",
                "scheme",
                "mean_error_px",
                "polygon_iou",
                *[f"{name}_error_px" for name in CORNER_NAMES],
            ]
        )
        for item in serializable_records:
            writer.writerow(
                [
                    item["frame_index"],
                    item["timestamp_ms"],
                    item["confidence"],
                    item["scheme"],
                    item["mean_error_px"],
                    item["polygon_iou"],
                    *item["corner_errors_px"],
                ]
            )
    _draw_preview(records, reference, config.output_dir / "worst_samples.jpg", config.preview_count)
    return report


def assert_quality(report: dict[str, object]) -> None:
    metrics = report["metrics"]
    if not isinstance(metrics, dict):
        raise AssertionError("evaluation report has no metrics")
    if float(metrics["detection_rate"]) < 0.95:
        raise AssertionError(f"court detection rate is too low: {metrics['detection_rate']}")
    if float(metrics["median_corner_error_ratio_of_image_diagonal"]) > 0.015:
        raise AssertionError(
            "median corner error exceeds 1.5% of the image diagonal: "
            f"{metrics['median_corner_error_ratio_of_image_diagonal']}"
        )
    if float(metrics["mean_polygon_iou"]) < 0.93:
        raise AssertionError(f"mean court polygon IoU is too low: {metrics['mean_polygon_iou']}")
    if float(metrics["p90_temporal_deviation_ratio_of_image_diagonal"]) > 0.01:
        raise AssertionError(
            "P90 temporal deviation exceeds 1% of the image diagonal: "
            f"{metrics['p90_temporal_deviation_ratio_of_image_diagonal']}"
        )


def _default_frame_logs() -> tuple[Path, ...]:
    return tuple(sorted((REPO_ROOT / "outputs" / "pyqt_debug").glob(DEFAULT_LOG_GLOB)))


@unittest.skipUnless(
    os.environ.get("WFBARNET_RUN_COURT_POSE_ACCURACY") == "1",
    "set WFBARNET_RUN_COURT_POSE_ACCURACY=1 to run the external GPU/model integration test",
)
class CourtPoseModelAccuracyTest(unittest.TestCase):
    def test_external_model_on_project_video(self) -> None:
        weights = Path(os.environ.get("COURT_POSE_WEIGHTS", str(DEFAULT_WEIGHTS)))
        video = Path(os.environ.get("COURT_POSE_VIDEO", str(DEFAULT_VIDEO)))
        output_dir = Path(os.environ.get("COURT_POSE_OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR)))
        samples = int(os.environ.get("COURT_POSE_SAMPLES", "24"))
        device = os.environ.get("COURT_POSE_DEVICE", "0")
        refine_white_lines = os.environ.get("COURT_POSE_REFINE_WHITE_LINES") == "1"
        report = evaluate_court_pose(
            EvaluationConfig(
                weights=weights,
                video=video,
                frame_logs=_default_frame_logs(),
                output_dir=output_dir,
                samples=samples,
                device=device,
                refine_white_lines=refine_white_lines,
            )
        )
        print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
        assert_quality(report)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a four-corner YOLO Pose model on a WFBARNet video.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--frame-log", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="0")
    parser.add_argument("--preview-count", type=int, default=12)
    parser.add_argument("--assert-quality", action="store_true")
    parser.add_argument("--refine-white-lines", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    frame_logs = tuple(args.frame_log) if args.frame_log else _default_frame_logs()
    report = evaluate_court_pose(
        EvaluationConfig(
            weights=args.weights,
            video=args.video,
            frame_logs=frame_logs,
            output_dir=args.output,
            samples=args.samples,
            imgsz=args.imgsz,
            confidence=args.conf,
            device=args.device,
            preview_count=args.preview_count,
            refine_white_lines=args.refine_white_lines,
        )
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.assert_quality:
        assert_quality(report)


if __name__ == "__main__":
    main()
