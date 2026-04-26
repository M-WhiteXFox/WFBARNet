from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}


@dataclass(slots=True)
class CourtKeypointDetection:
    frame_id: int
    timestamp_sec: float
    corners: list[list[float]]
    raw_keypoints: list[list[float]]
    keypoint_confidence: list[float]
    confidence: float
    area_ratio: float
    updated: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect four badminton court corner keypoints with a YOLO keypoint model."
    )
    parser.add_argument("--source", required=True, help="Image or video path.")
    parser.add_argument(
        "--weights",
        default=str(PROJECT_ROOT / "assets" / "weights" / "court" / "court_yolo_keypoints.pt"),
        help="YOLO keypoint model weights trained to predict court corners.",
    )
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "court_keypoints_demo"))
    parser.add_argument("--device", default="cpu", help="Ultralytics device, for example cpu, 0, cuda:0.")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO object confidence threshold.")
    parser.add_argument("--kp-conf", type=float, default=0.20, help="Minimum average keypoint confidence.")
    parser.add_argument("--min-area-ratio", type=float, default=0.02, help="Minimum court polygon area / frame area.")
    parser.add_argument(
        "--keypoint-indices",
        default="0,1,2,3",
        help="Comma-separated keypoint indices to use as the four court corners.",
    )
    parser.add_argument(
        "--order",
        default="auto",
        choices=["auto", "model"],
        help="auto orders corners as top-left, top-right, bottom-right, bottom-left.",
    )
    parser.add_argument("--detect-every", type=int, default=150, help="For video, run YOLO every N frames.")
    parser.add_argument("--max-frames", type=int, default=0, help="For video, stop after N frames; 0 means all.")
    parser.add_argument(
        "--smooth-alpha",
        type=float,
        default=0.25,
        help="For video updates, blend new corners with previous corners. 1.0 disables smoothing.",
    )
    parser.add_argument("--no-video", action="store_true", help="Do not save visualization video for video input.")
    return parser.parse_args()


def parse_indices(raw: str) -> list[int]:
    indices = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if len(indices) != 4:
        raise ValueError("--keypoint-indices must contain exactly four indices.")
    return indices


def load_yolo(weights: str):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Missing dependency: pip install ultralytics") from exc

    weight_path = Path(weights)
    if not weight_path.is_file():
        raise FileNotFoundError(f"YOLO court keypoint weights not found: {weight_path}")
    return YOLO(str(weight_path))


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS


def to_numpy(value: object) -> np.ndarray:
    if value is None:
        return np.array([])
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def order_corners(points: np.ndarray) -> np.ndarray:
    sums = points[:, 0] + points[:, 1]
    diffs = points[:, 0] - points[:, 1]
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = points[int(np.argmin(sums))]
    ordered[2] = points[int(np.argmax(sums))]
    ordered[1] = points[int(np.argmax(diffs))]
    ordered[3] = points[int(np.argmin(diffs))]
    return ordered


def polygon_area_ratio(corners: np.ndarray, frame_shape: tuple[int, ...]) -> float:
    frame_area = float(frame_shape[0] * frame_shape[1])
    if frame_area <= 0:
        return 0.0
    area = abs(float(cv2.contourArea(corners.astype(np.float32))))
    return area / frame_area


def geometry_is_valid(corners: np.ndarray, frame_shape: tuple[int, ...], min_area_ratio: float) -> bool:
    height, width = frame_shape[:2]
    if corners.shape != (4, 2):
        return False
    if not np.isfinite(corners).all():
        return False
    if np.any(corners[:, 0] < -width * 0.05) or np.any(corners[:, 0] > width * 1.05):
        return False
    if np.any(corners[:, 1] < -height * 0.05) or np.any(corners[:, 1] > height * 1.05):
        return False
    return polygon_area_ratio(corners, frame_shape) >= min_area_ratio


def select_best_detection(result: object, indices: list[int]) -> tuple[np.ndarray, np.ndarray, float] | None:
    keypoints_obj = getattr(result, "keypoints", None)
    if keypoints_obj is None:
        return None

    xy = to_numpy(getattr(keypoints_obj, "xy", None))
    if xy.size == 0 or xy.ndim != 3:
        return None

    conf = to_numpy(getattr(keypoints_obj, "conf", None))
    if conf.size == 0:
        conf = np.ones(xy.shape[:2], dtype=np.float32)

    if max(indices) >= xy.shape[1]:
        raise ValueError(f"Model returned {xy.shape[1]} keypoints, cannot use indices {indices}.")

    box_conf = np.ones((xy.shape[0],), dtype=np.float32)
    boxes = getattr(result, "boxes", None)
    if boxes is not None:
        raw_box_conf = to_numpy(getattr(boxes, "conf", None))
        if raw_box_conf.size == xy.shape[0]:
            box_conf = raw_box_conf.astype(np.float32)

    best_index = -1
    best_score = -1.0
    for det_index in range(xy.shape[0]):
        kp_conf = conf[det_index, indices].astype(np.float32)
        score = float(kp_conf.mean() * box_conf[det_index])
        if score > best_score:
            best_score = score
            best_index = det_index

    if best_index < 0:
        return None

    points = xy[best_index, indices].astype(np.float32)
    kp_conf = conf[best_index, indices].astype(np.float32)
    return points, kp_conf, best_score


def detect_court_corners(
    model: object,
    frame: np.ndarray,
    frame_id: int,
    timestamp_sec: float,
    args: argparse.Namespace,
    indices: list[int],
    previous_corners: np.ndarray | None = None,
) -> tuple[CourtKeypointDetection | None, np.ndarray | None]:
    results = model.predict(frame, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)
    if not results:
        return None, previous_corners

    selected = select_best_detection(results[0], indices)
    if selected is None:
        return None, previous_corners

    raw_points, kp_conf, score = selected
    corners = order_corners(raw_points) if args.order == "auto" else raw_points.copy()
    average_kp_conf = float(kp_conf.mean())
    area_ratio = polygon_area_ratio(corners, frame.shape)
    updated = average_kp_conf >= args.kp_conf and geometry_is_valid(corners, frame.shape, args.min_area_ratio)

    output_corners = corners
    if updated and previous_corners is not None:
        alpha = float(np.clip(args.smooth_alpha, 0.0, 1.0))
        output_corners = previous_corners * (1.0 - alpha) + corners * alpha

    if updated:
        previous_corners = output_corners
    elif previous_corners is not None:
        output_corners = previous_corners

    detection = CourtKeypointDetection(
        frame_id=frame_id,
        timestamp_sec=timestamp_sec,
        corners=output_corners.round(2).tolist(),
        raw_keypoints=raw_points.round(2).tolist(),
        keypoint_confidence=kp_conf.round(4).tolist(),
        confidence=round(float(score), 4),
        area_ratio=round(float(area_ratio), 4),
        updated=bool(updated),
    )
    return detection, previous_corners


def draw_detection(frame: np.ndarray, detection: CourtKeypointDetection | None) -> np.ndarray:
    canvas = frame.copy()
    if detection is None:
        cv2.putText(canvas, "court: no detection", (24, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 80, 255), 2)
        return canvas

    corners = np.asarray(detection.corners, dtype=np.int32)
    color = (20, 220, 80) if detection.updated else (0, 180, 255)
    cv2.polylines(canvas, [corners.reshape(-1, 1, 2)], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
    labels = ["TL", "TR", "BR", "BL"]
    for label, point in zip(labels, corners):
        x, y = int(point[0]), int(point[1])
        cv2.circle(canvas, (x, y), 6, color, -1, lineType=cv2.LINE_AA)
        cv2.putText(canvas, label, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    status = "updated" if detection.updated else "held"
    text = f"court: {status} conf={detection.confidence:.2f} area={detection.area_ratio:.3f}"
    cv2.putText(canvas, text, (24, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return canvas


def write_json(path: Path, detections: Iterable[CourtKeypointDetection]) -> None:
    data = [asdict(item) for item in detections]
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def run_image(model: object, source: Path, output_dir: Path, args: argparse.Namespace, indices: list[int]) -> None:
    frame = cv2.imread(str(source))
    if frame is None:
        raise RuntimeError(f"Could not read image: {source}")

    detection, _ = detect_court_corners(model, frame, 0, 0.0, args, indices)
    vis = draw_detection(frame, detection)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = source.stem
    cv2.imwrite(str(output_dir / f"{stem}_court_keypoints.jpg"), vis)
    write_json(output_dir / f"{stem}_court_keypoints.json", [detection] if detection else [])

    if detection is None or not detection.updated:
        print("[warn] No valid court keypoints detected.")
        return
    print(json.dumps(asdict(detection), indent=2))


def run_video(model: object, source: Path, output_dir: Path, args: argparse.Namespace, indices: list[int]) -> None:
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    output_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    if not args.no_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_dir / f"{source.stem}_court_keypoints.mp4"), fourcc, fps, (width, height))

    detections: list[CourtKeypointDetection] = []
    latest_detection: CourtKeypointDetection | None = None
    previous_corners: np.ndarray | None = None
    frame_id = 0
    detect_every = max(1, int(args.detect_every))
    max_frames = max(0, int(args.max_frames))

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if max_frames and frame_id >= max_frames:
            break

        if frame_id % detect_every == 0:
            detection, previous_corners = detect_court_corners(
                model=model,
                frame=frame,
                frame_id=frame_id,
                timestamp_sec=frame_id / fps,
                args=args,
                indices=indices,
                previous_corners=previous_corners,
            )
            if detection is not None:
                latest_detection = detection
                detections.append(detection)

        if writer is not None:
            writer.write(draw_detection(frame, latest_detection))

        frame_id += 1

    cap.release()
    if writer is not None:
        writer.release()

    write_json(output_dir / f"{source.stem}_court_keypoints.json", detections)
    updates = sum(1 for item in detections if item.updated)
    print(f"[done] frames={frame_id} detections={len(detections)} updates={updates}")
    print(f"[out] {output_dir}")


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    if not source.is_file():
        raise FileNotFoundError(f"Source not found: {source}")

    model = load_yolo(args.weights)
    indices = parse_indices(args.keypoint_indices)
    output_dir = Path(args.output_dir)

    if is_image(source):
        run_image(model, source, output_dir, args, indices)
    elif is_video(source):
        run_video(model, source, output_dir, args, indices)
    else:
        raise ValueError(f"Unsupported source type: {source.suffix}")


if __name__ == "__main__":
    main()
