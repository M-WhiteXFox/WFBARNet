from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.court.opencv_court_homography_core import compute_homographies
from src.models.pose_branch import PoseBranch
from src.models.track_branch import TrackBranch
from src.postprocess.pose import CourtPoseTargetTracker
from src.utils.structures import PersonPoseResult
from src.utils.video import iter_video_frame_windows, probe_video


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build fixed TrackNet candidate and person-box caches for trajectory evaluation.",
    )
    parser.add_argument("datasets", nargs="+", help="Dataset stems, for example: 1 2 10-1")
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "Dataset")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--court-meta",
        type=Path,
        default=ROOT / "outputs" / "trajectory_filter_tuning" / "cache" / "cache_meta.json",
        help="Existing cache metadata whose fixed court corners should be reused.",
    )
    parser.add_argument(
        "--track-weight",
        type=Path,
        default=ROOT / "assets" / "weights" / "track" / "model_best.pt",
    )
    parser.add_argument(
        "--pose-weight",
        type=Path,
        default=ROOT / "assets" / "weights" / "pose" / "yolo26s-pose.pt",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--pose-stride", type=int, default=2)
    return parser.parse_args()


def _person_bboxes(poses: list[PersonPoseResult]) -> list[list[float]]:
    return [
        [float(value) for value in pose.bbox[:4]]
        for pose in poses
        if len(pose.bbox) >= 4
    ]


def main() -> None:
    args = _parse_args()
    if args.batch_size < 1 or args.pose_stride < 1:
        raise ValueError("--batch-size and --pose-stride must be positive")
    for weight in (args.track_weight, args.pose_weight):
        if not weight.is_file():
            raise FileNotFoundError(f"Model weight not found: {weight}")

    court_meta = json.loads(args.court_meta.read_text(encoding="utf-8"))
    corners = np.asarray(court_meta["court_corners"], dtype=np.float32)
    if corners.shape != (4, 2):
        raise ValueError(f"Expected four court corners in {args.court_meta}")
    court_to_image_h, image_to_court_h = compute_homographies(corners)
    if court_to_image_h is None or image_to_court_h is None:
        raise ValueError(f"Invalid court corners in {args.court_meta}")
    court_prediction = {
        "valid": True,
        "corners": corners.astype(float).tolist(),
        "court_to_image_h": court_to_image_h.astype(float).tolist(),
        "image_to_court_h": image_to_court_h.astype(float).tolist(),
    }

    track_branch = TrackBranch(
        model_weight=str(args.track_weight),
        device=args.device,
        input_size=(512, 288),
        score_thr=0.35,
        max_candidates=5,
        candidate_score_thr_ratio=0.6,
    )
    pose_branch = PoseBranch(
        backend="yolo26s-pose",
        model_weight=str(args.pose_weight),
        device=args.device,
        conf_thr=0.35,
        max_persons=12,
        yolo_imgsz=960,
        yolo_crop_pose=True,
        yolo_crop_imgsz=640,
        yolo_crop_padding=0.30,
        yolo_crop_min_box_conf=0.45,
        yolo_max_pose_crops=8,
        yolo_court_filter=True,
        yolo_court_required=False,
        yolo_court_margin=30.0,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_meta: dict[str, object] = {
        "device": args.device,
        "court_source_meta": str(args.court_meta.resolve()),
        "court_corners": corners.astype(float).tolist(),
        "track_weight": str(args.track_weight.resolve()),
        "pose_weight": str(args.pose_weight.resolve()),
        "track_score_threshold": 0.35,
        "track_max_candidates": 5,
        "track_candidate_score_ratio": 0.6,
        "pose_stride": args.pose_stride,
        "datasets": {},
    }

    for name in args.datasets:
        video_path = args.dataset_dir / f"{name}.mp4"
        if not video_path.is_file():
            raise FileNotFoundError(f"Dataset video not found: {video_path}")
        metadata = probe_video(str(video_path))
        pose_tracker = CourtPoseTargetTracker(
            max_missing_frames=max(args.pose_stride, int(round(metadata.fps * 0.35))),
            court_margin=30.0,
            detection_smoothing=0.78,
            velocity_smoothing=0.50,
            court_required=False,
            predict_missing_motion=True,
            motion_prediction_scale=0.55,
        )

        cache_path = args.output_dir / f"{name}_model_cache.jsonl"
        temporary_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        processed = 0
        started = perf_counter()
        batch: list[tuple[int, np.ndarray, list[np.ndarray]]] = []
        progress = tqdm(total=metadata.frame_count or None, desc=f"Cache {name}")

        def flush_batch(handle: object) -> None:
            nonlocal processed
            if not batch:
                return
            candidates_batch = track_branch.infer_batch_candidate_results(
                [window for _, _, window in batch]
            )
            for (frame_id, frame, _), candidates in zip(batch, candidates_batch):
                detections = (
                    pose_branch.infer(frame, court_prediction=court_prediction)
                    if frame_id % args.pose_stride == 0
                    else []
                )
                poses = pose_tracker.update(
                    detections,
                    court_prediction,
                    frame_shape=frame.shape,
                )
                record = {
                    "frame_id": frame_id,
                    "candidates": [asdict(candidate) for candidate in candidates],
                    "person_bboxes": _person_bboxes(poses),
                }
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                processed += 1
            progress.update(len(batch))
            batch.clear()

        try:
            with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
                for item in iter_video_frame_windows(str(video_path)):
                    batch.append(item)
                    if len(batch) >= args.batch_size:
                        flush_batch(handle)
                flush_batch(handle)
            temporary_path.replace(cache_path)
        finally:
            progress.close()

        elapsed = perf_counter() - started
        if metadata.frame_count and processed != metadata.frame_count:
            raise RuntimeError(
                f"Cache frame count mismatch for {name}: {processed} != {metadata.frame_count}"
            )
        datasets = output_meta["datasets"]
        assert isinstance(datasets, dict)
        datasets[name] = {
            "video": str(video_path.resolve()),
            "fps": metadata.fps,
            "width": metadata.width,
            "height": metadata.height,
            "frame_count": processed,
            "cache": str(cache_path.resolve()),
            "elapsed_s": elapsed,
            "processing_fps": processed / elapsed if elapsed > 0 else 0.0,
        }

    meta_path = args.output_dir / "cache_meta.json"
    meta_path.write_text(
        json.dumps(output_meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(meta_path)


if __name__ == "__main__":
    main()
