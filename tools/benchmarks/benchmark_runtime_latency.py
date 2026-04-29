from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import statistics
import sys
from time import perf_counter
from typing import Any

import cv2
import numpy as np
import torch

try:
    from PyQt6.QtGui import QImage
except Exception:
    QImage = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.pose_branch import PoseBranch
from src.models.track_branch import TrackBranch
from src.postprocess.track_filter import BallTrackFilter
from src.utils.structures import FrameResult, TrackResult
from src.utils.visualize import TrackTrailRenderer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark runtime latency for the PyQt6 inference pipeline.")
    parser.add_argument("--source", required=True, help="Video file to benchmark.")
    parser.add_argument("--frames", type=int, default=300, help="Number of frames/windows to measure.")
    parser.add_argument("--warmup", type=int, default=30, help="Frames/windows to run before collecting timings.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--track-weight", default=str(PROJECT_ROOT / "assets" / "weights" / "track" / "model_best.pt"))
    parser.add_argument("--pose-weight", default=str(PROJECT_ROOT / "assets" / "weights" / "pose" / "yolo26s-pose.pt"))
    parser.add_argument("--input-width", type=int, default=512)
    parser.add_argument("--input-height", type=int, default=288)
    parser.add_argument("--score-thr", type=float, default=0.35)
    parser.add_argument("--pose-stride", type=int, default=3)
    parser.add_argument("--no-track", action="store_true")
    parser.add_argument("--no-pose", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--qimage", action="store_true", help="Also time the Qt display copy used by the PyQt runtime.")
    return parser.parse_args()


def resolve_device(raw: str) -> str:
    if raw == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if raw == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    return raw


def sync_if_cuda(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def qt_display_copy(frame: np.ndarray) -> Any:
    if QImage is not None and hasattr(QImage.Format, "Format_BGR888"):
        height, width = frame.shape[:2]
        image = QImage(
            frame.data,
            width,
            height,
            frame.strides[0],
            QImage.Format.Format_BGR888,
        )
        image._buffer = frame
        return image

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if QImage is None:
        return rgb.copy()

    height, width = rgb.shape[:2]
    image = QImage(
        rgb.data,
        width,
        height,
        rgb.strides[0],
        QImage.Format.Format_RGB888,
    )
    image._buffer = rgb
    return image


class Timer:
    def __init__(self, timings: dict[str, list[float]], name: str, device: str = "cpu") -> None:
        self.timings = timings
        self.name = name
        self.device = device
        self.start = 0.0

    def __enter__(self) -> None:
        sync_if_cuda(self.device)
        self.start = perf_counter()

    def __exit__(self, exc_type, exc, tb) -> None:
        sync_if_cuda(self.device)
        self.timings[self.name].append((perf_counter() - self.start) * 1000.0)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return ordered[index]


def print_summary(timings: dict[str, list[float]]) -> None:
    print("\nLatency summary (ms)")
    print("-" * 76)
    print(f"{'stage':<24} {'count':>7} {'avg':>10} {'p50':>10} {'p95':>10} {'max':>10}")
    print("-" * 76)
    for name, values in timings.items():
        if not values:
            continue
        avg = statistics.fmean(values)
        print(
            f"{name:<24} {len(values):>7} "
            f"{avg:>10.2f} {percentile(values, 50):>10.2f} "
            f"{percentile(values, 95):>10.2f} {max(values):>10.2f}"
        )
    total_values = timings.get("total", [])
    if total_values:
        fps = 1000.0 / statistics.fmean(total_values)
        print("-" * 76)
        print(f"Estimated end-to-end FPS: {fps:.2f}")


def read_initial_window(cap: cv2.VideoCapture) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ok, first = cap.read()
    if not ok or first is None:
        raise RuntimeError("Could not read first frame.")
    ok, second = cap.read()
    if not ok or second is None:
        second = first.copy()
    return first.copy(), first, second


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    source = Path(args.source)
    if not source.is_file():
        raise FileNotFoundError(f"Video not found: {source}")

    track_enabled = not args.no_track
    pose_enabled = not args.no_pose
    render_enabled = not args.no_render

    print(f"[env] torch={torch.__version__} cuda={torch.cuda.is_available()} device={device}")
    print(f"[source] {source}")

    track_branch: TrackBranch | None = None
    pose_branch: PoseBranch | None = None

    if track_enabled:
        track_branch = TrackBranch(
            model_weight=str(Path(args.track_weight)),
            device=device,
            input_size=(args.input_width, args.input_height),
            score_thr=args.score_thr,
        )
        print(f"[track] backend={track_branch.backend_name} weight={args.track_weight}")

    if pose_enabled:
        pose_branch = PoseBranch(
            backend="yolo26s-pose",
            model_weight=str(Path(args.pose_weight)),
            device=device,
            conf_thr=0.35,
            max_persons=2,
        )
        print(f"[pose] weight={args.pose_weight}")

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 25.0

    prev_frame, current_frame, next_frame = read_initial_window(cap)
    track_filter = BallTrackFilter(fps=fps)
    renderer = TrackTrailRenderer(fps=fps)
    last_pose: list[Any] = []
    timings: dict[str, list[float]] = defaultdict(list)
    collected = 0
    processed = 0
    total_target = args.warmup + args.frames

    while processed < total_target:
        ok, incoming_frame = cap.read()
        if not ok or incoming_frame is None:
            break

        collect = processed >= args.warmup
        local_timings = timings if collect else defaultdict(list)

        with Timer(local_timings, "total", device):
            if track_enabled and track_branch is not None:
                with Timer(local_timings, "track_infer_total", device):
                    candidates = track_branch.infer_candidate_results([prev_frame, current_frame, next_frame])
                with Timer(local_timings, "track_filter"):
                    track = track_filter.update_candidates(candidates)
            else:
                track = TrackResult(ball_xy=[-1.0, -1.0], visible=0, score=0.0)

            if pose_enabled and pose_branch is not None and processed % max(1, args.pose_stride) == 0:
                with Timer(local_timings, "pose_infer", device):
                    last_pose = pose_branch.infer(current_frame)

            if render_enabled:
                with Timer(local_timings, "render"):
                    result = FrameResult(frame_id=processed, pose=last_pose, track=track)
                    vis_frame = renderer.draw(current_frame, result)
            else:
                vis_frame = current_frame

            if args.qimage:
                with Timer(local_timings, "qimage_like_convert"):
                    _ = qt_display_copy(vis_frame)

        if collect:
            collected += 1

        prev_frame = current_frame
        current_frame = next_frame
        next_frame = incoming_frame
        processed += 1

    cap.release()
    print(f"[done] collected={collected} warmup={args.warmup} processed={processed}")
    print_summary(timings)


if __name__ == "__main__":
    main()
