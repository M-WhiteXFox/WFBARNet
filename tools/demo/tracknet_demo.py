from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import torch

from src.models.track_branch import TrackBranch
from src.postprocess.track_filter import BallTrackFilter
from src.utils.structures import FrameResult
from src.utils.visualize import TrackTrailRenderer


def get_available_devices():
    devices = ["cpu"]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            devices.append(f"cuda:{i}")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append("mps")
    return devices


def get_device_label(device: str) -> str:
    if device == "cpu":
        return "CPU"
    elif device.startswith("cuda"):
        gpu_id = device.split(":")[1] if ":" in device else "0"
        gpu_name = torch.cuda.get_device_name(int(gpu_id)) if torch.cuda.is_available() else "GPU"
        return f"GPU ({gpu_name})"
    elif device == "mps":
        return "Apple MPS"
    return device


def main():
    video_path = PROJECT_ROOT / "videos" / "test3.mp4"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(f"[TrackNet Demo] 开始 TrackNet 实时推理")
    print(f"[TrackNet Demo] 视频路径: {video_path}")
    print(f"[TrackNet Demo] 推理设备: {get_device_label(device)}")

    if not video_path.exists():
        print(f"[Error] 视频文件不存在: {video_path}")
        return

    print("[TrackNet Demo] 加载模型...")
    track_branch = TrackBranch(
        model_weight=str(PROJECT_ROOT / "assets" / "weights" / "track" / "model_best.pt"),
        device=device,
        input_size=(512, 288),
        score_thr=0.35,
    )
    print("[TrackNet Demo] 模型加载完成")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[Error] 无法打开视频: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[TrackNet Demo] 视频信息: {width}x{height}, {fps:.1f} FPS, {total_frames} 帧")
    print("[TrackNet Demo] 按 'q' 或 ESC 退出")

    cv2.namedWindow("TrackNet Demo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("TrackNet Demo", 960, 540)

    ok, first_frame = cap.read()
    if not ok:
        print("[Error] 无法读取视频帧")
        return

    ok, second_frame = cap.read()
    if not ok:
        second_frame = first_frame.copy()

    prev_frame = first_frame.copy()
    curr_frame = first_frame
    next_frame = second_frame
    frame_id = 0
    ema_fps = 0.0
    tick_frequency = cv2.getTickFrequency()
    frame_count = 0
    track_filter = BallTrackFilter(fps=fps)
    trail_renderer = TrackTrailRenderer(fps=fps, history_seconds=0.5)

    while True:
        start_tick = cv2.getTickCount()

        candidates = track_branch.infer_candidate_results([prev_frame, curr_frame, next_frame])
        track = track_filter.update_candidates(candidates)

        result = FrameResult(frame_id=frame_id, pose=[], track=track)
        vis_frame = trail_renderer.draw(curr_frame, result)
        if track.visible:
            x, y = map(int, track.ball_xy)
            track_text = f"Track: Visible (x={x}, y={y}, score={track.score:.2f})"
        else:
            cv2.putText(
                vis_frame,
                "ball lost",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 120, 255),
                2,
            )
            track_text = "Track: Lost"

        end_tick = cv2.getTickCount()
        elapsed = max((end_tick - start_tick) / tick_frequency, 1e-6)
        instant_fps = 1.0 / elapsed
        ema_fps = instant_fps if ema_fps == 0.0 else 0.9 * ema_fps + 0.1 * instant_fps

        cv2.putText(
            vis_frame,
            f"FPS: {ema_fps:.1f}",
            (16, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (40, 220, 40),
            2,
        )
        cv2.putText(
            vis_frame,
            f"Frame: {frame_id}/{total_frames}",
            (16, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        status_text = f"[{frame_id}/{total_frames}] FPS: {ema_fps:.1f} | {track_text}"
        print(f"\r{status_text}", end="", flush=True)

        cv2.imshow("TrackNet Demo", vis_frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

        if frame_count >= total_frames - 2:
            break

        prev_frame = curr_frame
        curr_frame = next_frame
        ok, incoming = cap.read()
        if not ok:
            break
        next_frame = incoming
        frame_id += 1
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print("\n[TrackNet Demo] 推理完成")


if __name__ == "__main__":
    main()
