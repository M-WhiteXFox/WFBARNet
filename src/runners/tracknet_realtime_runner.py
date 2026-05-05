from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.models.track_branch import TrackBranch
from src.postprocess.tracknet_v3_filter import create_tracknet_v3_ball_track_filter
from src.utils.exporters import export_csv, export_json, export_npy, export_track_debug_csv
from src.utils.structures import FrameResult, TrackResult
from src.utils.visualize import TrackTrailRenderer


def _parse_capture_source(source: str) -> str | int:
    if source.isdigit():
        return int(source)
    return source


@dataclass
class TrackNetRealtimeRunner:
    track_branch: TrackBranch
    output_dir: Path
    display: bool = True
    save_video: bool = True
    window_name: str = "TrackNet Realtime"
    max_frames: Optional[int] = None

    def run(
        self,
        source: str,
        save_json: bool = True,
        save_csv: bool = True,
        save_npy: bool = True,
    ) -> list[FrameResult]:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(_parse_capture_source(source))
        if not cap.isOpened():
            raise FileNotFoundError(f"Unable to open realtime source: {source}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

        writer = None
        if self.save_video:
            writer = cv2.VideoWriter(
                str(self.output_dir / "tracknet_realtime_vis.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )

        results: list[FrameResult] = []

        ok, first_frame = cap.read()
        if not ok:
            cap.release()
            if writer is not None:
                writer.release()
            raise RuntimeError("The realtime source opened but returned no frames.")

        ok, second_frame = cap.read()
        if not ok:
            second_frame = first_frame.copy()

        prev_frame = first_frame.copy()
        curr_frame = first_frame
        next_frame = second_frame
        frame_id = 0
        ema_fps = 0.0
        tick_frequency = cv2.getTickFrequency()
        track_filter = create_tracknet_v3_ball_track_filter(fps=fps, debug_enabled=True)
        trail_renderer = TrackTrailRenderer(fps=fps, history_seconds=0.5)

        while True:
            start_tick = cv2.getTickCount()

            candidates = self.track_branch.infer_candidate_results([prev_frame, curr_frame, next_frame])
            track = track_filter.update_candidates(candidates, frame_shape=curr_frame.shape)
            result = FrameResult(frame_id=frame_id, pose=[], track=track)
            results.append(result)

            vis_frame = self._draw_overlay(curr_frame.copy(), result, ema_fps, trail_renderer)
            if writer is not None:
                writer.write(vis_frame)
            if self.display:
                cv2.imshow(self.window_name, vis_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

            end_tick = cv2.getTickCount()
            elapsed = max((end_tick - start_tick) / tick_frequency, 1e-6)
            instant_fps = 1.0 / elapsed
            ema_fps = instant_fps if ema_fps == 0.0 else 0.9 * ema_fps + 0.1 * instant_fps

            if self.max_frames is not None and len(results) >= self.max_frames:
                break

            prev_frame = curr_frame
            curr_frame = next_frame
            ok, incoming = cap.read()
            if not ok:
                if frame_id == 0:
                    break
                next_frame = curr_frame.copy()
                frame_id += 1
                final_candidates = self.track_branch.infer_candidate_results([prev_frame, curr_frame, next_frame])
                final_track = track_filter.update_candidates(final_candidates, frame_shape=curr_frame.shape)
                final_result = FrameResult(frame_id=frame_id, pose=[], track=final_track)
                results.append(final_result)
                final_vis = self._draw_overlay(curr_frame.copy(), final_result, ema_fps, trail_renderer)
                if writer is not None:
                    writer.write(final_vis)
                if self.display:
                    cv2.imshow(self.window_name, final_vis)
                    cv2.waitKey(1)
                break
            next_frame = incoming
            frame_id += 1

        cap.release()
        if writer is not None:
            writer.release()
        if self.display:
            cv2.destroyAllWindows()

        if save_json:
            export_json(results, self.output_dir / "tracknet_realtime_results.json")
        if save_csv:
            export_csv(results, self.output_dir / "tracknet_realtime_results.csv")
        if save_npy:
            export_npy(results, self.output_dir / "tracknet_realtime_results.npy")
        export_track_debug_csv(track_filter.debug_records, self.output_dir / "tracknet_realtime_debug.csv")

        return results

    def _draw_overlay(
        self,
        frame: np.ndarray,
        result: FrameResult,
        fps: float,
        trail_renderer: TrackTrailRenderer,
    ) -> np.ndarray:
        frame = trail_renderer.draw(frame, result)
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (16, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (40, 220, 40),
            2,
        )
        cv2.putText(
            frame,
            f"Frame: {result.frame_id}",
            (16, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        return frame

    def _draw_track(self, frame: np.ndarray, track: TrackResult) -> None:
        if track.visible:
            x, y = map(int, track.ball_xy)
            cv2.circle(frame, (x, y), 8, (0, 0, 255), 2)
            cv2.circle(frame, (x, y), 14, (0, 255, 255), 2)
            cv2.putText(
                frame,
                f"ball {track.score:.2f}",
                (x + 10, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )
        else:
            cv2.putText(
                frame,
                "ball lost",
                (16, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 120, 255),
                2,
            )
