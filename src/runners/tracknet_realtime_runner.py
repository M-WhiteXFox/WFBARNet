from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Optional

import cv2
import numpy as np

from src.models.track_branch import TrackBranch
from src.postprocess.adaptive_track import AUTO_TRACK_ROUTE, AdaptiveTrackPostProcessor
from src.postprocess.track_filter import TrackFilterAlgorithm
from src.utils.exporters import export_csv, export_json, export_npy, export_track_debug_csv
from src.utils.structures import FrameResult, TrackResult
from src.utils.visualize import TrackTrailRenderer


TRACK_STATE_RESET_GAP_SECONDS = 0.75


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
    postprocess_route: str = AUTO_TRACK_ROUTE

    def run(
        self,
        source: str,
        save_json: bool = True,
        save_csv: bool = True,
        save_npy: bool = True,
    ) -> list[FrameResult]:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        capture_source = _parse_capture_source(source)
        live_source = isinstance(capture_source, int)
        cap = cv2.VideoCapture(capture_source)
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
        first_capture_time = perf_counter() if live_source else 0.0
        if not ok:
            cap.release()
            if writer is not None:
                writer.release()
            raise RuntimeError("The realtime source opened but returned no frames.")

        ok, second_frame = cap.read()
        second_capture_time = perf_counter() if live_source else 1.0 / fps
        if not ok:
            second_frame = first_frame.copy()
            second_capture_time = first_capture_time + 1.0 / fps

        prev_frame = first_frame.copy()
        curr_frame = first_frame
        next_frame = second_frame
        curr_capture_time = first_capture_time
        next_capture_time = second_capture_time
        previous_track_time: float | None = None
        frame_id = 0
        ema_fps = 0.0
        tick_frequency = cv2.getTickFrequency()
        track_postprocessor = AdaptiveTrackPostProcessor(
            fps=fps,
            route=self.postprocess_route,
            reliable_context=False,
        )
        trail_renderer = TrackTrailRenderer(fps=fps, history_seconds=0.5)
        lagged_debug_records: list[dict[str, object]] = []

        def consume_lagged_frame(lagged_frame) -> bool:
            result = FrameResult(
                frame_id=int(lagged_frame.payload["frame_id"]),
                pose=[],
                track=lagged_frame.track,
            )
            results.append(result)
            if lagged_frame.debug_record is not None:
                lagged_debug_records.append(lagged_frame.debug_record)
            vis_frame = self._draw_overlay(
                lagged_frame.payload["frame"].copy(),
                result,
                ema_fps,
                trail_renderer,
            )
            if writer is not None:
                writer.write(vis_frame)
            if not self.display:
                return False
            cv2.imshow(self.window_name, vis_frame)
            key = cv2.waitKey(1) & 0xFF
            return key in (27, ord("q"))

        while True:
            start_tick = cv2.getTickCount()

            candidates = self.track_branch.infer_candidate_results([prev_frame, curr_frame, next_frame])
            track_dt = _frame_step_seconds(curr_capture_time, previous_track_time, fps)
            if track_dt > TRACK_STATE_RESET_GAP_SECONDS:
                track_postprocessor.reset()
                track_dt = 1.0 / fps
            previous_track_time = curr_capture_time
            lagged_frames = track_postprocessor.push(
                candidates,
                dt=track_dt,
                frame_shape=curr_frame.shape,
                payload={"frame_id": frame_id, "frame": curr_frame},
            )
            if any(consume_lagged_frame(frame) for frame in lagged_frames):
                break

            end_tick = cv2.getTickCount()
            elapsed = max((end_tick - start_tick) / tick_frequency, 1e-6)
            instant_fps = 1.0 / elapsed
            ema_fps = instant_fps if ema_fps == 0.0 else 0.9 * ema_fps + 0.1 * instant_fps

            if self.max_frames is not None and len(results) >= self.max_frames:
                break

            prev_frame = curr_frame
            curr_frame = next_frame
            curr_capture_time = next_capture_time
            ok, incoming = cap.read()
            incoming_capture_time = perf_counter() if live_source else (frame_id + 2) / fps
            if not ok:
                if frame_id == 0:
                    break
                next_frame = curr_frame.copy()
                frame_id += 1
                final_candidates = self.track_branch.infer_candidate_results([prev_frame, curr_frame, next_frame])
                final_dt = _frame_step_seconds(curr_capture_time, previous_track_time, fps)
                if final_dt > TRACK_STATE_RESET_GAP_SECONDS:
                    track_postprocessor.reset()
                    final_dt = 1.0 / fps
                final_lagged_frames = track_postprocessor.push(
                    final_candidates,
                    dt=final_dt,
                    frame_shape=curr_frame.shape,
                    payload={"frame_id": frame_id, "frame": curr_frame},
                )
                for final_lagged_frame in final_lagged_frames:
                    if consume_lagged_frame(final_lagged_frame):
                        break
                for pending_frame in track_postprocessor.flush():
                    if consume_lagged_frame(pending_frame):
                        break
                break
            next_frame = incoming
            next_capture_time = incoming_capture_time
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
        export_track_debug_csv(lagged_debug_records, self.output_dir / "tracknet_realtime_debug.csv")

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


def _frame_step_seconds(current_time: float, previous_time: float | None, fps: float) -> float:
    fallback = 1.0 / fps if fps > 0 else 1.0 / 25.0
    if previous_time is None:
        return fallback
    elapsed = float(current_time) - float(previous_time)
    return elapsed if elapsed > 0.0 else fallback


def _reset_filter_state_preserving_debug(track_filter: TrackFilterAlgorithm) -> None:
    debug_records = list(track_filter.debug_records)
    track_filter.reset()
    track_filter.debug_records.extend(debug_records)
