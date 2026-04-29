from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

from src.builders.bst_input_builder import BSTInputBuilder
from src.models.pose_branch import PoseBranch
from src.models.track_branch import TrackBranch
from src.postprocess.track_filter import BallTrackFilter
from src.utils.exporters import export_csv, export_json, export_npy
from src.utils.structures import FrameResult
from src.utils.video import iter_frame_windows, iter_video_frame_windows, load_frames, probe_video
from src.utils.visualize import TrackTrailRenderer, save_visualization_video


@dataclass
class UnifiedRunner:
    pose_branch: PoseBranch
    track_branch: TrackBranch
    output_dir: Path
    device: str = "cpu"
    execution_mode: str = "serial"

    def run(
        self,
        source: str,
        save_json: bool = True,
        save_csv: bool = True,
        save_npy: bool = True,
        save_vis: bool = True,
        save_bst: bool = True,
    ) -> list[FrameResult]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if not Path(source).is_dir():
            results = self._run_video_stream(source, save_vis=save_vis)
            self._export_results(results, save_json, save_csv, save_npy, save_bst)
            return results

        frames = load_frames(source)
        if not frames:
            raise FileNotFoundError(f"No frames loaded from source: {source}")

        if self.execution_mode == "cuda_stream" and self.device.startswith("cuda") and torch.cuda.is_available():
            results = self._run_cuda_stream(frames)
        else:
            results = self._run_serial(frames)

        self._export_results(results, save_json, save_csv, save_npy, save_bst)
        if save_vis:
            save_visualization_video(frames, results, self.output_dir / "unified_vis.mp4")
        return results

    def _export_results(
        self,
        results: list[FrameResult],
        save_json: bool,
        save_csv: bool,
        save_npy: bool,
        save_bst: bool,
    ) -> None:
        if save_json:
            export_json(results, self.output_dir / "unified_results.json")
        if save_csv:
            export_csv(results, self.output_dir / "unified_results.csv")
        if save_npy:
            export_npy(results, self.output_dir / "unified_results.npy")
        if save_bst:
            BSTInputBuilder(normalize=False).save(results, self.output_dir / "bst_input.npy")

    def _run_serial(self, frames: list) -> list[FrameResult]:
        outputs: list[FrameResult] = []
        track_filter = BallTrackFilter()
        for frame_id, frame, window in tqdm(list(iter_frame_windows(frames)), desc="Unified inference"):
            pose = self.pose_branch.infer(frame)
            raw_track = self.track_branch.infer_result(window)
            track = track_filter.update(raw_track, frame_shape=frame.shape)
            outputs.append(FrameResult(frame_id=frame_id, pose=pose, track=track))
        return outputs

    def _run_video_stream(self, source: str, save_vis: bool) -> list[FrameResult]:
        metadata = probe_video(source)
        track_filter = BallTrackFilter(fps=metadata.fps)
        trail_renderer = TrackTrailRenderer(fps=metadata.fps, history_seconds=3.0)
        writer = None
        if save_vis:
            writer = cv2.VideoWriter(
                str(self.output_dir / "unified_vis.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                metadata.fps,
                (metadata.width, metadata.height),
            )

        progress_total = metadata.frame_count if metadata.frame_count > 0 else None
        outputs: list[FrameResult] = []
        try:
            for frame_id, frame, window in tqdm(
                iter_video_frame_windows(source),
                total=progress_total,
                desc="Unified inference",
            ):
                pose = self.pose_branch.infer(frame)
                raw_track = self.track_branch.infer_result(window)
                track = track_filter.update(raw_track, frame_shape=frame.shape)
                result = FrameResult(frame_id=frame_id, pose=pose, track=track)
                outputs.append(result)
                if writer is not None:
                    writer.write(trail_renderer.draw(frame, result))
        finally:
            if writer is not None:
                writer.release()

        if not outputs:
            raise FileNotFoundError(f"No frames loaded from source: {source}")
        return outputs

    def _run_cuda_stream(self, frames: list) -> list[FrameResult]:
        pose_stream = torch.cuda.Stream()
        track_stream = torch.cuda.Stream()
        outputs: list[FrameResult] = []
        track_filter = BallTrackFilter()
        for frame_id, frame, window in tqdm(list(iter_frame_windows(frames)), desc="Unified inference (dual stream)"):
            with torch.cuda.stream(pose_stream):
                pose = self.pose_branch.infer(frame)
            with torch.cuda.stream(track_stream):
                raw_track = self.track_branch.infer_result(window)
            torch.cuda.synchronize()
            track = track_filter.update(raw_track, frame_shape=frame.shape)
            outputs.append(FrameResult(frame_id=frame_id, pose=pose, track=track))
        return outputs
