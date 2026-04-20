from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm

from src.builders.bst_input_builder import BSTInputBuilder
from src.models.pose_branch import PoseBranch
from src.models.track_branch import TrackBranch
from src.utils.exporters import export_csv, export_json, export_npy
from src.utils.structures import FrameResult
from src.utils.video import iter_frame_windows, load_frames
from src.utils.visualize import save_visualization_video


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
        frames = load_frames(source)
        if not frames:
            raise FileNotFoundError(f"No frames loaded from source: {source}")

        if self.execution_mode == "cuda_stream" and self.device.startswith("cuda") and torch.cuda.is_available():
            results = self._run_cuda_stream(frames)
        else:
            results = self._run_serial(frames)

        if save_json:
            export_json(results, self.output_dir / "unified_results.json")
        if save_csv:
            export_csv(results, self.output_dir / "unified_results.csv")
        if save_npy:
            export_npy(results, self.output_dir / "unified_results.npy")
        if save_vis:
            save_visualization_video(frames, results, self.output_dir / "unified_vis.mp4")
        if save_bst:
            BSTInputBuilder(normalize=False).save(results, self.output_dir / "bst_input.npy")
        return results

    def _run_serial(self, frames: list) -> list[FrameResult]:
        outputs: list[FrameResult] = []
        for frame_id, frame, window in tqdm(list(iter_frame_windows(frames)), desc="Unified inference"):
            pose = self.pose_branch.infer(frame)
            _, track = self.track_branch.infer(window)
            outputs.append(FrameResult(frame_id=frame_id, pose=pose, track=track))
        return outputs

    def _run_cuda_stream(self, frames: list) -> list[FrameResult]:
        pose_stream = torch.cuda.Stream()
        track_stream = torch.cuda.Stream()
        outputs: list[FrameResult] = []
        for frame_id, frame, window in tqdm(list(iter_frame_windows(frames)), desc="Unified inference (dual stream)"):
            with torch.cuda.stream(pose_stream):
                pose = self.pose_branch.infer(frame)
            with torch.cuda.stream(track_stream):
                _, track = self.track_branch.infer(window)
            torch.cuda.synchronize()
            outputs.append(FrameResult(frame_id=frame_id, pose=pose, track=track))
        return outputs
