from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from src.models.pose_branch import PoseBranch
from src.utils.exporters import export_csv, export_json, export_npy
from src.utils.structures import FrameResult, TrackResult
from src.utils.video import load_frames
from src.utils.visualize import save_visualization_video


@dataclass
class PoseVideoRunner:
    pose_branch: PoseBranch
    output_dir: Path

    def run(
        self,
        source: str,
        save_json: bool = True,
        save_csv: bool = True,
        save_npy: bool = True,
        save_vis: bool = True,
    ) -> list[FrameResult]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        frames = load_frames(source)
        if not frames:
            raise FileNotFoundError(f"No frames loaded from source: {source}")

        results: list[FrameResult] = []
        for frame_id, frame in enumerate(tqdm(frames, desc="Pose inference")):
            pose = self.pose_branch.infer(frame)
            track = TrackResult(ball_xy=[0.0, 0.0], visible=0, score=0.0, heatmap_shape=[])
            results.append(FrameResult(frame_id=frame_id, pose=pose, track=track))

        if save_json:
            export_json(results, self.output_dir / "pose_results.json")
        if save_csv:
            export_csv(results, self.output_dir / "pose_results.csv")
        if save_npy:
            export_npy(results, self.output_dir / "pose_results.npy")
        if save_vis:
            save_visualization_video(frames, results, self.output_dir / "pose_vis.mp4")
        return results
