from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
from tqdm import tqdm

from src.models.pose_branch import PoseBranch
from src.utils.exporters import export_csv, export_json, export_npy
from src.utils.structures import FrameResult, TrackResult
from src.utils.video import iter_video_frames, load_frames, probe_video
from src.utils.visualize import draw_result, save_visualization_video


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
        if not Path(source).is_dir():
            results = self._run_video_stream(source, save_vis=save_vis)
            self._export_results(results, save_json, save_csv, save_npy)
            return results

        frames = load_frames(source)
        if not frames:
            raise FileNotFoundError(f"No frames loaded from source: {source}")

        results: list[FrameResult] = []
        for frame_id, frame in enumerate(tqdm(frames, desc="Pose inference")):
            pose = self.pose_branch.infer(frame)
            track = TrackResult(ball_xy=[0.0, 0.0], visible=0, score=0.0, heatmap_shape=[])
            results.append(FrameResult(frame_id=frame_id, pose=pose, track=track))

        self._export_results(results, save_json, save_csv, save_npy)
        if save_vis:
            save_visualization_video(frames, results, self.output_dir / "pose_vis.mp4")
        return results

    def _run_video_stream(self, source: str, save_vis: bool) -> list[FrameResult]:
        metadata = probe_video(source)
        writer = None
        if save_vis:
            writer = cv2.VideoWriter(
                str(self.output_dir / "pose_vis.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                metadata.fps,
                (metadata.width, metadata.height),
            )

        results: list[FrameResult] = []
        try:
            for frame_id, frame in tqdm(
                iter_video_frames(source),
                total=metadata.frame_count if metadata.frame_count > 0 else None,
                desc="Pose inference",
            ):
                pose = self.pose_branch.infer(frame)
                track = TrackResult(ball_xy=[0.0, 0.0], visible=0, score=0.0, heatmap_shape=[])
                result = FrameResult(frame_id=frame_id, pose=pose, track=track)
                results.append(result)
                if writer is not None:
                    writer.write(draw_result(frame, result))
        finally:
            if writer is not None:
                writer.release()

        if not results:
            raise FileNotFoundError(f"No frames loaded from source: {source}")
        return results

    def _export_results(
        self,
        results: list[FrameResult],
        save_json: bool,
        save_csv: bool,
        save_npy: bool,
    ) -> None:
        if save_json:
            export_json(results, self.output_dir / "pose_results.json")
        if save_csv:
            export_csv(results, self.output_dir / "pose_results.csv")
        if save_npy:
            export_npy(results, self.output_dir / "pose_results.npy")
