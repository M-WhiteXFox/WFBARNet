from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
from tqdm import tqdm

from src.models.track_branch import TrackBranch
from src.postprocess.track_filter import BallTrackFilter
from src.utils.exporters import export_csv, export_json, export_npy
from src.utils.structures import FrameResult
from src.utils.video import iter_frame_windows, iter_video_frame_windows, load_frames, probe_video
from src.utils.visualize import TrackTrailRenderer, save_visualization_video


@dataclass
class TrackVideoRunner:
    track_branch: TrackBranch
    output_dir: Path

    def run(
        self,
        source: str,
        save_json: bool = True,
        save_csv: bool = True,
        save_npy: bool = True,
        save_vis: bool = True,
        max_frames: int | None = None,
    ) -> list[FrameResult]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        source_path = Path(source)
        if source_path.is_dir():
            return self._run_frame_directory(
                source=source,
                save_json=save_json,
                save_csv=save_csv,
                save_npy=save_npy,
                save_vis=save_vis,
                max_frames=max_frames,
            )
        return self._run_video_stream(
            source=source,
            save_json=save_json,
            save_csv=save_csv,
            save_npy=save_npy,
            save_vis=save_vis,
            max_frames=max_frames,
        )

    def _run_frame_directory(
        self,
        source: str,
        save_json: bool,
        save_csv: bool,
        save_npy: bool,
        save_vis: bool,
        max_frames: int | None,
    ) -> list[FrameResult]:
        frames = load_frames(source)
        if not frames:
            raise FileNotFoundError(f"No frames loaded from source: {source}")
        if max_frames is not None:
            frames = frames[:max_frames]
        results: list[FrameResult] = []
        track_filter = BallTrackFilter()
        for frame_id, frame, window in tqdm(list(iter_frame_windows(frames)), desc="Track inference"):
            raw_track = self.track_branch.infer_result(window)
            track = track_filter.update(raw_track, frame_shape=frame.shape)
            results.append(FrameResult(frame_id=frame_id, pose=[], track=track))

        self._export_results(results, save_json, save_csv, save_npy)
        if save_vis:
            save_visualization_video(frames, results, self.output_dir / "track_vis.mp4")
        return results

    def _run_video_stream(
        self,
        source: str,
        save_json: bool,
        save_csv: bool,
        save_npy: bool,
        save_vis: bool,
        max_frames: int | None,
    ) -> list[FrameResult]:
        metadata = probe_video(source)

        writer = None
        if save_vis:
            writer = cv2.VideoWriter(
                str(self.output_dir / "track_vis.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                metadata.fps,
                (metadata.width, metadata.height),
            )

        results: list[FrameResult] = []
        track_filter = BallTrackFilter(fps=metadata.fps)
        trail_renderer = TrackTrailRenderer(fps=metadata.fps, history_seconds=3.0)
        progress_total = metadata.frame_count if metadata.frame_count > 0 else None
        if max_frames is not None:
            progress_total = min(progress_total, max_frames) if progress_total is not None else max_frames
        progress = tqdm(total=progress_total, desc="Track inference")
        try:
            for frame_id, curr_frame, window in iter_video_frame_windows(source, max_frames=max_frames):
                raw_track = self.track_branch.infer_result(window)
                track = track_filter.update(raw_track, frame_shape=curr_frame.shape)
                result = FrameResult(frame_id=frame_id, pose=[], track=track)
                results.append(result)
                if writer is not None:
                    writer.write(trail_renderer.draw(curr_frame, result))
                progress.update(1)
        finally:
            progress.close()
            if writer is not None:
                writer.release()

        if not results:
            raise RuntimeError("The video opened but returned no frames.")

        self._export_results(results, save_json, save_csv, save_npy)
        return results

    def _export_results(
        self,
        results: list[FrameResult],
        save_json: bool,
        save_csv: bool,
        save_npy: bool,
    ) -> None:
        if save_json:
            export_json(results, self.output_dir / "track_results.json")
        if save_csv:
            export_csv(results, self.output_dir / "track_results.csv")
        if save_npy:
            export_npy(results, self.output_dir / "track_results.npy")
