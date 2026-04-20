from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
from tqdm import tqdm

from src.models.track_branch import TrackBranch
from src.utils.exporters import export_csv, export_json, export_npy
from src.utils.structures import FrameResult
from src.utils.video import iter_frame_windows, load_frames
from src.utils.visualize import draw_result, save_visualization_video


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
        for frame_id, _, window in tqdm(list(iter_frame_windows(frames)), desc="Track inference"):
            _, track = self.track_branch.infer(window)
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
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise FileNotFoundError(f"Unable to open video source: {source}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        writer = None
        if save_vis:
            writer = cv2.VideoWriter(
                str(self.output_dir / "track_vis.mp4"),
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
            raise RuntimeError("The video opened but returned no frames.")

        ok, second_frame = cap.read()
        if not ok:
            second_frame = first_frame.copy()

        prev_frame = first_frame.copy()
        curr_frame = first_frame
        next_frame = second_frame
        frame_id = 0

        progress_total = total_frames if total_frames > 0 else None
        if max_frames is not None:
            progress_total = max_frames
        progress = tqdm(total=progress_total, desc="Track inference")
        while True:
            _, track = self.track_branch.infer([prev_frame, curr_frame, next_frame])
            result = FrameResult(frame_id=frame_id, pose=[], track=track)
            results.append(result)
            if writer is not None:
                writer.write(draw_result(curr_frame, result))
            progress.update(1)
            if max_frames is not None and len(results) >= max_frames:
                break

            prev_frame = curr_frame
            curr_frame = next_frame
            ok, incoming = cap.read()
            if not ok:
                if frame_id > 0:
                    frame_id += 1
                    next_frame = curr_frame.copy()
                    _, final_track = self.track_branch.infer([prev_frame, curr_frame, next_frame])
                    final_result = FrameResult(frame_id=frame_id, pose=[], track=final_track)
                    results.append(final_result)
                    if writer is not None:
                        writer.write(draw_result(curr_frame, final_result))
                    progress.update(1)
                break
            next_frame = incoming
            frame_id += 1

        progress.close()
        cap.release()
        if writer is not None:
            writer.release()

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
