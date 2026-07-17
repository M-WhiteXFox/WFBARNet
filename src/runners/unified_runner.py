from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

from src.builders.bst_input_builder import BSTInputBuilder
from src.models.pose_branch import PoseBranch
from src.models.track_branch import TrackBranch
from src.postprocess.adaptive_track import AUTO_TRACK_ROUTE, AdaptiveTrackPostProcessor
from src.utils.exporters import export_csv, export_json, export_npy, export_track_debug_csv
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
    postprocess_route: str = AUTO_TRACK_ROUTE
    track_debug_records: list[dict[str, object]] = field(default_factory=list, init=False, repr=False)

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
        export_track_debug_csv(self.track_debug_records, self.output_dir / "unified_track_debug.csv")

    def _run_serial(self, frames: list) -> list[FrameResult]:
        outputs: list[FrameResult] = []
        track_postprocessor = AdaptiveTrackPostProcessor(
            fps=25.0,
            route=self.postprocess_route,
            reliable_context=True,
        )
        debug_records: list[dict[str, object]] = []
        for frame_id, frame, window in tqdm(list(iter_frame_windows(frames)), desc="Unified inference"):
            pose = self.pose_branch.infer(frame)
            candidates = self.track_branch.infer_candidate_results(window)
            lagged_frames = track_postprocessor.push(
                candidates,
                frame_shape=frame.shape,
                person_bboxes=_pose_bboxes(pose),
                payload={"frame_id": frame_id, "pose": pose},
            )
            for lagged_frame in lagged_frames:
                self._consume_lagged(lagged_frame, outputs, debug_records)
        for lagged_frame in track_postprocessor.flush():
            self._consume_lagged(lagged_frame, outputs, debug_records)
        self.track_debug_records = debug_records
        return outputs

    def _run_video_stream(self, source: str, save_vis: bool) -> list[FrameResult]:
        metadata = probe_video(source)
        track_postprocessor = AdaptiveTrackPostProcessor(
            fps=metadata.fps,
            route=self.postprocess_route,
            reliable_context=True,
        )
        debug_records: list[dict[str, object]] = []
        trail_renderer = TrackTrailRenderer(fps=metadata.fps, history_seconds=0.5)
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
                candidates = self.track_branch.infer_candidate_results(window)
                lagged_frames = track_postprocessor.push(
                    candidates,
                    frame_shape=frame.shape,
                    person_bboxes=_pose_bboxes(pose),
                    payload={"frame_id": frame_id, "pose": pose, "frame": frame},
                )
                for lagged_frame in lagged_frames:
                    self._consume_lagged(
                        lagged_frame,
                        outputs,
                        debug_records,
                        trail_renderer=trail_renderer,
                        writer=writer,
                    )
            for lagged_frame in track_postprocessor.flush():
                self._consume_lagged(
                    lagged_frame,
                    outputs,
                    debug_records,
                    trail_renderer=trail_renderer,
                    writer=writer,
                )
        finally:
            if writer is not None:
                writer.release()

        if not outputs:
            raise FileNotFoundError(f"No frames loaded from source: {source}")
        self.track_debug_records = debug_records
        return outputs

    def _run_cuda_stream(self, frames: list) -> list[FrameResult]:
        pose_stream = torch.cuda.Stream()
        track_stream = torch.cuda.Stream()
        outputs: list[FrameResult] = []
        track_postprocessor = AdaptiveTrackPostProcessor(
            fps=25.0,
            route=self.postprocess_route,
            reliable_context=True,
        )
        debug_records: list[dict[str, object]] = []
        for frame_id, frame, window in tqdm(list(iter_frame_windows(frames)), desc="Unified inference (dual stream)"):
            with torch.cuda.stream(pose_stream):
                pose = self.pose_branch.infer(frame)
            with torch.cuda.stream(track_stream):
                candidates = self.track_branch.infer_candidate_results(window)
            torch.cuda.synchronize()
            lagged_frames = track_postprocessor.push(
                candidates,
                frame_shape=frame.shape,
                person_bboxes=_pose_bboxes(pose),
                payload={"frame_id": frame_id, "pose": pose},
            )
            for lagged_frame in lagged_frames:
                self._consume_lagged(lagged_frame, outputs, debug_records)
        for lagged_frame in track_postprocessor.flush():
            self._consume_lagged(lagged_frame, outputs, debug_records)
        self.track_debug_records = debug_records
        return outputs

    @staticmethod
    def _consume_lagged(
        lagged_frame,
        outputs: list[FrameResult],
        debug_records: list[dict[str, object]],
        *,
        trail_renderer: TrackTrailRenderer | None = None,
        writer: cv2.VideoWriter | None = None,
    ) -> None:
        result = FrameResult(
            frame_id=int(lagged_frame.payload["frame_id"]),
            pose=lagged_frame.payload["pose"],
            track=lagged_frame.track,
        )
        outputs.append(result)
        if lagged_frame.debug_record is not None:
            debug_records.append(lagged_frame.debug_record)
        if writer is not None and trail_renderer is not None:
            writer.write(trail_renderer.draw(lagged_frame.payload["frame"], result))


def _pose_bboxes(poses: object) -> list[tuple[float, float, float, float]]:
    if not isinstance(poses, list):
        return []

    bboxes: list[tuple[float, float, float, float]] = []
    for pose in poses:
        bbox = getattr(pose, "bbox", None)
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            continue
        try:
            x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
        except (TypeError, ValueError):
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        bboxes.append((x1, y1, x2, y2))
    return bboxes
