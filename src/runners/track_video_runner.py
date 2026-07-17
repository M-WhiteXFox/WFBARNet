from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
from tqdm import tqdm

from src.models.track_branch import TrackBranch
from src.postprocess.adaptive_track import AUTO_TRACK_ROUTE, AdaptiveTrackPostProcessor
from src.utils.exporters import export_csv, export_json, export_npy, export_track_debug_csv
from src.utils.structures import FrameResult
from src.utils.video import iter_frame_windows, iter_video_frame_windows, load_frames, probe_video
from src.utils.visualize import TrackTrailRenderer, save_visualization_video


@dataclass
class TrackVideoRunner:
    track_branch: TrackBranch
    output_dir: Path
    batch_size: int = 8
    postprocess_route: str = AUTO_TRACK_ROUTE

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
        track_postprocessor = AdaptiveTrackPostProcessor(
            fps=25.0,
            route=self.postprocess_route,
            reliable_context=False,
        )
        debug_records: list[dict[str, object]] = []
        windows = list(iter_frame_windows(frames))
        progress = tqdm(total=len(windows), desc="Track inference")
        try:
            for start in range(0, len(windows), self._batch_size()):
                batch = windows[start : start + self._batch_size()]
                candidate_batch = self.track_branch.infer_batch_candidate_results([window for _, _, window in batch])
                for (frame_id, frame, _), candidates in zip(batch, candidate_batch):
                    lagged_frames = track_postprocessor.push(
                        candidates,
                        frame_shape=frame.shape,
                        payload={"frame_id": frame_id},
                    )
                    for lagged_frame in lagged_frames:
                        results.append(
                            FrameResult(
                                frame_id=int(lagged_frame.payload["frame_id"]),
                                pose=[],
                                track=lagged_frame.track,
                            )
                        )
                        if lagged_frame.debug_record is not None:
                            debug_records.append(lagged_frame.debug_record)
                progress.update(len(batch))
        finally:
            progress.close()
        for lagged_frame in track_postprocessor.flush():
            results.append(
                FrameResult(
                    frame_id=int(lagged_frame.payload["frame_id"]),
                    pose=[],
                    track=lagged_frame.track,
                )
            )
            if lagged_frame.debug_record is not None:
                debug_records.append(lagged_frame.debug_record)

        self._export_results(results, save_json, save_csv, save_npy)
        self._export_debug(debug_records)
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
        track_postprocessor = AdaptiveTrackPostProcessor(
            fps=metadata.fps,
            route=self.postprocess_route,
            reliable_context=False,
        )
        debug_records: list[dict[str, object]] = []
        trail_renderer = TrackTrailRenderer(fps=metadata.fps, history_seconds=0.5)
        progress_total = metadata.frame_count if metadata.frame_count > 0 else None
        if max_frames is not None:
            progress_total = min(progress_total, max_frames) if progress_total is not None else max_frames
        progress = tqdm(total=progress_total, desc="Track inference")
        batch: list[tuple[int, object, list[object]]] = []
        try:
            for frame_id, curr_frame, window in iter_video_frame_windows(source, max_frames=max_frames):
                batch.append((frame_id, curr_frame, window))
                if len(batch) >= self._batch_size():
                    self._process_video_batch(
                        batch,
                        track_postprocessor,
                        trail_renderer,
                        results,
                        debug_records,
                        writer,
                    )
                    progress.update(len(batch))
                    batch.clear()
            if batch:
                self._process_video_batch(
                    batch,
                    track_postprocessor,
                    trail_renderer,
                    results,
                    debug_records,
                    writer,
                )
                progress.update(len(batch))
            for lagged_frame in track_postprocessor.flush():
                self._consume_video_lagged(
                    lagged_frame,
                    trail_renderer,
                    results,
                    debug_records,
                    writer,
                )
        finally:
            progress.close()
            if writer is not None:
                writer.release()

        if not results:
            raise RuntimeError("The video opened but returned no frames.")

        self._export_results(results, save_json, save_csv, save_npy)
        self._export_debug(debug_records)
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

    def _export_debug(self, records: list[dict[str, object]]) -> None:
        export_track_debug_csv(records, self.output_dir / "track_debug.csv")

    def _batch_size(self) -> int:
        return max(1, int(self.batch_size))

    def _process_video_batch(
        self,
        batch: list[tuple[int, object, list[object]]],
        track_postprocessor: AdaptiveTrackPostProcessor,
        trail_renderer: TrackTrailRenderer,
        results: list[FrameResult],
        debug_records: list[dict[str, object]],
        writer: cv2.VideoWriter | None,
    ) -> None:
        candidate_batch = self.track_branch.infer_batch_candidate_results([window for _, _, window in batch])
        for (frame_id, curr_frame, _), candidates in zip(batch, candidate_batch):
            lagged_frames = track_postprocessor.push(
                candidates,
                frame_shape=curr_frame.shape,
                payload={"frame_id": frame_id, "frame": curr_frame},
            )
            for lagged_frame in lagged_frames:
                self._consume_video_lagged(
                    lagged_frame,
                    trail_renderer,
                    results,
                    debug_records,
                    writer,
                )

    @staticmethod
    def _consume_video_lagged(
        lagged_frame,
        trail_renderer: TrackTrailRenderer,
        results: list[FrameResult],
        debug_records: list[dict[str, object]],
        writer: cv2.VideoWriter | None,
    ) -> None:
        result = FrameResult(
            frame_id=int(lagged_frame.payload["frame_id"]),
            pose=[],
            track=lagged_frame.track,
        )
        results.append(result)
        if lagged_frame.debug_record is not None:
            debug_records.append(lagged_frame.debug_record)
        if writer is not None:
            writer.write(trail_renderer.draw(lagged_frame.payload["frame"], result))
