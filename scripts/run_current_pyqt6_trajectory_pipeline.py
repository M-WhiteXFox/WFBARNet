from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from PyQt6.QtCore import QCoreApplication


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps.pyqt6.controllers import analysis_controller_runtime as runtime
from src.models.pose_branch import PoseBranch
from src.models.track_branch import TrackBranch
from src.postprocess.trajectory_events import RealtimeTrajectoryEventDetector
from src.utils.exporters import frame_result_log_record


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the current PyQt6 batch trajectory pipeline and capture its frame log."
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=ROOT / "videos" / "3834a6e48f7efc440e0c29fbefb24e6e.mp4",
    )
    parser.add_argument(
        "--track-weight",
        type=Path,
        default=ROOT / "assets" / "weights" / "track" / "model_best.pt",
    )
    parser.add_argument(
        "--pose-weight",
        type=Path,
        default=ROOT / "assets" / "weights" / "pose" / "yolo26s-pose.pt",
    )
    parser.add_argument(
        "--frame-log",
        type=Path,
        default=ROOT / "outputs" / "current_pyqt6" / "3834_current_pyqt6_frame_log.jsonl",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "outputs" / "current_pyqt6" / "3834_current_pyqt6_summary.json",
    )
    return parser.parse_args()


class _DetectorCapture:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.handle = path.open("w", encoding="utf-8", newline="\n")
        self.rows = 0
        self.events: dict[tuple[str, int], dict[str, Any]] = {}

    def detector_type(self) -> type:
        capture = self

        class CapturingTrajectoryEventDetector:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self._detector = RealtimeTrajectoryEventDetector(*args, **kwargs)

            def update(
                self,
                frame_result: Any,
                *,
                timestamp_ms: int | None = None,
                frame_shape: Any = None,
                court_prediction: Any = None,
            ) -> dict[str, object] | None:
                event = self._detector.update(
                    frame_result,
                    timestamp_ms=timestamp_ms,
                    frame_shape=frame_shape,
                    court_prediction=court_prediction,
                )
                hit_event = event if isinstance(event, dict) and event.get("event_type") == "hit" else None
                landing_event = (
                    event if isinstance(event, dict) and event.get("event_type") == "landing" else None
                )
                record = frame_result_log_record(
                    frame_result,
                    timestamp_ms=timestamp_ms,
                    court_prediction=court_prediction,
                    hit_event=hit_event,
                    trajectory_event=event,
                    landing_event=landing_event,
                )
                capture.handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                capture.rows += 1
                if isinstance(event, dict):
                    event_type = str(event.get("event_type", ""))
                    event_frame = int(event.get("frame_id", frame_result.frame_id))
                    capture.events[(event_type, event_frame)] = dict(event)
                return event

        return CapturingTrajectoryEventDetector

    def close(self) -> None:
        self.handle.close()


def _track_branch(weight: Path, device: str) -> TrackBranch:
    return TrackBranch(
        model_weight=str(weight),
        device=device,
        input_size=(512, 288),
        score_thr=0.35,
    )


def _pose_branch(weight: Path, device: str) -> PoseBranch:
    return PoseBranch(
        backend="yolo26s-pose",
        model_weight=str(weight),
        device=device,
        conf_thr=0.35,
        max_persons=runtime.POSE_CANDIDATE_LIMIT,
        yolo_imgsz=runtime.POSE_YOLO_IMGSZ,
        yolo_crop_pose=True,
        yolo_crop_imgsz=runtime.POSE_CROP_IMGSZ,
        yolo_crop_padding=runtime.POSE_CROP_PADDING,
        yolo_crop_min_box_conf=runtime.POSE_CROP_MIN_BOX_CONF,
        yolo_max_pose_crops=runtime.POSE_MAX_CROPS,
        yolo_court_filter=True,
        yolo_court_required=False,
        yolo_court_margin=runtime.POSE_COURT_MARGIN_CM,
    )


def main() -> None:
    args = _parse_args()
    for path in (args.video, args.track_weight, args.pose_weight):
        if not path.is_file():
            raise FileNotFoundError(path)

    app = QCoreApplication.instance() or QCoreApplication([])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    capture = _DetectorCapture(args.frame_log)
    original_detector_type = runtime.RealtimeTrajectoryEventDetector
    started = perf_counter()
    try:
        runtime.RealtimeTrajectoryEventDetector = capture.detector_type()
        worker = runtime.BatchInferenceWorker(
            str(args.video.parent),
            _track_branch(args.track_weight, device),
            _pose_branch(args.pose_weight, device),
            pose_stride=runtime.POSE_INFERENCE_STRIDE,
            track_enabled=True,
            pose_enabled=True,
            bst_model=None,
        )
        record = worker._process_video(args.video.resolve(), index=0, total=1)
    finally:
        runtime.RealtimeTrajectoryEventDetector = original_detector_type
        capture.close()
        app.processEvents()

    elapsed = perf_counter() - started
    summary = {
        "source_video": str(args.video.resolve()),
        "frame_log": str(args.frame_log.resolve()),
        "pipeline": {
            "implementation": "apps.pyqt6.controllers.analysis_controller_runtime.BatchInferenceWorker",
            "track_weight": str(args.track_weight.resolve()),
            "pose_weight": str(args.pose_weight.resolve()),
            "device": device,
            "track_score_thr": 0.35,
            "pose_stride": runtime.POSE_INFERENCE_STRIDE,
            "track_filter": "create_tracknet_v3_ball_track_filter",
            "fixed_lag_ms": 300,
            "event_detector": "RealtimeTrajectoryEventDetector",
        },
        "captured_rows": capture.rows,
        "events": [
            event
            for _, event in sorted(
                capture.events.items(),
                key=lambda item: (item[0][1], item[0][0]),
            )
        ],
        "elapsed_seconds": elapsed,
        "pyqt6_record": record,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
