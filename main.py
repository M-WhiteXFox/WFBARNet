from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from dataclasses import fields
from pathlib import Path
from typing import Optional

from src.models.pose_branch import PoseBranch
from src.models.track_branch import TrackBranch
from src.runners.pose_video_runner import PoseVideoRunner
from src.runners.tracknet_realtime_runner import TrackNetRealtimeRunner
from src.runners.track_video_runner import TrackVideoRunner
from src.runners.unified_runner import UnifiedRunner
from src.utils.device import resolve_device


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass
class RuntimeConfig:
    pipeline: str = "track_only"
    source: str = ""
    output_dir: str = str(PROJECT_ROOT / "outputs" / "run")
    device: str = "auto"
    execution_mode: str = "serial"
    save_json: bool = True
    save_csv: bool = True
    save_npy: bool = True
    save_vis: bool = True
    pose_backend: str = "yolo26s-pose"
    pose_config: str = ""
    pose_weight: str = str(PROJECT_ROOT / "assets" / "weights" / "pose" / "yolo26s-pose.pt")
    pose_bbox_mode: str = "whole_image"
    pose_det_config: Optional[str] = None
    pose_det_weight: Optional[str] = None
    pose_input_size: tuple[int, int] = (192, 256)
    pose_conf_thr: float = 0.3
    max_persons: int = 2
    track_weight: str = str(PROJECT_ROOT / "assets" / "weights" / "track" / "model_best.pt")
    track_input_size: tuple[int, int] = (512, 288)
    track_score_thr: float = 0.5
    track_max_frames: Optional[int] = None
    realtime_display: bool = True
    realtime_save_video: bool = True
    realtime_window_name: str = "TrackNet Realtime"
    realtime_max_frames: Optional[int] = None
    save_bst_input: bool = True
    extra: dict = field(default_factory=dict)


USER_CONFIG = RuntimeConfig(
    source="",
)


def runtime_config_from_mapping(data: dict[str, object]) -> RuntimeConfig:
    field_names = {item.name for item in fields(RuntimeConfig)}
    known: dict[str, object] = {}
    extra: dict[str, object] = {}
    for key, value in data.items():
        if key in field_names and key != "extra":
            known[key] = value
        else:
            extra[key] = value

    for key in ("pose_input_size", "track_input_size"):
        value = known.get(key)
        if isinstance(value, list) and len(value) == 2:
            known[key] = (int(value[0]), int(value[1]))

    config = RuntimeConfig(**known)
    config.extra.update(extra)
    return config


def load_runtime_config(path: str | Path) -> RuntimeConfig:
    config_path = Path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Runtime config must be a JSON object: {config_path}")
    return runtime_config_from_mapping(payload)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run WFBARNet GUI or command-line inference.")
    parser.add_argument("--config", default="", help="Path to a JSON runtime config.")
    parser.add_argument("--source", default=None, help="Video path, camera index, or frame directory.")
    parser.add_argument("--pipeline", default=None, choices=["track_only", "pose_only", "track_realtime", "unified"])
    parser.add_argument("--output-dir", default=None, help="Directory for exported inference outputs.")
    parser.add_argument("--device", default=None, help="Inference device: auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--pose-weight", default=None, help="YOLO/MMPose pose checkpoint path.")
    parser.add_argument("--track-weight", default=None, help="TrackNet checkpoint or TensorRT engine path.")
    parser.add_argument("--no-vis", action="store_true", help="Disable visualization video export.")
    return parser.parse_args(argv)


def build_runtime_config_from_args(argv: list[str] | None = None) -> RuntimeConfig:
    args = parse_args(argv)
    config = load_runtime_config(args.config) if args.config else RuntimeConfig()
    overrides = {
        "source": args.source,
        "pipeline": args.pipeline,
        "output_dir": args.output_dir,
        "device": args.device,
        "pose_weight": args.pose_weight,
        "track_weight": args.track_weight,
    }
    for key, value in overrides.items():
        if value is not None:
            setattr(config, key, value)
    if args.no_vis:
        config.save_vis = False
    return config


def launch_pyqt_app() -> int:
    from apps.pyqt6.main import main as pyqt_main

    return pyqt_main()


def build_runner(config: RuntimeConfig) -> UnifiedRunner:
    device = resolve_device(config.device)
    pose_branch = PoseBranch(
        backend=config.pose_backend,
        device=device,
        model_config=config.pose_config,
        model_weight=config.pose_weight,
        det_config=config.pose_det_config,
        det_weight=config.pose_det_weight,
        bbox_mode=config.pose_bbox_mode,
        input_size=config.pose_input_size,
        conf_thr=config.pose_conf_thr,
        max_persons=config.max_persons,
    )
    track_branch = TrackBranch(
        model_weight=config.track_weight,
        device=device,
        input_size=config.track_input_size,
        score_thr=config.track_score_thr,
    )
    return UnifiedRunner(
        pose_branch=pose_branch,
        track_branch=track_branch,
        output_dir=Path(config.output_dir),
        device=device,
        execution_mode=config.execution_mode,
    )


def build_track_realtime_runner(config: RuntimeConfig) -> TrackNetRealtimeRunner:
    device = resolve_device(config.device)
    track_branch = TrackBranch(
        model_weight=config.track_weight,
        device=device,
        input_size=config.track_input_size,
        score_thr=config.track_score_thr,
    )
    return TrackNetRealtimeRunner(
        track_branch=track_branch,
        output_dir=Path(config.output_dir),
        display=config.realtime_display,
        save_video=config.realtime_save_video,
        window_name=config.realtime_window_name,
        max_frames=config.realtime_max_frames,
    )


def build_pose_runner(config: RuntimeConfig) -> PoseVideoRunner:
    device = resolve_device(config.device)
    pose_branch = PoseBranch(
        backend=config.pose_backend,
        device=device,
        model_config=config.pose_config,
        model_weight=config.pose_weight,
        det_config=config.pose_det_config,
        det_weight=config.pose_det_weight,
        bbox_mode=config.pose_bbox_mode,
        input_size=config.pose_input_size,
        conf_thr=config.pose_conf_thr,
        max_persons=config.max_persons,
    )
    return PoseVideoRunner(
        pose_branch=pose_branch,
        output_dir=Path(config.output_dir),
    )


def build_track_runner(config: RuntimeConfig) -> TrackVideoRunner:
    device = resolve_device(config.device)
    track_branch = TrackBranch(
        model_weight=config.track_weight,
        device=device,
        input_size=config.track_input_size,
        score_thr=config.track_score_thr,
    )
    return TrackVideoRunner(
        track_branch=track_branch,
        output_dir=Path(config.output_dir),
    )


def run_pipeline(config: RuntimeConfig) -> None:
    if config.pipeline == "track_realtime":
        runner = build_track_realtime_runner(config)
        runner.run(
            source=config.source,
            save_json=config.save_json,
            save_csv=config.save_csv,
            save_npy=config.save_npy,
        )
        return

    if config.pipeline == "pose_only":
        runner = build_pose_runner(config)
        runner.run(
            source=config.source,
            save_json=config.save_json,
            save_csv=config.save_csv,
            save_npy=config.save_npy,
            save_vis=config.save_vis,
            max_frames=config.track_max_frames,
        )
        return

    if config.pipeline == "track_only":
        runner = build_track_runner(config)
        runner.run(
            source=config.source,
            save_json=config.save_json,
            save_csv=config.save_csv,
            save_npy=config.save_npy,
            save_vis=config.save_vis,
        )
        return

    runner = build_runner(config)
    runner.run(
        source=config.source,
        save_json=config.save_json,
        save_csv=config.save_csv,
        save_npy=config.save_npy,
        save_vis=config.save_vis,
        save_bst=config.save_bst_input,
    )


def main(argv: list[str] | None = None) -> int:
    config = build_runtime_config_from_args(argv)
    if not config.source:
        return launch_pyqt_app()

    run_pipeline(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
