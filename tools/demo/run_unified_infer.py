from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from main import RuntimeConfig, build_runner


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified badminton inference demo")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "demo"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--execution-mode", default="serial", choices=["serial", "cuda_stream"])
    parser.add_argument("--pose-backend", default="mmpose")
    parser.add_argument("--pose-config", default=str(PROJECT_ROOT / "tools" / "mmpose" / "configs" / "rtmpose-s_8xb256-420e_coco-256x192.py"))
    parser.add_argument("--pose-weight", default=str(PROJECT_ROOT / "assets" / "weights" / "pose" / "rtmpose-s_8xb256-420e_coco-256x192.pth"))
    parser.add_argument("--track-weight", default=str(PROJECT_ROOT / "assets" / "weights" / "track" / "model_best.pt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RuntimeConfig(
        source=args.source,
        output_dir=args.output_dir,
        device=args.device,
        execution_mode=args.execution_mode,
        pose_backend=args.pose_backend,
        pose_config=args.pose_config,
        pose_weight=args.pose_weight,
        track_weight=args.track_weight,
    )
    runner = build_runner(config)
    runner.run(
        source=config.source,
        save_json=True,
        save_csv=True,
        save_npy=True,
        save_vis=True,
    )


if __name__ == "__main__":
    main()
