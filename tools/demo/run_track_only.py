from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from main import RuntimeConfig, build_track_runner


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track-only video inference")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "track_only"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--track-weight", default=str(PROJECT_ROOT / "assets" / "weights" / "track" / "model_best.pt"))
    parser.add_argument("--score-thr", type=float, default=0.5)
    parser.add_argument("--input-width", type=int, default=512)
    parser.add_argument("--input-height", type=int, default=288)
    parser.add_argument("--max-frames", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RuntimeConfig(
        pipeline="track_only",
        source=args.source,
        output_dir=args.output_dir,
        device=args.device,
        track_weight=args.track_weight,
        track_input_size=(args.input_width, args.input_height),
        track_score_thr=args.score_thr,
    )
    runner = build_track_runner(config)
    runner.run(
        source=config.source,
        save_json=True,
        save_csv=True,
        save_npy=True,
        save_vis=True,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
