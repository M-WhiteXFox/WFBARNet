from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from time import perf_counter

import torch
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.track_branch import TrackBranch
from src.utils.video import iter_video_frame_windows, probe_video


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache TrackNet candidates for TrackNetV3 raw_data/raw_data2 videos.",
    )
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="TrackNetV3 repository containing raw_data and raw_data2.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--track-weight",
        type=Path,
        default=ROOT / "assets" / "weights" / "track" / "model_best.pt",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--split", action="append", choices=("raw_data", "raw_data2"))
    return parser.parse_args()


def _line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def main() -> None:
    args = _parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if not args.track_weight.is_file():
        raise FileNotFoundError(f"Model weight not found: {args.track_weight}")

    splits = args.split or ["raw_data", "raw_data2"]
    videos: list[tuple[str, Path]] = []
    for split in splits:
        split_dir = args.dataset_root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Dataset split not found: {split_dir}")
        for video_path in sorted(split_dir.glob("*.mp4")):
            label_path = video_path.with_suffix(".csv")
            if not label_path.is_file():
                raise FileNotFoundError(f"Missing paired annotation: {label_path}")
            videos.append((split, video_path))

    track_branch = TrackBranch(
        model_weight=str(args.track_weight),
        device=args.device,
        input_size=(512, 288),
        score_thr=0.35,
        max_candidates=5,
        candidate_score_thr_ratio=0.6,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, object] = {
        "dataset_root": str(args.dataset_root.resolve()),
        "track_weight": str(args.track_weight.resolve()),
        "device": args.device,
        "batch_size": args.batch_size,
        "track_input_size": [512, 288],
        "track_score_threshold": 0.35,
        "track_max_candidates": 5,
        "track_candidate_score_ratio": 0.6,
        "splits": {},
    }

    for split, video_path in videos:
        video = probe_video(str(video_path))
        cache_dir = args.output_dir / split
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{video_path.stem}_model_cache.jsonl"
        if cache_path.is_file() and _line_count(cache_path) == video.frame_count:
            split_meta = metadata["splits"].setdefault(split, {})
            assert isinstance(split_meta, dict)
            split_meta[video_path.stem] = {
                "video": str(video_path.resolve()),
                "annotation": str(video_path.with_suffix(".csv").resolve()),
                "cache": str(cache_path.resolve()),
                "fps": video.fps,
                "width": video.width,
                "height": video.height,
                "frame_count": video.frame_count,
                "cached": True,
            }
            print(f"Reuse {cache_path}")
            continue

        temporary_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        processed = 0
        started = perf_counter()
        batch: list[tuple[int, object, list[object]]] = []
        progress = tqdm(total=video.frame_count or None, desc=f"{split}/{video_path.stem}")

        def flush_batch(handle: object) -> None:
            nonlocal processed
            if not batch:
                return
            candidate_batch = track_branch.infer_batch_candidate_results(
                [window for _, _, window in batch]
            )
            for (frame_id, _, _), candidates in zip(batch, candidate_batch):
                record = {
                    "frame_id": frame_id,
                    "candidates": [asdict(candidate) for candidate in candidates],
                }
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")
                processed += 1
            progress.update(len(batch))
            batch.clear()

        try:
            with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
                for item in iter_video_frame_windows(str(video_path)):
                    batch.append(item)
                    if len(batch) >= args.batch_size:
                        flush_batch(handle)
                flush_batch(handle)
            temporary_path.replace(cache_path)
        finally:
            progress.close()
        if video.frame_count and processed != video.frame_count:
            raise RuntimeError(
                f"Cache frame count mismatch for {video_path}: {processed} != {video.frame_count}"
            )

        elapsed = perf_counter() - started
        split_meta = metadata["splits"].setdefault(split, {})
        assert isinstance(split_meta, dict)
        split_meta[video_path.stem] = {
            "video": str(video_path.resolve()),
            "annotation": str(video_path.with_suffix(".csv").resolve()),
            "cache": str(cache_path.resolve()),
            "fps": video.fps,
            "width": video.width,
            "height": video.height,
            "frame_count": processed,
            "elapsed_s": elapsed,
            "processing_fps": processed / elapsed if elapsed > 0 else 0.0,
            "cached": False,
        }
        (args.output_dir / "cache_meta.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    meta_path = args.output_dir / "cache_meta.json"
    meta_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(meta_path.resolve())


if __name__ == "__main__":
    main()
