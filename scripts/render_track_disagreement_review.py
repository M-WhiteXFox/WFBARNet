from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "tracknetv3_candidate_graph"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect track disagreements into a review video.")
    parser.add_argument(
        "--video",
        type=Path,
        default=OUTPUT_DIR / "BD_New_19_current_vs_graph.mp4",
    )
    parser.add_argument(
        "--per-frame",
        type=Path,
        default=OUTPUT_DIR / "BD_New_19_comparison_per_frame.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / "BD_New_19_uncertain_frames_review.mp4",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=OUTPUT_DIR / "BD_New_19_uncertain_frames_manifest.csv",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=OUTPUT_DIR / "BD_New_19_uncertain_frames_summary.json",
    )
    parser.add_argument("--output-fps", type=float, default=2.0)
    return parser.parse_args()


def _is_true(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def _load_disagreements(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if _is_true(row["material_difference"])]
    if not rows:
        raise RuntimeError(f"no material differences found in {path}")
    return rows


def _difference_type(row: dict[str, str]) -> tuple[str, tuple[int, int, int]]:
    current_visible = _is_true(row["current_visible"])
    graph_visible = _is_true(row["graph_visible"])
    if current_visible and not graph_visible:
        return "CURRENT ONLY", (75, 85, 245)
    if graph_visible and not current_visible:
        return "GRAPH ONLY", (245, 210, 60)
    distance = float(row["method_distance_px"])
    return f"POSITION GAP {distance:.1f}px", (45, 165, 255)


def _status_text(row: dict[str, str]) -> str:
    current = "VISIBLE" if _is_true(row["current_visible"]) else "MISSING"
    graph = "VISIBLE" if _is_true(row["graph_visible"]) else "MISSING"
    return f"CURRENT: {current}    GRAPH: {graph}"


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "review_index",
        "frame_id",
        "time_seconds",
        "difference_type",
        "current_visible",
        "current_source",
        "graph_visible",
        "graph_candidate_rank",
        "method_distance_px",
        "verdict",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for review_index, row in enumerate(rows, start=1):
            writer.writerow(
                {
                    "review_index": review_index,
                    "frame_id": row["frame_id"],
                    "time_seconds": row["time_seconds"],
                    "difference_type": _difference_type(row)[0],
                    "current_visible": row["current_visible"],
                    "current_source": row["current_source"],
                    "graph_visible": row["graph_visible"],
                    "graph_candidate_rank": row["graph_candidate_rank"],
                    "method_distance_px": row["method_distance_px"],
                    "verdict": "",
                    "notes": "",
                }
            )


def _render(
    video_path: Path,
    output_path: Path,
    rows: list[dict[str, str]],
    output_fps: float,
) -> dict[str, object]:
    if output_fps <= 0.0:
        raise ValueError("output_fps must be positive")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open comparison video: {video_path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    footer_height = 70
    output_size = (width, height + footer_height)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        output_size,
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"could not create review video: {output_path}")

    rows_by_frame = {int(row["frame_id"]): row for row in rows}
    written = 0
    frame_id = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            row = rows_by_frame.get(frame_id)
            if row is not None:
                review_index = written + 1
                difference, color = _difference_type(row)
                review_frame = cv2.copyMakeBorder(
                    frame,
                    0,
                    footer_height,
                    0,
                    0,
                    cv2.BORDER_CONSTANT,
                    value=(14, 17, 21),
                )
                cv2.putText(
                    review_frame,
                    (
                        f"REVIEW {review_index:02d}/{len(rows):02d}    "
                        f"SOURCE FRAME {frame_id:03d}    "
                        f"TIME {float(row['time_seconds']):05.2f}s    {difference}"
                    ),
                    (22, height + 29),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.68,
                    color,
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    review_frame,
                    _status_text(row),
                    (22, height + 56),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (225, 229, 234),
                    1,
                    cv2.LINE_AA,
                )
                writer.write(review_frame)
                written += 1
            frame_id += 1
    finally:
        capture.release()
        writer.release()
    if written != len(rows):
        raise RuntimeError(f"rendered {written} review frames, expected {len(rows)}")
    return {
        "frames": written,
        "fps": output_fps,
        "width": output_size[0],
        "height": output_size[1],
        "duration_seconds": written / output_fps,
        "source_fps": source_fps,
    }


def _verify(path: Path, expected: dict[str, object]) -> dict[str, object]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open review video: {path}")
    decoded = 0
    sample_stddev: dict[str, float] = {}
    sample_ids = {0, int(expected["frames"]) // 2, int(expected["frames"]) - 1}
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if decoded in sample_ids:
            sample_stddev[str(decoded)] = float(frame.std())
        decoded += 1
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if decoded != int(expected["frames"]):
        raise RuntimeError(f"decoded {decoded} review frames, expected {expected['frames']}")
    if width != int(expected["width"]) or height != int(expected["height"]):
        raise RuntimeError("review video size mismatch")
    if abs(fps - float(expected["fps"])) > 0.01:
        raise RuntimeError("review video fps mismatch")
    if len(sample_stddev) != len(sample_ids) or any(value <= 0.0 for value in sample_stddev.values()):
        raise RuntimeError("review video contains an invalid sample frame")
    return {
        "frames_decoded": decoded,
        "fps": fps,
        "width": width,
        "height": height,
        "sample_stddev": sample_stddev,
    }


def main() -> None:
    args = _parse_args()
    rows = _load_disagreements(args.per_frame)
    _write_manifest(args.manifest, rows)
    render = _render(args.video, args.output, rows, float(args.output_fps))
    verification = _verify(args.output, render)
    summary = {
        "source_video": str(args.video.resolve()),
        "source_per_frame": str(args.per_frame.resolve()),
        "output": str(args.output.resolve()),
        "manifest": str(args.manifest.resolve()),
        "selected_source_frames": [int(row["frame_id"]) for row in rows],
        "render": render,
        "verification": verification,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
