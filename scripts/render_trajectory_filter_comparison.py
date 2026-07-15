from __future__ import annotations

import argparse
import csv
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render baseline and literature-filter trajectory comparison.")
    parser.add_argument("--video", type=Path, default=ROOT / "Dataset" / "2.mp4")
    parser.add_argument(
        "--per-frame",
        type=Path,
        default=ROOT / "outputs" / "literature_trajectory_experiment" / "methods" / "2" / "per_frame.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "assets" / "temp" / "trajectory_filter_comparison_dataset2.mp4",
    )
    parser.add_argument("--start-frame", type=int, default=43)
    parser.add_argument("--end-frame", type=int, default=90)
    parser.add_argument("--output-fps", type=float, default=8.0)
    return parser.parse_args()


def _optional_float(value: str) -> float | None:
    return float(value) if value.strip() else None


def _load_rows(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            frame_id = int(raw["frame_id"])
            rows[frame_id] = {
                "frame_id": frame_id,
                "gt": (_optional_float(raw["gt_x"]), _optional_float(raw["gt_y"])),
                "baseline": (
                    _optional_float(raw["baseline_x"]),
                    _optional_float(raw["baseline_y"]),
                ),
                "baseline_error": _optional_float(raw["baseline_error_px"]),
                "combined": (
                    _optional_float(raw["combined_x"]),
                    _optional_float(raw["combined_y"]),
                ),
                "combined_error": _optional_float(raw["combined_error_px"]),
                "combined_source": raw["combined_source"],
            }
    return rows


def _valid_point(point: tuple[float | None, float | None]) -> bool:
    return point[0] is not None and point[1] is not None


def _scaled_point(
    point: tuple[float | None, float | None],
    scale_x: float,
    scale_y: float,
) -> tuple[int, int] | None:
    if not _valid_point(point):
        return None
    assert point[0] is not None and point[1] is not None
    return int(round(point[0] * scale_x)), int(round(point[1] * scale_y))


def _draw_trail(
    image: np.ndarray,
    points: deque[tuple[int, int] | None],
    color: tuple[int, int, int],
) -> None:
    visible = list(points)
    for index in range(1, len(visible)):
        first = visible[index - 1]
        second = visible[index]
        if first is None or second is None:
            continue
        alpha = index / max(1, len(visible) - 1)
        trail_color = tuple(int(component * (0.30 + 0.70 * alpha)) for component in color)
        cv2.line(image, first, second, trail_color, 2, cv2.LINE_AA)


def _draw_ground_truth(image: np.ndarray, point: tuple[int, int] | None) -> None:
    if point is None:
        return
    cv2.circle(image, point, 9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(image, (point[0] - 12, point[1]), (point[0] + 12, point[1]), (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(image, (point[0], point[1] - 12), (point[0], point[1] + 12), (255, 255, 255), 2, cv2.LINE_AA)


def _draw_prediction(
    image: np.ndarray,
    point: tuple[int, int] | None,
    color: tuple[int, int, int],
) -> None:
    if point is None:
        return
    cv2.circle(image, point, 7, (20, 20, 20), -1, cv2.LINE_AA)
    cv2.circle(image, point, 6, color, -1, cv2.LINE_AA)
    cv2.circle(image, point, 9, color, 2, cv2.LINE_AA)


def _status(error: float | None, point: tuple[int, int] | None) -> tuple[str, tuple[int, int, int]]:
    if point is None:
        return "MISSING", (180, 180, 180)
    if error is None:
        return "NO GT", (180, 180, 180)
    if error <= 20.0:
        return f"OK  {error:.1f}px", (90, 220, 120)
    return f"DRIFT  {error:.1f}px", (90, 90, 245)


def _draw_panel(
    frame: np.ndarray,
    *,
    title: str,
    frame_id: int,
    gt: tuple[int, int] | None,
    prediction: tuple[int, int] | None,
    error: float | None,
    color: tuple[int, int, int],
    trail: deque[tuple[int, int] | None],
    repair_source: str | None = None,
) -> np.ndarray:
    panel = frame.copy()
    overlay = panel.copy()
    cv2.rectangle(overlay, (0, 0), (panel.shape[1], 58), (15, 18, 22), -1)
    cv2.rectangle(overlay, (0, panel.shape[0] - 46), (panel.shape[1], panel.shape[0]), (15, 18, 22), -1)
    cv2.addWeighted(overlay, 0.78, panel, 0.22, 0.0, panel)
    _draw_trail(panel, trail, color)
    _draw_ground_truth(panel, gt)
    _draw_prediction(panel, prediction, color)

    status_text, status_color = _status(error, prediction)
    cv2.putText(panel, title, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(
        panel,
        f"FRAME {frame_id:03d}",
        (panel.shape[1] - 160, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (210, 215, 220),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(panel, status_text, (20, panel.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.62, status_color, 2, cv2.LINE_AA)
    if repair_source and repair_source != "baseline":
        label = repair_source.replace("_", " ").upper()
        cv2.putText(
            panel,
            label,
            (205, panel.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (80, 225, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.rectangle(panel, (3, 3), (panel.shape[1] - 4, panel.shape[0] - 4), (80, 225, 255), 3)
    return panel


def render_comparison(
    video_path: Path,
    per_frame_path: Path,
    output_path: Path,
    *,
    start_frame: int,
    end_frame: int,
    output_fps: float,
) -> dict[str, Any]:
    rows = _load_rows(per_frame_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    pane_width = 960
    pane_height = round(source_height * pane_width / source_width)
    output_size = (pane_width * 2, pane_height)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(output_fps),
        output_size,
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create video: {output_path}")

    scale_x = pane_width / source_width
    scale_y = pane_height / source_height
    baseline_trail: deque[tuple[int, int] | None] = deque(maxlen=14)
    combined_trail: deque[tuple[int, int] | None] = deque(maxlen=14)
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    written = 0
    for frame_id in range(start_frame, end_frame + 1):
        ok, source_frame = capture.read()
        if not ok:
            break
        row = rows.get(frame_id)
        if row is None:
            continue
        resized = cv2.resize(source_frame, (pane_width, pane_height), interpolation=cv2.INTER_AREA)
        gt = _scaled_point(row["gt"], scale_x, scale_y)
        baseline = _scaled_point(row["baseline"], scale_x, scale_y)
        combined = _scaled_point(row["combined"], scale_x, scale_y)
        baseline_trail.append(baseline)
        combined_trail.append(combined)
        left = _draw_panel(
            resized,
            title="CURRENT BASELINE",
            frame_id=frame_id,
            gt=gt,
            prediction=baseline,
            error=row["baseline_error"],
            color=(70, 85, 245),
            trail=baseline_trail,
        )
        right = _draw_panel(
            resized,
            title="LITERATURE FILTER  (3-FRAME LAG)",
            frame_id=frame_id,
            gt=gt,
            prediction=combined,
            error=row["combined_error"],
            color=(245, 210, 60),
            trail=combined_trail,
            repair_source=row["combined_source"],
        )
        writer.write(np.hstack((left, right)))
        written += 1

    capture.release()
    writer.release()
    if written == 0 or not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError("No comparison frames were rendered")
    return {
        "output": str(output_path.resolve()),
        "frames": written,
        "fps": float(output_fps),
        "duration_seconds": written / float(output_fps),
        "width": output_size[0],
        "height": output_size[1],
    }


def main() -> None:
    args = _parse_args()
    result = render_comparison(
        args.video,
        args.per_frame,
        args.output,
        start_frame=int(args.start_frame),
        end_frame=int(args.end_frame),
        output_fps=float(args.output_fps),
    )
    print(result)


if __name__ == "__main__":
    main()
