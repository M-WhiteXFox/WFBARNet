from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.court import opencv_court_homography_core as _court_core
from src.court.courtkeynet_model import COURTKEYNET_MODEL_CONFIG, CourtKeyNet
from src.court.opencv_court_detector import CourtLinePrediction


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(slots=True)
class CourtKeyNetConfig:
    weights: str = "assets/weights/courtkeynet/CourtKeyNet.safetensors"
    imgsz: int = 640
    confidence_threshold: float = 0.50
    confirmation_frames: int = 3
    max_corner_shift_ratio: float = 0.035
    device: str = ""


def resolve_courtkeynet_weights(raw: str | Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        expected = path.resolve()
    else:
        project_root = PROJECT_ROOT.resolve()
        expected = (project_root / path).resolve()
        try:
            expected.relative_to(project_root)
        except ValueError as exc:
            raise FileNotFoundError(
                f"CourtKeyNet weights not found at expected path: {expected}"
            ) from exc
    if expected.is_file():
        return expected
    raise FileNotFoundError(f"CourtKeyNet weights not found at expected path: {expected}")


def heatmap_courtkeynet_confidence(
    heatmaps: torch.Tensor,
    keypoints: torch.Tensor | None = None,
) -> torch.Tensor:
    del keypoints
    _validate_heatmaps(heatmaps)
    batch_size, point_count = heatmaps.shape[:2]
    probabilities = F.softmax(heatmaps.reshape(batch_size, point_count, -1), dim=-1)
    peak_probabilities = probabilities.amax(dim=-1)
    per_keypoint = ((peak_probabilities - 0.05) / (0.40 - 0.05)).clamp(0.0, 1.0)
    return per_keypoint.mean(dim=1)


def entropy_courtkeynet_confidence(heatmaps: torch.Tensor) -> torch.Tensor:
    _validate_heatmaps(heatmaps)
    batch_size, point_count, height, width = heatmaps.shape
    probabilities = F.softmax(heatmaps.reshape(batch_size, point_count, -1), dim=-1)
    entropy = -(probabilities * torch.log(probabilities + 1e-10)).sum(dim=-1)
    max_entropy = math.log(height * width)
    if max_entropy <= 0.0:
        return torch.ones((batch_size,), dtype=heatmaps.dtype, device=heatmaps.device)
    return 1.0 - (entropy / max_entropy).mean(dim=1)


def geometric_courtkeynet_confidence(
    keypoints: torch.Tensor,
) -> torch.Tensor:
    if keypoints.ndim != 3 or keypoints.shape[1:] != (4, 2):
        raise ValueError("CourtKeyNet keypoints must have shape Bx4x2.")
    if not torch.isfinite(keypoints).all().item():
        raise ValueError("CourtKeyNet keypoints must contain only finite values.")

    def cross_2d(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        return first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]

    edges = torch.stack(
        [
            keypoints[:, 1] - keypoints[:, 0],
            keypoints[:, 2] - keypoints[:, 1],
            keypoints[:, 3] - keypoints[:, 2],
            keypoints[:, 0] - keypoints[:, 3],
        ],
        dim=1,
    )
    crosses = torch.stack(
        [
            cross_2d(edges[:, 0], edges[:, 1]),
            cross_2d(edges[:, 1], edges[:, 2]),
            cross_2d(edges[:, 2], edges[:, 3]),
            cross_2d(edges[:, 3], edges[:, 0]),
        ],
        dim=1,
    )
    convex = ((crosses > 0).all(dim=1) | (crosses < 0).all(dim=1)).to(keypoints.dtype)

    x = keypoints[:, :, 0]
    y = keypoints[:, :, 1]
    area = 0.5 * torch.abs(
        x[:, 0] * (y[:, 1] - y[:, 3])
        + x[:, 1] * (y[:, 2] - y[:, 0])
        + x[:, 2] * (y[:, 3] - y[:, 1])
        + x[:, 3] * (y[:, 0] - y[:, 2])
    )
    area_valid = ((area > 0.01) & (area < 0.95)).to(keypoints.dtype)

    width_top = torch.linalg.vector_norm(keypoints[:, 1] - keypoints[:, 0], dim=1)
    width_bottom = torch.linalg.vector_norm(keypoints[:, 2] - keypoints[:, 3], dim=1)
    height_left = torch.linalg.vector_norm(keypoints[:, 3] - keypoints[:, 0], dim=1)
    height_right = torch.linalg.vector_norm(keypoints[:, 2] - keypoints[:, 1], dim=1)
    average_width = (width_top + width_bottom) / 2.0
    average_height = (height_left + height_right) / 2.0
    aspect_ratio = average_width / (average_height + 1e-6)
    aspect_valid = ((aspect_ratio > 0.3) & (aspect_ratio < 5.0)).to(keypoints.dtype)

    center = keypoints.mean(dim=1)
    top_left_valid = (
        (keypoints[:, 0, 0] < center[:, 0]) & (keypoints[:, 0, 1] < center[:, 1])
    ).to(keypoints.dtype)
    top_right_valid = (
        (keypoints[:, 1, 0] > center[:, 0]) & (keypoints[:, 1, 1] < center[:, 1])
    ).to(keypoints.dtype)
    bottom_right_valid = (
        (keypoints[:, 2, 0] > center[:, 0]) & (keypoints[:, 2, 1] > center[:, 1])
    ).to(keypoints.dtype)
    bottom_left_valid = (
        (keypoints[:, 3, 0] < center[:, 0]) & (keypoints[:, 3, 1] > center[:, 1])
    ).to(keypoints.dtype)
    position_valid = (
        top_left_valid + top_right_valid + bottom_right_valid + bottom_left_valid
    ) / 4.0

    return 0.30 * convex + 0.20 * area_valid + 0.20 * aspect_valid + 0.30 * position_valid


def combined_courtkeynet_confidence(
    heatmaps: torch.Tensor,
    keypoints: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    heatmap = heatmap_courtkeynet_confidence(heatmaps, keypoints)
    geometry = geometric_courtkeynet_confidence(keypoints)
    entropy = entropy_courtkeynet_confidence(heatmaps)
    if heatmap.shape != geometry.shape or heatmap.shape != entropy.shape:
        raise ValueError("CourtKeyNet confidence inputs must use the same batch size.")
    combined = 0.40 * heatmap + 0.40 * geometry + 0.20 * entropy
    return combined, {
        "heatmap": heatmap,
        "geometry": geometry,
        "entropy": entropy,
    }


class CourtKeyNetLineDetector:
    def __init__(self, config: CourtKeyNetConfig | None = None, *, model: Any | None = None) -> None:
        self.config = config or CourtKeyNetConfig()
        self._model = model
        self._model_prepared = False
        self._weights_path: Path | None = None
        self._device = _resolve_device(self.config.device)
        self._latest_prediction: CourtLinePrediction | None = None
        self._rejected_count = 0

    def reset(self) -> None:
        self._latest_prediction = None
        self._rejected_count = 0

    def latest_prediction(self) -> CourtLinePrediction | None:
        return self._latest_prediction

    def predict(
        self,
        frame: np.ndarray,
        frame_id: int,
        timestamp_ms: int,
        *,
        force: bool = False,
    ) -> CourtLinePrediction:
        del force
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("CourtKeyNet prediction expects a three-channel BGR image frame.")
        if int(self.config.imgsz) <= 0:
            raise ValueError("CourtKeyNet imgsz must be positive.")

        frame_id = int(frame_id)
        timestamp_ms = max(0, int(timestamp_ms))
        height, width = frame.shape[:2]
        started_at = perf_counter()
        candidate_confidence: float | None = None
        components: dict[str, float] = {}

        try:
            model = self._ensure_model()
            inputs = _preprocess_courtkeynet_frame(frame, int(self.config.imgsz), self._device)
            with torch.inference_mode():
                output = model(inputs)
            heatmaps, refined_points = _validated_model_output(output)
            combined, score_components = combined_courtkeynet_confidence(heatmaps, refined_points)
            candidate_confidence = float(combined[0].detach().cpu().item())
            components = {
                "courtkeynet_heatmap_confidence": _tensor_scalar(score_components["heatmap"]),
                "courtkeynet_geometry_confidence": _tensor_scalar(score_components["geometry"]),
                "courtkeynet_entropy_confidence": _tensor_scalar(score_components["entropy"]),
                "courtkeynet_combined_confidence": candidate_confidence,
                "courtkeynet_confidence_threshold": float(self.config.confidence_threshold),
            }

            normalized = refined_points[0].detach().to(device="cpu", dtype=torch.float32).numpy()
            source_points = normalized * np.array([float(width), float(height)], dtype=np.float32)
            corners = _court_core.order_quad_points(source_points)
            if corners is None:
                raise ValueError("CourtKeyNet refined points do not form a valid quadrilateral.")
            court_to_image_h, image_to_court_h = _court_core.compute_homographies(corners)
            if court_to_image_h is None or image_to_court_h is None:
                raise ValueError("CourtKeyNet quadrilateral could not produce finite homographies.")

            projected_lines = _court_core.project_template_lines(court_to_image_h)
            keypoint_template = _court_core.template_keypoints_for_scheme("8")
            keypoint_names = [name for name, _ in keypoint_template]
            template_points = np.asarray([point for _, point in keypoint_template], dtype=np.float32)
            keypoints = _court_core.project_points(template_points, court_to_image_h)
            geometry = _prediction_geometry(
                corners=corners,
                keypoint_names=keypoint_names,
                keypoints=keypoints,
                court_to_image_h=court_to_image_h,
                image_to_court_h=image_to_court_h,
                projected_lines=projected_lines,
            )
        except (KeyError, TypeError, ValueError) as exc:
            self._rejected_count += 1
            prediction = self._invalid_prediction(
                frame_id=frame_id,
                timestamp_ms=timestamp_ms,
                source_size=(int(width), int(height)),
                candidate_confidence=candidate_confidence,
                reason=str(exc),
                components=components,
                detect_ms=(perf_counter() - started_at) * 1000.0,
            )
            self._latest_prediction = prediction
            return prediction

        threshold = float(self.config.confidence_threshold)
        confirmation_frames = max(1, int(self.config.confirmation_frames))
        detect_ms = (perf_counter() - started_at) * 1000.0
        if candidate_confidence < threshold:
            self._rejected_count += 1
            prediction = CourtLinePrediction(
                frame_id=frame_id,
                timestamp_ms=timestamp_ms,
                source_size=(int(width), int(height)),
                valid=False,
                attempted=True,
                updated=False,
                update_type="courtkeynet rejected",
                status="courtkeynet confidence below threshold",
                confidence=0.0,
                candidate_confidence=candidate_confidence,
                reason=f"confidence {candidate_confidence:.3f} below threshold {threshold:.3f}",
                scheme="courtkeynet",
                metrics={"components": components},
                detect_ms=float(detect_ms),
                rejected_count=int(self._rejected_count),
                **geometry,
            )
        elif confirmation_frames > 1:
            status = f"courtkeynet confirmation 1/{confirmation_frames}"
            prediction = CourtLinePrediction(
                frame_id=frame_id,
                timestamp_ms=timestamp_ms,
                source_size=(int(width), int(height)),
                valid=False,
                attempted=True,
                updated=False,
                update_type="courtkeynet confirmation",
                status=status,
                confidence=0.0,
                candidate_confidence=candidate_confidence,
                reason=status,
                scheme="courtkeynet",
                metrics={"components": components},
                detect_ms=float(detect_ms),
                rejected_count=int(self._rejected_count),
                **geometry,
            )
        else:
            prediction = CourtLinePrediction(
                frame_id=frame_id,
                timestamp_ms=timestamp_ms,
                source_size=(int(width), int(height)),
                valid=True,
                attempted=True,
                updated=True,
                update_type="courtkeynet detection",
                status="courtkeynet detection",
                confidence=candidate_confidence,
                candidate_confidence=candidate_confidence,
                reason="courtkeynet detection",
                scheme="courtkeynet",
                metrics={"components": components},
                detect_ms=float(detect_ms),
                rejected_count=int(self._rejected_count),
                **geometry,
            )

        self._latest_prediction = prediction
        return prediction

    def _ensure_model(self) -> Any:
        if self._model is None:
            try:
                from safetensors.torch import load_file
            except ImportError as exc:
                raise RuntimeError("Missing dependency: safetensors.") from exc

            self._weights_path = resolve_courtkeynet_weights(self.config.weights)
            model = CourtKeyNet(COURTKEYNET_MODEL_CONFIG)
            state_dict = load_file(str(self._weights_path), device="cpu")
            model.load_state_dict(state_dict, strict=True)
            self._model = model

        if not self._model_prepared:
            self._model = self._model.to(self._device)
            self._model.eval()
            self._model_prepared = True
        return self._model

    def _invalid_prediction(
        self,
        *,
        frame_id: int,
        timestamp_ms: int,
        source_size: tuple[int, int],
        candidate_confidence: float | None,
        reason: str,
        components: dict[str, float],
        detect_ms: float,
    ) -> CourtLinePrediction:
        return CourtLinePrediction(
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            source_size=source_size,
            valid=False,
            attempted=True,
            updated=False,
            update_type="courtkeynet rejected",
            status="courtkeynet candidate rejected",
            confidence=0.0,
            candidate_confidence=candidate_confidence,
            reason=reason,
            scheme="courtkeynet",
            corners=[],
            keypoints=[],
            court_to_image_h=[],
            image_to_court_h=[],
            projected_lines={},
            metrics={"components": components},
            detect_ms=float(detect_ms),
            rejected_count=int(self._rejected_count),
        )


def _validate_heatmaps(heatmaps: torch.Tensor) -> None:
    if not isinstance(heatmaps, torch.Tensor):
        raise TypeError("CourtKeyNet heatmaps must be a torch.Tensor.")
    if heatmaps.ndim != 4 or heatmaps.shape[1] != 4 or heatmaps.shape[2] <= 0 or heatmaps.shape[3] <= 0:
        raise ValueError("CourtKeyNet heatmaps must have shape Bx4xHxW.")
    if not heatmaps.is_floating_point():
        raise ValueError("CourtKeyNet heatmaps must use a floating-point dtype.")
    if not torch.isfinite(heatmaps).all().item():
        raise ValueError("CourtKeyNet heatmaps must contain only finite values.")


def _validated_model_output(output: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(output, Mapping):
        raise TypeError("CourtKeyNet model output must be a mapping.")
    if "heatmaps" not in output or "kpts_refined" not in output:
        raise ValueError("CourtKeyNet model output requires heatmaps and kpts_refined.")
    heatmaps = output["heatmaps"]
    refined_points = output["kpts_refined"]
    _validate_heatmaps(heatmaps)
    if heatmaps.shape[0] != 1:
        raise ValueError("CourtKeyNet heatmaps must contain exactly one frame.")
    if not isinstance(refined_points, torch.Tensor):
        raise TypeError("CourtKeyNet kpts_refined must be a torch.Tensor.")
    if refined_points.shape != (1, 4, 2):
        raise ValueError("CourtKeyNet kpts_refined must have shape 1x4x2.")
    if not refined_points.is_floating_point():
        raise ValueError("CourtKeyNet kpts_refined must use a floating-point dtype.")
    if not torch.isfinite(refined_points).all().item():
        raise ValueError("CourtKeyNet kpts_refined must contain only finite values.")
    if heatmaps.device != refined_points.device:
        raise ValueError("CourtKeyNet heatmaps and kpts_refined must be on the same device.")
    return heatmaps, refined_points


def _resolve_device(raw: str) -> torch.device:
    value = str(raw).strip()
    if value:
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _preprocess_courtkeynet_frame(
    frame: np.ndarray,
    imgsz: int,
    device: torch.device,
) -> torch.Tensor:
    resized = cv2.resize(frame, (imgsz, imgsz))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return (
        torch.from_numpy(rgb)
        .permute(2, 0, 1)
        .contiguous()
        .to(device=device, dtype=torch.float32)
        .div_(255.0)
        .unsqueeze(0)
    )


def _tensor_scalar(value: torch.Tensor) -> float:
    return float(value.reshape(-1)[0].detach().cpu().item())


def _prediction_geometry(
    *,
    corners: np.ndarray,
    keypoint_names: list[str],
    keypoints: np.ndarray,
    court_to_image_h: np.ndarray,
    image_to_court_h: np.ndarray,
    projected_lines: dict[str, np.ndarray],
) -> dict[str, Any]:
    return {
        "corners": _points_to_list(corners),
        "keypoints": _keypoints_to_list(keypoint_names, keypoints),
        "court_to_image_h": _matrix_to_list(court_to_image_h),
        "image_to_court_h": _matrix_to_list(image_to_court_h),
        "projected_lines": _projected_lines_to_list(projected_lines),
    }


def _points_to_list(points: Any) -> list[list[float]]:
    array = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    return [[float(x), float(y)] for x, y in array if np.isfinite(x) and np.isfinite(y)]


def _keypoints_to_list(names: list[str], points: Any) -> list[dict[str, Any]]:
    return [
        {"name": str(names[index]), "point": point}
        for index, point in enumerate(_points_to_list(points))
    ]


def _matrix_to_list(matrix: Any) -> list[list[float]]:
    array = np.asarray(matrix, dtype=np.float64)
    if array.shape != (3, 3) or not np.isfinite(array).all():
        return []
    return [[float(value) for value in row] for row in array]


def _projected_lines_to_list(lines: dict[str, Any]) -> dict[str, list[list[float]]]:
    return {str(name): _points_to_list(points) for name, points in lines.items()}
