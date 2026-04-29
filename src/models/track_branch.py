from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import warnings

import numpy as np
import torch

from src.models.tensorrt_track_backend import TensorRTTrackBackend
from src.models.tracknet_v3 import TrackNetV3
from src.postprocess.track import (
    decode_track_heatmap,
    decode_track_heatmap_batch,
    decode_track_heatmap_candidate_batch,
    decode_track_heatmap_candidates,
)
from src.preprocess.track import preprocess_track_window, preprocess_track_batch
from src.utils.device import resolve_device
from src.utils.structures import TrackResult


@dataclass
class TrackBranch:
    model_weight: str | None = None
    device: str = "cpu"
    input_size: tuple[int, int] = (512, 288)
    score_thr: float = 0.5
    max_candidates: int = 5
    candidate_score_thr_ratio: float = 0.6
    allow_random_weights: bool = False

    def __post_init__(self) -> None:
        self.device = resolve_device(self.device)
        self.backend_name = self._resolve_backend_name()
        if self.backend_name == "tensorrt":
            self._use_amp = False
            self._use_channels_last = False
            self.model = TensorRTTrackBackend(str(self.model_weight), self.device)
            return

        self._use_amp = "cuda" in self.device
        self._use_channels_last = self._use_amp and torch.cuda.is_available()
        self.model = TrackNetV3().to(self.device).eval()
        if self._use_amp:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
            self.model = self.model.half()
        if self._use_channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)
        if self.model_weight and Path(self.model_weight).exists():
            state = torch.load(self.model_weight, map_location=self.device)
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            elif isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            if isinstance(state, dict):
                cleaned = {k.replace("module.", ""): v for k, v in state.items()}
                missing, unexpected = self.model.load_state_dict(cleaned, strict=False)
                if missing or unexpected:
                    warnings.warn(
                        f"TrackNet checkpoint did not load cleanly. missing={len(missing)} unexpected={len(unexpected)}",
                        stacklevel=2,
                    )
        else:
            message = f"TrackNet weight file not found: {self.model_weight}"
            if not self.allow_random_weights:
                raise FileNotFoundError(
                    f"{message}. Provide a valid checkpoint or set allow_random_weights=True for explicit test/demo runs."
                )
            warnings.warn(f"{message}. Running with random weights.", stacklevel=2)

    def _resolve_backend_name(self) -> str:
        suffix = Path(self.model_weight).suffix.lower() if self.model_weight else ""
        if suffix == ".engine":
            return "tensorrt"
        return "pytorch"

    def _prepare_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if self._use_amp:
            tensor = tensor.half()
        if self._use_channels_last:
            tensor = tensor.contiguous(memory_format=torch.channels_last)
        return tensor

    @torch.inference_mode()
    def infer(self, frames: Sequence[np.ndarray]) -> tuple[np.ndarray, TrackResult]:
        if self.backend_name == "tensorrt":
            with self.model.stream_context():
                tensor, meta = preprocess_track_window(frames, self.input_size, self.device)
                heatmaps = self.model.infer(tensor)
            decoded = decode_track_heatmap(heatmaps, meta, self.score_thr)
            return heatmaps, decoded

        tensor, meta = preprocess_track_window(frames, self.input_size, self.device)
        tensor = self._prepare_tensor(tensor)
        heatmaps = self.model(tensor).float().detach().cpu().numpy()
        decoded = decode_track_heatmap(heatmaps, meta, self.score_thr)
        return heatmaps, decoded

    @torch.inference_mode()
    def infer_result(self, frames: Sequence[np.ndarray]) -> TrackResult:
        if self.backend_name == "tensorrt":
            with self.model.stream_context():
                tensor, meta = preprocess_track_window(frames, self.input_size, self.device)
                heatmap = self.predict_heatmap_planes(tensor).squeeze(0)
            return decode_track_heatmap(heatmap, meta, self.score_thr)

        tensor, meta = preprocess_track_window(frames, self.input_size, self.device)
        heatmap = self.predict_heatmap_planes(tensor).squeeze(0)
        return decode_track_heatmap(heatmap, meta, self.score_thr)

    @torch.inference_mode()
    def infer_candidate_results(self, frames: Sequence[np.ndarray]) -> list[TrackResult]:
        if self.backend_name == "tensorrt":
            with self.model.stream_context():
                tensor, meta = preprocess_track_window(frames, self.input_size, self.device)
                heatmap = self.predict_heatmap_planes(tensor).squeeze(0)
            return decode_track_heatmap_candidates(
                heatmap,
                meta,
                self.score_thr,
                max_candidates=self.max_candidates,
                candidate_score_thr=self._candidate_score_thr(),
            )

        tensor, meta = preprocess_track_window(frames, self.input_size, self.device)
        heatmap = self.predict_heatmap_planes(tensor).squeeze(0)
        return decode_track_heatmap_candidates(
            heatmap,
            meta,
            self.score_thr,
            max_candidates=self.max_candidates,
            candidate_score_thr=self._candidate_score_thr(),
        )

    @torch.inference_mode()
    def infer_batch(self, batch_frames: list[Sequence[np.ndarray]]) -> tuple[np.ndarray, list[TrackResult]]:
        if self.backend_name == "tensorrt":
            with self.model.stream_context():
                tensor, metas = preprocess_track_batch(batch_frames, self.input_size, self.device)
                heatmaps = self._infer_tensorrt_batch(tensor)
            decoded_batch = decode_track_heatmap_batch(heatmaps, metas, self.score_thr)
            return heatmaps, decoded_batch

        tensor, metas = preprocess_track_batch(batch_frames, self.input_size, self.device)
        tensor = self._prepare_tensor(tensor)
        heatmaps = self.model(tensor).float().detach().cpu().numpy()
        decoded_batch = decode_track_heatmap_batch(heatmaps, metas, self.score_thr)
        return heatmaps, decoded_batch

    @torch.inference_mode()
    def infer_batch_results(self, batch_frames: list[Sequence[np.ndarray]]) -> list[TrackResult]:
        if self.backend_name == "tensorrt":
            with self.model.stream_context():
                tensor, metas = preprocess_track_batch(batch_frames, self.input_size, self.device)
                heatmaps = self.predict_heatmap_planes(tensor)
            return decode_track_heatmap_batch(heatmaps, metas, self.score_thr)

        tensor, metas = preprocess_track_batch(batch_frames, self.input_size, self.device)
        heatmaps = self.predict_heatmap_planes(tensor)
        return decode_track_heatmap_batch(heatmaps, metas, self.score_thr)

    @torch.inference_mode()
    def infer_batch_candidate_results(self, batch_frames: list[Sequence[np.ndarray]]) -> list[list[TrackResult]]:
        if self.backend_name == "tensorrt":
            with self.model.stream_context():
                tensor, metas = preprocess_track_batch(batch_frames, self.input_size, self.device)
                heatmaps = self.predict_heatmap_planes(tensor)
            return decode_track_heatmap_candidate_batch(
                heatmaps,
                metas,
                self.score_thr,
                max_candidates=self.max_candidates,
                candidate_score_thr=self._candidate_score_thr(),
            )

        tensor, metas = preprocess_track_batch(batch_frames, self.input_size, self.device)
        heatmaps = self.predict_heatmap_planes(tensor)
        return decode_track_heatmap_candidate_batch(
            heatmaps,
            metas,
            self.score_thr,
            max_candidates=self.max_candidates,
            candidate_score_thr=self._candidate_score_thr(),
        )

    @torch.inference_mode()
    def predict_heatmap_planes(self, tensor: torch.Tensor) -> np.ndarray:
        if self.backend_name == "tensorrt":
            heatmaps = self._infer_tensorrt_batch(tensor)
            if heatmaps.ndim == 4:
                return heatmaps[:, 1]
            if heatmaps.ndim == 3:
                return heatmaps
            raise ValueError(f"Unexpected TensorRT heatmap shape: {heatmaps.shape}")

        tensor = self._prepare_tensor(tensor)
        return self.model(tensor)[:, 1].float().detach().cpu().numpy()

    def _infer_tensorrt_batch(self, tensor: torch.Tensor) -> np.ndarray:
        if tensor.shape[0] <= 1:
            return self.model.infer(tensor)
        outputs = [self.model.infer(tensor[index : index + 1]) for index in range(tensor.shape[0])]
        return np.concatenate(outputs, axis=0)

    def _candidate_score_thr(self) -> float:
        ratio = min(max(float(self.candidate_score_thr_ratio), 0.0), 1.0)
        return float(self.score_thr) * ratio
