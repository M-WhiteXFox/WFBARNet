from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import warnings

import numpy as np
import torch

from src.models.tracknet_v3 import TrackNetV3
from src.postprocess.track import decode_track_heatmap, decode_track_heatmap_batch
from src.preprocess.track import preprocess_track_window, preprocess_track_batch
from src.utils.device import resolve_device
from src.utils.structures import TrackResult


@dataclass
class TrackBranch:
    model_weight: str | None = None
    device: str = "cpu"
    input_size: tuple[int, int] = (512, 288)
    score_thr: float = 0.5
    allow_random_weights: bool = False

    def __post_init__(self) -> None:
        self.device = resolve_device(self.device)
        self._use_amp = "cuda" in self.device
        self.model = TrackNetV3().to(self.device).eval()
        if self._use_amp:
            self.model = self.model.half()
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

    @torch.no_grad()
    def infer(self, frames: Sequence[np.ndarray]) -> tuple[np.ndarray, TrackResult]:
        tensor, meta = preprocess_track_window(frames, self.input_size, self.device)
        if self._use_amp:
            tensor = tensor.half()
        heatmaps = self.model(tensor).float().detach().cpu().numpy()
        decoded = decode_track_heatmap(heatmaps, meta, self.score_thr)
        return heatmaps, decoded

    @torch.no_grad()
    def infer_batch(self, batch_frames: list[Sequence[np.ndarray]]) -> tuple[np.ndarray, list[TrackResult]]:
        tensor, metas = preprocess_track_batch(batch_frames, self.input_size, self.device)
        if self._use_amp:
            tensor = tensor.half()
        heatmaps = self.model(tensor).float().detach().cpu().numpy()
        decoded_batch = decode_track_heatmap_batch(heatmaps, metas, self.score_thr)
        return heatmaps, decoded_batch
