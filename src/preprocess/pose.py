from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np
import torch


@dataclass
class PosePreprocessMeta:
    orig_size: tuple[int, int]
    resized_size: tuple[int, int]
    scale_x: float
    scale_y: float


def preprocess_pose_frame(
    image: np.ndarray,
    input_size: Tuple[int, int],
    device: str,
) -> tuple[torch.Tensor, PosePreprocessMeta]:
    in_w, in_h = input_size
    orig_h, orig_w = image.shape[:2]
    resized = cv2.resize(image, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    meta = PosePreprocessMeta(
        orig_size=(orig_w, orig_h),
        resized_size=(in_w, in_h),
        scale_x=orig_w / float(in_w),
        scale_y=orig_h / float(in_h),
    )
    return tensor, meta
