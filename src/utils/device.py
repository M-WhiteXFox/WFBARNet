from __future__ import annotations

import torch


def resolve_device(device: str | None) -> str:
    """Resolve user-facing device aliases into concrete torch device strings."""
    normalized = (device or "auto").strip().lower()
    if normalized in {"", "auto"}:
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but it is not available in this environment.")
    if normalized == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None or not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested, but it is not available in this environment.")
    return device or "cpu"
