"""Hardware enforcement: hard-fail when the current GPU isn't on the allowlist.

Latency, throughput, and cost numbers are all GPU-dependent. Publishing them
requires standardising the silicon — this module makes that consistency a
runtime invariant rather than a documentation note.

Read order:
  1. pricing.yaml -> self_hosted.allowed_gpus
  2. torch.cuda.get_device_name(0)
  3. Substring match (case-insensitive) — accommodates trivial naming variation
     like "NVIDIA GeForce RTX 3090" vs "NVIDIA GeForce RTX 3090 Ti".
"""
from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).parent.parent.parent


class GpuMismatchError(RuntimeError):
    """Raised when the active GPU isn't on the configured allowlist."""


def _load_allowed_gpus() -> list[str]:
    """Read the allowlist from pricing.yaml. Empty list means 'no enforcement'."""
    path = REPO_ROOT / "configs" / "pricing.yaml"
    if not path.exists():
        return []
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    self_hosted = cfg.get("self_hosted") or {}
    allowed = self_hosted.get("allowed_gpus")
    if not allowed:
        return []
    return [str(g) for g in allowed]


def get_current_gpu_name() -> str | None:
    """Return the active CUDA device name, or None if no CUDA / no torch."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_name(0)
    except Exception:
        return None


def check_allowed_gpu(skip: bool = False) -> None:
    """Validate the active GPU against the allowlist.

    Raises GpuMismatchError when the GPU isn't on the allowlist. Silently
    no-ops in these (intentional) cases:
      - skip=True (e.g. smoke-test mode)
      - The allowlist is empty (enforcement disabled)
      - No CUDA device is available (e.g. CPU-only test environments)

    Substring matching is used so "NVIDIA GeForce RTX 3090" matches both the
    plain card and "RTX 3090 Ti" variants without churning the allowlist.
    """
    if skip:
        return
    allowed = _load_allowed_gpus()
    if not allowed:
        return
    current = get_current_gpu_name()
    if current is None:
        # No GPU detected — let downstream code raise its own clearer error
        # about missing CUDA rather than failing the consistency check first.
        return
    current_lc = current.lower()
    for name in allowed:
        if name.lower() in current_lc or current_lc in name.lower():
            return
    raise GpuMismatchError(
        f"Active GPU {current!r} is not on the allowlist {allowed!r}. "
        f"Refusing to run — latency, throughput, and cost numbers depend on "
        f"the GPU model and must not be mixed across hardware in published "
        f"results. To allow this GPU, add it to configs/pricing.yaml "
        f"self_hosted.allowed_gpus. To disable enforcement entirely, set "
        f"allowed_gpus: []."
    )
