"""Runtime loader for the MultiScaleDeformableAttention native extension."""

from __future__ import annotations

import importlib
import warnings

from .cpu_fallback import build_cpu_fallback

_LOAD_ERROR = None

try:
    MultiScaleDeformableAttention = importlib.import_module("MultiScaleDeformableAttention")
except Exception as exc:  # pragma: no cover - runtime import guard
    _LOAD_ERROR = exc
    warnings.warn(
        "MultiScaleDeformableAttention CUDA extension not found. Falling back to a CPU-only stub. "
        "Install the ops package with an environment that satisfies the native build prerequisites to enable acceleration.",
        stacklevel=2,
    )
    MultiScaleDeformableAttention = build_cpu_fallback(exc)
else:
    MultiScaleDeformableAttention = MultiScaleDeformableAttention


__all__ = [
    "MultiScaleDeformableAttention",
    "load_multi_scale_deformable_attention",
    "is_native_extension_available",
]


def load_multi_scale_deformable_attention():
    """Return the best available implementation (native or CPU fallback)."""

    return MultiScaleDeformableAttention


def is_native_extension_available() -> bool:
    """Expose whether the optimized CUDA/C++ extension was imported successfully."""

    return _LOAD_ERROR is None
