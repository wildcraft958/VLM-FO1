"""CPU-only fallback for the MultiScaleDeformableAttention extension.

This module exposes the same symbols as the native CUDA/C++ extension but raises a
clear error that explains how to enable the optimized implementation.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional


class _CPUFallbackModule:
    """Mimics the native extension API while surfacing actionable guidance."""

    def __init__(self, load_error: Optional[BaseException] = None) -> None:
        self._load_error = load_error

    def _raise(self, fn_name: str) -> None:
        hints: Iterable[str] = (
            "Install a matching PyTorch wheel (GPU or CPU) before building the ops package.",
            "If you need CUDA acceleration, ensure the CUDA toolkit headers (e.g., cusparse.h) are available and rerun `pip install -e . --no-build-isolation` inside detect_tools/upn/ops.",
            "For quick experiments without the native build you can use `ms_deform_attn_core_pytorch` inside `ms_deform_attn_func.py`.",
        )
        message = [
            f"MultiScaleDeformableAttention.{fn_name} is unavailable because the native extension failed to load.",
        ]
        message.extend(f"- {hint}" for hint in hints)
        if self._load_error:
            message.append(f"- Original load error: {self._load_error}")
        raise RuntimeError("\n".join(message)) from self._load_error

    def ms_deform_attn_forward(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - runtime guard
        self._raise("ms_deform_attn_forward")

    def ms_deform_attn_backward(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - runtime guard
        self._raise("ms_deform_attn_backward")


def build_cpu_fallback(load_error: Optional[BaseException] = None) -> _CPUFallbackModule:
    """Factory that returns a module-shaped fallback instance."""

    return _CPUFallbackModule(load_error=load_error)
