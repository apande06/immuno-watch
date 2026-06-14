"""Inference layer for ImmunoWatch.

Exposes the real-time :class:`InferenceEngine` (the continuous scorer) and the
:class:`AlertExplainer` (SHAP attribution + plain-language alert generation).

Both are torch-backed and imported lazily (PEP 562) so importing this package has
no hard import-time cost until a symbol is actually used.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

_LAZY = {"InferenceEngine": "inference.engine", "AlertExplainer": "inference.explainer"}

if TYPE_CHECKING:
    from inference.engine import InferenceEngine
    from inference.explainer import AlertExplainer


def __getattr__(name: str):
    """Lazily import torch-backed symbols on first access (PEP 562)."""
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_path), name)


__all__ = ["InferenceEngine", "AlertExplainer"]
