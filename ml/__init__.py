"""Machine-learning layer for ImmunoWatch.

Exposes the personal-baseline anomaly detector (LSTM autoencoder), the infection
risk predictor (Temporal Transformer), the federated-learning simulation, the
preprocessing pipeline, and the evaluation suite.

The torch-backed symbols are imported **lazily** (PEP 562) so that lightweight,
torch-free consumers — e.g. the preprocessing pipeline or the data simulator — can
``import ml.preprocessing`` without paying the cost of (or requiring) PyTorch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Torch-free symbols are safe to import eagerly.
from ml.preprocessing import MODEL_FEATURES, BiosignalPreprocessor, make_windows

# Map of lazily-loaded names -> their defining module.
_LAZY: dict[str, str] = {
    "BaselineTrainer": "ml.baseline",
    "LSTMAutoencoder": "ml.baseline",
    "InfectionRiskTransformer": "ml.predictor",
    "PositionalEncoding": "ml.predictor",
    "PredictorTrainer": "ml.predictor",
}

if TYPE_CHECKING:  # for type checkers / IDEs only
    from ml.baseline import BaselineTrainer, LSTMAutoencoder
    from ml.predictor import (
        InfectionRiskTransformer,
        PositionalEncoding,
        PredictorTrainer,
    )


def __getattr__(name: str):
    """Lazily import torch-backed symbols on first access (PEP 562)."""
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, name)


__all__ = [
    "BiosignalPreprocessor",
    "MODEL_FEATURES",
    "make_windows",
    "BaselineTrainer",
    "LSTMAutoencoder",
    "InfectionRiskTransformer",
    "PositionalEncoding",
    "PredictorTrainer",
]
