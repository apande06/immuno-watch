"""Infection-risk predictor — a multi-output Temporal Transformer (PyTorch).

Clinical purpose:
    The autoencoder answers "is this patient deviating from *their* normal?". This
    model answers the harder clinical question: "is that deviation an infection,
    how severe, and how long until it presents clinically?". Self-attention lets
    it learn the cross-sensor temporal pattern that is the earliest reliable
    signal in immunocompromised patients — HRV degrading 2-4 hours *before*
    temperature rises. Three output heads give the care team not just IF something
    is wrong, but HOW SEVERE and HOW MUCH TIME they have to act.

Technical purpose:
    Input projection -> sinusoidal positional encoding -> TransformerEncoder ->
    global average pool -> {risk, severity, time-to-event} heads. Trained on
    time-ordered windows pooled across patients (each window already z-scored to
    its own patient's baseline, so they are comparable).

Framework: PyTorch — research-grade flexibility for the custom multi-output
architecture; exported via ONNX to TFLite for chip deployment (see ml/export.py).

Reference: Vaswani et al., "Attention Is All You Need", NeurIPS 2017.

Usage:
    python ml/predictor.py
"""

from __future__ import annotations

import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path

# --- path bootstrap ---
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import constants as C
from exceptions import ModelNotTrainedError
from ml.preprocessing import MODEL_FEATURES, BiosignalPreprocessor, make_windows

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("immunowatch.predictor")

REAL_EVENTS: frozenset[str] = frozenset({"infection", "neutropenic_crisis", "viral_mild"})


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding following Vaswani et al. (2017).

    Critical for the Transformer to understand temporal order of readings. Without
    it the model treats timesteps as a bag of features rather than a sequence —
    losing the key clinical signal that HRV degrades BEFORE temperature rises.
    """

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encodings to a (batch, seq, d_model) tensor."""
        return x + self.pe[:, : x.size(1)]


def _head(d_model: int) -> nn.Sequential:
    """Shared two-layer head pattern: Linear(d,32) -> ReLU -> Linear(32,1)."""
    return nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 1))


class InfectionRiskTransformer(nn.Module):
    """Predicts infection risk, severity, and time-to-event from 6-hour windows.

    Note:
        The risk head emits a *logit* (no terminal sigmoid). Training uses
        ``BCEWithLogitsLoss`` with ``pos_weight`` for the 10:1 class imbalance,
        which is numerically stabler than Sigmoid+BCE; the sigmoid is applied at
        inference in :meth:`predict`.
    """

    def __init__(
        self,
        n_sensors: int = C.N_SENSORS,
        d_model: int = C.TRANSFORMER_D_MODEL,
        n_heads: int = C.TRANSFORMER_HEADS,
        n_layers: int = C.TRANSFORMER_LAYERS,
        ffn_dim: int = C.TRANSFORMER_FFN_DIM,
        dropout: float = C.TRANSFORMER_DROPOUT,
        max_len: int = 512,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(n_sensors, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.risk_head = _head(d_model)
        self.severity_head = _head(d_model)
        self.time_to_event_head = _head(d_model)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the encoder and three heads.

        Args:
            x: Tensor of shape (batch, seq, n_sensors).

        Returns:
            Dict with ``risk_logit``, ``severity`` (raw), and ``time_to_event``
            (raw hours), each of shape (batch,).
        """
        h = self.input_projection(x)
        h = self.positional_encoding(h)
        h = self.encoder(h)
        pooled = h.mean(dim=1)  # global average pooling over the sequence
        return {
            "risk_logit": self.risk_head(pooled).squeeze(-1),
            "severity": self.severity_head(pooled).squeeze(-1),
            "time_to_event": self.time_to_event_head(pooled).squeeze(-1),
        }

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Inference helper returning bounded clinical quantities."""
        out = self.forward(x)
        return {
            "risk_score": torch.sigmoid(out["risk_logit"]),
            "severity": out["severity"].clamp(0.0, C.MAX_SEVERITY),
            "time_to_event": out["time_to_event"].clamp(0.0, C.MAX_TIME_TO_EVENT_H),
        }


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------
@dataclass
class WindowedDataset:
    """Container for windowed model inputs and their three targets."""

    x: np.ndarray          # (N, window, 3)
    y_risk: np.ndarray     # (N,) in {0, 1}
    y_severity: np.ndarray  # (N,) in [0, 10]
    y_tte: np.ndarray      # (N,) in [0, 48] hours
    end_index: np.ndarray  # (N,) sample index of each window end (for time split)

    def __len__(self) -> int:
        return len(self.x)

    def subset(self, mask: np.ndarray) -> "WindowedDataset":
        return WindowedDataset(
            self.x[mask], self.y_risk[mask], self.y_severity[mask],
            self.y_tte[mask], self.end_index[mask],
        )


def build_patient_windows(patient_id: str) -> WindowedDataset:
    """Preprocess one patient and build labelled 6-hour windows.

    Labelling (clinical rationale in module docstring):
        * risk = 1 if a real infection event is active at, or begins within the
          6-hour horizon after, the window's end.
        * severity = max event severity over the window + horizon span.
        * time_to_event = hours until the next event onset (0 if already active),
          capped at 48h.

    Args:
        patient_id: Patient to window.

    Returns:
        A :class:`WindowedDataset` for this patient.
    """
    pre = BiosignalPreprocessor(patient_id)
    frame, _ = pre.fit_transform()

    feats = frame[list(MODEL_FEATURES)].to_numpy(dtype=np.float32)
    labels = frame["event_label"].to_numpy()
    severity = frame["severity"].to_numpy(dtype=np.float32)
    real_mask = np.isin(labels, list(REAL_EVENTS))
    onsets = np.where((~np.r_[False, real_mask[:-1]]) & real_mask)[0]

    window = C.PREDICTOR_WINDOW_MINUTES
    horizon = C.PREDICTOR_LABEL_HORIZON_MINUTES
    stride = C.PREDICTOR_WINDOW_STRIDE
    n = len(frame)

    xs, yr, ys, yt, ends = [], [], [], [], []
    for end in range(window - 1, n, stride):
        start = end - window + 1
        future_end = min(end + horizon, n - 1)
        span = real_mask[start : future_end + 1]
        positive = bool(span.any())

        sev = float(severity[start : future_end + 1].max()) if positive else 0.0

        if real_mask[end]:
            tte = 0.0
        else:
            nxt = onsets[onsets >= end]
            tte_min = (nxt[0] - end) if len(nxt) else C.MAX_TIME_TO_EVENT_H * 60
            tte = min(float(tte_min) / 60.0, C.MAX_TIME_TO_EVENT_H)

        xs.append(feats[start : end + 1])
        yr.append(1.0 if positive else 0.0)
        ys.append(sev)
        yt.append(tte)
        ends.append(end)

    return WindowedDataset(
        x=np.asarray(xs, dtype=np.float32),
        y_risk=np.asarray(yr, dtype=np.float32),
        y_severity=np.asarray(ys, dtype=np.float32),
        y_tte=np.asarray(yt, dtype=np.float32),
        end_index=np.asarray(ends, dtype=np.int64),
    )


def time_split(ds: WindowedDataset, train: float = 0.70, val: float = 0.15) -> tuple:
    """Split a single patient's windows chronologically (no shuffling).

    Clinical note:
        Random shuffling would leak future readings into training and inflate
        metrics catastrophically — the model would "peek" at the infection it is
        meant to forecast. We split strictly by time.
    """
    order = np.argsort(ds.end_index)
    n = len(order)
    i_tr, i_va = int(n * train), int(n * (train + val))
    return ds.subset(order[:i_tr]), ds.subset(order[i_tr:i_va]), ds.subset(order[i_va:])


def build_global_dataset() -> tuple[WindowedDataset, WindowedDataset, WindowedDataset]:
    """Build pooled train/val/test sets across every patient (per-patient time split)."""
    tr_parts, va_parts, te_parts = [], [], []
    for patient_id in C.PATIENT_ARCHETYPES:
        ds = build_patient_windows(patient_id)
        tr, va, te = time_split(ds)
        tr_parts.append(tr)
        va_parts.append(va)
        te_parts.append(te)
        logger.info(
            "[%s] windows: train=%d (%.1f%% pos) val=%d test=%d",
            patient_id, len(tr), 100 * tr.y_risk.mean() if len(tr) else 0, len(va), len(te),
        )
    return (_concat(tr_parts), _concat(va_parts), _concat(te_parts))


def _concat(parts: list[WindowedDataset]) -> WindowedDataset:
    return WindowedDataset(
        x=np.concatenate([p.x for p in parts]),
        y_risk=np.concatenate([p.y_risk for p in parts]),
        y_severity=np.concatenate([p.y_severity for p in parts]),
        y_tte=np.concatenate([p.y_tte for p in parts]),
        end_index=np.concatenate([p.end_index for p in parts]),
    )


# ---------------------------------------------------------------------------
# Multi-task loss
# ---------------------------------------------------------------------------
class MultiTaskLoss(nn.Module):
    """Weighted sum: 0.5*BCE(risk) + 0.3*MSE(severity) + 0.2*MAE(time_to_event)."""

    def __init__(self, pos_weight: float = C.CLASS_WEIGHT_RATIO) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()

    def forward(self, out: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]) -> torch.Tensor:
        risk = self.bce(out["risk_logit"], targets["risk"])
        sev = self.mse(out["severity"], targets["severity"])
        tte = self.mae(out["time_to_event"], targets["tte"])
        return C.LOSS_WEIGHT_RISK * risk + C.LOSS_WEIGHT_SEVERITY * sev + C.LOSS_WEIGHT_TTE * tte


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class PredictorTrainer:
    """Trains and evaluates the global infection-risk Transformer."""

    def __init__(self, device: str | None = None) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = InfectionRiskTransformer().to(self.device)

    def train(self) -> dict[str, float]:
        """Full training + evaluation pipeline; saves model and metrics."""
        torch.manual_seed(C.RANDOM_SEED)
        np.random.seed(C.RANDOM_SEED)
        train_ds, val_ds, test_ds = build_global_dataset()

        loader = DataLoader(
            _to_tensors(train_ds), batch_size=C.PREDICTOR_BATCH_SIZE, shuffle=True
        )
        criterion = MultiTaskLoss().to(self.device)
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=C.PREDICTOR_LR, weight_decay=C.PREDICTOR_WEIGHT_DECAY
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=C.PREDICTOR_MAX_EPOCHS)

        history: dict[str, list[float]] = {"train": [], "val": []}
        best_val, best_state = float("inf"), None
        for epoch in range(1, C.PREDICTOR_MAX_EPOCHS + 1):
            train_loss = self._run_epoch(loader, criterion, optimizer)
            val_loss = self._eval_loss(val_ds, criterion)
            scheduler.step()
            history["train"].append(train_loss)
            history["val"].append(val_loss)
            if val_loss < best_val:
                best_val, best_state = val_loss, {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            logger.info("epoch %02d | train %.4f | val %.4f", epoch, train_loss, val_loss)

        if best_state is not None:
            self.model.load_state_dict(best_state)

        metrics = self.evaluate(test_ds)
        self._save(metrics)
        self._plot_training_curve(history)
        return metrics

    def _run_epoch(self, loader: DataLoader, criterion: nn.Module, optimizer) -> float:
        self.model.train()
        total, n = 0.0, 0
        for x, risk, sev, tte in loader:
            x = x.to(self.device)
            targets = {"risk": risk.to(self.device), "severity": sev.to(self.device), "tte": tte.to(self.device)}
            optimizer.zero_grad()
            out = self.model(x)
            loss = criterion(out, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * x.size(0)
            n += x.size(0)
        return total / max(n, 1)

    @torch.no_grad()
    def _eval_loss(self, ds: WindowedDataset, criterion: nn.Module) -> float:
        if len(ds) == 0:
            return float("nan")
        self.model.eval()
        x = torch.from_numpy(ds.x).to(self.device)
        targets = {
            "risk": torch.from_numpy(ds.y_risk).to(self.device),
            "severity": torch.from_numpy(ds.y_severity).to(self.device),
            "tte": torch.from_numpy(ds.y_tte).to(self.device),
        }
        return criterion(self.model(x), targets).item()

    @torch.no_grad()
    def evaluate(self, ds: WindowedDataset) -> dict[str, float]:
        """Compute AUC-ROC, F1, precision, recall, and confusion matrix on a set."""
        self.model.eval()
        x = torch.from_numpy(ds.x).to(self.device)
        probs = torch.sigmoid(self.model(x)["risk_logit"]).cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        y = ds.y_risk.astype(int)

        auc = float(roc_auc_score(y, probs)) if len(np.unique(y)) > 1 else float("nan")
        cm = confusion_matrix(y, preds, labels=[0, 1]).tolist()
        metrics = {
            "auc_roc": auc,
            "f1": float(f1_score(y, preds, zero_division=0)),
            "precision": float(precision_score(y, preds, zero_division=0)),
            "recall": float(recall_score(y, preds, zero_division=0)),
            "n_test": int(len(y)),
            "positives": int(y.sum()),
            "confusion_matrix": cm,
        }
        logger.info(
            "TEST | AUC=%.3f F1=%.3f P=%.3f R=%.3f (n=%d, pos=%d)",
            metrics["auc_roc"], metrics["f1"], metrics["precision"], metrics["recall"],
            metrics["n_test"], metrics["positives"],
        )
        return metrics

    def _save(self, metrics: dict) -> None:
        C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), C.MODELS_DIR / "predictor.pt")
        with open(C.MODELS_DIR / "predictor_metrics.json", "w") as fh:
            json.dump(metrics, fh, indent=2)
        logger.info("Saved predictor.pt + predictor_metrics.json")

    def _plot_training_curve(self, history: dict[str, list[float]]) -> None:
        out_dir = C.REPORTS_DIR / "predictor"
        out_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(history["train"], label="train", color="#3b82f6")
        ax.plot(history["val"], label="validation", color="#f59e0b")
        ax.set_title("Infection-risk Transformer — training")
        ax.set_xlabel("epoch")
        ax.set_ylabel("multi-task loss")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "training_curve.png", dpi=120)
        plt.close(fig)

    def load(self) -> "PredictorTrainer":
        """Load the trained global predictor.

        Raises:
            ModelNotTrainedError: If ``models/predictor.pt`` is missing.
        """
        ckpt = C.MODELS_DIR / "predictor.pt"
        if not ckpt.exists():
            raise ModelNotTrainedError("predictor", str(ckpt))
        self.model.load_state_dict(torch.load(ckpt, map_location=self.device))
        self.model.eval()
        return self


def _to_tensors(ds: WindowedDataset) -> TensorDataset:
    return TensorDataset(
        torch.from_numpy(ds.x),
        torch.from_numpy(ds.y_risk),
        torch.from_numpy(ds.y_severity),
        torch.from_numpy(ds.y_tte),
    )


if __name__ == "__main__":
    PredictorTrainer().train()
