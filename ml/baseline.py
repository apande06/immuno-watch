"""Personal-baseline anomaly detector — a PyTorch LSTM autoencoder.

Clinical purpose:
    Every immunocompromised patient has a *personal* normal. A transplant
    recipient's resting HRV, a chemo patient's nadir temperature — these differ so
    much that any population threshold is useless. This model is trained
    exclusively on an individual's first 14 healthy days, so it learns *their*
    physiology. When infection perturbs that physiology, the reconstruction error
    spikes — often hours before a population rule (e.g. the 38.3C fever floor)
    would ever fire.

Technical purpose:
    Encoder-bottleneck-decoder LSTM. The 2-hour input window is compressed to a
    16-dim bottleneck and reconstructed; mean squared reconstruction error,
    normalised by a per-patient 95th-percentile threshold, is the anomaly score.

Framework: PyTorch — chosen for flexibility in defining the custom
encoder-bottleneck-decoder architecture and for compatibility with TorchScript /
ONNX export for edge deployment.

Reference: Malhotra et al., "LSTM-based Encoder-Decoder for Multi-sensor Anomaly
Detection", ICML 2016 Anomaly Detection Workshop.

Usage:
    python ml/baseline.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# --- path bootstrap so `python ml/baseline.py` can import project modules ---
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import constants as C
from exceptions import ModelNotTrainedError
from ml.preprocessing import MODEL_FEATURES, BiosignalPreprocessor, make_windows

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("immunowatch.baseline")


class LSTMAutoencoder(nn.Module):
    """Patient-specific anomaly detector trained only on healthy baseline data.

    Architecture:
        Encoder LSTM -> Linear bottleneck (ReLU) -> Decoder LSTM -> reconstruction.

    Input shape:  (batch, 120, 3) — 2-hour window, 3 sensors.
    Output shape: (batch, 120, 3) — reconstructed signal.

    Clinical rationale:
        Training only on normal physiology means the model learns THIS patient's
        healthy patterns. Infection-driven shifts produce high reconstruction
        error before symptoms emerge, because the model has never seen that
        pattern during training.
    """

    def __init__(
        self,
        n_sensors: int = C.N_SENSORS,
        hidden_size: int = C.LSTM_HIDDEN_SIZE,
        num_layers: int = C.LSTM_LAYERS,
        bottleneck: int = C.BOTTLENECK_SIZE,
        dropout: float = C.LSTM_DROPOUT,
        window: int = C.BASELINE_WINDOW_MINUTES,
    ) -> None:
        super().__init__()
        self.window = window
        self.n_sensors = n_sensors

        self.encoder = nn.LSTM(
            input_size=n_sensors,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.bottleneck = nn.Sequential(nn.Linear(hidden_size, bottleneck), nn.ReLU())
        self.decoder_input = nn.Linear(bottleneck, hidden_size)
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_size, n_sensors)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct a batch of sensor windows.

        Args:
            x: Tensor of shape (batch, window, n_sensors).

        Returns:
            Reconstruction of identical shape.
        """
        _, (hidden, _) = self.encoder(x)
        latent = self.bottleneck(hidden[-1])              # (batch, bottleneck)
        seed = self.decoder_input(latent)                 # (batch, hidden)
        repeated = seed.unsqueeze(1).repeat(1, x.size(1), 1)  # (batch, window, hidden)
        decoded, _ = self.decoder(repeated)
        return self.output(decoded)


class BaselineTrainer:
    """Manages training lifecycle, checkpointing, and threshold calibration."""

    def __init__(self, patient_id: str, device: str | None = None) -> None:
        self.patient_id = patient_id
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = LSTMAutoencoder().to(self.device)
        self.threshold: float = 1.0
        self.baseline_stats = None  # set during train(); used by _save_baseline_stats
        self.model_dir = C.MODELS_DIR / patient_id
        self.report_dir = C.REPORTS_DIR / patient_id

    # ------------------------------------------------------------ training
    def train(self) -> dict[str, float]:
        """Run the full baseline training + calibration pipeline.

        Returns:
            Summary metrics: best validation loss, calibrated threshold, epochs run.
        """
        torch.manual_seed(C.RANDOM_SEED)
        pre = BiosignalPreprocessor(self.patient_id)
        frame, _ = pre.fit_transform()
        self.baseline_stats = pre.baseline_stats

        train_w, val_w, calib_w = self._split_windows(frame)
        logger.info(
            "[%s] windows -> train=%d val=%d calib=%d",
            self.patient_id, len(train_w), len(val_w), len(calib_w),
        )

        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(train_w)),
            batch_size=C.BASELINE_BATCH_SIZE,
            shuffle=True,
        )
        val_tensor = torch.from_numpy(val_w).to(self.device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=C.BASELINE_LR, weight_decay=C.BASELINE_WEIGHT_DECAY
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=C.BASELINE_SCHED_PATIENCE
        )

        history: dict[str, list[float]] = {"train": [], "val": []}
        best_val = float("inf")
        best_state = None
        patience = 0

        for epoch in range(1, C.BASELINE_MAX_EPOCHS + 1):
            train_loss = self._run_epoch(train_loader, criterion, optimizer)
            val_loss = self._validate(val_tensor, criterion)
            scheduler.step(val_loss)
            history["train"].append(train_loss)
            history["val"].append(val_loss)

            if val_loss < best_val - 1e-6:
                best_val, best_state, patience = val_loss, self._snapshot(), 0
            else:
                patience += 1
            logger.info(
                "[%s] epoch %02d | train %.5f | val %.5f | best %.5f",
                self.patient_id, epoch, train_loss, val_loss, best_val,
            )
            if patience >= C.BASELINE_EARLY_STOP_PATIENCE:
                logger.info("[%s] early stopping at epoch %d", self.patient_id, epoch)
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.threshold = self._calibrate_threshold(calib_w)
        self._save_artifacts()
        self._plot_training_curve(history)
        return {
            "best_val_loss": best_val,
            "threshold": self.threshold,
            "epochs": len(history["train"]),
        }

    def _run_epoch(self, loader: DataLoader, criterion: nn.Module, optimizer) -> float:
        self.model.train()
        total, n = 0.0, 0
        for (batch,) in loader:
            batch = batch.to(self.device)
            optimizer.zero_grad()
            recon = self.model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            total += loss.item() * batch.size(0)
            n += batch.size(0)
        return total / max(n, 1)

    @torch.no_grad()
    def _validate(self, val_tensor: torch.Tensor, criterion: nn.Module) -> float:
        self.model.eval()
        if val_tensor.numel() == 0:
            return float("nan")
        return criterion(self.model(val_tensor), val_tensor).item()

    # ----------------------------------------------------- window selection
    def _split_windows(self, frame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split baseline-period windows into train / val / threshold-calibration.

        Train on days 1-12, hold out days 12-14 for threshold calibration, and
        carve a small time-ordered validation slice from training for early
        stopping. Windows overlapping +/-30 min of any labelled event are dropped.
        """
        start = frame["timestamp"].iloc[0]
        train_cut = start + (C.BASELINE_TRAINING_DAYS - 2) * np.timedelta64(1, "D")
        calib_cut = start + C.BASELINE_TRAINING_DAYS * np.timedelta64(1, "D")

        feats = frame[list(MODEL_FEATURES)].to_numpy(dtype=np.float32)
        event_mask = self._event_exclusion_mask(frame)

        train_region = (frame["timestamp"] < train_cut).to_numpy()
        calib_region = ((frame["timestamp"] >= train_cut) & (frame["timestamp"] < calib_cut)).to_numpy()

        train_all = self._windows_from_region(feats, train_region & ~event_mask, C.BASELINE_TRAIN_STRIDE)
        calib = self._windows_from_region(feats, calib_region & ~event_mask, C.BASELINE_TRAIN_STRIDE)

        # Time-ordered 85/15 split of training windows for early stopping.
        split = max(int(len(train_all) * 0.85), 1)
        return train_all[:split], train_all[split:], calib

    def _event_exclusion_mask(self, frame) -> np.ndarray:
        """Boolean mask marking samples within +/-30 min of any labelled event."""
        is_event = (frame["event_label"] != "normal").to_numpy()
        if not is_event.any():
            return np.zeros(len(frame), dtype=bool)
        pad = C.EVENT_EXCLUSION_MINUTES
        dilated = is_event.copy()
        idx = np.where(is_event)[0]
        for i in idx:
            dilated[max(0, i - pad) : min(len(frame), i + pad + 1)] = True
        return dilated

    @staticmethod
    def _windows_from_region(feats: np.ndarray, region: np.ndarray, stride: int) -> np.ndarray:
        """Build windows only from maximal contiguous runs where ``region`` is True."""
        windows: list[np.ndarray] = []
        n = len(region)
        i = 0
        while i < n:
            if not region[i]:
                i += 1
                continue
            j = i
            while j < n and region[j]:
                j += 1
            segment = feats[i:j]
            w = make_windows(segment, C.BASELINE_WINDOW_MINUTES, stride)
            if len(w):
                windows.append(w)
            i = j
        if not windows:
            return np.empty((0, C.BASELINE_WINDOW_MINUTES, C.N_SENSORS), dtype=np.float32)
        return np.concatenate(windows, axis=0)

    # --------------------------------------------------- threshold + scoring
    @torch.no_grad()
    def _calibrate_threshold(self, calib_w: np.ndarray) -> float:
        """Set the anomaly threshold to the 95th pct of normal reconstruction error."""
        self.model.eval()
        if len(calib_w) == 0:
            logger.warning("[%s] no calibration windows; threshold defaults to 1.0", self.patient_id)
            return 1.0
        errors = self._reconstruction_errors(calib_w)
        threshold = float(np.percentile(errors, C.ANOMALY_THRESHOLD_PERCENTILE))
        return max(threshold, 1e-6)

    @torch.no_grad()
    def _reconstruction_errors(self, windows: np.ndarray) -> np.ndarray:
        """Per-window mean squared reconstruction error."""
        self.model.eval()
        tensor = torch.from_numpy(windows).to(self.device)
        recon = self.model(tensor)
        err = ((recon - tensor) ** 2).mean(dim=(1, 2))
        return err.cpu().numpy()

    @torch.no_grad()
    def get_anomaly_score(self, window: np.ndarray) -> float:
        """Return the reconstruction error of a window normalised by the threshold.

        Args:
            window: Scaled sensor window of shape (120, 3).

        Returns:
            Score where >1.0 indicates an anomaly relative to this patient's
            calibrated normal. Clamped into [0, 1] for blending in the engine via
            a soft saturation so a 2x-threshold error maps near 1.0.
        """
        arr = np.asarray(window, dtype=np.float32)[None, ...]
        err = float(self._reconstruction_errors(arr)[0])
        ratio = err / self.threshold
        # Soft-saturating map: ratio 0->0, 1->~0.5, large->~1 (bounded for blending).
        return float(1.0 - np.exp(-np.log(2.0) * ratio))

    # ------------------------------------------------------------- persistence
    def _snapshot(self) -> dict:
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

    def _save_artifacts(self) -> None:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.model_dir / "baseline.pt")
        with open(self.model_dir / "threshold.json", "w") as fh:
            json.dump({"patient_id": self.patient_id, "threshold": self.threshold}, fh, indent=2)
        self._save_baseline_stats()
        logger.info("[%s] saved baseline.pt + threshold.json + baseline_stats.json", self.patient_id)

    def _save_baseline_stats(self) -> None:
        """Persist learned baseline statistics in PatientBaseline JSON shape."""
        stats = self.baseline_stats
        if stats is None:  # pragma: no cover - train() always sets this first
            return
        payload = {
            "patient_id": self.patient_id,
            "archetype": self.patient_id,
            "baseline_temp": stats.temp_mean,
            "baseline_temp_sd": stats.temp_std,
            "baseline_impedance": stats.impedance_mean,
            "baseline_impedance_sd": stats.impedance_std,
            "baseline_hrv": stats.hrv_mean,
            "baseline_hrv_sd": stats.hrv_std,
            "anomaly_threshold": self.threshold,
        }
        (self.model_dir / "baseline_stats.json").write_text(json.dumps(payload, indent=2))

    def load(self) -> "BaselineTrainer":
        """Load a previously trained model and threshold from disk.

        Raises:
            ModelNotTrainedError: If the checkpoint is missing.
        """
        ckpt = self.model_dir / "baseline.pt"
        if not ckpt.exists():
            raise ModelNotTrainedError(f"baseline[{self.patient_id}]", str(ckpt))
        self.model.load_state_dict(torch.load(ckpt, map_location=self.device))
        self.model.eval()
        thr_path = self.model_dir / "threshold.json"
        if thr_path.exists():
            self.threshold = float(json.loads(thr_path.read_text())["threshold"])
        return self

    def _plot_training_curve(self, history: dict[str, list[float]]) -> None:
        self.report_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(history["train"], label="train", color="#3b82f6")
        ax.plot(history["val"], label="validation", color="#f59e0b")
        ax.set_title(f"Baseline LSTM-AE training — {self.patient_id}")
        ax.set_xlabel("epoch")
        ax.set_ylabel("MSE reconstruction loss")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.report_dir / "baseline_training.png", dpi=120)
        plt.close(fig)


def train_all_baselines() -> dict[str, dict[str, float]]:
    """Train a personal baseline model for every patient archetype."""
    results: dict[str, dict[str, float]] = {}
    for patient_id in C.PATIENT_ARCHETYPES:
        logger.info("=== Training baseline for %s ===", patient_id)
        results[patient_id] = BaselineTrainer(patient_id).train()
    logger.info("All baseline models trained: %s", {k: round(v["threshold"], 5) for k, v in results.items()})
    return results


async def _persist_baselines_to_db() -> None:
    """Upsert every saved baseline_stats.json into the database."""
    import json

    from data.database import session_scope, upsert_patient_baseline

    async with session_scope() as session:
        for patient_id in C.PATIENT_ARCHETYPES:
            path = C.MODELS_DIR / patient_id / "baseline_stats.json"
            if not path.exists():
                continue
            stats = json.loads(path.read_text())
            await upsert_patient_baseline(session, patient_id, stats)
    logger.info("Persisted patient baselines to the database")


if __name__ == "__main__":
    import asyncio

    train_all_baselines()
    asyncio.run(_persist_baselines_to_db())
