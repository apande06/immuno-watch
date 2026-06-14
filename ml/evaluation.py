"""Comprehensive evaluation suite — publication-quality plots for ImmunoWatch.

Clinical purpose:
    A monitoring model that cannot be evaluated cannot be trusted with a
    neutropenic patient. This module quantifies, per model, how well the system
    separates infection from normal physiology and how well its probabilities are
    calibrated — the two properties a clinician must believe before acting on an
    alert.

Technical purpose:
    Generates ROC/PR curves, score distributions, reconstruction-error heatmaps,
    confusion matrices, calibration curves, and the federated generalisation
    comparison, then compiles them into a single PDF report under ``reports/``.

Usage:
    python ml/evaluation.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# --- path bootstrap ---
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

import constants as C
from ml.baseline import BaselineTrainer
from ml.predictor import (
    PredictorTrainer,
    REAL_EVENTS,
    build_global_dataset,
    build_patient_windows,
)
from ml.preprocessing import MODEL_FEATURES, BiosignalPreprocessor, make_windows

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("immunowatch.evaluation")

sns.set_theme(style="whitegrid")


class ModelEvaluator:
    """Generates the full evaluation plot suite and a compiled PDF report."""

    def __init__(self) -> None:
        self.figures: list[plt.Figure] = []

    # ---------------------------------------------------------- baseline
    def evaluate_baseline(self, patient_id: str) -> plt.Figure | None:
        """ROC, PR, score distribution, and reconstruction-error heatmap.

        Args:
            patient_id: Patient whose baseline model to evaluate.

        Returns:
            The composed figure, or ``None`` if the model/ data is unavailable.
        """
        try:
            trainer = BaselineTrainer(patient_id).load()
        except Exception as exc:  # pragma: no cover - missing artifacts
            logger.warning("Skipping baseline eval for %s: %s", patient_id, exc)
            return None

        frame, _ = BiosignalPreprocessor(patient_id).fit_transform()
        feats = frame[list(MODEL_FEATURES)].to_numpy(dtype=np.float32)
        windows = make_windows(feats, C.BASELINE_WINDOW_MINUTES, stride=C.BASELINE_TRAIN_STRIDE)
        if len(windows) == 0:
            return None

        errors = trainer._reconstruction_errors(windows)
        # A window is "anomalous" if it overlaps any real infection event.
        real = np.isin(frame["event_label"].to_numpy(), list(REAL_EVENTS))
        labels = np.array(
            [
                int(real[i * C.BASELINE_TRAIN_STRIDE : i * C.BASELINE_TRAIN_STRIDE + C.BASELINE_WINDOW_MINUTES].any())
                for i in range(len(windows))
            ]
        )

        fig, axes = plt.subplots(2, 2, figsize=(13, 10))
        fig.suptitle(f"Baseline LSTM-AE evaluation — {patient_id}", fontsize=14)

        if labels.sum() and (labels == 0).sum():
            fpr, tpr, _ = roc_curve(labels, errors)
            axes[0, 0].plot(fpr, tpr, color="#3b82f6", label=f"AUC={auc(fpr, tpr):.3f}")
            axes[0, 0].plot([0, 1], [0, 1], "--", color="#9ca3af")
            axes[0, 0].set(title="ROC", xlabel="FPR", ylabel="TPR")
            axes[0, 0].legend()

            prec, rec, _ = precision_recall_curve(labels, errors)
            axes[0, 1].plot(rec, prec, color="#10b981")
            axes[0, 1].set(title="Precision-Recall", xlabel="recall", ylabel="precision")

            axes[1, 0].hist(errors[labels == 0], bins=40, alpha=0.7, label="normal", color="#10b981")
            axes[1, 0].hist(errors[labels == 1], bins=40, alpha=0.7, label="anomaly", color="#ef4444")
            axes[1, 0].axvline(trainer.threshold, color="#f59e0b", ls="--", label="threshold")
            axes[1, 0].set(title="Reconstruction-error distribution", xlabel="MSE", ylabel="count")
            axes[1, 0].legend()
        else:
            for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
                ax.text(0.5, 0.5, "insufficient anomaly windows", ha="center")

        self._reconstruction_heatmap(axes[1, 1], frame, windows, errors)
        fig.tight_layout()
        self.figures.append(fig)
        return fig

    def _reconstruction_heatmap(self, ax, frame, windows, errors) -> None:
        """Day x hour heatmap of mean reconstruction error with event overlay."""
        end_idx = np.arange(len(windows)) * C.BASELINE_TRAIN_STRIDE + C.BASELINE_WINDOW_MINUTES - 1
        end_idx = np.clip(end_idx, 0, len(frame) - 1)
        ts = frame["timestamp"].to_numpy()
        start = frame["timestamp"].iloc[0]
        days = ((ts[end_idx] - np.datetime64(start)) / np.timedelta64(1, "D")).astype(int)
        hours = ((ts[end_idx] - np.datetime64(start)) / np.timedelta64(1, "h")).astype(int) % 24

        grid = np.full((C.SIMULATION_DAYS, 24), np.nan)
        for d, h, e in zip(days, hours, errors):
            if 0 <= d < C.SIMULATION_DAYS:
                grid[d, h] = e if np.isnan(grid[d, h]) else max(grid[d, h], e)
        im = ax.imshow(grid, aspect="auto", cmap="magma", origin="lower")
        ax.set(title="Reconstruction error (day x hour)", xlabel="hour", ylabel="day")
        ax.figure.colorbar(im, ax=ax, fraction=0.046)

    # --------------------------------------------------------- predictor
    def evaluate_predictor(self) -> plt.Figure | None:
        """Confusion matrix, ROC with bootstrap CI, calibration, and F1."""
        try:
            trainer = PredictorTrainer().load()
        except Exception as exc:  # pragma: no cover
            logger.warning("Skipping predictor eval: %s", exc)
            return None

        _, _, test = build_global_dataset()
        if len(test) == 0:
            return None
        with torch.no_grad():
            probs = torch.sigmoid(trainer.model(torch.from_numpy(test.x))["risk_logit"]).numpy()
        y = test.y_risk.astype(int)
        preds = (probs >= 0.5).astype(int)

        fig, axes = plt.subplots(2, 2, figsize=(13, 10))
        fig.suptitle("Infection-risk Transformer evaluation", fontsize=14)

        cm = confusion_matrix(y, preds, labels=[0, 1])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=axes[0, 0],
            xticklabels=["normal", "infection"], yticklabels=["normal", "infection"],
        )
        axes[0, 0].set(title="Confusion matrix", xlabel="predicted", ylabel="actual")

        if len(np.unique(y)) > 1:
            fpr, tpr, _ = roc_curve(y, probs)
            lo, hi, point = self._bootstrap_auc_ci(y, probs)
            axes[0, 1].plot(fpr, tpr, color="#3b82f6", label=f"AUC={point:.3f} [{lo:.3f}, {hi:.3f}]")
            axes[0, 1].plot([0, 1], [0, 1], "--", color="#9ca3af")
            axes[0, 1].set(title="ROC (95% bootstrap CI)", xlabel="FPR", ylabel="TPR")
            axes[0, 1].legend()

            frac_pos, mean_pred = calibration_curve(y, probs, n_bins=10, strategy="quantile")
            axes[1, 0].plot(mean_pred, frac_pos, "o-", color="#10b981", label="model")
            axes[1, 0].plot([0, 1], [0, 1], "--", color="#9ca3af", label="ideal")
            axes[1, 0].set(title="Calibration", xlabel="predicted probability", ylabel="observed frequency")
            axes[1, 0].legend()

        f1_pos = f1_score(y, preds, pos_label=1, zero_division=0)
        f1_neg = f1_score(y, preds, pos_label=0, zero_division=0)
        axes[1, 1].bar(["normal", "infection"], [f1_neg, f1_pos], color=["#10b981", "#ef4444"])
        axes[1, 1].set(title="Per-class F1", ylim=(0, 1), ylabel="F1")
        fig.tight_layout()
        self.figures.append(fig)
        return fig

    @staticmethod
    def _bootstrap_auc_ci(y: np.ndarray, probs: np.ndarray, n: int = 500) -> tuple[float, float, float]:
        """95% bootstrap confidence interval for AUC-ROC."""
        rng = np.random.default_rng(C.RANDOM_SEED)
        point = roc_auc_score(y, probs)
        scores = []
        for _ in range(n):
            idx = rng.integers(0, len(y), len(y))
            if len(np.unique(y[idx])) < 2:
                continue
            scores.append(roc_auc_score(y[idx], probs[idx]))
        if not scores:
            return point, point, point
        return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5)), float(point)

    # --------------------------------------------------------- federated
    def evaluate_federated(self) -> plt.Figure | None:
        """Before/after loss curves and cross-patient generalisation bar chart."""
        path = C.MODELS_DIR / "federated_results.json"
        if not path.exists():
            logger.warning("Skipping federated eval: %s not found (run ml/federated.py)", path)
            return None
        data = json.loads(path.read_text())
        history, summary = data["history"], data["summary"]
        patients = list(summary)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle("Federated learning evaluation", fontsize=14)
        rounds = range(1, data["rounds"] + 1)
        for pid in patients:
            axes[0].plot(rounds, history[pid]["federated"], "o-", label=f"{pid} (fed)")
            axes[0].plot(rounds, history[pid]["local"], "x--", alpha=0.5, label=f"{pid} (local)")
        axes[0].set(title="Validation loss across rounds", xlabel="round", ylabel="loss")
        axes[0].legend(fontsize=8)

        x = np.arange(len(patients))
        local = [summary[p]["auc_local"] for p in patients]
        fed = [summary[p]["auc_federated"] for p in patients]
        axes[1].bar(x - 0.2, local, 0.4, label="local-only", color="#9ca3af")
        axes[1].bar(x + 0.2, fed, 0.4, label="federated", color="#3b82f6")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(patients, rotation=20, fontsize=8)
        axes[1].set(title="Cross-patient AUC", ylabel="AUC", ylim=(0, 1))
        axes[1].legend()
        fig.tight_layout()
        self.figures.append(fig)
        return fig

    # ------------------------------------------------------------ report
    def generate_full_report(self) -> Path:
        """Compile every generated figure into one PDF under ``reports/``."""
        self.figures.clear()
        for pid in C.PATIENT_ARCHETYPES:
            self.evaluate_baseline(pid)
        self.evaluate_predictor()
        self.evaluate_federated()

        C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = C.REPORTS_DIR / "immunowatch_model_report.pdf"
        with PdfPages(out_path) as pdf:
            self._cover_page(pdf)
            for fig in self.figures:
                pdf.savefig(fig)
        for fig in self.figures:
            plt.close(fig)
        logger.info("Compiled report with %d figures -> %s", len(self.figures), out_path)
        return out_path

    def _cover_page(self, pdf: PdfPages) -> None:
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.5, 0.62, "ImmunoWatch", ha="center", fontsize=30, weight="bold")
        fig.text(0.5, 0.56, "Model Evaluation Report", ha="center", fontsize=18)
        fig.text(0.5, 0.50, "Personal-baseline LSTM-AE | Infection-risk Transformer | FedAvg", ha="center", fontsize=11)
        metrics_path = C.MODELS_DIR / "predictor_metrics.json"
        if metrics_path.exists():
            m = json.loads(metrics_path.read_text())
            fig.text(
                0.5, 0.40,
                f"Predictor test AUC={m.get('auc_roc', float('nan')):.3f}  "
                f"F1={m.get('f1', float('nan')):.3f}  "
                f"recall={m.get('recall', float('nan')):.3f}",
                ha="center", fontsize=12,
            )
        pdf.savefig(fig)
        plt.close(fig)


if __name__ == "__main__":
    ModelEvaluator().generate_full_report()
