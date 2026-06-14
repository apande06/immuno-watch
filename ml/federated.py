"""Federated learning simulation for ImmunoWatch (FedAvg across patients).

Clinical purpose:
    Biosignal streams are among the most sensitive data a person has. In a real
    deployment, each implanted chip would train locally on its owner's data and
    transmit only model weight updates — never raw readings. This preserves
    privacy while still letting a rare infection pattern observed in one patient
    improve every other patient's model.

Technical purpose:
    Simulates FedAvg across the three patient models to demonstrate the privacy
    architecture and quantify the generalisation gain from federation versus
    purely-local training.

Reference: McMahan et al., "Communication-Efficient Learning of Deep Networks
from Decentralized Data", AISTATS 2017 (the original FedAvg paper).

Usage:
    python ml/federated.py
"""

from __future__ import annotations

import copy
import logging
import sys
from pathlib import Path

# --- path bootstrap ---
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

import constants as C
from ml.predictor import (
    InfectionRiskTransformer,
    MultiTaskLoss,
    WindowedDataset,
    _to_tensors,
    build_patient_windows,
    time_split,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("immunowatch.federated")

StateDict = dict[str, torch.Tensor]


def fedavg(state_dicts: list[StateDict], weights: list[int]) -> StateDict:
    """Weighted average of model state dicts (the FedAvg aggregation step).

    Args:
        state_dicts: Per-client model parameters.
        weights: Per-client dataset sizes (the aggregation weights).

    Returns:
        The aggregated state dict.
    """
    total = float(sum(weights)) or 1.0
    avg: StateDict = {}
    for key in state_dicts[0]:
        stacked = sum(sd[key].float() * (w / total) for sd, w in zip(state_dicts, weights))
        avg[key] = stacked.to(state_dicts[0][key].dtype)
    return avg


class FederatedSimulation:
    """Runs FedAvg across the patient models and reports the generalisation gain."""

    def __init__(self, device: str | None = None) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.criterion = MultiTaskLoss().to(self.device)
        self.patients = list(C.PATIENT_ARCHETYPES)

    # ----------------------------------------------------------- data
    def _load_splits(self) -> dict[str, tuple[WindowedDataset, WindowedDataset]]:
        splits: dict[str, tuple[WindowedDataset, WindowedDataset]] = {}
        for pid in self.patients:
            tr, va, _te = time_split(build_patient_windows(pid))
            splits[pid] = (tr, va)
        return splits

    # ----------------------------------------------------------- training utils
    def _train_local(self, model: nn.Module, ds: WindowedDataset, epochs: int) -> None:
        if len(ds) == 0:
            return
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=C.PREDICTOR_LR, weight_decay=C.PREDICTOR_WEIGHT_DECAY)
        loader = DataLoader(_to_tensors(ds), batch_size=C.PREDICTOR_BATCH_SIZE, shuffle=True)
        for _ in range(epochs):
            for x, risk, sev, tte in loader:
                x = x.to(self.device)
                targets = {"risk": risk.to(self.device), "severity": sev.to(self.device), "tte": tte.to(self.device)}
                optimizer.zero_grad()
                loss = self.criterion(model(x), targets)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

    @torch.no_grad()
    def _val_loss(self, model: nn.Module, ds: WindowedDataset) -> float:
        if len(ds) == 0:
            return float("nan")
        model.eval()
        x = torch.from_numpy(ds.x).to(self.device)
        targets = {
            "risk": torch.from_numpy(ds.y_risk).to(self.device),
            "severity": torch.from_numpy(ds.y_severity).to(self.device),
            "tte": torch.from_numpy(ds.y_tte).to(self.device),
        }
        return self.criterion(model(x), targets).item()

    @torch.no_grad()
    def _auc(self, model: nn.Module, ds: WindowedDataset) -> float:
        if len(ds) == 0 or len(np.unique(ds.y_risk)) < 2:
            return float("nan")
        model.eval()
        x = torch.from_numpy(ds.x).to(self.device)
        probs = torch.sigmoid(model(x)["risk_logit"]).cpu().numpy()
        return float(roc_auc_score(ds.y_risk.astype(int), probs))

    # ----------------------------------------------------------- driver
    def run(self) -> dict[str, dict[str, float]]:
        """Execute the federated rounds and the local-only control.

        Returns:
            Per-patient ``{"auc_local": ..., "auc_federated": ...}`` summary.
        """
        torch.manual_seed(C.RANDOM_SEED)
        splits = self._load_splits()

        # Shared initialisation so every model starts from identical weights.
        seed_model = InfectionRiskTransformer().to(self.device)
        init_state = copy.deepcopy(seed_model.state_dict())

        fed_models = {pid: self._clone(init_state) for pid in self.patients}
        local_models = {pid: self._clone(init_state) for pid in self.patients}

        history = {pid: {"federated": [], "local": []} for pid in self.patients}

        for rnd in range(1, C.FEDERATED_ROUNDS + 1):
            # --- federated: local train then aggregate ---
            for pid in self.patients:
                self._train_local(fed_models[pid], splits[pid][0], C.FEDERATED_LOCAL_EPOCHS)
            sizes = [len(splits[pid][0]) for pid in self.patients]
            aggregated = fedavg([fed_models[pid].state_dict() for pid in self.patients], sizes)
            for pid in self.patients:
                fed_models[pid].load_state_dict(aggregated)

            # --- local-only control: train without aggregation ---
            for pid in self.patients:
                self._train_local(local_models[pid], splits[pid][0], C.FEDERATED_LOCAL_EPOCHS)

            for pid in self.patients:
                history[pid]["federated"].append(self._val_loss(fed_models[pid], splits[pid][1]))
                history[pid]["local"].append(self._val_loss(local_models[pid], splits[pid][1]))
            logger.info(
                "Round %d | federated val loss: %s",
                rnd, {pid: round(history[pid]["federated"][-1], 4) for pid in self.patients},
            )

        summary = self._summarise(fed_models, local_models, splits)
        self._plot(history)
        self._save(fed_models)
        self._save_results(history, summary)
        return summary

    def _summarise(self, fed_models, local_models, splits) -> dict[str, dict[str, float]]:
        summary: dict[str, dict[str, float]] = {}
        logger.info("%-20s %-16s %-16s", "patient", "AUC (local-only)", "AUC (federated)")
        for pid in self.patients:
            val = splits[pid][1]
            auc_local = self._auc(local_models[pid], val)
            auc_fed = self._auc(fed_models[pid], val)
            summary[pid] = {"auc_local": auc_local, "auc_federated": auc_fed}
            logger.info("%-20s %-16.3f %-16.3f", pid, auc_local, auc_fed)
        return summary

    def _plot(self, history: dict[str, dict[str, list[float]]]) -> None:
        out_dir = C.REPORTS_DIR / "federated"
        out_dir.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, len(self.patients), figsize=(5 * len(self.patients), 4), sharey=True)
        if len(self.patients) == 1:
            axes = [axes]
        rounds = range(1, C.FEDERATED_ROUNDS + 1)
        for ax, pid in zip(axes, self.patients):
            ax.plot(rounds, history[pid]["local"], "o-", label="local-only", color="#9ca3af")
            ax.plot(rounds, history[pid]["federated"], "o-", label="federated", color="#3b82f6")
            ax.set_title(pid)
            ax.set_xlabel("federation round")
            ax.set_xticks(list(rounds))
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("validation loss")
        axes[0].legend()
        fig.suptitle("Federated vs. local-only validation loss")
        fig.tight_layout()
        fig.savefig(out_dir / "federated_rounds.png", dpi=120)
        plt.close(fig)

    def _save(self, fed_models) -> None:
        # All federated models share the aggregated weights; persist one copy.
        any_pid = self.patients[0]
        torch.save(fed_models[any_pid].state_dict(), C.MODELS_DIR / "federated_predictor.pt")
        logger.info("Saved aggregated model to models/federated_predictor.pt")

    def _save_results(self, history: dict, summary: dict) -> None:
        """Persist round-by-round history + AUC summary for the evaluation report."""
        import json

        payload = {"history": history, "summary": summary, "rounds": C.FEDERATED_ROUNDS}
        (C.MODELS_DIR / "federated_results.json").write_text(json.dumps(payload, indent=2))

    def _clone(self, state: StateDict) -> InfectionRiskTransformer:
        model = InfectionRiskTransformer().to(self.device)
        model.load_state_dict(copy.deepcopy(state))
        return model


if __name__ == "__main__":
    FederatedSimulation().run()
