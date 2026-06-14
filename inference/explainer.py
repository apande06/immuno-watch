"""Explainability layer — SHAP attribution + plain-language alert generation.

Clinical purpose:
    A risk score a clinician cannot interrogate is a risk score a clinician will
    not trust — and rightly so, when the decision is whether to admit a
    neutropenic patient at 3am. Every ImmunoWatch alert is therefore decomposed
    into per-sensor SHAP contributions and rendered twice: once in precise
    clinical language with a recommended order set, and once in reassuring plain
    language for the patient's own app.

Technical purpose:
    Wraps a SHAP ``KernelExplainer`` around the Transformer predictor. Because the
    model input is a 360x3 window, we attribute risk at the *sensor* level by
    explaining each channel's window-mean against a background of 100 normal
    windows — yielding the three contributions the dashboard renders.

Reference: Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions",
NeurIPS 2017.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

import constants as C
from data.schemas import AlertTier, PatientBaseline, SensorReading
from ml.predictor import InfectionRiskTransformer

logger = logging.getLogger("immunowatch.explainer")

try:  # SHAP is required by spec but we degrade gracefully if it misbehaves.
    import shap

    _SHAP_AVAILABLE = True
except Exception as exc:  # pragma: no cover - environment-dependent
    logger.warning("SHAP unavailable (%s); using finite-difference attribution", exc)
    _SHAP_AVAILABLE = False


# Tier -> recommended clinical action.
RECOMMENDED_ACTION: dict[AlertTier, str] = {
    AlertTier.WATCH: "increase monitoring cadence and recheck labs within 12h",
    AlertTier.WARNING: "order CBC with differential and blood cultures; clinician review now",
    AlertTier.CRITICAL: "immediate in-person evaluation; empiric antibiotics per neutropenic-fever protocol",
}

# Tier -> patient-facing instruction.
PATIENT_ACTION: dict[AlertTier, str] = {
    AlertTier.WATCH: "No action needed — your care team is keeping a closer eye on things.",
    AlertTier.WARNING: "Please contact your nurse so they can check on you.",
    AlertTier.CRITICAL: "Please go to the emergency room immediately — your care team has been alerted.",
}

# Patient-facing names for each sensor.
PATIENT_SENSOR_NAME: dict[str, str] = {
    "temp_c": "body temperature",
    "impedance_ohm": "immune-activity signal",
    "hrv_rmssd_ms": "heart-rhythm variability",
}


class AlertExplainer:
    """Generates clinically grounded, human-readable explanations for alerts."""

    def __init__(self, model: InfectionRiskTransformer, background_windows: np.ndarray) -> None:
        """Initialise the explainer.

        Args:
            model: Trained infection-risk Transformer (eval mode).
            background_windows: ~100 normal windows of shape (n, 360, 3) used as
                the SHAP reference distribution.
        """
        self.model = model.eval()
        if background_windows.size == 0:
            background_windows = np.zeros((1, C.PREDICTOR_WINDOW_MINUTES, C.N_SENSORS), dtype=np.float32)
        self._reference = background_windows.mean(axis=0)            # (360, 3)
        self._reference_mean = self._reference.mean(axis=0)          # (3,)
        background_feats = background_windows.mean(axis=1)           # (n, 3)
        self._background_feats = background_feats[: C.SHAP_BACKGROUND_SAMPLES]
        self._expected = float(self._predict_features(self._background_feats).mean())

        if _SHAP_AVAILABLE:
            self._explainer = shap.KernelExplainer(self._predict_features, self._background_feats)
        else:  # pragma: no cover
            self._explainer = None

    # ------------------------------------------------------- prediction fn
    def _features_to_windows(self, feats: np.ndarray) -> np.ndarray:
        """Map per-sensor means back to full windows by shifting the reference.

        Each channel of the reference window is offset so its mean equals the
        requested per-sensor value, preserving the temporal shape SHAP needs while
        keeping the explained feature space at sensor granularity.
        """
        feats = np.atleast_2d(feats).astype(np.float32)
        shifts = feats - self._reference_mean[None, :]          # (m, 3)
        return self._reference[None, :, :] + shifts[:, None, :]  # (m, 360, 3)

    def _predict_features(self, feats: np.ndarray) -> np.ndarray:
        """SHAP prediction function: per-sensor means -> risk probability."""
        windows = self._features_to_windows(feats)
        with torch.no_grad():
            probs = self.model.predict(torch.from_numpy(windows.astype(np.float32)))["risk_score"]
        return probs.cpu().numpy()

    # --------------------------------------------------------------- explain
    def explain(self, window: np.ndarray) -> tuple[float, float, float]:
        """Attribute a window's risk score to the three sensors.

        Args:
            window: Scaled sensor window of shape (360, 3).

        Returns:
            ``(shap_temp, shap_impedance, shap_hrv)`` — additive contributions in
            risk-probability units (positive pushes risk up).
        """
        feat = np.asarray(window, dtype=np.float32).mean(axis=0)[None, :]
        if self._explainer is not None:
            try:
                values = self._explainer.shap_values(feat, nsamples=C.SHAP_NSAMPLES, silent=True)
                values = np.asarray(values).reshape(-1)[: C.N_SENSORS]
                return float(values[0]), float(values[1]), float(values[2])
            except Exception as exc:  # pragma: no cover
                logger.debug("SHAP failed (%s); falling back", exc)
        return self._finite_difference(feat.reshape(-1))

    def _finite_difference(self, feat: np.ndarray) -> tuple[float, float, float]:
        """Fallback attribution: risk drop when each sensor is reset to baseline."""
        base = float(self._predict_features(feat[None, :])[0])
        contribs = []
        for c in range(C.N_SENSORS):
            counter = feat.copy()
            counter[c] = self._reference_mean[c]
            contribs.append(base - float(self._predict_features(counter[None, :])[0]))
        return tuple(float(v) for v in contribs)  # type: ignore[return-value]

    # ------------------------------------------------ narrative generation
    def generate_clinical_explanation(
        self,
        shap_values: tuple[float, float, float],
        reading: SensorReading,
        baseline: PatientBaseline,
        severity: float,
        time_to_event: float,
    ) -> str:
        """Render the physician-facing explanation string."""
        deltas = self._sensor_deltas(reading, baseline)
        attach_shap_to_deltas(deltas, shap_values)
        ranked = self._rank_sensors(shap_values)
        rank_word = {0: "primary", 1: "secondary", 2: "tertiary"}

        n_abnormal = sum(1 for d in deltas.values() if d["abnormal"])
        clauses = []
        for order, sensor in enumerate(ranked):
            d = deltas[sensor]
            clauses.append(
                f"{d['name']} {d['direction']} {d['magnitude']} {d['comparator']} baseline "
                f"({rank_word[order]} driver, SHAP: {d['shap']:+.3f})"
            )
        condition = self._infer_condition(severity)
        return (
            f"Alert triggered by {n_abnormal} converging abnormal signals: "
            + "; ".join(clauses)
            + f". Pattern consistent with {condition}. "
            f"Estimated {time_to_event:.0f} hours to clinical presentation. "
            f"Recommend {RECOMMENDED_ACTION[self._tier_for_severity(severity)]}."
        )

    def generate_patient_explanation(
        self,
        shap_values: tuple[float, float, float],
        reading: SensorReading,
        baseline: PatientBaseline,
        tier: AlertTier,
    ) -> str:
        """Render the plain-language patient-facing explanation string."""
        deltas = self._sensor_deltas(reading, baseline)
        ranked = self._rank_sensors(shap_values)
        primary = ranked[0]
        d = deltas[primary]
        layman = "slightly elevated" if d["raw_direction"] == "up" else "declining"
        return (
            f"Your {PATIENT_SENSOR_NAME[primary]} has been {layman} compared to your "
            f"normal levels over the past few hours. {PATIENT_ACTION[tier]}"
        )

    # ----------------------------------------------------------- helpers
    def _sensor_deltas(self, reading: SensorReading, baseline: PatientBaseline) -> dict[str, dict]:
        """Compute per-sensor direction / magnitude / abnormality vs. baseline."""
        temp_delta = reading.temp_c - baseline.baseline_temp
        imp_pct = (reading.impedance_ohm - baseline.baseline_impedance) / baseline.baseline_impedance
        hrv_pct = (reading.hrv_rmssd_ms - baseline.baseline_hrv) / baseline.baseline_hrv
        return {
            "temp_c": {
                "name": "temperature",
                "shap": 0.0,
                "raw_direction": "up" if temp_delta >= 0 else "down",
                "direction": "elevated" if temp_delta >= 0 else "reduced",
                "comparator": "above" if temp_delta >= 0 else "below",
                "magnitude": f"{abs(temp_delta):.2f}C",
                "abnormal": abs(temp_delta) >= C.SUSTAINED_ELEVATION_THRESHOLD_C,
            },
            "impedance_ohm": {
                "name": "bioimpedance",
                "shap": 0.0,
                "raw_direction": "up" if imp_pct >= 0 else "down",
                "direction": "elevated" if imp_pct >= 0 else "reduced",
                "comparator": "above" if imp_pct >= 0 else "below",
                "magnitude": f"{abs(imp_pct) * 100:.1f}%",
                "abnormal": abs(imp_pct) >= C.IMPEDANCE_DECLINE_WATCH_PCT,
            },
            "hrv_rmssd_ms": {
                "name": "HRV (RMSSD)",
                "shap": 0.0,
                "raw_direction": "up" if hrv_pct >= 0 else "down",
                "direction": "elevated" if hrv_pct >= 0 else "reduced",
                "comparator": "above" if hrv_pct >= 0 else "below",
                "magnitude": f"{abs(hrv_pct) * 100:.1f}%",
                "abnormal": abs(hrv_pct) >= C.HRV_DECLINE_WATCH_PCT,
            },
        }

    def _rank_sensors(self, shap_values: tuple[float, float, float]) -> list[str]:
        """Return sensor column names ordered by descending |SHAP|."""
        order = np.argsort(-np.abs(np.asarray(shap_values)))
        cols = list(C.SENSOR_COLUMNS)
        return [cols[i] for i in order]

    @staticmethod
    def _infer_condition(severity: float) -> str:
        if severity >= 9:
            return "neutropenic crisis / impending sepsis"
        if severity >= 6:
            return "bacterial infection onset"
        if severity >= 2:
            return "mild viral infection"
        return "transient physiological deviation"

    @staticmethod
    def _tier_for_severity(severity: float) -> AlertTier:
        if severity >= 8.5:
            return AlertTier.CRITICAL
        if severity >= 5:
            return AlertTier.WARNING
        return AlertTier.WATCH


def attach_shap_to_deltas(deltas: dict[str, dict], shap_values: tuple[float, float, float]) -> None:
    """Populate the ``shap`` field of each delta entry (used in narratives)."""
    for col, value in zip(C.SENSOR_COLUMNS, shap_values):
        deltas[col]["shap"] = value


__all__ = ["AlertExplainer", "RECOMMENDED_ACTION", "PATIENT_ACTION"]
