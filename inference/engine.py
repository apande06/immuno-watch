"""Real-time inference loop for ImmunoWatch continuous monitoring.

Clinical purpose:
    This is the component that actually watches the patient. It fuses two
    complementary detectors — the personal-baseline autoencoder (exquisitely
    sensitive to *this* patient's deviations) and the cross-patient Transformer
    (higher specificity for true infection) — into one combined risk score, maps
    it onto the clinical escalation ladder, and suppresses repeat alerts to fight
    the alarm fatigue that demonstrably causes missed events in monitored units.

Technical purpose:
    Maintains a 6-hour rolling buffer per patient, scores each incoming reading,
    attaches a SHAP explanation, deduplicates, and persists alerts.

Combined score: ``final = 0.4 * anomaly_score + 0.6 * risk_score``.

Alert deduplication rationale: Cvach, "Monitor Alarm Fatigue", Biomed Instrum
Technol, 2012.
"""

from __future__ import annotations

import logging
import pickle
from collections import deque
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import torch

import constants as C
from data.database import insert_alert, session_scope
from data.schemas import Alert, AlertTier, PatientBaseline, PatientStatus, SensorReading
from exceptions import InsufficientDataError, PatientNotFoundError
from inference.explainer import AlertExplainer
from ml.baseline import BaselineTrainer
from ml.predictor import InfectionRiskTransformer, PredictorTrainer
from ml.preprocessing import make_windows

logger = logging.getLogger("immunowatch.engine")


class InferenceEngine:
    """Stateful real-time scorer for every monitored patient."""

    def __init__(self, device: str | None = None) -> None:
        self.device = torch.device(device or "cpu")
        self.patients: list[str] = list(C.PATIENT_ARCHETYPES)

        self.buffers: dict[str, deque[SensorReading]] = {
            pid: deque(maxlen=C.PREDICTOR_WINDOW_MINUTES) for pid in self.patients
        }
        self.scalers: dict[str, object] = {}
        self.baselines: dict[str, BaselineTrainer] = {}
        self.baseline_stats: dict[str, PatientBaseline] = {}
        self.last_alert: dict[tuple[str, AlertTier], datetime] = {}
        self.status: dict[str, PatientStatus] = {}

        self.predictor: Optional[InfectionRiskTransformer] = self._load_predictor()
        self._load_patient_artifacts()
        self.explainer: Optional[AlertExplainer] = self._build_explainer()
        self._prefill_buffers()
        logger.info("InferenceEngine ready for %d patients", len(self.patients))

    # ------------------------------------------------------------ loading
    def _load_predictor(self) -> Optional[InfectionRiskTransformer]:
        try:
            return PredictorTrainer(device=str(self.device)).load().model
        except Exception as exc:
            logger.warning("Predictor not loaded (%s); risk scoring degraded", exc)
            return None

    def _load_patient_artifacts(self) -> None:
        for pid in self.patients:
            # Per-patient scaler (for scaling incoming raw readings).
            scaler_path = C.MODELS_DIR / pid / "scaler.pkl"
            if scaler_path.exists():
                with open(scaler_path, "rb") as fh:
                    self.scalers[pid] = pickle.load(fh)
            # Per-patient baseline autoencoder + calibrated threshold.
            try:
                self.baselines[pid] = BaselineTrainer(pid, device=str(self.device)).load()
            except Exception as exc:
                logger.warning("[%s] baseline model not loaded: %s", pid, exc)
            # Learned baseline statistics for explanations / status.
            self.baseline_stats[pid] = self._load_baseline_stats(pid)

    def _load_baseline_stats(self, pid: str) -> PatientBaseline:
        """Load saved baseline stats JSON, falling back to the archetype priors."""
        import json

        path = C.MODELS_DIR / pid / "baseline_stats.json"
        if path.exists():
            data = json.loads(path.read_text())
            return PatientBaseline(**data)
        cfg = C.PATIENT_ARCHETYPES[pid]
        return PatientBaseline(
            patient_id=pid,
            archetype=pid,
            baseline_temp=float(cfg["temp_baseline_c"]),
            baseline_temp_sd=float(cfg["temp_sd_c"]),
            baseline_impedance=float(cfg["impedance_baseline_ohm"]),
            baseline_impedance_sd=float(cfg["impedance_sd_ohm"]),
            baseline_hrv=float(cfg["hrv_baseline_ms"]),
            baseline_hrv_sd=float(cfg["hrv_sd_ms"]),
        )

    def _build_explainer(self) -> Optional[AlertExplainer]:
        if self.predictor is None:
            return None
        background = self._load_background_windows()
        try:
            return AlertExplainer(self.predictor, background)
        except Exception as exc:  # pragma: no cover
            logger.warning("Explainer unavailable: %s", exc)
            return None

    def _load_background_windows(self) -> np.ndarray:
        """Assemble ~100 normal scaled windows for the SHAP background."""
        for pid in self.patients:
            if pid not in self.scalers:
                continue
            csv = C.PATIENT_DATA_DIR / f"{pid}.csv"
            if not csv.exists():
                continue
            frame = pd.read_csv(csv, parse_dates=["timestamp"]).dropna(subset=list(C.SENSOR_COLUMNS))
            normal = frame[frame["event_label"] == "normal"]
            raw = normal[list(C.SENSOR_COLUMNS)].to_numpy()
            scaled = self.scalers[pid].transform(raw).astype(np.float32)
            windows = make_windows(scaled, C.PREDICTOR_WINDOW_MINUTES, stride=300)
            if len(windows):
                return windows[: C.SHAP_BACKGROUND_SAMPLES]
        return np.zeros((1, C.PREDICTOR_WINDOW_MINUTES, C.N_SENSORS), dtype=np.float32)

    def _prefill_buffers(self) -> None:
        """Seed each patient buffer with their most recent 6 hours from the CSV."""
        for pid in self.patients:
            csv = C.PATIENT_DATA_DIR / f"{pid}.csv"
            if not csv.exists():
                continue
            frame = pd.read_csv(csv, parse_dates=["timestamp"]).dropna(subset=list(C.SENSOR_COLUMNS))
            # Use the last normal stretch so the engine starts from a stable state.
            normal = frame[frame["event_label"] == "normal"].tail(C.PREDICTOR_WINDOW_MINUTES)
            for row in normal.itertuples(index=False):
                self.buffers[pid].append(
                    SensorReading(
                        patient_id=pid,
                        timestamp=row.timestamp.to_pydatetime(),
                        temp_c=float(row.temp_c),
                        impedance_ohm=float(row.impedance_ohm),
                        hrv_rmssd_ms=float(row.hrv_rmssd_ms),
                    )
                )
            if self.buffers[pid]:
                self._refresh_status(pid, risk_score=0.0, tier=None)

    # ---------------------------------------------------------- scoring
    def _scaled_window(self, pid: str, n: int) -> np.ndarray:
        readings = list(self.buffers[pid])[-n:]
        raw = np.array(
            [[r.temp_c, r.impedance_ohm, r.hrv_rmssd_ms] for r in readings], dtype=np.float32
        )
        scaler = self.scalers.get(pid)
        return scaler.transform(raw).astype(np.float32) if scaler is not None else raw

    def _anomaly_score(self, pid: str) -> Optional[float]:
        if pid not in self.baselines or len(self.buffers[pid]) < C.BASELINE_WINDOW_MINUTES:
            return None
        window = self._scaled_window(pid, C.BASELINE_WINDOW_MINUTES)
        return self.baselines[pid].get_anomaly_score(window)

    def _risk_outputs(self, pid: str) -> Optional[dict[str, float]]:
        if self.predictor is None or len(self.buffers[pid]) < C.PREDICTOR_WINDOW_MINUTES:
            return None
        window = self._scaled_window(pid, C.PREDICTOR_WINDOW_MINUTES)
        tensor = torch.from_numpy(window[None, ...]).to(self.device)
        with torch.no_grad():
            out = self.predictor.predict(tensor)
        return {
            "risk_score": float(out["risk_score"].item()),
            "severity": float(out["severity"].item()),
            "time_to_event": float(out["time_to_event"].item()),
        }

    @staticmethod
    def _tier_for_score(score: float) -> Optional[AlertTier]:
        if score >= C.CRITICAL_THRESHOLD:
            return AlertTier.CRITICAL
        if score >= C.WARNING_THRESHOLD:
            return AlertTier.WARNING
        if score >= C.WATCH_THRESHOLD:
            return AlertTier.WATCH
        return None

    # ---------------------------------------------------------- main loop
    async def process_reading(self, patient_id: str, reading: SensorReading) -> Optional[Alert]:
        """Ingest one reading, score it, and emit an Alert if one is warranted.

        Args:
            patient_id: Patient the reading belongs to.
            reading: Validated sensor reading.

        Returns:
            A persisted :class:`Alert` if a (non-deduplicated) alert fired, else
            ``None``.

        Raises:
            PatientNotFoundError: If ``patient_id`` is not monitored.
        """
        if patient_id not in self.buffers:
            raise PatientNotFoundError(patient_id)

        self.buffers[patient_id].append(reading)

        anomaly = self._anomaly_score(patient_id)
        risk = self._risk_outputs(patient_id)

        final_score, severity, tte = self._combine(anomaly, risk)
        tier = self._tier_for_score(final_score)
        self._refresh_status(patient_id, risk_score=final_score, tier=tier, reading=reading)

        if tier is None or self._is_duplicate(patient_id, tier, reading.timestamp):
            return None

        alert = self._build_alert(patient_id, reading, final_score, severity, tte, tier)
        async with session_scope() as session:
            await insert_alert(session, alert)
        self.last_alert[(patient_id, tier)] = reading.timestamp
        logger.info("[%s] %s alert | score=%.2f severity=%.1f", patient_id, tier.value, final_score, severity)
        return alert

    def _combine(
        self, anomaly: Optional[float], risk: Optional[dict[str, float]]
    ) -> tuple[float, float, float]:
        """Blend the two detectors and derive severity / time-to-event."""
        if anomaly is not None and risk is not None:
            final = C.ANOMALY_SCORE_WEIGHT * anomaly + C.RISK_SCORE_WEIGHT * risk["risk_score"]
            return final, risk["severity"], risk["time_to_event"]
        if risk is not None:
            return risk["risk_score"], risk["severity"], risk["time_to_event"]
        if anomaly is not None:
            return anomaly, anomaly * C.MAX_SEVERITY, C.MAX_TIME_TO_EVENT_H * (1 - anomaly)
        return 0.0, 0.0, C.MAX_TIME_TO_EVENT_H

    def _is_duplicate(self, pid: str, tier: AlertTier, ts: datetime) -> bool:
        last = self.last_alert.get((pid, tier))
        return last is not None and (ts - last) < timedelta(minutes=C.ALERT_DEDUP_MINUTES)

    def _build_alert(
        self,
        pid: str,
        reading: SensorReading,
        score: float,
        severity: float,
        tte: float,
        tier: AlertTier,
    ) -> Alert:
        baseline = self.baseline_stats[pid]
        shap_vals = (0.0, 0.0, 0.0)
        clinical = f"{tier.value}: combined risk {score:.2f}, estimated severity {severity:.1f}."
        patient_msg = "Your readings have changed from your normal baseline; your care team is aware."

        if self.explainer is not None and len(self.buffers[pid]) >= C.PREDICTOR_WINDOW_MINUTES:
            window = self._scaled_window(pid, C.PREDICTOR_WINDOW_MINUTES)
            shap_vals = self.explainer.explain(window)
            clinical = self.explainer.generate_clinical_explanation(
                shap_vals, reading, baseline, severity, tte
            )
            patient_msg = self.explainer.generate_patient_explanation(
                shap_vals, reading, baseline, tier
            )

        return Alert(
            patient_id=pid,
            timestamp=reading.timestamp,
            risk_score=round(float(np.clip(score, 0.0, 1.0)), 4),
            severity=round(float(np.clip(severity, 0.0, C.MAX_SEVERITY)), 2),
            tier=tier,
            shap_temp=round(shap_vals[0], 4),
            shap_impedance=round(shap_vals[1], 4),
            shap_hrv=round(shap_vals[2], 4),
            clinical_explanation=clinical,
            patient_explanation=patient_msg,
        )

    # ---------------------------------------------------------- status
    def _refresh_status(
        self,
        pid: str,
        risk_score: float,
        tier: Optional[AlertTier],
        reading: Optional[SensorReading] = None,
    ) -> None:
        baseline = self.baseline_stats[pid]
        last_reading = reading or (self.buffers[pid][-1] if self.buffers[pid] else None)
        self.status[pid] = PatientStatus(
            patient_id=pid,
            archetype=baseline.archetype,
            current_risk_score=round(float(np.clip(risk_score, 0.0, 1.0)), 4),
            current_tier=tier,
            baseline_temp=baseline.baseline_temp,
            baseline_impedance=baseline.baseline_impedance,
            baseline_hrv=baseline.baseline_hrv,
            last_reading=last_reading,
            last_updated=datetime.utcnow(),
        )

    def get_current_status(self, patient_id: str) -> PatientStatus:
        """Return the latest status snapshot for a patient.

        Raises:
            PatientNotFoundError: If the patient is not monitored.
        """
        if patient_id not in self.status:
            if patient_id not in self.buffers:
                raise PatientNotFoundError(patient_id)
            self._refresh_status(patient_id, 0.0, None)
        return self.status[patient_id]

    def get_all_statuses(self) -> list[PatientStatus]:
        """Return status snapshots for every monitored patient."""
        return [self.get_current_status(pid) for pid in self.patients]

    def get_baseline(self, patient_id: str) -> PatientBaseline:
        """Return a patient's learned baseline statistics."""
        if patient_id not in self.baseline_stats:
            raise PatientNotFoundError(patient_id)
        return self.baseline_stats[patient_id]

    # ---------------------------------------------------------- simulation
    async def simulate_infection_event(self, patient_id: str) -> list[Alert]:
        """Inject 60 minutes of infection-pattern readings and process them live.

        Clinical note:
            The injected pattern reproduces the canonical early-infection signature
            — HRV declining first, then temperature rising and impedance falling —
            which drives the combined score up through WATCH -> WARNING -> CRITICAL
            exactly as a real onset would.

        Returns:
            The alerts generated during the cascade (one per tier reached).

        Raises:
            PatientNotFoundError: If the patient is not monitored.
            InsufficientDataError: If the buffer is too short to score.
        """
        if patient_id not in self.buffers:
            raise PatientNotFoundError(patient_id)
        if len(self.buffers[patient_id]) < C.BASELINE_WINDOW_MINUTES:
            raise InsufficientDataError(
                patient_id, len(self.buffers[patient_id]), C.BASELINE_WINDOW_MINUTES
            )

        baseline = self.baseline_stats[patient_id]
        rng = np.random.default_rng(C.RANDOM_SEED)
        last_ts = self.buffers[patient_id][-1].timestamp
        ev = C.ANOMALY_EVENTS["infection"]
        alerts: list[Alert] = []

        for minute in range(1, 61):
            frac = minute / 60.0
            temp = baseline.baseline_temp + float(ev["temp_rise_c"]) * frac + rng.normal(0, C.TEMP_NOISE_C)
            hrv = baseline.baseline_hrv * (1 - float(ev["hrv_drop_pct"]) * frac) + rng.normal(0, C.HRV_NOISE_MS)
            imp = baseline.baseline_impedance * (1 - float(ev["impedance_drop_pct"]) * frac) + rng.normal(0, C.IMPEDANCE_NOISE_OHM)
            reading = SensorReading(
                patient_id=patient_id,
                timestamp=last_ts + timedelta(minutes=minute),
                temp_c=float(np.clip(temp, C.TEMP_MIN_C, C.TEMP_MAX_C)),
                impedance_ohm=float(np.clip(imp, C.IMPEDANCE_MIN_OHM, C.IMPEDANCE_MAX_OHM)),
                hrv_rmssd_ms=float(np.clip(hrv, C.HRV_MIN_MS, C.HRV_MAX_MS)),
            )
            alert = await self.process_reading(patient_id, reading)
            if alert is not None:
                alerts.append(alert)

        logger.info("[%s] simulation produced %d alerts: %s", patient_id, len(alerts), [a.tier.value for a in alerts])
        return alerts


__all__ = ["InferenceEngine"]
