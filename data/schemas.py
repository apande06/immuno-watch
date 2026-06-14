"""Pydantic v2 data models — the typed contract between every ImmunoWatch layer.

Clinical purpose:
    These models encode what a *valid* clinical observation looks like. A core
    temperature of 50C or an HRV of -3ms is physiologically impossible and almost
    certainly a sensor fault; rejecting it at the boundary prevents corrupt data
    from ever reaching the models or a clinician's screen.

Technical purpose:
    Shared schemas used by the simulator, database ORM, inference engine, and
    FastAPI request/response bodies. Field validators enforce physiological ranges
    once, in one place, so no downstream code has to re-check them.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

import constants as C


class AlertTier(str, Enum):
    """Three-tier clinical escalation ladder.

    Clinical note:
        Tiers map to concrete care-team actions: WATCH = increase monitoring
        cadence; WARNING = clinician review + labs; CRITICAL = immediate
        in-person evaluation per neutropenic-fever protocol.
    """

    WATCH = "WATCH"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class SensorReading(BaseModel):
    """Single sensor snapshot from the implanted chip.

    Clinical note:
        The three channels are deliberately complementary: temperature is the
        classic but *lagging* indicator, HRV is the earliest *leading* indicator,
        and impedance tracks the inflammatory fluid shift in between.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "patient_id": "chemo_nadir_01",
                "timestamp": "2026-01-15T14:32:00",
                "temp_c": 36.7,
                "impedance_ohm": 418.2,
                "hrv_rmssd_ms": 27.4,
            }
        }
    )

    patient_id: str = Field(..., min_length=1, description="Stable patient identifier")
    timestamp: datetime = Field(..., description="UTC timestamp of the reading")
    temp_c: float = Field(
        ..., ge=C.TEMP_MIN_C, le=C.TEMP_MAX_C, description="Core temperature in Celsius"
    )
    impedance_ohm: float = Field(
        ...,
        ge=C.IMPEDANCE_MIN_OHM,
        le=C.IMPEDANCE_MAX_OHM,
        description="Bioelectrical impedance in Ohms",
    )
    hrv_rmssd_ms: float = Field(
        ..., ge=C.HRV_MIN_MS, le=C.HRV_MAX_MS, description="HRV RMSSD in milliseconds"
    )

    @field_validator("temp_c", "impedance_ohm", "hrv_rmssd_ms")
    @classmethod
    def _reject_nan(cls, value: float) -> float:
        """Reject NaN/inf at the boundary (range checks let NaN slip through)."""
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("sensor value must be a finite number")
        return value


class Alert(BaseModel):
    """Clinical alert with SHAP explainability fields.

    Clinical note:
        Every alert carries *both* a physician-grade explanation (precise values,
        medical terminology, recommended order set) and a plain-language patient
        explanation. The SHAP contributions make the model's reasoning auditable.
    """

    patient_id: str = Field(..., min_length=1)
    timestamp: datetime
    risk_score: float = Field(..., ge=0.0, le=1.0)
    severity: float = Field(..., ge=0.0, le=C.MAX_SEVERITY)
    tier: AlertTier
    shap_temp: float = Field(..., description="SHAP contribution from temperature")
    shap_impedance: float = Field(..., description="SHAP contribution from impedance")
    shap_hrv: float = Field(..., description="SHAP contribution from HRV")
    clinical_explanation: str = Field(..., description="For the physician dashboard")
    patient_explanation: str = Field(..., description="Plain language for the patient app")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "patient_id": "chemo_nadir_01",
                "timestamp": "2026-01-15T03:12:00",
                "risk_score": 0.91,
                "severity": 8.4,
                "tier": "CRITICAL",
                "shap_temp": 0.21,
                "shap_impedance": 0.18,
                "shap_hrv": 0.47,
                "clinical_explanation": "Alert triggered by 3 converging abnormal signals...",
                "patient_explanation": "Your heart rhythm variability has been declining...",
            }
        }
    )


class PatientStatus(BaseModel):
    """Current health-status snapshot for the dashboard."""

    patient_id: str
    archetype: str
    current_risk_score: float = Field(..., ge=0.0, le=1.0)
    current_tier: Optional[AlertTier] = None
    baseline_temp: float
    baseline_impedance: float
    baseline_hrv: float
    last_reading: Optional[SensorReading] = None
    last_updated: datetime


class TrendPoint(BaseModel):
    """One bucket in a patient's time-aggregated risk trend."""

    timestamp: datetime
    risk_score: float = Field(..., ge=0.0, le=1.0)
    tier: Optional[AlertTier] = None


class PatientBaseline(BaseModel):
    """Learned per-patient baseline statistics (days 1-14).

    Clinical note:
        This is the heart of the personalised approach — the reference against
        which every later reading is judged. ``anomaly_threshold`` is the 95th
        percentile of the autoencoder's reconstruction error on healthy data.
    """

    patient_id: str
    archetype: str
    baseline_temp: float
    baseline_temp_sd: float
    baseline_impedance: float
    baseline_impedance_sd: float
    baseline_hrv: float
    baseline_hrv_sd: float
    anomaly_threshold: float = Field(
        default=1.0, description="95th-pct reconstruction error on normal data"
    )
    trained_at: Optional[datetime] = None


def export_json_schemas() -> dict[str, dict]:
    """Return the JSON Schema for each public model (used in docs/tests).

    Returns:
        Mapping of model name to its JSON Schema dict.
    """
    return {
        model.__name__: model.model_json_schema()
        for model in (SensorReading, Alert, PatientStatus, TrendPoint, PatientBaseline)
    }


__all__ = [
    "AlertTier",
    "SensorReading",
    "Alert",
    "PatientStatus",
    "TrendPoint",
    "PatientBaseline",
    "export_json_schemas",
]


if __name__ == "__main__":  # pragma: no cover - manual schema inspection helper
    import json

    print(json.dumps(export_json_schemas(), indent=2, default=str))
