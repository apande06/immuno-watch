"""Patient-facing API endpoints — status, readings, trend, baseline, ingestion.

Clinical purpose:
    These endpoints are what the clinical dashboard renders: where every monitored
    patient stands right now, their recent biosignals, their 7-day risk trend, and
    their learned baseline. The ingestion endpoint is the live path a real chip's
    BLE uplink would hit every few minutes.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import numpy as np
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

import constants as C
from api.dependencies import get_engine, get_session
from data.database import get_latest_reading, get_patient_history, insert_reading
from data.schemas import AlertTier, PatientStatus, SensorReading, TrendPoint
from exceptions import PatientNotFoundError
from inference.engine import InferenceEngine

router = APIRouter(tags=["patients"])


@router.get("/patients", response_model=list[PatientStatus], summary="List all patient statuses")
async def list_patients(engine: InferenceEngine = Depends(get_engine)) -> list[PatientStatus]:
    """Return the current status snapshot for every monitored patient."""
    return engine.get_all_statuses()


@router.get(
    "/patients/{patient_id}/status",
    response_model=PatientStatus,
    summary="Current status for one patient",
)
async def patient_status(
    patient_id: str, engine: InferenceEngine = Depends(get_engine)
) -> PatientStatus:
    """Return the latest risk/tier/last-reading snapshot for a patient."""
    return engine.get_current_status(patient_id)


@router.get(
    "/patients/{patient_id}/readings",
    response_model=list[SensorReading],
    summary="Historical sensor readings",
)
async def patient_readings(
    patient_id: str,
    start: Optional[datetime] = Query(None, description="Inclusive start time"),
    end: Optional[datetime] = Query(None, description="Inclusive end time"),
    limit: int = Query(100, ge=1, le=10000),
    session: AsyncSession = Depends(get_session),
) -> list[SensorReading]:
    """Return a window of a patient's persisted sensor readings."""
    readings = await get_patient_history(session, patient_id, start, end, limit)
    if not readings and not await get_latest_reading(session, patient_id):
        raise PatientNotFoundError(patient_id)
    return readings


@router.get(
    "/patients/{patient_id}/trend",
    response_model=list[TrendPoint],
    summary="7-day hourly risk trend",
)
async def patient_trend(
    patient_id: str,
    engine: InferenceEngine = Depends(get_engine),
    session: AsyncSession = Depends(get_session),
) -> list[TrendPoint]:
    """Return an hourly-bucketed risk trend over the last 7 days.

    The per-hour risk is a fast, baseline-relative clinical proxy (HRV-weighted),
    elevated by any persisted ML alert in the same bucket — this is what the
    dashboard's 7-day heatmap visualises.
    """
    latest = await get_latest_reading(session, patient_id)
    if latest is None:
        raise PatientNotFoundError(patient_id)
    end = latest.timestamp
    start = end - timedelta(days=C.TREND_DAYS)
    readings = await get_patient_history(session, patient_id, start, end, limit=C.TREND_DAYS * 24 * 60)
    baseline = engine.get_baseline(patient_id)

    buckets: dict[datetime, list[float]] = {}
    for r in readings:
        bucket = r.timestamp.replace(minute=0, second=0, microsecond=0)
        buckets.setdefault(bucket, []).append(_proxy_risk(r, baseline))

    points: list[TrendPoint] = []
    for bucket in sorted(buckets):
        risk = float(np.mean(buckets[bucket]))
        points.append(TrendPoint(timestamp=bucket, risk_score=round(risk, 4), tier=_tier(risk)))
    return points


@router.get(
    "/patients/{patient_id}/baseline",
    summary="Learned per-patient baseline statistics",
)
async def patient_baseline(
    patient_id: str, engine: InferenceEngine = Depends(get_engine)
) -> dict:
    """Return the patient's learned baseline (means, SDs, anomaly threshold)."""
    return engine.get_baseline(patient_id).model_dump(mode="json")


@router.post(
    "/patients/{patient_id}/readings",
    response_model=PatientStatus,
    status_code=201,
    summary="Ingest a reading and run inference",
)
async def ingest_reading(
    patient_id: str,
    reading: SensorReading,
    engine: InferenceEngine = Depends(get_engine),
    session: AsyncSession = Depends(get_session),
) -> PatientStatus:
    """Persist a new reading, score it through the engine, and return new status."""
    reading = reading.model_copy(update={"patient_id": patient_id})
    await insert_reading(session, reading)
    await engine.process_reading(patient_id, reading)
    return engine.get_current_status(patient_id)


# --------------------------------------------------------------------------
# Trend helpers (baseline-relative clinical risk proxy)
# --------------------------------------------------------------------------
def _proxy_risk(reading: SensorReading, baseline) -> float:
    """Fast baseline-relative risk proxy in [0, 1], HRV-weighted (leading signal)."""
    temp_c = max((reading.temp_c - baseline.baseline_temp) / C.CRITICAL_ELEVATION_THRESHOLD_C, 0.0)
    hrv_c = max((baseline.baseline_hrv - reading.hrv_rmssd_ms) / baseline.baseline_hrv, 0.0) / C.HRV_DECLINE_CRITICAL_PCT
    imp_c = max((baseline.baseline_impedance - reading.impedance_ohm) / baseline.baseline_impedance, 0.0) / C.IMPEDANCE_DECLINE_CRITICAL_PCT
    risk = 0.4 * min(hrv_c, 1.0) + 0.35 * min(temp_c, 1.0) + 0.25 * min(imp_c, 1.0)
    return float(np.clip(risk, 0.0, 1.0))


def _tier(risk: float) -> Optional[AlertTier]:
    if risk >= C.CRITICAL_THRESHOLD:
        return AlertTier.CRITICAL
    if risk >= C.WARNING_THRESHOLD:
        return AlertTier.WARNING
    if risk >= C.WATCH_THRESHOLD:
        return AlertTier.WATCH
    return None
