"""Alert API endpoints — recent per-patient alerts and the cross-patient triage feed.

Clinical purpose:
    ``/alerts/critical`` is the charge-nurse view: every actively CRITICAL patient
    across the whole monitored population, newest first, so the sickest patient is
    never buried. The per-patient feed backs the dashboard's alert timeline with
    its SHAP attributions and dual explanations.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import get_session
from data.database import get_critical_alerts, get_recent_alerts
from data.schemas import Alert, AlertTier

router = APIRouter(tags=["alerts"])


@router.get(
    "/patients/{patient_id}/alerts",
    response_model=list[Alert],
    summary="Recent alerts for a patient",
)
async def patient_alerts(
    patient_id: str,
    hours: int = Query(24, ge=1, le=720, description="Look-back window in hours"),
    tier: Optional[AlertTier] = Query(None, description="Optional tier filter"),
    session: AsyncSession = Depends(get_session),
) -> list[Alert]:
    """Return a patient's alerts within the look-back window, newest first."""
    return await get_recent_alerts(session, patient_id, hours=hours, tier=tier)


@router.get(
    "/alerts/critical",
    response_model=list[Alert],
    summary="All active CRITICAL alerts across patients",
)
async def critical_alerts(
    hours: int = Query(24, ge=1, le=720),
    session: AsyncSession = Depends(get_session),
) -> list[Alert]:
    """Return every CRITICAL alert across all patients in the window, newest first."""
    return await get_critical_alerts(session, hours=hours)
