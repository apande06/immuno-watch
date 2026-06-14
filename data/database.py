"""Async persistence layer — SQLAlchemy 2.0 ORM over SQLite (aiosqlite).

Clinical purpose:
    Every reading and every alert must be durably recorded. In a real deployment
    this is the medico-legal audit trail: which signals were observed, what the
    model concluded, when the care team was notified. We therefore persist the
    raw readings, the learned per-patient baselines, and every alert with its full
    SHAP attribution.

Technical purpose:
    SQLAlchemy 2.0's async ORM with ``aiosqlite`` lets the FastAPI event loop
    perform non-blocking database I/O. SQLite keeps the portfolio project
    zero-config while the async session factory mirrors how a production Postgres
    deployment would be wired.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any, Optional

from sqlalchemy import (
    DateTime,
    Float,
    Index,
    String,
    delete,
    select,
)
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

import constants as C
from data.schemas import Alert, AlertTier, PatientBaseline, SensorReading

logger = logging.getLogger("immunowatch.database")


class Base(DeclarativeBase):
    """Declarative base for all ImmunoWatch ORM models."""


class SensorReadingORM(Base):
    """Persisted raw sensor reading."""

    __tablename__ = "sensor_readings"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    patient_id: Mapped[str] = mapped_column(String(64), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)
    temp_c: Mapped[float] = mapped_column(Float)
    impedance_ohm: Mapped[float] = mapped_column(Float)
    hrv_rmssd_ms: Mapped[float] = mapped_column(Float)

    __table_args__ = (
        Index("ix_reading_patient_time", "patient_id", "timestamp"),
    )

    def to_schema(self) -> SensorReading:
        """Convert this row to its Pydantic schema."""
        return SensorReading(
            patient_id=self.patient_id,
            timestamp=self.timestamp,
            temp_c=self.temp_c,
            impedance_ohm=self.impedance_ohm,
            hrv_rmssd_ms=self.hrv_rmssd_ms,
        )


class AlertORM(Base):
    """Persisted clinical alert with SHAP attribution."""

    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    patient_id: Mapped[str] = mapped_column(String(64), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)
    risk_score: Mapped[float] = mapped_column(Float)
    severity: Mapped[float] = mapped_column(Float)
    tier: Mapped[str] = mapped_column(String(16), index=True)
    shap_temp: Mapped[float] = mapped_column(Float)
    shap_impedance: Mapped[float] = mapped_column(Float)
    shap_hrv: Mapped[float] = mapped_column(Float)
    clinical_explanation: Mapped[str] = mapped_column(String)
    patient_explanation: Mapped[str] = mapped_column(String)

    __table_args__ = (
        Index("ix_alert_patient_time", "patient_id", "timestamp"),
    )

    def to_schema(self) -> Alert:
        """Convert this row to its Pydantic schema."""
        return Alert(
            patient_id=self.patient_id,
            timestamp=self.timestamp,
            risk_score=self.risk_score,
            severity=self.severity,
            tier=AlertTier(self.tier),
            shap_temp=self.shap_temp,
            shap_impedance=self.shap_impedance,
            shap_hrv=self.shap_hrv,
            clinical_explanation=self.clinical_explanation,
            patient_explanation=self.patient_explanation,
        )


class PatientBaselineORM(Base):
    """Persisted per-patient learned baseline statistics."""

    __tablename__ = "patient_baselines"

    patient_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    archetype: Mapped[str] = mapped_column(String(32))
    baseline_temp: Mapped[float] = mapped_column(Float)
    baseline_temp_sd: Mapped[float] = mapped_column(Float)
    baseline_impedance: Mapped[float] = mapped_column(Float)
    baseline_impedance_sd: Mapped[float] = mapped_column(Float)
    baseline_hrv: Mapped[float] = mapped_column(Float)
    baseline_hrv_sd: Mapped[float] = mapped_column(Float)
    anomaly_threshold: Mapped[float] = mapped_column(Float, default=1.0)
    trained_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    def to_schema(self) -> PatientBaseline:
        """Convert this row to its Pydantic schema."""
        return PatientBaseline(
            patient_id=self.patient_id,
            archetype=self.archetype,
            baseline_temp=self.baseline_temp,
            baseline_temp_sd=self.baseline_temp_sd,
            baseline_impedance=self.baseline_impedance,
            baseline_impedance_sd=self.baseline_impedance_sd,
            baseline_hrv=self.baseline_hrv,
            baseline_hrv_sd=self.baseline_hrv_sd,
            anomaly_threshold=self.anomaly_threshold,
            trained_at=self.trained_at,
        )


# ---------------------------------------------------------------------------
# Engine / session factory
# ---------------------------------------------------------------------------
_engine = create_async_engine(C.DATABASE_URL, echo=False, future=True)
async_session_factory = async_sessionmaker(_engine, expire_on_commit=False, class_=AsyncSession)


def get_engine():
    """Return the process-wide async engine."""
    return _engine


async def create_db_and_tables() -> None:
    """Create all tables if they do not exist (idempotent).

    Clinical note:
        Safe to call on every startup; existing data is never dropped.
    """
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database schema ready at %s", C.DATABASE_PATH)


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """Provide a transactional async session as a context manager.

    Yields:
        An :class:`AsyncSession` that is committed on success and rolled back on
        any exception.
    """
    session = async_session_factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


# ---------------------------------------------------------------------------
# CRUD helpers
# ---------------------------------------------------------------------------
async def insert_reading(session: AsyncSession, reading: SensorReading) -> SensorReadingORM:
    """Insert a single sensor reading.

    Args:
        session: Active async session.
        reading: Validated sensor reading to persist.

    Returns:
        The persisted ORM row (flushed, with primary key populated).
    """
    row = SensorReadingORM(
        patient_id=reading.patient_id,
        timestamp=reading.timestamp,
        temp_c=reading.temp_c,
        impedance_ohm=reading.impedance_ohm,
        hrv_rmssd_ms=reading.hrv_rmssd_ms,
    )
    session.add(row)
    await session.flush()
    return row


async def bulk_insert_readings(
    session: AsyncSession, readings: list[SensorReading]
) -> int:
    """Efficiently insert many readings (used by the simulator).

    Args:
        session: Active async session.
        readings: Validated readings to persist.

    Returns:
        Number of rows inserted.
    """
    rows = [
        SensorReadingORM(
            patient_id=r.patient_id,
            timestamp=r.timestamp,
            temp_c=r.temp_c,
            impedance_ohm=r.impedance_ohm,
            hrv_rmssd_ms=r.hrv_rmssd_ms,
        )
        for r in readings
    ]
    session.add_all(rows)
    await session.flush()
    return len(rows)


async def insert_alert(session: AsyncSession, alert: Alert) -> AlertORM:
    """Persist a clinical alert.

    Args:
        session: Active async session.
        alert: Validated alert to persist.

    Returns:
        The persisted ORM row.
    """
    row = AlertORM(
        patient_id=alert.patient_id,
        timestamp=alert.timestamp,
        risk_score=alert.risk_score,
        severity=alert.severity,
        tier=alert.tier.value,
        shap_temp=alert.shap_temp,
        shap_impedance=alert.shap_impedance,
        shap_hrv=alert.shap_hrv,
        clinical_explanation=alert.clinical_explanation,
        patient_explanation=alert.patient_explanation,
    )
    session.add(row)
    await session.flush()
    return row


async def get_patient_history(
    session: AsyncSession,
    patient_id: str,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    limit: int = 100,
) -> list[SensorReading]:
    """Return a patient's readings within an optional time window.

    Args:
        session: Active async session.
        patient_id: Patient to query.
        start: Inclusive lower time bound (optional).
        end: Inclusive upper time bound (optional).
        limit: Maximum number of (most-recent) readings to return.

    Returns:
        Readings ordered by ascending timestamp.
    """
    stmt = select(SensorReadingORM).where(SensorReadingORM.patient_id == patient_id)
    if start is not None:
        stmt = stmt.where(SensorReadingORM.timestamp >= start)
    if end is not None:
        stmt = stmt.where(SensorReadingORM.timestamp <= end)
    # Take the most recent `limit` rows, then return them in chronological order.
    stmt = stmt.order_by(SensorReadingORM.timestamp.desc()).limit(limit)
    result = await session.execute(stmt)
    rows = list(result.scalars().all())
    rows.reverse()
    return [row.to_schema() for row in rows]


async def get_latest_reading(
    session: AsyncSession, patient_id: str
) -> Optional[SensorReading]:
    """Return a patient's most recent reading, if any."""
    stmt = (
        select(SensorReadingORM)
        .where(SensorReadingORM.patient_id == patient_id)
        .order_by(SensorReadingORM.timestamp.desc())
        .limit(1)
    )
    row = (await session.execute(stmt)).scalars().first()
    return row.to_schema() if row else None


async def get_recent_alerts(
    session: AsyncSession,
    patient_id: str,
    hours: int = 24,
    tier: Optional[AlertTier] = None,
) -> list[Alert]:
    """Return recent alerts for a patient.

    Args:
        session: Active async session.
        patient_id: Patient to query.
        hours: Look-back window in hours.
        tier: Optional tier filter.

    Returns:
        Alerts ordered newest-first.
    """
    cutoff = _utcnow() - timedelta(hours=hours)
    stmt = (
        select(AlertORM)
        .where(AlertORM.patient_id == patient_id)
        .where(AlertORM.timestamp >= cutoff)
    )
    if tier is not None:
        stmt = stmt.where(AlertORM.tier == tier.value)
    stmt = stmt.order_by(AlertORM.timestamp.desc())
    result = await session.execute(stmt)
    return [row.to_schema() for row in result.scalars().all()]


async def get_critical_alerts(session: AsyncSession, hours: int = 24) -> list[Alert]:
    """Return all CRITICAL alerts across every patient in the window."""
    cutoff = _utcnow() - timedelta(hours=hours)
    stmt = (
        select(AlertORM)
        .where(AlertORM.tier == AlertTier.CRITICAL.value)
        .where(AlertORM.timestamp >= cutoff)
        .order_by(AlertORM.timestamp.desc())
    )
    result = await session.execute(stmt)
    return [row.to_schema() for row in result.scalars().all()]


async def upsert_patient_baseline(
    session: AsyncSession, patient_id: str, baseline_stats: dict[str, Any]
) -> PatientBaselineORM:
    """Insert or update a patient's learned baseline.

    Args:
        session: Active async session.
        patient_id: Patient the baseline belongs to.
        baseline_stats: Mapping of baseline fields (see PatientBaselineORM columns).

    Returns:
        The persisted/updated ORM row.
    """
    existing = await session.get(PatientBaselineORM, patient_id)
    if existing is None:
        existing = PatientBaselineORM(patient_id=patient_id)
        session.add(existing)
    for field, value in baseline_stats.items():
        if hasattr(existing, field):
            setattr(existing, field, value)
    existing.trained_at = baseline_stats.get("trained_at", _utcnow())
    await session.flush()
    return existing


async def get_patient_baseline(
    session: AsyncSession, patient_id: str
) -> Optional[PatientBaseline]:
    """Return a patient's stored baseline, if one has been trained."""
    row = await session.get(PatientBaselineORM, patient_id)
    return row.to_schema() if row else None


async def list_patient_ids(session: AsyncSession) -> list[str]:
    """Return the distinct patient ids that have any readings."""
    stmt = select(SensorReadingORM.patient_id).distinct().order_by(SensorReadingORM.patient_id)
    result = await session.execute(stmt)
    return [pid for pid in result.scalars().all()]


async def clear_patient_readings(session: AsyncSession, patient_id: str) -> int:
    """Delete all readings for a patient (used when re-running the simulator)."""
    result = await session.execute(
        delete(SensorReadingORM).where(SensorReadingORM.patient_id == patient_id)
    )
    return int(result.rowcount or 0)


def _utcnow() -> datetime:
    """Naive UTC now (SQLite stores naive datetimes consistently)."""
    return datetime.utcnow()


__all__ = [
    "Base",
    "SensorReadingORM",
    "AlertORM",
    "PatientBaselineORM",
    "async_session_factory",
    "get_engine",
    "create_db_and_tables",
    "session_scope",
    "insert_reading",
    "bulk_insert_readings",
    "insert_alert",
    "get_patient_history",
    "get_latest_reading",
    "get_recent_alerts",
    "get_critical_alerts",
    "upsert_patient_baseline",
    "get_patient_baseline",
    "list_patient_ids",
    "clear_patient_readings",
]
