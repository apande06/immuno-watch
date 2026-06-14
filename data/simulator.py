"""Synthetic biosignal stream generator for immunocompromised patients.

Clinical purpose:
    Real implanted-sensor data from immunocompromised patients during infection
    onset is scarce and privacy-protected. This simulator produces physiologically
    faithful 30-day streams so the models can be developed and validated. Fidelity
    matters: the circadian temperature rhythm, sleep-related HRV elevation, and —
    most importantly — the *leading* HRV decline that precedes fever are all
    modelled, because those are the exact patterns the ML system must learn to
    detect before a clinician would.

Technical purpose:
    Emits one CSV per patient under ``data/patients/`` and bulk-inserts every
    valid reading into the SQLite database. Deterministic given the master seed in
    :mod:`constants`.

Usage:
    python data/simulator.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

# --- path bootstrap so `python data/simulator.py` can import project modules ---
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import constants as C
from data.database import (
    bulk_insert_readings,
    clear_patient_readings,
    create_db_and_tables,
    session_scope,
)
from data.schemas import SensorReading

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("immunowatch.simulator")

SIM_START = datetime(2026, 1, 1, 0, 0, 0)


def _hour_of_day(index: pd.DatetimeIndex) -> np.ndarray:
    """Fractional hour-of-day (0-24) for each timestamp."""
    return index.hour.to_numpy() + index.minute.to_numpy() / 60.0


def _circadian_temperature(hours: np.ndarray) -> np.ndarray:
    """Circadian core-temperature offset in Celsius.

    Peaks at +amplitude near 18:00 and troughs near 06:00 using a cosine — the
    standard human circadian temperature model (Refinetti, *Physiol Behav*, 2010).

    Args:
        hours: Fractional hour-of-day per sample.

    Returns:
        Per-sample additive temperature offset.
    """
    phase = 2 * np.pi * (hours - C.CIRCADIAN_TEMP_PEAK_HOUR) / 24.0
    return C.CIRCADIAN_TEMP_AMPLITUDE_C * np.cos(phase)


def _hrv_diurnal_multiplier(hours: np.ndarray) -> np.ndarray:
    """Multiplicative HRV modulation: higher in sleep, lower in active hours."""
    mult = np.ones_like(hours)
    sleep = (hours >= 0) & (hours < 6)
    activity = (hours >= 9) & (hours < 17)
    mult[sleep] += C.HRV_SLEEP_BOOST_PCT
    mult[activity] -= C.HRV_ACTIVITY_SUPPRESS_PCT
    return mult


def _impedance_diurnal_multiplier(hours: np.ndarray) -> np.ndarray:
    """Multiplicative impedance drift: +/-2% across the day, trough near 15:00."""
    # cos is maximal at h=15, so negate to make 15:00 the trough.
    phase = 2 * np.pi * (hours - 15.0) / 24.0
    return 1.0 - C.IMPEDANCE_DIURNAL_DRIFT_PCT * np.cos(phase)


def _event_envelope(
    n: int, onset_idx: int, ramp_min: int, end_idx: int, recovery_min: int
) -> np.ndarray:
    """Build a [0, 1] deviation envelope for one sensor during one event.

    The shape is: 0 before ``onset_idx``; linear ramp to 1 over ``ramp_min``;
    held at 1 until ``end_idx``; linear recovery to 0 over ``recovery_min``.

    Args:
        n: Length of the full signal.
        onset_idx: Sample index where the deviation begins.
        ramp_min: Minutes to reach peak deviation.
        end_idx: Sample index where recovery begins.
        recovery_min: Minutes to return to baseline.

    Returns:
        Envelope array of length ``n``.
    """
    env = np.zeros(n, dtype=float)
    ramp_min = max(ramp_min, 1)
    recovery_min = max(recovery_min, 1)

    ramp_end = min(onset_idx + ramp_min, n)
    if ramp_end > onset_idx:
        env[onset_idx:ramp_end] = np.linspace(0.0, 1.0, ramp_end - onset_idx, endpoint=False)

    hold_end = min(end_idx, n)
    if hold_end > ramp_end:
        env[ramp_end:hold_end] = 1.0

    rec_end = min(hold_end + recovery_min, n)
    if rec_end > hold_end:
        env[hold_end:rec_end] = np.linspace(1.0, 0.0, rec_end - hold_end, endpoint=False)
    return env


def _generate_patient_frame(
    patient_id: str, archetype: str, rng: np.random.Generator
) -> pd.DataFrame:
    """Generate the full 30-day signal frame for one patient.

    Args:
        patient_id: Identifier to stamp on every row.
        archetype: One of :data:`constants.PATIENT_ARCHETYPES`.
        rng: Seeded NumPy generator for reproducibility.

    Returns:
        DataFrame with timestamp, the three sensors, ``event_label``,
        ``severity`` and ``patient_archetype`` columns. Dropout cells are NaN.
    """
    cfg = C.PATIENT_ARCHETYPES[archetype]
    n = C.SIMULATION_DAYS * C.READINGS_PER_DAY
    index = pd.date_range(SIM_START, periods=n, freq=f"{C.SAMPLE_INTERVAL_MINUTES}min")
    hours = _hour_of_day(index)

    # --- healthy baseline physiology -------------------------------------
    temp = (
        float(cfg["temp_baseline_c"])
        + _circadian_temperature(hours)
        + rng.normal(0.0, C.TEMP_NOISE_C, n)
    )
    hrv = (
        float(cfg["hrv_baseline_ms"]) * _hrv_diurnal_multiplier(hours)
        + rng.normal(0.0, C.HRV_NOISE_MS, n)
    )
    impedance = (
        float(cfg["impedance_baseline_ohm"]) * _impedance_diurnal_multiplier(hours)
        + rng.normal(0.0, C.IMPEDANCE_NOISE_OHM, n)
    )

    event_label = np.array(["normal"] * n, dtype=object)
    severity = np.zeros(n, dtype=float)

    # --- inject anomaly events, spaced >= MIN_EVENT_SPACING_DAYS apart ----
    # Anchored after the 14-day baseline window so the learned baseline stays
    # clean, then spaced exactly 5 days apart.
    schedule = [
        ("infection", 14, 6 * 60),          # day 14, temp onset 06:00
        ("viral_mild", 19, 14 * 60),        # day 19, onset 14:00
        ("neutropenic_crisis", 24, 3 * 60),  # day 24, onset 03:00
        ("false_alarm", 29, 16 * 60),       # day 29, onset 16:00
    ]
    n_events = 0
    for label, day, minute_of_day in schedule:
        anchor = day * C.READINGS_PER_DAY + minute_of_day
        if anchor >= n:
            continue
        ev = C.ANOMALY_EVENTS[label]
        n_events += 1

        if label == "false_alarm":
            # Single-minute thermistor artifact: large but NOT sustained.
            temp[anchor] += float(ev["temp_rise_c"])
            event_label[anchor] = label
            severity[anchor] = float(ev["severity"])
            continue

        total = int(ev["total_minutes"])
        recovery = 120
        # Temperature and impedance start at the anchor; HRV LEADS by hrv_lead.
        hrv_onset = max(anchor - int(ev["hrv_lead_minutes"]), 0)
        end_idx = anchor + total

        env_temp = _event_envelope(n, anchor, int(ev["temp_ramp_minutes"]), end_idx, recovery)
        env_imp = _event_envelope(
            n, anchor, int(ev["impedance_ramp_minutes"]), end_idx, recovery
        )
        env_hrv = _event_envelope(n, hrv_onset, int(ev["hrv_ramp_minutes"]), end_idx, recovery)

        temp = temp + float(ev["temp_rise_c"]) * env_temp
        impedance = impedance * (1.0 - float(ev["impedance_drop_pct"]) * env_imp)
        hrv = hrv * (1.0 - float(ev["hrv_drop_pct"]) * env_hrv)

        # Label the active window (from the leading HRV onset to recovery start).
        label_end = min(end_idx, n)
        event_label[hrv_onset:label_end] = label
        severity[hrv_onset:label_end] = float(ev["severity"])

    # --- clip to physiological ranges (matches the Pydantic schema) -------
    temp = np.clip(temp, C.TEMP_MIN_C, C.TEMP_MAX_C)
    impedance = np.clip(impedance, C.IMPEDANCE_MIN_OHM, C.IMPEDANCE_MAX_OHM)
    hrv = np.clip(hrv, C.HRV_MIN_MS, C.HRV_MAX_MS)

    frame = pd.DataFrame(
        {
            "timestamp": index,
            "temp_c": np.round(temp, 3),
            "impedance_ohm": np.round(impedance, 2),
            "hrv_rmssd_ms": np.round(hrv, 2),
            "event_label": event_label,
            "severity": severity,
            "patient_archetype": archetype,
        }
    )

    # --- inject realistic sensor dropout (whole-reading gaps) -------------
    n_missing = int(round(C.MISSING_DATA_RATE * n))
    if n_missing > 0:
        drop_idx = rng.choice(n, size=n_missing, replace=False)
        frame.loc[drop_idx, list(C.SENSOR_COLUMNS)] = np.nan

    return frame


def _frame_to_readings(patient_id: str, frame: pd.DataFrame) -> list[SensorReading]:
    """Convert non-dropout rows of a patient frame into validated readings."""
    valid = frame.dropna(subset=list(C.SENSOR_COLUMNS))
    readings: list[SensorReading] = []
    for row in valid.itertuples(index=False):
        readings.append(
            SensorReading(
                patient_id=patient_id,
                timestamp=row.timestamp.to_pydatetime(),
                temp_c=float(row.temp_c),
                impedance_ohm=float(row.impedance_ohm),
                hrv_rmssd_ms=float(row.hrv_rmssd_ms),
            )
        )
    return readings


async def _persist_readings(patient_id: str, readings: list[SensorReading]) -> None:
    """Replace any prior readings for the patient and bulk-insert the new set."""
    async with session_scope() as session:
        await clear_patient_readings(session, patient_id)
        # Chunk inserts to keep individual transactions modest in size.
        chunk = 5000
        for start in range(0, len(readings), chunk):
            await bulk_insert_readings(session, readings[start : start + chunk])


async def generate_all() -> None:
    """Generate, persist, and summarise data for every patient archetype."""
    C.PATIENT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    await create_db_and_tables()

    logger.info("Generating %d-day streams for %d patients", C.SIMULATION_DAYS, len(C.PATIENT_ARCHETYPES))
    for offset, archetype in enumerate(C.PATIENT_ARCHETYPES):
        patient_id = archetype  # one representative patient per archetype
        rng = np.random.default_rng(C.RANDOM_SEED + offset)
        frame = _generate_patient_frame(patient_id, archetype, rng)

        csv_path = C.PATIENT_DATA_DIR / f"{patient_id}.csv"
        frame.to_csv(csv_path, index=False)

        readings = _frame_to_readings(patient_id, frame)
        await _persist_readings(patient_id, readings)

        n_missing = int(frame[list(C.SENSOR_COLUMNS)].isna().any(axis=1).sum())
        n_events = int((frame["event_label"] != "normal").any())
        events_present = sorted(set(frame["event_label"]) - {"normal"})
        logger.info(
            "[%s] %d readings | %d persisted | %d dropout rows | events: %s",
            patient_id,
            len(frame),
            len(readings),
            n_missing,
            ", ".join(events_present) or "none",
        )

    logger.info("Simulation complete. CSVs in %s", C.PATIENT_DATA_DIR)


def main() -> None:
    """Console entry point."""
    asyncio.run(generate_all())


if __name__ == "__main__":
    main()
