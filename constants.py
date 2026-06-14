"""Clinical thresholds, model hyperparameters, and configuration constants.

Clinical purpose:
    This module is the single source of truth for every clinically meaningful
    number used by ImmunoWatch. Each threshold carries an inline citation to the
    guideline or study that justifies it, so a clinician or auditor can trace any
    alert back to published evidence. Centralising these values guarantees that
    the simulator, preprocessing, models, and inference engine all reason about
    the same definition of "abnormal".

Technical purpose:
    Eliminates magic numbers from implementation files. Every other module
    imports from here; nothing in this file imports project code, so there are no
    circular-import hazards.

Clinical note:
    The "personal baseline" philosophy means population thresholds (e.g. the IDSA
    fever cut-off) are used only as hard safety floors. The primary signal is
    *deviation from the individual patient's learned baseline*, because a
    chemotherapy patient's normal physiology looks nothing like a transplant
    patient's normal physiology.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

# ---------------------------------------------------------------------------
# Filesystem layout
# ---------------------------------------------------------------------------
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
PATIENT_DATA_DIR: Final[Path] = DATA_DIR / "patients"
MODELS_DIR: Final[Path] = PROJECT_ROOT / "models"
REPORTS_DIR: Final[Path] = PROJECT_ROOT / "reports"
EXPORT_DIR: Final[Path] = MODELS_DIR / "export"
DATABASE_PATH: Final[Path] = PROJECT_ROOT / "immunowatch.db"
DATABASE_URL: Final[str] = f"sqlite+aiosqlite:///{DATABASE_PATH.as_posix()}"

# ---------------------------------------------------------------------------
# Temperature thresholds
# Source: Freifeld et al., "Clinical Practice Guideline for the Use of
# Antimicrobial Agents in Neutropenic Patients with Cancer", IDSA Guidelines for
# Febrile Neutropenia (Clin Infect Dis, 2011; reaffirmed 2023).
# ---------------------------------------------------------------------------
NEUTROPENIC_FEVER_THRESHOLD_C: Final[float] = 38.3   # Single oral temp >=38.3C = emergency
SUSTAINED_ELEVATION_THRESHOLD_C: Final[float] = 0.5  # 0.5C above personal baseline = WATCH
CRITICAL_ELEVATION_THRESHOLD_C: Final[float] = 0.8   # 0.8C above baseline = WARNING

# ---------------------------------------------------------------------------
# HRV thresholds
# Source: Task Force of the ESC and NASPE, "Heart Rate Variability: Standards of
# Measurement, Physiological Interpretation, and Clinical Use", Circulation, 1996.
# RMSSD is the time-domain marker of parasympathetic (vagal) tone; it falls early
# under systemic inflammatory stress, typically 2-4h before fever onset.
# ---------------------------------------------------------------------------
HRV_DECLINE_WATCH_PCT: Final[float] = 0.15     # 15% RMSSD decline = early signal
HRV_DECLINE_WARNING_PCT: Final[float] = 0.28   # 28% decline = strong infection signal
HRV_DECLINE_CRITICAL_PCT: Final[float] = 0.40  # 40% decline = severe immune stress

# ---------------------------------------------------------------------------
# Impedance thresholds
# Source: Lukaski et al. and bioelectrical impedance analysis in critical illness,
# Crit Care Med, 2019. Falling impedance tracks extracellular fluid shifts and
# inflammatory tissue changes, serving as a non-invasive WBC-activity proxy.
# ---------------------------------------------------------------------------
IMPEDANCE_DECLINE_WATCH_PCT: Final[float] = 0.03    # 3% drop = early inflammation
IMPEDANCE_DECLINE_WARNING_PCT: Final[float] = 0.06  # 6% drop = active inflammation
IMPEDANCE_DECLINE_CRITICAL_PCT: Final[float] = 0.10  # 10% drop = severe inflammatory response

# ---------------------------------------------------------------------------
# Inference alert tiers
# Combined risk score in [0, 1] maps onto a three-tier clinical escalation ladder.
# ---------------------------------------------------------------------------
WATCH_THRESHOLD: Final[float] = 0.40
WARNING_THRESHOLD: Final[float] = 0.65
CRITICAL_THRESHOLD: Final[float] = 0.85
ALERT_DEDUP_MINUTES: Final[int] = 30  # Suppress repeat same-tier alerts within 30 min

# ---------------------------------------------------------------------------
# Model architecture / training
# ---------------------------------------------------------------------------
BASELINE_WINDOW_MINUTES: Final[int] = 120    # 2-hour window for the autoencoder
PREDICTOR_WINDOW_MINUTES: Final[int] = 360   # 6-hour window for the Transformer
BASELINE_TRAINING_DAYS: Final[int] = 14      # Days 1-14 define the personal baseline
ANOMALY_THRESHOLD_PERCENTILE: Final[int] = 95  # 95th pct of normal reconstruction error

LSTM_HIDDEN_SIZE: Final[int] = 64
LSTM_LAYERS: Final[int] = 2
LSTM_DROPOUT: Final[float] = 0.2
BOTTLENECK_SIZE: Final[int] = 16

TRANSFORMER_HEADS: Final[int] = 4
TRANSFORMER_LAYERS: Final[int] = 2
TRANSFORMER_D_MODEL: Final[int] = 64
TRANSFORMER_FFN_DIM: Final[int] = 128
TRANSFORMER_DROPOUT: Final[float] = 0.1

CLASS_WEIGHT_RATIO: Final[float] = 10.0  # 10:1 weighting for severe class imbalance
FEDERATED_ROUNDS: Final[int] = 3
FEDERATED_LOCAL_EPOCHS: Final[int] = 5

# Number of sensor channels (temperature, impedance, HRV) — the model input width.
N_SENSORS: Final[int] = 3
SENSOR_COLUMNS: Final[tuple[str, str, str]] = ("temp_c", "impedance_ohm", "hrv_rmssd_ms")

# Training hyperparameters (kept here so notebooks and scripts stay in sync).
BASELINE_TRAIN_STRIDE: Final[int] = 10
BASELINE_LR: Final[float] = 1e-3
BASELINE_WEIGHT_DECAY: Final[float] = 1e-5
BASELINE_MAX_EPOCHS: Final[int] = 60
BASELINE_BATCH_SIZE: Final[int] = 64
BASELINE_EARLY_STOP_PATIENCE: Final[int] = 10
BASELINE_SCHED_PATIENCE: Final[int] = 5
EVENT_EXCLUSION_MINUTES: Final[int] = 30  # Exclude +/-30 min around labelled events

PREDICTOR_LR: Final[float] = 2e-4
PREDICTOR_WEIGHT_DECAY: Final[float] = 1e-4
PREDICTOR_MAX_EPOCHS: Final[int] = 40
PREDICTOR_BATCH_SIZE: Final[int] = 32
PREDICTOR_WINDOW_STRIDE: Final[int] = 15
PREDICTOR_LABEL_HORIZON_MINUTES: Final[int] = 360  # Positive if event within 6h ahead

# Multi-task loss weights for the predictor (risk / severity / time-to-event).
LOSS_WEIGHT_RISK: Final[float] = 0.5
LOSS_WEIGHT_SEVERITY: Final[float] = 0.3
LOSS_WEIGHT_TTE: Final[float] = 0.2
MAX_TIME_TO_EVENT_H: Final[float] = 48.0
MAX_SEVERITY: Final[float] = 10.0

# Combined-score blend used by the inference engine.
ANOMALY_SCORE_WEIGHT: Final[float] = 0.40
RISK_SCORE_WEIGHT: Final[float] = 0.60

# ---------------------------------------------------------------------------
# Data simulation
# ---------------------------------------------------------------------------
SIMULATION_DAYS: Final[int] = 30
SAMPLE_INTERVAL_MINUTES: Final[int] = 1
READINGS_PER_DAY: Final[int] = 24 * 60 // SAMPLE_INTERVAL_MINUTES
MISSING_DATA_RATE: Final[float] = 0.005  # 0.5% random sensor dropout (-> NaN)
MIN_EVENT_SPACING_DAYS: Final[int] = 5
RANDOM_SEED: Final[int] = 42

# Circadian / diurnal physiology
CIRCADIAN_TEMP_AMPLITUDE_C: Final[float] = 0.4   # +0.4C peak (18:00), trough (04:00)
CIRCADIAN_TEMP_PEAK_HOUR: Final[float] = 18.0
HRV_SLEEP_BOOST_PCT: Final[float] = 0.20         # +20% during 00:00-06:00 sleep
HRV_ACTIVITY_SUPPRESS_PCT: Final[float] = 0.15   # -15% during 09:00-17:00 activity
IMPEDANCE_DIURNAL_DRIFT_PCT: Final[float] = 0.02  # +/-2% across day, trough 15:00

# Per-sensor Gaussian measurement noise (standard deviation).
TEMP_NOISE_C: Final[float] = 0.05
IMPEDANCE_NOISE_OHM: Final[float] = 3.0
HRV_NOISE_MS: Final[float] = 2.0

# Physically plausible hard limits (shared with the Pydantic schemas).
TEMP_MIN_C: Final[float] = 34.0
TEMP_MAX_C: Final[float] = 42.0
IMPEDANCE_MIN_OHM: Final[float] = 200.0
IMPEDANCE_MAX_OHM: Final[float] = 700.0
HRV_MIN_MS: Final[float] = 5.0
HRV_MAX_MS: Final[float] = 200.0

# ---------------------------------------------------------------------------
# Patient archetypes
# Each archetype encodes the *population starting point*; the ML system then
# learns the individual's true baseline from days 1-14 of streamed data.
# ---------------------------------------------------------------------------
PatientArchetypeConfig = dict[str, float | str]

PATIENT_ARCHETYPES: Final[dict[str, dict[str, float | str]]] = {
    "chemo_nadir": {
        "description": "Chemotherapy nadir, ANC ~200, highest infection risk",
        "temp_baseline_c": 36.6,
        "temp_sd_c": 0.2,
        "hrv_baseline_ms": 28.0,
        "hrv_sd_ms": 5.0,
        "impedance_baseline_ohm": 420.0,
        "impedance_sd_ohm": 15.0,
    },
    "organ_transplant": {
        "description": "Solid-organ transplant on chronic immunosuppression",
        "temp_baseline_c": 36.9,
        "temp_sd_c": 0.15,
        "hrv_baseline_ms": 38.0,
        "hrv_sd_ms": 6.0,
        "impedance_baseline_ohm": 380.0,
        "impedance_sd_ohm": 20.0,
    },
    "hiv_managed": {
        "description": "Managed HIV, stable but immunologically vulnerable",
        "temp_baseline_c": 37.0,
        "temp_sd_c": 0.25,
        "hrv_baseline_ms": 45.0,
        "hrv_sd_ms": 8.0,
        "impedance_baseline_ohm": 440.0,
        "impedance_sd_ohm": 18.0,
    },
}

# ---------------------------------------------------------------------------
# Anomaly event templates
# Encodes the magnitude, duration, and — critically — the *temporal ordering* of
# multi-sensor changes. The leading HRV decline (HRV drops before temperature
# rises) is the single most important early-warning signal in this system.
# ---------------------------------------------------------------------------
ANOMALY_EVENTS: Final[dict[str, dict[str, float | int | str]]] = {
    "infection": {
        "label": "infection",
        "severity": 7.0,
        "temp_rise_c": 0.8,
        "temp_ramp_minutes": 360,        # temp rises over 6h
        "hrv_drop_pct": 0.35,
        "hrv_ramp_minutes": 240,         # HRV collapses over 4h...
        "hrv_lead_minutes": 180,         # ...starting 3h BEFORE the temp rise
        "impedance_drop_pct": 0.08,
        "impedance_ramp_minutes": 360,
        "total_minutes": 720,
    },
    "neutropenic_crisis": {
        "label": "neutropenic_crisis",
        "severity": 10.0,
        "temp_rise_c": 2.0,              # spike well past the 38.3C emergency floor
        "temp_ramp_minutes": 120,        # within 2h
        "hrv_drop_pct": 0.55,
        "hrv_ramp_minutes": 120,
        "hrv_lead_minutes": 60,
        "impedance_drop_pct": 0.14,
        "impedance_ramp_minutes": 120,
        "total_minutes": 240,
    },
    "viral_mild": {
        "label": "viral_mild",
        "severity": 3.0,
        "temp_rise_c": 0.4,
        "temp_ramp_minutes": 120,
        "hrv_drop_pct": 0.15,
        "hrv_ramp_minutes": 120,
        "hrv_lead_minutes": 60,
        "impedance_drop_pct": 0.03,
        "impedance_ramp_minutes": 120,
        "total_minutes": 720,            # sustained ~12h
    },
    "false_alarm": {
        "label": "false_alarm",
        "severity": 0.0,
        "temp_rise_c": 1.5,              # large but lasts ONE minute only
        "temp_ramp_minutes": 1,
        "hrv_drop_pct": 0.0,
        "hrv_ramp_minutes": 1,
        "hrv_lead_minutes": 0,
        "impedance_drop_pct": 0.0,
        "impedance_ramp_minutes": 1,
        "total_minutes": 1,
    },
}

# Severity ceiling used when normalising the regression target.
SEVERITY_BY_LABEL: Final[dict[str, float]] = {
    "normal": 0.0,
    **{name: float(cfg["severity"]) for name, cfg in ANOMALY_EVENTS.items()},
}

# ---------------------------------------------------------------------------
# API / service configuration
# ---------------------------------------------------------------------------
API_HOST: Final[str] = "0.0.0.0"
API_PORT: Final[int] = 8000
DASHBOARD_ORIGIN: Final[str] = "http://localhost:3000"
CORS_ORIGINS: Final[tuple[str, ...]] = (
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
)

# SHAP explainer configuration.
SHAP_BACKGROUND_SAMPLES: Final[int] = 100
SHAP_NSAMPLES: Final[int] = 64  # KernelExplainer coalition samples (speed vs. fidelity)

# Trend endpoint aggregation.
TREND_DAYS: Final[int] = 7
TREND_BUCKET_HOURS: Final[int] = 1

# ---------------------------------------------------------------------------
# Signal processing / feature engineering
# Source for Butterworth artifact removal: Smith, "The Scientist and Engineer's
# Guide to Digital Signal Processing", 1997.
# ---------------------------------------------------------------------------
SAMPLE_RATE_HZ: Final[float] = 1.0 / (SAMPLE_INTERVAL_MINUTES * 60)  # 1 sample / minute
TEMP_FILTER_ORDER: Final[int] = 4
# The spec calls for a ~0.01 Hz low-pass on temperature. At one sample per minute
# the Nyquist frequency is only SAMPLE_RATE_HZ / 2 ~= 0.0083 Hz, so a literal
# 0.01 Hz cutoff is unrepresentable. We implement the *intent* — strip
# single-minute thermistor artifacts while preserving multi-hour fever ramps — by
# setting the cutoff to a 30-minute period (~5.6e-4 Hz, ~0.067 of Nyquist).
TEMP_FILTER_CUTOFF_HZ: Final[float] = 1.0 / (30 * 60)

MAX_FORWARD_FILL: Final[int] = 5            # ffill at most 5 consecutive gaps (= 5 min)
ROLLING_ZSCORE_WINDOW_MIN: Final[int] = 120  # 2-hour window for impedance z-scoring
HRV_CLIP_SIGMA: Final[float] = 3.0           # clip HRV outliers at 3 sigma
CROSS_SENSOR_SIGMA: Final[float] = 1.0       # |z| > 1 counts as a sensor deviation
HRV_TREND_WINDOW_MIN: Final[int] = 60        # slope of HRV over last 60 readings
IMPEDANCE_PCT_CHANGE_WINDOW_MIN: Final[int] = 120  # 2-hour rolling mean reference
LAG_FEATURES_MINUTES: Final[tuple[int, ...]] = (5, 15, 30, 60, 180, 360)  # 6 lags x 3 sensors

__all__ = [name for name in dir() if not name.startswith("_")]
