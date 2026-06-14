"""Clinical signal-processing and feature-engineering pipeline.

Clinical purpose:
    Raw implant data is noisy: thermistors throw single-sample artifacts, the
    radio drops packets, and every channel carries diurnal drift unrelated to
    infection. This pipeline cleans those nuisances *without erasing the signal we
    care about* — the slow, multi-hour physiological shift that precedes fever.
    Order matters clinically: we remove artifacts first, then normalise, then
    derive features, so artifacts never propagate into the derived signals a
    clinician would act on.

Technical purpose:
    Turns a per-patient CSV into (a) a clean, per-patient-scaled 3-channel sensor
    matrix that feeds both deep models and (b) a set of interpretable derived
    features used by the dashboard, SHAP context, and the training notebook.

All parameters are imported from :mod:`constants` — no magic numbers here.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler

import constants as C
from exceptions import InsufficientDataError

logger = logging.getLogger("immunowatch.preprocessing")

# The three scaled channels that feed the LSTM autoencoder and the Transformer.
MODEL_FEATURES: tuple[str, ...] = tuple(f"{col}_scaled" for col in C.SENSOR_COLUMNS)


@dataclass
class BaselineStats:
    """Per-sensor mean/SD learned from the 14-day baseline window."""

    temp_mean: float
    temp_std: float
    impedance_mean: float
    impedance_std: float
    hrv_mean: float
    hrv_std: float

    def as_db_dict(self, archetype: str) -> dict[str, float | str]:
        """Shape the stats for :func:`data.database.upsert_patient_baseline`."""
        return {
            "archetype": archetype,
            "baseline_temp": self.temp_mean,
            "baseline_temp_sd": self.temp_std,
            "baseline_impedance": self.impedance_mean,
            "baseline_impedance_sd": self.impedance_std,
            "baseline_hrv": self.hrv_mean,
            "baseline_hrv_sd": self.hrv_std,
        }


@dataclass
class BiosignalPreprocessor:
    """Clinical signal processing pipeline for ImmunoWatch sensor data.

    Preprocessing order matters clinically: artifact removal first, then
    normalization, then feature engineering. Running in the wrong order would
    propagate artifacts into derived features.

    Args:
        patient_id: Patient whose stream is being processed (controls scaler path).

    Attributes:
        baseline_stats: Per-sensor baseline statistics, populated after ``fit``.
        scaler: The fitted per-patient StandardScaler.
        engineered_features: Names of the interpretable derived feature columns.
    """

    patient_id: str
    baseline_stats: BaselineStats | None = None
    scaler: StandardScaler | None = field(default=None, repr=False)
    engineered_features: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ IO
    def load_csv(self) -> pd.DataFrame:
        """Load this patient's simulator CSV.

        Returns:
            DataFrame indexed by a continuous 1-minute timestamp.

        Raises:
            InsufficientDataError: If the CSV is missing or empty.
        """
        path = C.PATIENT_DATA_DIR / f"{self.patient_id}.csv"
        if not path.exists():
            raise InsufficientDataError(self.patient_id, have=0, need=1)
        frame = pd.read_csv(path, parse_dates=["timestamp"])
        if frame.empty:
            raise InsufficientDataError(self.patient_id, have=0, need=1)
        return frame.sort_values("timestamp").reset_index(drop=True)

    # ----------------------------------------------------- cleaning steps
    def _denoise_temperature(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply a zero-phase 4th-order Butterworth low-pass to temperature.

        Clinical note:
            ``filtfilt`` is zero-phase, so the smoothed fever onset is not shifted
            in time — critical, because a delayed onset estimate would erode the
            12-24h early-warning lead this whole system is built to provide.
        """
        nyquist = 0.5 * C.SAMPLE_RATE_HZ
        wn = C.TEMP_FILTER_CUTOFF_HZ / nyquist
        wn = float(np.clip(wn, 1e-4, 0.99))
        b, a = butter(C.TEMP_FILTER_ORDER, wn, btype="low")
        # filtfilt needs a gap-free series; interpolate transient NaNs first.
        series = frame["temp_c"].interpolate(limit_direction="both")
        padlen = 3 * max(len(a), len(b))
        if series.notna().sum() > padlen:
            frame["temp_c"] = filtfilt(b, a, series.to_numpy())
        else:  # pragma: no cover - tiny inputs only
            frame["temp_c"] = series.to_numpy()
        return frame

    def _fill_missing(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Forward-fill short gaps (<=5 min), then drop any remaining NaN rows."""
        cols = list(C.SENSOR_COLUMNS)
        frame[cols] = frame[cols].ffill(limit=C.MAX_FORWARD_FILL)
        before = len(frame)
        frame = frame.dropna(subset=cols).reset_index(drop=True)
        dropped = before - len(frame)
        if dropped:
            logger.debug("[%s] dropped %d rows with gaps > %d min", self.patient_id, dropped, C.MAX_FORWARD_FILL)
        return frame

    def _clip_hrv(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Validate RMSSD range and clip outliers at +/-3 sigma about its mean."""
        hrv = frame["hrv_rmssd_ms"]
        mu, sigma = float(hrv.mean()), float(hrv.std() or 1.0)
        lo = max(C.HRV_MIN_MS, mu - C.HRV_CLIP_SIGMA * sigma)
        hi = min(C.HRV_MAX_MS, mu + C.HRV_CLIP_SIGMA * sigma)
        frame["hrv_rmssd_ms"] = hrv.clip(lo, hi)
        return frame

    # --------------------------------------------------- baseline / scaler
    def _compute_baseline(self, frame: pd.DataFrame) -> BaselineStats:
        """Compute per-sensor baseline stats from the first 14 healthy days."""
        cutoff = frame["timestamp"].iloc[0] + pd.Timedelta(days=C.BASELINE_TRAINING_DAYS)
        window = frame[(frame["timestamp"] < cutoff) & (frame["event_label"] == "normal")]
        if len(window) < C.BASELINE_WINDOW_MINUTES:
            raise InsufficientDataError(self.patient_id, len(window), C.BASELINE_WINDOW_MINUTES)
        return BaselineStats(
            temp_mean=float(window["temp_c"].mean()),
            temp_std=float(window["temp_c"].std() or C.TEMP_NOISE_C),
            impedance_mean=float(window["impedance_ohm"].mean()),
            impedance_std=float(window["impedance_ohm"].std() or C.IMPEDANCE_NOISE_OHM),
            hrv_mean=float(window["hrv_rmssd_ms"].mean()),
            hrv_std=float(window["hrv_rmssd_ms"].std() or C.HRV_NOISE_MS),
        )

    def _fit_scaler(self, frame: pd.DataFrame) -> StandardScaler:
        """Fit a StandardScaler on the 14-day baseline window and persist it."""
        cutoff = frame["timestamp"].iloc[0] + pd.Timedelta(days=C.BASELINE_TRAINING_DAYS)
        window = frame[frame["timestamp"] < cutoff]
        scaler = StandardScaler().fit(window[list(C.SENSOR_COLUMNS)].to_numpy())
        out_dir = C.MODELS_DIR / self.patient_id
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "scaler.pkl", "wb") as fh:
            pickle.dump(scaler, fh)
        logger.info("[%s] fitted + saved per-patient scaler", self.patient_id)
        return scaler

    # ----------------------------------------------- feature engineering
    def _add_lag_features(self, frame: pd.DataFrame) -> list[str]:
        """Add t-5m..t-6h lag features for every sensor (18 columns)."""
        names: list[str] = []
        for col in C.SENSOR_COLUMNS:
            for lag in C.LAG_FEATURES_MINUTES:
                name = f"{col}_lag_{lag}m"
                frame[name] = frame[col].shift(lag)
                names.append(name)
        return names

    def _add_derived_features(self, frame: pd.DataFrame, stats: BaselineStats) -> list[str]:
        """Add the interpretable cross-sensor derived features."""
        names: list[str] = []

        # temp_delta_from_baseline: standardised deviation from the 14-day mean.
        frame["temp_delta_from_baseline"] = (
            frame["temp_c"] - stats.temp_mean
        ) / stats.temp_std
        names.append("temp_delta_from_baseline")

        # impedance_pct_change_2h: percent change vs. the 2-hour rolling mean.
        roll = frame["impedance_ohm"].rolling(C.IMPEDANCE_PCT_CHANGE_WINDOW_MIN, min_periods=1).mean()
        frame["impedance_pct_change_2h"] = (frame["impedance_ohm"] - roll) / roll * 100.0
        names.append("impedance_pct_change_2h")

        # hrv_trend_slope_1h: OLS slope of HRV over the last 60 readings.
        frame["hrv_trend_slope_1h"] = _rolling_slope(
            frame["hrv_rmssd_ms"].to_numpy(), C.HRV_TREND_WINDOW_MIN
        )
        names.append("hrv_trend_slope_1h")

        # rolling-z-scored impedance removes diurnal drift but keeps anomalies.
        imp = frame["impedance_ohm"]
        roll_mean = imp.rolling(C.ROLLING_ZSCORE_WINDOW_MIN, min_periods=1).mean()
        roll_std = imp.rolling(C.ROLLING_ZSCORE_WINDOW_MIN, min_periods=1).std().replace(0, np.nan)
        frame["impedance_rolling_z"] = ((imp - roll_mean) / roll_std).fillna(0.0)
        names.append("impedance_rolling_z")

        # cross_sensor_alarm: True when >=2 sensors deviate > 1 sigma from baseline.
        z_temp = (frame["temp_c"] - stats.temp_mean).abs() / stats.temp_std
        z_imp = (frame["impedance_ohm"] - stats.impedance_mean).abs() / stats.impedance_std
        z_hrv = (frame["hrv_rmssd_ms"] - stats.hrv_mean).abs() / stats.hrv_std
        deviating = (
            (z_temp > C.CROSS_SENSOR_SIGMA).astype(int)
            + (z_imp > C.CROSS_SENSOR_SIGMA).astype(int)
            + (z_hrv > C.CROSS_SENSOR_SIGMA).astype(int)
        )
        frame["cross_sensor_alarm"] = (deviating >= 2).astype(int)
        names.append("cross_sensor_alarm")
        return names

    # ------------------------------------------------------------- driver
    def fit_transform(self) -> tuple[pd.DataFrame, list[str]]:
        """Run the full pipeline for this patient.

        Returns:
            ``(frame, model_feature_names)`` where ``frame`` contains cleaned
            sensors, the scaled model channels, all engineered features, and the
            ``event_label``/``severity`` labels, and ``model_feature_names`` are
            the three scaled channels consumed by the deep models.

        Raises:
            InsufficientDataError: If the patient lacks a usable baseline window.
        """
        frame = self.load_csv()
        # 1) artifact removal -> 2) gap handling -> 3) HRV validation.
        frame = self._denoise_temperature(frame)
        frame = self._fill_missing(frame)
        frame = self._clip_hrv(frame)

        # 4) learn baseline + per-patient scaler from days 1-14.
        self.baseline_stats = self._compute_baseline(frame)
        self.scaler = self._fit_scaler(frame)

        # 5) scaled model channels (the deep-model sequence input).
        scaled = self.scaler.transform(frame[list(C.SENSOR_COLUMNS)].to_numpy())
        for i, name in enumerate(MODEL_FEATURES):
            frame[name] = scaled[:, i]

        # 6) interpretable derived + lag features (analysis / SHAP / notebook).
        derived = self._add_derived_features(frame, self.baseline_stats)
        lags = self._add_lag_features(frame)
        self.engineered_features = derived + lags

        # Lag features introduce NaNs in the first rows; fill for downstream use.
        frame[self.engineered_features] = frame[self.engineered_features].bfill().fillna(0.0)

        logger.info(
            "[%s] preprocessed %d rows | %d model channels | %d engineered features",
            self.patient_id,
            len(frame),
            len(MODEL_FEATURES),
            len(self.engineered_features),
        )
        return frame, list(MODEL_FEATURES)

    def transform_window(self, window: np.ndarray) -> np.ndarray:
        """Scale a raw (T, 3) sensor window with the fitted scaler (for inference).

        Args:
            window: Raw sensor values shaped ``(timesteps, 3)`` in the column
                order of :data:`constants.SENSOR_COLUMNS`.

        Returns:
            Scaled window of identical shape.

        Raises:
            ModelNotTrainedError: If no scaler has been fitted/loaded.
        """
        scaler = self._require_scaler()
        return scaler.transform(window)

    def _require_scaler(self) -> StandardScaler:
        from exceptions import ModelNotTrainedError

        if self.scaler is None:
            path = C.MODELS_DIR / self.patient_id / "scaler.pkl"
            if path.exists():
                with open(path, "rb") as fh:
                    self.scaler = pickle.load(fh)
            else:
                raise ModelNotTrainedError(f"scaler[{self.patient_id}]", str(path))
        return self.scaler


def make_windows(matrix: np.ndarray, window: int, stride: int = 1) -> np.ndarray:
    """Slice a (T, F) matrix into overlapping (N, window, F) windows.

    Args:
        matrix: 2-D array of shape (timesteps, features).
        window: Window length in timesteps.
        stride: Step between consecutive window starts.

    Returns:
        3-D array of shape (n_windows, window, features). Empty if T < window.
    """
    if matrix.shape[0] < window:
        return np.empty((0, window, matrix.shape[1]), dtype=matrix.dtype)
    starts = range(0, matrix.shape[0] - window + 1, stride)
    return np.stack([matrix[s : s + window] for s in starts]).astype(np.float32)


def _rolling_slope(values: np.ndarray, window: int) -> np.ndarray:
    """Vectorised OLS slope of ``values`` over a trailing window.

    Uses the closed-form slope with a fixed integer abscissa, computed as a
    convolution so it stays O(n) rather than O(n*window).

    Args:
        values: 1-D signal.
        window: Trailing window length.

    Returns:
        Per-sample slope; the first ``window-1`` samples are 0 (insufficient data).
    """
    n = len(values)
    out = np.zeros(n, dtype=float)
    if n < window:
        return out
    t = np.arange(window, dtype=float)
    t_centered = t - t.mean()
    s_xx = float((t_centered**2).sum())
    kernel = t_centered[::-1]  # convolution reverses the kernel
    num = np.convolve(values, kernel, mode="valid") / s_xx
    out[window - 1 :] = num
    return out


__all__ = [
    "BiosignalPreprocessor",
    "BaselineStats",
    "MODEL_FEATURES",
    "make_windows",
]
