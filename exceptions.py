"""Custom exception hierarchy for ImmunoWatch.

Clinical purpose:
    In a clinical-monitoring system, failure modes must be explicit and
    distinguishable. "We have never trained a baseline for this patient" is a
    fundamentally different operational state from "this patient does not exist"
    or "we lack enough data to score safely", and each demands a different
    response from the care team and the API layer.

Technical purpose:
    A small, well-typed exception hierarchy lets the API translate domain errors
    into precise HTTP status codes (404 vs. 409 vs. 422) and lets callers catch
    the specific condition they can recover from.
"""

from __future__ import annotations


class ImmunoWatchError(Exception):
    """Base class for all ImmunoWatch domain errors."""


class PatientNotFoundError(ImmunoWatchError):
    """Raised when a patient_id has no records in the system.

    Args:
        patient_id: The identifier that could not be located.
    """

    def __init__(self, patient_id: str) -> None:
        self.patient_id = patient_id
        super().__init__(f"No patient found with id '{patient_id}'.")


class InsufficientDataError(ImmunoWatchError):
    """Raised when there are too few readings to run a model safely.

    Clinical note:
        Scoring on an incomplete window would produce an unreliable risk estimate.
        For an immunocompromised patient, a *false negative* is potentially fatal,
        so we fail loudly rather than emit a low-confidence score.

    Args:
        patient_id: The patient whose buffer was too short.
        have: Number of readings available.
        need: Number of readings required.
    """

    def __init__(self, patient_id: str, have: int, need: int) -> None:
        self.patient_id = patient_id
        self.have = have
        self.need = need
        super().__init__(
            f"Patient '{patient_id}' has {have} readings but {need} are required."
        )


class ModelNotTrainedError(ImmunoWatchError):
    """Raised when an inference path needs a model artifact that is missing.

    Args:
        artifact: Human-readable name of the missing model/artifact.
        path: Filesystem path that was expected to contain the artifact.
    """

    def __init__(self, artifact: str, path: str | None = None) -> None:
        self.artifact = artifact
        self.path = path
        suffix = f" (expected at '{path}')" if path else ""
        super().__init__(f"Required model artifact '{artifact}' is not available{suffix}.")


__all__ = [
    "ImmunoWatchError",
    "PatientNotFoundError",
    "InsufficientDataError",
    "ModelNotTrainedError",
]
