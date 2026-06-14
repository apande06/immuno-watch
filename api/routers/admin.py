"""Administrative API endpoints — retraining, simulation, metrics, health.

Clinical purpose:
    Operational controls a clinical-engineering team would use: re-fit a patient's
    personal baseline as their treatment changes their physiology, drive an
    end-to-end infection simulation for validation/demo, and inspect model
    performance and system health before trusting the system on a live patient.
"""

from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends

import constants as C
from api.dependencies import get_engine
from data.schemas import Alert
from exceptions import ModelNotTrainedError
from inference.engine import InferenceEngine
from ml.baseline import BaselineTrainer

router = APIRouter(prefix="/admin", tags=["admin"])


def _retrain_baseline(patient_id: str) -> None:
    """Background worker: retrain and persist a patient's baseline model."""
    BaselineTrainer(patient_id).train()


@router.post(
    "/patients/{patient_id}/retrain",
    status_code=202,
    summary="Trigger personal-baseline retraining",
)
async def retrain_patient(
    patient_id: str,
    background_tasks: BackgroundTasks,
    engine: InferenceEngine = Depends(get_engine),
) -> dict:
    """Kick off baseline retraining for a patient as a background job."""
    engine.get_baseline(patient_id)  # validates the patient exists (404 otherwise)
    background_tasks.add_task(_retrain_baseline, patient_id)
    return {
        "status": "accepted",
        "patient_id": patient_id,
        "job": "baseline_retrain",
        "detail": "Retraining started; reload the engine to pick up new weights.",
    }


@router.post(
    "/simulate/{patient_id}/infection",
    response_model=Optional[Alert],
    summary="Simulate an infection event in real time",
)
async def simulate_infection(
    patient_id: str, engine: InferenceEngine = Depends(get_engine)
) -> Optional[Alert]:
    """Inject a 60-minute infection cascade and return the first alert generated."""
    alerts = await engine.simulate_infection_event(patient_id)
    return alerts[0] if alerts else None


@router.get("/model/metrics", summary="Predictor evaluation metrics")
async def model_metrics() -> dict:
    """Return the saved predictor test metrics (AUC, F1, precision, recall, CM).

    Raises:
        ModelNotTrainedError: If metrics have not been generated yet.
    """
    path = C.MODELS_DIR / "predictor_metrics.json"
    if not path.exists():
        raise ModelNotTrainedError("predictor_metrics", str(path))
    return json.loads(path.read_text())


@router.get("/health", summary="System health and model load status")
async def admin_health(engine: InferenceEngine = Depends(get_engine)) -> dict:
    """Return per-component load status for the running system."""
    return {
        "status": "ok",
        "patients": engine.patients,
        "predictor_loaded": engine.predictor is not None,
        "explainer_loaded": engine.explainer is not None,
        "baseline_models_loaded": sorted(engine.baselines),
        "scalers_loaded": sorted(engine.scalers),
        "buffered_readings": {pid: len(buf) for pid, buf in engine.buffers.items()},
    }
