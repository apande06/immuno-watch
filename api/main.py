"""ImmunoWatch FastAPI application — the clinical monitoring service.

Clinical purpose:
    This is the server a hospital's monitoring dashboard and a patient's phone app
    both talk to. On startup it loads every patient's personal baseline model and
    the shared infection-risk predictor, then serves status, trends, alerts, and a
    live ingestion path. Domain errors are returned as RFC 7807 Problem Details so
    client apps can react precisely (a missing patient is not a server fault).

Technical purpose:
    Wires the lifespan-managed :class:`InferenceEngine` and database, CORS for the
    dashboard, the three routers, structured exception handlers, and a health
    probe.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import constants as C
from api.routers import admin, alerts, patients
from data.database import create_db_and_tables
from exceptions import (
    ImmunoWatchError,
    InsufficientDataError,
    ModelNotTrainedError,
    PatientNotFoundError,
)
from inference.engine import InferenceEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("immunowatch.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise the database and inference engine on startup; clean up on shutdown."""
    logger.info("Starting ImmunoWatch API...")
    await create_db_and_tables()
    app.state.engine = InferenceEngine()
    logger.info("ImmunoWatch API ready")
    yield
    logger.info("Shutting down ImmunoWatch API")
    app.state.engine = None


app = FastAPI(
    title="ImmunoWatch API",
    description="AI-powered continuous health monitoring for immunocompromised patients",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(C.CORS_ORIGINS),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(patients.router)
app.include_router(alerts.router)
app.include_router(admin.router)


# ---------------------------------------------------------------------------
# RFC 7807 Problem Details exception handlers
# ---------------------------------------------------------------------------
def _problem(status: int, title: str, detail: str, request: Request, type_: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        media_type="application/problem+json",
        content={
            "type": f"https://immunowatch.health/problems/{type_}",
            "title": title,
            "status": status,
            "detail": detail,
            "instance": str(request.url.path),
        },
    )


@app.exception_handler(PatientNotFoundError)
async def _patient_not_found(request: Request, exc: PatientNotFoundError) -> JSONResponse:
    return _problem(404, "Patient not found", str(exc), request, "patient-not-found")


@app.exception_handler(InsufficientDataError)
async def _insufficient_data(request: Request, exc: InsufficientDataError) -> JSONResponse:
    return _problem(409, "Insufficient data", str(exc), request, "insufficient-data")


@app.exception_handler(ModelNotTrainedError)
async def _model_not_trained(request: Request, exc: ModelNotTrainedError) -> JSONResponse:
    return _problem(503, "Model not trained", str(exc), request, "model-not-trained")


@app.exception_handler(ImmunoWatchError)
async def _domain_error(request: Request, exc: ImmunoWatchError) -> JSONResponse:
    return _problem(400, "ImmunoWatch error", str(exc), request, "domain-error")


@app.get("/health", tags=["system"], summary="Liveness probe")
async def health(request: Request) -> dict:
    """Return basic liveness plus whether the engine finished loading."""
    engine = getattr(request.app.state, "engine", None)
    return {
        "status": "ok",
        "service": "immunowatch-api",
        "version": "1.0.0",
        "engine_ready": engine is not None,
    }
