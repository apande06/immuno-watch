"""FastAPI dependency providers — DB sessions and the shared inference engine.

Clinical purpose:
    Centralising these dependencies guarantees every request scores against the
    same loaded models and writes to the same audit database — there is exactly
    one source of truth for a patient's state.

Technical purpose:
    Async-session injection (one transactional session per request) and access to
    the process-wide :class:`InferenceEngine` held on ``app.state``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from data.database import async_session_factory
from inference.engine import InferenceEngine


async def get_session() -> AsyncIterator[AsyncSession]:
    """Yield a transactional async DB session, committing/rolling back at the end."""
    session = async_session_factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


def get_engine(request: Request) -> InferenceEngine:
    """Return the singleton inference engine initialised in the app lifespan."""
    return request.app.state.engine


__all__ = ["get_session", "get_engine"]
