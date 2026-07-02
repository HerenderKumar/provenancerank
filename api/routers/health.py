"""Liveness + readiness. Liveness is "process is up"; readiness checks the
things a request actually needs (ranker loaded, DB reachable, cache reachable)
and 503s if any are down, so a load balancer won't route to a cold instance.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request, Response
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_cache, get_ranker
from api.schemas import HealthResponse, ReadinessResponse
from core.config import get_settings
from db.base import get_session

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    s = get_settings()
    return HealthResponse(
        status="ok", service=s.service_name, version=s.service_version, environment=s.environment
    )


@router.get("/health/live")
async def live() -> dict:
    return {"status": "alive"}


@router.get("/health/ready", response_model=ReadinessResponse)
async def ready(
    request: Request, response: Response, session: AsyncSession = Depends(get_session)
) -> ReadinessResponse:
    ranker = get_ranker(request)
    cache = get_cache(request)

    db_ok = True
    try:
        await session.execute(text("SELECT 1"))
    except Exception:
        db_ok = False

    try:
        cache_ok = await cache.ping()
    except Exception:
        cache_ok = False

    checks = {"ranker": ranker.ready, "database": db_ok, "cache": cache_ok}
    ok = ranker.ready and db_ok
    if not ok:
        response.status_code = 503
    return ReadinessResponse(
        status="ready" if ok else "not_ready",
        ranker_ready=ranker.ready,
        candidates=len(ranker.matrix) if ranker.matrix is not None else 0,
        database=db_ok,
        cache=getattr(cache, "backend", "unknown"),
        checks=checks,
    )
