"""Admin: artifact status, precompute trigger, and audit/job inspection.
All admin-gated."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import AuthContext, get_ranker, require_admin
from api.schemas import ArtifactStatusResponse, JobStatusResponse
from core.logging import get_logger
from db.base import get_session
from db.models import AuditLog
from db.repositories import JobRepository

log = get_logger("api.admin")
router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/artifacts", response_model=ArtifactStatusResponse)
async def artifacts(
    request: Request, _: AuthContext = Depends(require_admin)
) -> ArtifactStatusResponse:
    return ArtifactStatusResponse(**get_ranker(request).health())


@router.post("/precompute", status_code=202)
async def trigger_precompute(request: Request, _: AuthContext = Depends(require_admin)) -> dict:
    """Kick off the offline precompute in a worker thread, then hot-reload the
    ranker. Fire-and-forget; poll /admin/artifacts for the new built_at."""
    ranker = get_ranker(request)

    def _work():
        import precompute

        precompute.main(["--no-st", "--bm25-backend", "inverted"])
        ranker.load()

    async def _run():
        try:
            await asyncio.to_thread(_work)
            log.info("admin.precompute_done")
        except Exception:
            log.exception("admin.precompute_failed")

    asyncio.create_task(_run())
    return {"status": "accepted", "message": "precompute started; poll /admin/artifacts"}


@router.get("/jobs", response_model=list[JobStatusResponse])
async def recent_jobs(
    _: AuthContext = Depends(require_admin), session: AsyncSession = Depends(get_session)
) -> list[JobStatusResponse]:
    from api.routers.rank import _to_status

    jobs = await JobRepository(session).list_recent(100)
    return [_to_status(j) for j in jobs]


@router.get("/audit")
async def audit_log(
    limit: int = 100,
    _: AuthContext = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
) -> list[dict]:
    rows = (
        await session.execute(
            select(AuditLog).order_by(AuditLog.created_at.desc()).limit(min(limit, 500))
        )
    ).scalars()
    return [
        {
            "action": r.action,
            "actor_id": r.actor_id,
            "target": r.target,
            "ip": r.ip,
            "meta": r.meta,
            "created_at": r.created_at.isoformat(),
        }
        for r in rows
    ]
