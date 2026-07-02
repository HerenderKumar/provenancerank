"""Developer endpoints — connect a GitHub account, watch the sync, read the
evidence graph.

connect-github kicks the ingestion off in the background (the API never blocks
on a GitHub crawl). We use an asyncio task here rather than Celery .delay() to
avoid nesting event loops in eager mode; in a real deployment you'd point this
at the Celery task instead — same sync_developer core either way.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import AuthContext, authenticate
from api.schemas import ConnectGithubRequest, ConnectGithubResponse, SyncStatusResponse
from core.exceptions import NotFoundError
from core.logging import get_logger
from db.base import get_session, session_scope
from db.repositories import DeveloperRepository, EvidenceRepository, SkillRepository
from graph.queries import GraphQuerier, get_backend
from ingestion.sync import sync_developer

log = get_logger("api.developers")
router = APIRouter(prefix="/developers", tags=["developers"])

_bg: set[asyncio.Task] = set()


def _spawn(coro) -> None:
    task = asyncio.create_task(coro)
    _bg.add(task)
    task.add_done_callback(_bg.discard)


async def _safe_sync(developer_id: str, username: str) -> None:
    try:
        await sync_developer(developer_id, username)
    except Exception as exc:
        log.warning("developer.sync_failed", developer=developer_id, reason=str(exc)[:160])
        async with session_scope() as s:
            await DeveloperRepository(s).set_status(developer_id, "error")


@router.post("/connect-github", response_model=ConnectGithubResponse, status_code=202)
async def connect_github(
    body: ConnectGithubRequest,
    _: AuthContext = Depends(authenticate),
    session: AsyncSession = Depends(get_session),
) -> ConnectGithubResponse:
    dev = await DeveloperRepository(session).upsert(body.github_username, body.display_name)
    await session.commit()
    _spawn(_safe_sync(dev.id, body.github_username))
    return ConnectGithubResponse(
        developer_id=dev.id, github_username=body.github_username, status="syncing"
    )


@router.post("/{developer_id}/sync", response_model=ConnectGithubResponse, status_code=202)
async def resync(
    developer_id: str,
    _: AuthContext = Depends(authenticate),
    session: AsyncSession = Depends(get_session),
) -> ConnectGithubResponse:
    dev = await DeveloperRepository(session).get_by_id(developer_id)
    if not dev or not dev.github_username:
        raise NotFoundError("developer not found or has no GitHub username")
    _spawn(_safe_sync(dev.id, dev.github_username))
    return ConnectGithubResponse(
        developer_id=dev.id, github_username=dev.github_username, status="syncing"
    )


@router.get("/{developer_id}/sync-status", response_model=SyncStatusResponse)
async def sync_status(
    developer_id: str,
    _: AuthContext = Depends(authenticate),
    session: AsyncSession = Depends(get_session),
) -> SyncStatusResponse:
    dev = await DeveloperRepository(session).get_by_id(developer_id)
    if not dev:
        raise NotFoundError("developer not found")
    artifact_count = await EvidenceRepository(session).count_for(developer_id)
    skills = await SkillRepository(session).list_for(developer_id)
    return SyncStatusResponse(
        developer_id=dev.id,
        github_username=dev.github_username,
        status=dev.sync_status,
        last_synced_at=dev.last_synced_at,
        artifact_count=artifact_count,
        skill_count=len(skills),
    )


@router.get("/{developer_id}/evidence-graph")
async def evidence_graph(
    developer_id: str,
    _: AuthContext = Depends(authenticate),
    session: AsyncSession = Depends(get_session),
) -> dict:
    dev = await DeveloperRepository(session).get_by_id(developer_id)
    if not dev:
        raise NotFoundError("developer not found")
    querier = GraphQuerier(await get_backend())
    subgraph = await querier.developer_subgraph(developer_id)
    artifacts = await EvidenceRepository(session).list_for(developer_id, limit=100)
    return {
        "developer": {
            "id": dev.id,
            "github_username": dev.github_username,
            "display_name": dev.display_name,
            "sync_status": dev.sync_status,
        },
        "graph": subgraph,
        "artifacts": [
            {
                "id": a.id,
                "source_type": a.source_type,
                "summary": a.summary,
                "complexity": a.complexity_score,
                "production_signal": a.production_signal,
                "problem_category": a.problem_category,
                "url": a.source_url,
                "content_hash": a.content_hash,
                "date": a.artifact_date.isoformat() if a.artifact_date else None,
            }
            for a in artifacts
        ],
    }
