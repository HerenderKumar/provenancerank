"""Ranking endpoints.

CQRS-ish split: POST /rank is the write path (kicks off a job, returns 202);
GET /rank/{id}/results is the read path (never recomputes, just reads the DB).
The SSE stream relays live progress for the UI's loop tracker.
"""

from __future__ import annotations

import csv
import io
import json

from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from api.deps import AuthContext, authenticate, enforce_rate_limit, get_jobs
from api.metrics import record_ranking
from api.schemas import JobCreatedResponse, JobStatusResponse, RankRequest, RankRow, ResultsResponse
from core.exceptions import NotFoundError
from db.base import get_session
from db.models import SUCCEEDED
from db.repositories import JobRepository

router = APIRouter(prefix="/rank", tags=["ranking"])


@router.post("", response_model=JobCreatedResponse, status_code=status.HTTP_202_ACCEPTED)
async def submit(
    body: RankRequest,
    request: Request,
    auth: AuthContext = Depends(enforce_rate_limit),
    session: AsyncSession = Depends(get_session),
) -> JobCreatedResponse:
    jobs = get_jobs(request)
    job_id = await jobs.submit(
        session,
        jd_text=body.jd_text,
        candidate_ids=body.candidate_ids,
        candidates=body.candidates,
        top_k=body.top_k,
        user_id=auth.subject,
    )
    base = str(request.url_for("get_status", job_id=job_id))
    return JobCreatedResponse(
        job_id=job_id, status="queued", stream_url=base + "/stream", results_url=base + "/results"
    )


@router.get("", response_model=list[JobStatusResponse])
async def list_jobs(
    _: AuthContext = Depends(authenticate), session: AsyncSession = Depends(get_session)
) -> list[JobStatusResponse]:
    jobs = await JobRepository(session).list_recent(50)
    return [_to_status(j) for j in jobs]


@router.get("/{job_id}", response_model=JobStatusResponse, name="get_status")
async def get_status(
    job_id: str,
    _: AuthContext = Depends(authenticate),
    session: AsyncSession = Depends(get_session),
) -> JobStatusResponse:
    job = await JobRepository(session).get(job_id)
    if not job:
        raise NotFoundError("job not found")
    return _to_status(job)


@router.get("/{job_id}/results", response_model=ResultsResponse)
async def get_results(
    job_id: str,
    _: AuthContext = Depends(authenticate),
    session: AsyncSession = Depends(get_session),
) -> ResultsResponse:
    repo = JobRepository(session)
    job = await repo.get(job_id)
    if not job:
        raise NotFoundError("job not found")
    rows = await repo.get_results(job_id)
    if job.status == SUCCEEDED and rows:
        record_ranking(
            job.source or "fast_path", job.cache_hit, job.elapsed_ms or 0.0, job.gate_stats or {}
        )
    return ResultsResponse(
        job_id=job_id,
        status=job.status,
        count=len(rows),
        results=[
            RankRow(candidate_id=r.candidate_id, rank=r.rank, score=r.score, reasoning=r.reasoning)
            for r in rows
        ],
    )


@router.get("/{job_id}/submission.csv")
async def download_csv(
    job_id: str,
    _: AuthContext = Depends(authenticate),
    session: AsyncSession = Depends(get_session),
) -> StreamingResponse:
    repo = JobRepository(session)
    job = await repo.get(job_id)
    if not job:
        raise NotFoundError("job not found")
    rows = await repo.get_results(job_id)
    buf = io.StringIO()
    w = csv.writer(buf, quoting=csv.QUOTE_MINIMAL)
    w.writerow(["candidate_id", "rank", "score", "reasoning"])
    for r in rows:
        w.writerow([r.candidate_id, r.rank, f"{r.score:.4f}", r.reasoning])
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="submission_{job_id}.csv"'},
    )


@router.get("/{job_id}/stream")
async def stream(
    job_id: str, request: Request, _: AuthContext = Depends(authenticate)
) -> EventSourceResponse:
    jobs = get_jobs(request)

    async def event_gen():
        async for event in jobs.subscribe(job_id):
            if await request.is_disconnected():
                break
            yield {"event": event.get("event", "message"), "data": json.dumps(event)}

    return EventSourceResponse(event_gen())


def _to_status(job) -> JobStatusResponse:
    return JobStatusResponse(
        job_id=job.id,
        status=job.status,
        source=job.source,
        cache_hit=job.cache_hit,
        candidates_count=job.candidates_count,
        top_k=job.top_k,
        elapsed_ms=job.elapsed_ms,
        gate_stats=job.gate_stats or {},
        coverage=job.coverage or {},
        error=job.error,
        created_at=job.created_at,
        finished_at=job.finished_at,
    )
