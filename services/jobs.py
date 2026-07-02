"""Job orchestration.

A ranking request returns immediately with a job id. Instead of spawning the
work inline, submit() enqueues a message; a QueueWorker pulls it and runs the
ranking in a worker thread (CPU work off the event loop). That decoupling is the
whole point of the queue: at scale the worker moves to its own fleet, unchanged.

Results are cached read-through by request hash, persisted to the DB, and
progress is published to in-memory channels the SSE endpoint subscribes to. The
consumer is idempotent — a duplicate delivery for an already-finished job is a
no-op — so at-least-once delivery is safe.
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import asdict

from core.logging import get_logger
from db.base import session_scope
from db.models import FAILED, SUCCEEDED
from db.repositories import AuditRepository, JobRepository
from services.cache import Cache
from services.queue import InMemoryQueue, Message, QueueWorker
from services.ranker import RankerService, RankResult

log = get_logger("services.jobs")


def _request_hash(jd_hash: str, candidate_ids: list[str] | None, top_k: int) -> str:
    h = hashlib.sha256()
    h.update(jd_hash.encode())
    h.update(str(top_k).encode())
    if candidate_ids:
        for cid in sorted(candidate_ids):
            h.update(cid.encode())
    return h.hexdigest()[:24]


class JobManager:
    def __init__(
        self,
        ranker: RankerService,
        cache: Cache,
        queue: InMemoryQueue | None = None,
        max_concurrency: int = 2,
    ):
        self.ranker = ranker
        self.cache = cache
        self.queue = queue or InMemoryQueue()
        self.worker = QueueWorker(self.queue, self._handle, concurrency=max_concurrency)
        self._channels: dict[str, list[asyncio.Queue]] = {}

    async def start(self) -> None:
        self.worker.start()

    async def stop(self) -> None:
        await self.worker.stop()

    # progress pub/sub the SSE endpoint subscribes to
    def _publish(self, job_id: str, event: dict) -> None:
        for q in self._channels.get(job_id, []):
            q.put_nowait(event)

    async def subscribe(self, job_id: str):
        q: asyncio.Queue = asyncio.Queue()
        self._channels.setdefault(job_id, []).append(q)
        try:
            while True:
                event = await q.get()
                yield event
                if event.get("event") in ("completed", "failed"):
                    break
        finally:
            self._channels.get(job_id, []).remove(q)
            if not self._channels.get(job_id):
                self._channels.pop(job_id, None)

    async def submit(
        self,
        session,
        *,
        jd_text: str | None,
        candidate_ids: list[str] | None,
        candidates: list[dict] | None,
        top_k: int,
        user_id: str | None,
    ) -> str:
        """Create the job row on the request's session, commit it (releasing the
        request's write lock), then enqueue a message for the worker."""
        jd_for_hash = jd_text or self.ranker.default_jd or ""
        jd_hash = hashlib.sha256(jd_for_hash.encode()).hexdigest()[:16]
        count = (
            len(candidates)
            if candidates is not None
            else (len(candidate_ids) if candidate_ids else self.ranker_count())
        )
        job = await JobRepository(session).create(
            jd_hash=jd_hash,
            candidates_count=count,
            top_k=top_k,
            created_by=user_id,
            params={"custom_jd": bool(jd_text), "live": candidates is not None},
        )
        job_id = job.id
        await session.commit()

        # interactive sandbox samples jump the queue ahead of full-pool jobs
        priority = 3 if candidates is not None else 5
        await self.queue.enqueue(
            {
                "job_id": job_id,
                "jd_text": jd_text,
                "candidate_ids": candidate_ids,
                "candidates": candidates,
                "top_k": top_k,
                "user_id": user_id,
                "jd_hash": jd_hash,
            },
            priority=priority,
        )
        return job_id

    def ranker_count(self) -> int:
        return len(self.ranker.matrix) if self.ranker.matrix is not None else 0

    async def _handle(self, msg: Message) -> None:
        p = msg.payload
        job_id = p["job_id"]
        # idempotent consumer: skip if this job already finished (duplicate delivery)
        async with session_scope() as s:
            job = await JobRepository(s).get(job_id)
            if job and job.status in (SUCCEEDED, FAILED):
                log.info("job.duplicate_skipped", job_id=job_id)
                return
        try:
            await self._execute(
                job_id,
                p["jd_text"],
                p["candidate_ids"],
                p["candidates"],
                p["top_k"],
                p["user_id"],
                p["jd_hash"],
            )
        except Exception as exc:
            # last attempt -> give up and mark the job failed; otherwise re-raise
            # so the worker requeues it (transient errors get another shot).
            if msg.attempts + 1 >= msg.max_attempts:
                log.exception("job.failed_final", job_id=job_id)
                await self.mark_failed(job_id, str(exc))
                return
            raise

    async def _execute(
        self, job_id, jd_text, candidate_ids, candidates, top_k, user_id, jd_hash
    ) -> None:
        self._publish(job_id, {"event": "status", "status": "running", "job_id": job_id})
        async with session_scope() as s:
            await JobRepository(s).mark_running(job_id)

        cache_key = f"rank:{_request_hash(jd_hash, candidate_ids, top_k)}"
        cache_hit = False
        cached = await self.cache.get(cache_key) if candidates is None else None
        if cached:
            result = RankResult(**cached)
            result.cache_hit = cache_hit = True
            self._publish(job_id, {"event": "progress", "stage": "cache_hit"})
        else:
            self._publish(job_id, {"event": "progress", "stage": "scoring"})
            result = await asyncio.to_thread(
                self.ranker.rank, jd_text, candidate_ids, candidates, top_k
            )
            if candidates is None:
                await self.cache.set(cache_key, asdict(result), ttl=self._cache_ttl())

        await self._persist(job_id, result, user_id)
        self._publish(
            job_id,
            {
                "event": "completed",
                "job_id": job_id,
                "summary": {
                    "elapsed_ms": result.elapsed_ms,
                    "source": result.source,
                    "cache_hit": cache_hit,
                    "top": result.rows[:3],
                },
            },
        )

    async def _persist(self, job_id: str, result: RankResult, user_id: str | None) -> None:
        async with session_scope() as s:
            repo = JobRepository(s)
            await repo.mark_done(
                job_id,
                status=SUCCEEDED,
                elapsed_ms=result.elapsed_ms,
                source=result.source,
                cache_hit=result.cache_hit,
                gate_stats=result.gate_stats,
                coverage=result.coverage,
            )
            await repo.save_results(
                job_id,
                [
                    {
                        "candidate_id": r["candidate_id"],
                        "rank": int(r["rank"]),
                        "score": float(r["score"]),
                        "reasoning": r["reasoning"],
                    }
                    for r in result.rows
                ],
            )
            await AuditRepository(s).log(
                "job.completed",
                actor_id=user_id,
                target=job_id,
                meta={"source": result.source, "cache_hit": result.cache_hit},
            )

    async def mark_failed(self, job_id: str, error: str) -> None:
        async with session_scope() as s:
            await JobRepository(s).mark_done(
                job_id,
                status=FAILED,
                elapsed_ms=0.0,
                source="error",
                cache_hit=False,
                gate_stats={},
                coverage={},
                error=error[:500],
            )
        self._publish(job_id, {"event": "failed", "job_id": job_id, "error": error[:200]})

    @staticmethod
    def _cache_ttl() -> int:
        from core.config import get_settings

        return get_settings().cache_ttl_seconds

    def stats(self) -> dict:
        return self.queue.stats()
