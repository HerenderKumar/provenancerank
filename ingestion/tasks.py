"""Celery task definitions.

In prod (real broker + workers) sync_developer_github fans out one
summarise_and_index task per artifact onto the intelligence queue - 200 commits
become 200 independent jobs processed concurrently, and a GitHub stall on the
ingestion queue can't block summarisation. In eager mode (dev/tests) there's no
broker, so we run the whole thing inline via the shared async core instead of
nesting event loops.

Failures retry up to max_retries; a task that's still failing is pushed to a
Redis dead-letter list that scan_dead_letters sweeps hourly - no silent drops.
"""

from __future__ import annotations

import asyncio
import datetime as dt

from celery import Task

from core.celery_app import celery_app
from core.config import get_settings
from core.logging import get_logger
from db.base import session_scope
from db.repositories import DeveloperRepository
from graph.queries import GraphWriter, get_backend
from ingestion.github_client import GitHubClient
from ingestion.sync import _iter_artifacts, summarise_and_index, sync_developer

log = get_logger("ingestion.tasks")
_DLQ_KEY = "pr:celery:dlq"


def _run(coro):
    return asyncio.run(coro)


def _push_dead_letter(name: str, args: tuple, error: str) -> None:
    try:
        import json

        import redis

        r = redis.from_url(get_settings().celery_broker_url)
        r.rpush(_DLQ_KEY, json.dumps({"task": name, "args": args, "error": error[:300]}))
    except Exception:
        log.error("dlq.push_failed", task=name)


class DeadLetterTask(Task):
    """On final failure, drop the message into a dead-letter list."""

    def on_failure(self, exc, task_id, args, kwargs, einfo):  # noqa: ANN001
        if self.request.retries >= (self.max_retries or 0):
            _push_dead_letter(self.name, args, str(exc))
            log.error("task.dead_lettered", task=self.name, error=str(exc)[:200])


@celery_app.task(
    base=DeadLetterTask,
    bind=True,
    name="ingestion.tasks.sync_developer_github",
    queue="ingestion",
    max_retries=3,
    autoretry_for=(Exception,),
    retry_backoff=True,
)
def sync_developer_github(self, developer_id: str, github_username: str) -> dict:
    if get_settings().celery_task_always_eager:
        # dev/test: inline, no loop-nesting from fan-out
        return _run(sync_developer(developer_id, github_username))
    return _run(_fanout(developer_id, github_username))


async def _fanout(developer_id: str, github_username: str) -> dict:
    async with session_scope() as s:
        await DeveloperRepository(s).set_status(developer_id, "syncing")
    signals = await GitHubClient().fetch_developer_signals(github_username)
    enqueued = 0
    for atype, raw, chash, url, date in _iter_artifacts(signals):
        if not raw.strip():
            continue
        summarise_and_index_task.delay(
            developer_id, atype, raw, chash, url, date.isoformat() if date else None
        )
        enqueued += 1
    async with session_scope() as s:
        await DeveloperRepository(s).mark_synced(developer_id)
    return {"enqueued": enqueued, "fetched": signals.total_artifacts()}


@celery_app.task(
    base=DeadLetterTask,
    bind=True,
    name="ingestion.tasks.summarise_and_index",
    queue="intelligence",
    max_retries=3,
    autoretry_for=(Exception,),
    retry_backoff=True,
)
def summarise_and_index_task(
    self,
    developer_id: str,
    artifact_type: str,
    raw_content: str,
    content_hash: str,
    source_url: str,
    artifact_date_iso: str | None,
) -> bool:
    async def run():
        writer = GraphWriter(await get_backend())
        date = dt.datetime.fromisoformat(artifact_date_iso) if artifact_date_iso else None
        async with session_scope() as s:
            return await summarise_and_index(
                s, writer, developer_id, artifact_type, raw_content, content_hash, source_url, date
            )

    return _run(run())


@celery_app.task(name="ingestion.tasks.sync_all_developers", queue="ingestion")
def sync_all_developers() -> int:
    async def run():
        async with session_scope() as s:
            devs = await DeveloperRepository(s).list_syncable()
            return [(d.id, d.github_username) for d in devs]

    targets = _run(run())
    for did, uname in targets:
        sync_developer_github.delay(did, uname)
    log.info("beat.sync_all_enqueued", developers=len(targets))
    return len(targets)


@celery_app.task(name="ingestion.tasks.scan_dead_letters", queue="ingestion")
def scan_dead_letters() -> int:
    try:
        import redis

        r = redis.from_url(get_settings().celery_broker_url)
        depth = r.llen(_DLQ_KEY)
        if depth:
            log.warning("dlq.alert", depth=depth, sample=r.lrange(_DLQ_KEY, 0, 4))
        return int(depth)
    except Exception as exc:
        log.warning("dlq.scan_skipped", reason=str(exc)[:120])
        return 0
