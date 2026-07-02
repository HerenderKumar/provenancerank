"""Async ingestion core, shared by the API (await directly) and the Celery
tasks (run via asyncio.run in the worker). Keeping the real logic here means the
two entry points can't drift.

Flow: fetch GitHub signals -> for each artifact, summarise -> write evidence to
Postgres -> upsert skills into the graph -> bump confidence. Idempotent on
content_hash so re-running a sync is a no-op, and graph writes are best-effort
(the relational DB is the source of truth; a graph outage doesn't lose data).
"""

from __future__ import annotations

import datetime as dt

from core.logging import get_logger
from db.base import session_scope
from db.repositories import DeveloperRepository, EvidenceRepository, SkillRepository
from graph.queries import GraphWriter, get_backend
from ingestion.github_client import DeveloperSignals, GitHubClient
from intelligence.confidence_scorer import EvidenceRecord, compute_confidence
from intelligence.skill_extractor import category_of
from intelligence.summariser import summarise_artifact

log = get_logger("ingestion.sync")


def _commit_content(c) -> str:
    return f"{c.message}\n(+{c.additions} -{c.deletions} in {c.repo})"


def _pr_content(p) -> str:
    return "\n".join([p.title, p.body or "", *p.reviews]).strip()


def _issue_content(i) -> str:
    return "\n".join([i.title, i.body or "", *i.comments]).strip()


def _iter_artifacts(sig: DeveloperSignals):
    """Yield (artifact_type, raw_content, content_hash, url, date)."""
    for c in sig.commits:
        yield "commit_diff", _commit_content(c), c.content_hash, c.url, c.authored_at
    for p in sig.pull_requests:
        yield "pr_review", _pr_content(p), p.content_hash, p.url, p.created_at
    for i in sig.issues:
        yield "issue_thread", _issue_content(i), i.content_hash, i.url, i.created_at


async def summarise_and_index(
    session,
    writer: GraphWriter,
    developer_id: str,
    artifact_type: str,
    raw_content: str,
    content_hash: str,
    source_url: str,
    artifact_date: dt.datetime | None,
) -> bool:
    """Index one artifact. Returns False if it was already indexed (idempotent)."""
    evidence_repo = EvidenceRepository(session)
    if await evidence_repo.exists(content_hash):
        return False

    summary = await summarise_artifact(artifact_type, raw_content)  # type: ignore[arg-type]
    artifact = await evidence_repo.create(
        developer_id=developer_id,
        source_type=artifact_type,
        source_url=source_url,
        content_hash=content_hash,
        raw_content=raw_content[:8000],
        summary=summary.what_was_done,
        complexity_score=summary.complexity,
        problem_category=summary.problem_category,
        production_signal=summary.production_signal,
        artifact_date=artifact_date,
    )

    skill_repo = SkillRepository(session)
    when = artifact_date or dt.datetime.now(dt.timezone.utc)
    for skill in summary.skills_demonstrated:
        contribution = compute_confidence(
            [EvidenceRecord(artifact_type, skill, when, summary.complexity)], skill
        )
        rec = await skill_repo.upsert(developer_id, skill, category_of(skill), contribution, when)
        await skill_repo.link_evidence(rec.id, artifact.id)
        try:
            await writer.upsert_skill_with_evidence(
                developer_id=developer_id,
                skill_name=skill,
                artifact_id=artifact.id,
                source_type=artifact_type,
                complexity=summary.complexity,
                confidence=contribution,
                artifact_date=artifact_date,
                summary=summary.what_was_done,
                url=source_url,
                production_signal=summary.production_signal,
                skill_category=category_of(skill),
            )
        except Exception as exc:  # graph is best-effort; evidence already persisted
            log.warning("sync.graph_write_skipped", reason=str(exc)[:120])
    return True


async def sync_developer(
    developer_id: str, github_username: str, since: dt.datetime | None = None
) -> dict:
    backend = await get_backend()
    writer = GraphWriter(backend)
    await writer.upsert_developer(developer_id, github_username=github_username)

    async with session_scope() as s:
        await DeveloperRepository(s).set_status(developer_id, "syncing")

    client = GitHubClient()
    signals = await client.fetch_developer_signals(github_username, since=since)

    indexed = 0
    skipped = 0
    async with session_scope() as s:
        for artifact_type, raw, chash, url, date in _iter_artifacts(signals):
            if not raw.strip():
                continue
            did = await summarise_and_index(
                s, writer, developer_id, artifact_type, raw, chash, url, date
            )
            indexed += int(did)
            skipped += int(not did)

    async with session_scope() as s:
        await DeveloperRepository(s).mark_synced(developer_id)

    log.info(
        "sync.complete",
        developer=developer_id,
        indexed=indexed,
        skipped=skipped,
        fetched=signals.total_artifacts(),
    )
    return {"indexed": indexed, "skipped": skipped, "fetched": signals.total_artifacts()}
