"""Repository layer — all DB access goes through here, so routers/services never
hand-write SQL and the query surface stays in one place.
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import (
    ApiKey,
    AuditLog,
    Developer,
    EvidenceArtifact,
    JobRun,
    RankingResult,
    SkillEvidenceLink,
    SkillRecord,
    User,
)


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class UserRepository:
    def __init__(self, session: AsyncSession):
        self.s = session

    async def get_by_email(self, email: str) -> User | None:
        return (
            await self.s.execute(select(User).where(User.email == email.lower()))
        ).scalar_one_or_none()

    async def get_by_id(self, user_id: str) -> User | None:
        return await self.s.get(User, user_id)

    async def create(self, email: str, hashed_password: str, role: str) -> User:
        user = User(email=email.lower(), hashed_password=hashed_password, role=role)
        self.s.add(user)
        await self.s.flush()
        return user

    async def count(self) -> int:
        return (await self.s.execute(select(func.count(User.id)))).scalar_one()


class ApiKeyRepository:
    def __init__(self, session: AsyncSession):
        self.s = session

    async def get_active_by_hash(self, key_hash: str) -> ApiKey | None:
        row = (
            await self.s.execute(
                select(ApiKey).where(ApiKey.key_hash == key_hash, ApiKey.is_active.is_(True))
            )
        ).scalar_one_or_none()
        if row and row.expires_at and row.expires_at < _now():
            return None
        return row

    async def touch(self, key: ApiKey) -> None:
        key.last_used_at = _now()
        await self.s.flush()

    async def create(
        self, owner_id: str, key_hash: str, name: str, last4: str, scopes: str
    ) -> ApiKey:
        key = ApiKey(owner_id=owner_id, key_hash=key_hash, name=name, last4=last4, scopes=scopes)
        self.s.add(key)
        await self.s.flush()
        return key

    async def list_for_owner(self, owner_id: str) -> list[ApiKey]:
        return list(
            (
                await self.s.execute(
                    select(ApiKey)
                    .where(ApiKey.owner_id == owner_id)
                    .order_by(ApiKey.created_at.desc())
                )
            ).scalars()
        )

    async def revoke(self, key_id: str, owner_id: str) -> bool:
        res = await self.s.execute(
            update(ApiKey)
            .where(ApiKey.id == key_id, ApiKey.owner_id == owner_id)
            .values(is_active=False)
        )
        return res.rowcount > 0


class JobRepository:
    def __init__(self, session: AsyncSession):
        self.s = session

    async def create(
        self, jd_hash: str, candidates_count: int, top_k: int, created_by: str | None, params: dict
    ) -> JobRun:
        job = JobRun(
            jd_hash=jd_hash,
            candidates_count=candidates_count,
            top_k=top_k,
            created_by=created_by,
            params=params,
        )
        self.s.add(job)
        await self.s.flush()
        return job

    async def get(self, job_id: str) -> JobRun | None:
        return await self.s.get(JobRun, job_id)

    async def mark_running(self, job_id: str) -> None:
        await self.s.execute(
            update(JobRun).where(JobRun.id == job_id).values(status="running", started_at=_now())
        )

    async def mark_done(
        self,
        job_id: str,
        *,
        status: str,
        elapsed_ms: float,
        source: str,
        cache_hit: bool,
        gate_stats: dict,
        coverage: dict,
        error: str | None = None,
    ) -> None:
        await self.s.execute(
            update(JobRun)
            .where(JobRun.id == job_id)
            .values(
                status=status,
                finished_at=_now(),
                elapsed_ms=elapsed_ms,
                source=source,
                cache_hit=cache_hit,
                gate_stats=gate_stats,
                coverage=coverage,
                error=error,
            )
        )

    async def save_results(self, job_id: str, rows: list[dict]) -> None:
        self.s.add_all([RankingResult(job_id=job_id, **r) for r in rows])
        await self.s.flush()

    async def get_results(self, job_id: str) -> list[RankingResult]:
        return list(
            (
                await self.s.execute(
                    select(RankingResult)
                    .where(RankingResult.job_id == job_id)
                    .order_by(RankingResult.rank)
                )
            ).scalars()
        )

    async def list_recent(self, limit: int = 50) -> list[JobRun]:
        return list(
            (
                await self.s.execute(select(JobRun).order_by(JobRun.created_at.desc()).limit(limit))
            ).scalars()
        )


class AuditRepository:
    def __init__(self, session: AsyncSession):
        self.s = session

    async def log(
        self,
        action: str,
        *,
        actor_id: str | None = None,
        target: str | None = None,
        ip: str | None = None,
        meta: dict | None = None,
    ) -> None:
        self.s.add(
            AuditLog(action=action, actor_id=actor_id, target=target, ip=ip, meta=meta or {})
        )
        await self.s.flush()


# ---------------------------------------------------------------------------
# Live ingestion layer
# ---------------------------------------------------------------------------


class DeveloperRepository:
    def __init__(self, session: AsyncSession):
        self.s = session

    async def get_by_id(self, dev_id: str) -> Developer | None:
        return await self.s.get(Developer, dev_id)

    async def get_by_username(self, username: str) -> Developer | None:
        return (
            await self.s.execute(select(Developer).where(Developer.github_username == username))
        ).scalar_one_or_none()

    async def upsert(
        self, username: str, display_name: str | None = None, avatar_url: str | None = None
    ) -> Developer:
        dev = await self.get_by_username(username)
        if dev is None:
            dev = Developer(
                github_username=username, display_name=display_name, avatar_url=avatar_url
            )
            self.s.add(dev)
        else:
            dev.display_name = display_name or dev.display_name
            dev.avatar_url = avatar_url or dev.avatar_url
        await self.s.flush()
        return dev

    async def set_status(self, dev_id: str, status: str) -> None:
        await self.s.execute(
            update(Developer).where(Developer.id == dev_id).values(sync_status=status)
        )

    async def mark_synced(self, dev_id: str) -> None:
        await self.s.execute(
            update(Developer)
            .where(Developer.id == dev_id)
            .values(sync_status="done", last_synced_at=_now())
        )

    async def list_syncable(self) -> list[Developer]:
        return list(
            (
                await self.s.execute(
                    select(Developer).where(
                        Developer.github_username.is_not(None),
                        Developer.sync_status != "syncing",
                    )
                )
            ).scalars()
        )


class EvidenceRepository:
    def __init__(self, session: AsyncSession):
        self.s = session

    async def exists(self, content_hash: str) -> bool:
        return (
            await self.s.execute(
                select(EvidenceArtifact.id).where(EvidenceArtifact.content_hash == content_hash)
            )
        ).first() is not None

    async def create(self, **fields) -> EvidenceArtifact:
        art = EvidenceArtifact(**fields)
        self.s.add(art)
        await self.s.flush()
        return art

    async def count_for(self, dev_id: str) -> int:
        return (
            await self.s.execute(
                select(func.count(EvidenceArtifact.id)).where(
                    EvidenceArtifact.developer_id == dev_id
                )
            )
        ).scalar_one()

    async def list_for(self, dev_id: str, limit: int = 100) -> list[EvidenceArtifact]:
        return list(
            (
                await self.s.execute(
                    select(EvidenceArtifact)
                    .where(EvidenceArtifact.developer_id == dev_id)
                    .order_by(EvidenceArtifact.artifact_date.desc())
                    .limit(limit)
                )
            ).scalars()
        )


class SkillRepository:
    def __init__(self, session: AsyncSession):
        self.s = session

    async def upsert(
        self,
        developer_id: str,
        skill_name: str,
        category: str,
        confidence: float,
        artifact_date: dt.datetime | None,
    ) -> SkillRecord:
        rec = (
            await self.s.execute(
                select(SkillRecord).where(
                    SkillRecord.developer_id == developer_id,
                    SkillRecord.skill_name == skill_name,
                )
            )
        ).scalar_one_or_none()
        # `confidence` is this artifact's contribution; it accumulates (clamped).
        if rec is None:
            rec = SkillRecord(
                developer_id=developer_id,
                skill_name=skill_name,
                skill_category=category,
                confidence_score=min(confidence, 1.0),
                evidence_count=1,
                last_seen_at=artifact_date,
            )
            self.s.add(rec)
        else:
            rec.confidence_score = min(rec.confidence_score + confidence, 1.0)
            rec.evidence_count += 1
            last = rec.last_seen_at
            if last is not None and last.tzinfo is None:  # sqlite reads back naive
                last = last.replace(tzinfo=dt.timezone.utc)
            if artifact_date and (last is None or artifact_date > last):
                rec.last_seen_at = artifact_date
        await self.s.flush()
        return rec

    async def link_evidence(self, skill_id: str, artifact_id: str, weight: float = 1.0) -> None:
        self.s.add(
            SkillEvidenceLink(
                skill_id=skill_id, artifact_id=artifact_id, contribution_weight=weight
            )
        )
        await self.s.flush()

    async def list_for(self, developer_id: str) -> list[SkillRecord]:
        return list(
            (
                await self.s.execute(
                    select(SkillRecord)
                    .where(SkillRecord.developer_id == developer_id)
                    .order_by(SkillRecord.confidence_score.desc())
                )
            ).scalars()
        )
