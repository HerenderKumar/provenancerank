"""ORM models. UUID string PKs and portable JSON columns so the same schema runs
on SQLite (dev/test) and Postgres (prod) without edits.
"""

from __future__ import annotations

import datetime as dt
import uuid

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.base import Base


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


# job status values
QUEUED, RUNNING, SUCCEEDED, FAILED = "queued", "running", "succeeded", "failed"
# roles
ROLE_RECRUITER, ROLE_ADMIN = "recruiter", "admin"


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(20), default=ROLE_RECRUITER)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)

    api_keys: Mapped[list[ApiKey]] = relationship(
        back_populates="owner", cascade="all, delete-orphan"
    )


class ApiKey(Base):
    __tablename__ = "api_keys"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    key_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(100))
    last4: Mapped[str] = mapped_column(String(8))
    scopes: Mapped[str] = mapped_column(String(255), default="rank:read,rank:write")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)
    last_used_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    expires_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    owner: Mapped[User] = relationship(back_populates="api_keys")


class JobRun(Base):
    __tablename__ = "job_runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    status: Mapped[str] = mapped_column(String(16), default=QUEUED, index=True)
    jd_hash: Mapped[str] = mapped_column(String(64), index=True)
    candidates_count: Mapped[int] = mapped_column(Integer, default=0)
    top_k: Mapped[int] = mapped_column(Integer, default=100)
    source: Mapped[str] = mapped_column(String(20), default="fast_path")  # fast_path|live
    cache_hit: Mapped[bool] = mapped_column(Boolean, default=False)
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    gate_stats: Mapped[dict] = mapped_column(JSON, default=dict)
    coverage: Mapped[dict] = mapped_column(JSON, default=dict)
    elapsed_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_by: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=_now, index=True
    )
    started_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    results: Mapped[list[RankingResult]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )


class RankingResult(Base):
    __tablename__ = "ranking_results"
    __table_args__ = (Index("ix_result_job_rank", "job_id", "rank"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("job_runs.id", ondelete="CASCADE"), index=True)
    candidate_id: Mapped[str] = mapped_column(String(20), index=True)
    rank: Mapped[int] = mapped_column(Integer)
    score: Mapped[float] = mapped_column(Float)
    reasoning: Mapped[str] = mapped_column(Text, default="")
    features: Mapped[dict] = mapped_column(JSON, default=dict)

    job: Mapped[JobRun] = relationship(back_populates="results")


class AuditLog(Base):
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    actor_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    action: Mapped[str] = mapped_column(String(64), index=True)
    target: Mapped[str | None] = mapped_column(String(128), nullable=True)
    ip: Mapped[str | None] = mapped_column(String(64), nullable=True)
    meta: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=_now, index=True
    )


# ===========================================================================
# Live ingestion layer (addendum). Mirrors db/init.sql; portable across
# sqlite (dev) and Postgres (prod). The pgvector embedding_records table is
# Postgres-only and lives in init.sql, not here.
# ===========================================================================

# developer sync states
SYNC_PENDING, SYNC_SYNCING, SYNC_DONE, SYNC_ERROR = "pending", "syncing", "done", "error"


class Developer(Base):
    __tablename__ = "developers"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    github_username: Mapped[str | None] = mapped_column(String(255), unique=True, index=True)
    display_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    avatar_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)
    last_synced_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    sync_status: Mapped[str] = mapped_column(String(50), default=SYNC_PENDING)

    artifacts: Mapped[list[EvidenceArtifact]] = relationship(
        back_populates="developer", cascade="all, delete-orphan"
    )
    skills: Mapped[list[SkillRecord]] = relationship(
        back_populates="developer", cascade="all, delete-orphan"
    )


class EvidenceArtifact(Base):
    __tablename__ = "evidence_artifacts"
    __table_args__ = (Index("idx_evidence_developer", "developer_id"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    developer_id: Mapped[str] = mapped_column(
        ForeignKey("developers.id", ondelete="CASCADE"), index=True
    )
    source_type: Mapped[str] = mapped_column(String(50))
    source_url: Mapped[str] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    raw_content: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    complexity_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    problem_category: Mapped[str | None] = mapped_column(String(50), nullable=True)
    production_signal: Mapped[bool] = mapped_column(Boolean, default=False)
    artifact_date: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    ingested_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)

    developer: Mapped[Developer] = relationship(back_populates="artifacts")


class SkillRecord(Base):
    __tablename__ = "skill_records"
    __table_args__ = (UniqueConstraint("developer_id", "skill_name", name="uq_dev_skill"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    developer_id: Mapped[str] = mapped_column(
        ForeignKey("developers.id", ondelete="CASCADE"), index=True
    )
    skill_name: Mapped[str] = mapped_column(String(255))
    skill_category: Mapped[str | None] = mapped_column(String(100), nullable=True)
    confidence_score: Mapped[float] = mapped_column(Float, default=0.0)
    evidence_count: Mapped[int] = mapped_column(Integer, default=0)
    last_seen_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    developer: Mapped[Developer] = relationship(back_populates="skills")


class SkillEvidenceLink(Base):
    __tablename__ = "skill_evidence_links"

    skill_id: Mapped[str] = mapped_column(
        ForeignKey("skill_records.id", ondelete="CASCADE"), primary_key=True
    )
    artifact_id: Mapped[str] = mapped_column(
        ForeignKey("evidence_artifacts.id", ondelete="CASCADE"), primary_key=True
    )
    contribution_weight: Mapped[float] = mapped_column(Float, default=1.0)
