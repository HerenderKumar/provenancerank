"""Request/response models. Pydantic does the validation at the edge so routers
deal in typed objects, not raw dicts."""

from __future__ import annotations

import datetime as dt
import re

from pydantic import BaseModel, Field, field_validator

_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


# ---- auth ----


class RegisterRequest(BaseModel):
    email: str
    password: str = Field(min_length=8, max_length=128)
    role: str = "recruiter"

    @field_validator("email")
    @classmethod
    def _email(cls, v: str) -> str:
        if not _EMAIL.match(v):
            raise ValueError("invalid email")
        return v.lower()

    @field_validator("role")
    @classmethod
    def _role(cls, v: str) -> str:
        if v not in ("recruiter", "admin"):
            raise ValueError("role must be recruiter or admin")
        return v


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class UserOut(BaseModel):
    id: str
    email: str
    role: str
    created_at: dt.datetime


class ApiKeyCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    scopes: str = "rank:read,rank:write"


class ApiKeyCreatedResponse(BaseModel):
    id: str
    name: str
    api_key: str  # shown exactly once
    last4: str
    scopes: str


class ApiKeyOut(BaseModel):
    id: str
    name: str
    last4: str
    scopes: str
    is_active: bool
    created_at: dt.datetime
    last_used_at: dt.datetime | None = None


# ---- ranking ----


class RankRequest(BaseModel):
    jd_text: str | None = Field(
        default=None, max_length=20000, description="custom JD; omit to use the released JD"
    )
    candidate_ids: list[str] | None = Field(default=None, description="subset to rank")
    candidates: list[dict] | None = Field(
        default=None, description="ad-hoc candidate profiles (sandbox sample)"
    )
    top_k: int = Field(default=100, ge=1, le=100)

    @field_validator("candidates")
    @classmethod
    def _limit_live(cls, v):
        if v is not None and len(v) > 2000:
            raise ValueError("live sample capped at 2000 candidates")
        return v


class JobCreatedResponse(BaseModel):
    job_id: str
    status: str
    stream_url: str
    results_url: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    source: str | None = None
    cache_hit: bool = False
    candidates_count: int = 0
    top_k: int = 100
    elapsed_ms: float | None = None
    gate_stats: dict = {}
    coverage: dict = {}
    error: str | None = None
    created_at: dt.datetime | None = None
    finished_at: dt.datetime | None = None


class RankRow(BaseModel):
    candidate_id: str
    rank: int
    score: float
    reasoning: str


class ResultsResponse(BaseModel):
    job_id: str
    status: str
    count: int
    results: list[RankRow]


# ---- ops ----


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    environment: str


class ReadinessResponse(BaseModel):
    status: str
    ranker_ready: bool
    candidates: int
    database: bool
    cache: str
    checks: dict


class ArtifactStatusResponse(BaseModel):
    ready: bool
    candidates: int
    has_embeddings: bool
    has_bm25: bool
    jd_source: str
    embedding_backend: str | None = None
    built_at: str | None = None


# ---- live ingestion layer ----


class ConnectGithubRequest(BaseModel):
    github_username: str = Field(min_length=1, max_length=100)
    display_name: str | None = None


class ConnectGithubResponse(BaseModel):
    developer_id: str
    github_username: str
    status: str


class SyncStatusResponse(BaseModel):
    developer_id: str
    github_username: str | None = None
    status: str
    last_synced_at: dt.datetime | None = None
    artifact_count: int
    skill_count: int


class NLSearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=500)


class EvidenceItem(BaseModel):
    artifact_id: str | None = None
    source_type: str | None = None
    summary: str | None = None
    complexity: int | None = None
    production_signal: bool | None = None
    url: str | None = None
    date: str | None = None


class SearchCandidate(BaseModel):
    developer_id: str
    github_username: str | None = None
    display_name: str | None = None
    confidence: float
    evidence_count: int = 0
    evidence: list[dict] = []


class NLSearchResponse(BaseModel):
    query_interpreted: dict
    candidates: list[SearchCandidate]
