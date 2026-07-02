"""All tunables in one place, read from env / .env.

Uses pydantic-settings when it's available, otherwise a small dataclass that
reads os.environ directly — same field names either way. get_settings() is
cached so we read the environment once.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

# Repo root = parent of this file's parent (``core/`` -> repo root).
REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DEFAULT = str(REPO_ROOT / "artifacts")

_FIELDS: dict[str, object] = {
    # ---- LLM (offline precompute only — never used during ranking) ----
    "google_api_key": "",
    "jd_deconstruct_model": "gemini-1.5-flash",
    # ---- Embedding model (local, no API) ----
    "embedding_model": "all-MiniLM-L6-v2",
    "embedding_dim": 384,
    "embedding_batch_size": 512,
    "allow_remote_embeddings": False,  # if False, never download a model
    # ---- Paths ----
    "artifacts_dir": ARTIFACTS_DEFAULT,
    "candidates_file": str(REPO_ROOT / "candidates.jsonl"),
    "jd_file": str(REPO_ROOT / "data" / "job_description.txt"),
    "model_path": str(REPO_ROOT / "ml" / "models" / "fit_scorer_v1.pkl"),
    # ---- Final-score merge weights (must sum to 1.0) ----
    "ml_score_weight": 0.40,
    "retrieval_score_weight": 0.35,
    "signal_score_weight": 0.25,
    # ---- Behavioural sub-composite weights ----
    "availability_weight": 0.35,
    "engagement_weight": 0.35,
    "trust_weight": 0.20,
    "github_weight": 0.10,
    # ---- Loop agent ----
    "max_loop_iterations": 4,
    "confidence_threshold": 0.80,
    "coverage_gap_threshold": 0.60,
    # ---- Pipeline ----
    "stream_batch_size": 1000,
    "top_k": 100,
    "recency_half_life_days": 90.0,
    "random_seed": 42,
    # ---- API ----
    "api_host": "0.0.0.0",
    "api_port": 8000,
    "redis_url": "redis://redis:6379/0",
    "cache_ttl_seconds": 3600,
    "cache_enabled": True,
    "log_level": "INFO",
    # ---- environment / service identity ----
    "environment": "development",  # development | staging | production
    "service_name": "provenancerank-api",
    "service_version": "1.0.0",
    # ---- database (dev: sqlite; prod: postgresql+asyncpg://...) ----
    "database_url": "sqlite+aiosqlite:///./provenancerank.db",
    "db_pool_size": 10,
    "db_max_overflow": 20,
    "db_echo": False,
    # ---- auth / security ----
    "jwt_secret": "dev-only-change-me-to-a-32+char-random-secret",
    "jwt_algorithm": "HS256",
    "jwt_access_ttl_minutes": 30,
    "jwt_refresh_ttl_days": 7,
    "api_key_header": "X-API-Key",
    "bootstrap_admin_email": "admin@provenancerank.local",
    "bootstrap_admin_password": "changeme-admin",
    # ---- CORS (comma-separated origins) ----
    "cors_origins": "http://localhost:5173,http://localhost:3000",
    # ---- rate limiting ----
    "rate_limit_enabled": True,
    "rate_limit_per_minute": 120,
    "rate_limit_burst": 30,
    # ---- observability ----
    "metrics_enabled": True,
    "otel_enabled": False,
    "otel_exporter_endpoint": "http://jaeger:4318/v1/traces",
    # ---- live ingestion layer (addendum) ----
    "github_token": "",
    "github_graphql_url": "https://api.github.com/graphql",
    "stackoverflow_api_key": "",
    "max_repos_per_developer": 20,
    "max_commits_per_repo": 50,
    "sync_interval_hours": 6,
    # evidence confidence formula
    "evidence_half_life_days": 365.0,
    "evidence_normaliser": 10.0,
    "evidence_confidence_bonus": 0.15,
    # knowledge graph (Neo4j; falls back to in-memory when disabled/unreachable)
    "neo4j_enabled": False,
    "neo4j_uri": "bolt://neo4j:7687",
    "neo4j_user": "neo4j",
    "neo4j_password": "provenancerank",
    # hybrid graph retrieval: fuse exact skill traversal with semantic (vector)
    # search over evidence summaries, so paraphrased queries still recall.
    "graph_hybrid_enabled": True,
    "graph_vector_top_k": 50,
    "graph_rrf_k": 60,
    # celery (eager == run inline, no worker needed; prod sets it false)
    "celery_broker_url": "redis://redis:6379/1",
    "celery_result_backend": "redis://redis:6379/2",
    "celery_task_always_eager": True,
    "summariser_model": "gemini-1.5-flash",
    # ---- LLM provider (JD parse, summaries, grading, NL query) ----
    # auto order: GLM (if key) -> local Ollama (if up) -> Gemini (if key) -> heuristic.
    "llm_provider": "auto",  # auto | glm | ollama | gemini | none
    "glm_api_key": "",  # Z.ai / Zhipu GLM (OpenAI-compatible); primary when set
    "glm_base_url": "https://api.z.ai/api/paas/v4",
    "glm_model": "glm-5.2",
    "ollama_base_url": "http://localhost:11434",  # Docker: http://host.docker.internal:11434
    "ollama_model": "llama3.1:8b",  # any model you've `ollama pull`-ed
    "llm_timeout": 60,
    # ---- Accuracy layer (all offline in precompute; cached into the matrix) ----
    # The LambdaMART ranker is written to model_path (overwrites the proxy
    # regressor), so predict_fit loads it with no change downstream.
    "pseudo_label_sample": 1500,  # how many head rows Gemini grades (rest heuristic)
    # LLM label-grading is OFF by default: a Google key enables the nice JD parse,
    # but NOT 75 blocking Gemini calls that freeze precompute. Opt in explicitly.
    "pseudo_label_llm_enabled": False,
    "pseudo_label_llm_timeout": 30,  # seconds per Gemini grading call
    "reranker_enabled": True,
    "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "rerank_top_k": 300,  # rerank only the head — that's where NDCG@10/50 lives
    "rerank_weight": 0.15,  # blend weight where a rerank score exists
    "tournament_enabled": True,
    "tournament_top_n": 60,  # pairwise Bradley-Terry over the very top
    "tournament_weight": 0.10,
    # LLM head-scoring (offline; the LLM reads the top-K vs the JD, cached + blended)
    "llm_head_scoring_enabled": False,  # opt in (needs an LLM at precompute time)
    "llm_head_top_k": 250,  # only score the head, where NDCG@10/50 is decided
    "llm_score_weight": 0.15,  # blend weight where an llm_fit_score exists
    "llm_score_batch": 20,  # candidates per LLM call
    # ---- Performance (speed levers; none change which features run) ----
    "compute_device": "auto",  # auto | cpu | cuda | mps  (transformer models)
    "use_onnx": False,  # ONNX Runtime backend for embedder + cross-encoder
    "onnx_quantized": True,  # prefer the int8 weights when ONNX is on
    "ranker_neg_ratio": 4,  # negatives kept per positive when training the ranker
    "ranker_min_train_rows": 20000,  # floor so we never train on too little
    "ranker_early_stop_rounds": 40,  # stop boosting once NDCG plateaus
}

# Env var name -> field name (env names are upper-cased field names by default).
_ENV_ALIASES = {"GOOGLE_API_KEY": "google_api_key"}


def _coerce(default: object, raw: str) -> object:
    if isinstance(default, bool):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(default, int) and not isinstance(default, bool):
        return int(float(raw))
    if isinstance(default, float):
        return float(raw)
    return raw


try:
    from pydantic import Field
    from pydantic_settings import BaseSettings, SettingsConfigDict

    class Settings(BaseSettings):  # type: ignore[misc]
        """Validated settings loaded from environment and ``.env``."""

        model_config = SettingsConfigDict(
            env_file=os.environ.get("ENV_FILE", ".env"),
            env_file_encoding="utf-8",
            case_sensitive=False,
            extra="ignore",
        )

        google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")
        jd_deconstruct_model: str = "gemini-1.5-flash"
        embedding_model: str = "all-MiniLM-L6-v2"
        embedding_dim: int = 384
        embedding_batch_size: int = 512
        allow_remote_embeddings: bool = False
        artifacts_dir: str = ARTIFACTS_DEFAULT
        candidates_file: str = _FIELDS["candidates_file"]  # type: ignore[assignment]
        jd_file: str = _FIELDS["jd_file"]  # type: ignore[assignment]
        model_path: str = _FIELDS["model_path"]  # type: ignore[assignment]
        ml_score_weight: float = 0.40
        retrieval_score_weight: float = 0.35
        signal_score_weight: float = 0.25
        availability_weight: float = 0.35
        engagement_weight: float = 0.35
        trust_weight: float = 0.20
        github_weight: float = 0.10
        max_loop_iterations: int = 4
        confidence_threshold: float = 0.80
        coverage_gap_threshold: float = 0.60
        stream_batch_size: int = 1000
        top_k: int = 100
        recency_half_life_days: float = 90.0
        random_seed: int = 42
        api_host: str = "0.0.0.0"
        api_port: int = 8000
        redis_url: str = "redis://redis:6379/0"
        cache_ttl_seconds: int = 3600
        cache_enabled: bool = True
        log_level: str = "INFO"
        environment: str = "development"
        service_name: str = "provenancerank-api"
        service_version: str = "1.0.0"
        database_url: str = "sqlite+aiosqlite:///./provenancerank.db"
        db_pool_size: int = 10
        db_max_overflow: int = 20
        db_echo: bool = False
        jwt_secret: str = "dev-only-change-me-to-a-32+char-random-secret"
        jwt_algorithm: str = "HS256"
        jwt_access_ttl_minutes: int = 30
        jwt_refresh_ttl_days: int = 7
        api_key_header: str = "X-API-Key"
        bootstrap_admin_email: str = "admin@provenancerank.local"
        bootstrap_admin_password: str = "changeme-admin"
        cors_origins: str = "http://localhost:5173,http://localhost:3000"
        rate_limit_enabled: bool = True
        rate_limit_per_minute: int = 120
        rate_limit_burst: int = 30
        metrics_enabled: bool = True
        otel_enabled: bool = False
        otel_exporter_endpoint: str = "http://jaeger:4318/v1/traces"
        # live ingestion layer
        github_token: str = Field(default="", alias="GITHUB_TOKEN")
        github_graphql_url: str = "https://api.github.com/graphql"
        stackoverflow_api_key: str = ""
        max_repos_per_developer: int = 20
        max_commits_per_repo: int = 50
        sync_interval_hours: int = 6
        evidence_half_life_days: float = 365.0
        evidence_normaliser: float = 10.0
        evidence_confidence_bonus: float = 0.15
        neo4j_enabled: bool = False
        neo4j_uri: str = "bolt://neo4j:7687"
        neo4j_user: str = "neo4j"
        neo4j_password: str = "provenancerank"
        graph_hybrid_enabled: bool = True
        graph_vector_top_k: int = 50
        graph_rrf_k: int = 60
        celery_broker_url: str = "redis://redis:6379/1"
        celery_result_backend: str = "redis://redis:6379/2"
        celery_task_always_eager: bool = True
        summariser_model: str = "gemini-1.5-flash"
        # LLM provider
        llm_provider: str = "auto"
        glm_api_key: str = Field(default="", alias="GLM_API_KEY")
        glm_base_url: str = "https://api.z.ai/api/paas/v4"
        glm_model: str = "glm-5.2"
        ollama_base_url: str = "http://localhost:11434"
        ollama_model: str = "llama3.1:8b"
        llm_timeout: int = 60
        # accuracy layer
        pseudo_label_sample: int = 1500
        pseudo_label_llm_enabled: bool = False
        pseudo_label_llm_timeout: int = 30
        reranker_enabled: bool = True
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        rerank_top_k: int = 300
        rerank_weight: float = 0.15
        tournament_enabled: bool = True
        tournament_top_n: int = 60
        tournament_weight: float = 0.10
        llm_head_scoring_enabled: bool = False
        llm_head_top_k: int = 250
        llm_score_weight: float = 0.15
        llm_score_batch: int = 20
        # performance
        compute_device: str = "auto"
        use_onnx: bool = False
        onnx_quantized: bool = True
        ranker_neg_ratio: int = 4
        ranker_min_train_rows: int = 20000
        ranker_early_stop_rounds: int = 40

        @property
        def artifacts(self) -> Path:
            return Path(self.artifacts_dir)

        @property
        def cors_origin_list(self) -> list[str]:
            return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

        @property
        def is_production(self) -> bool:
            return self.environment.lower() == "production"

    _USING_PYDANTIC = True

except Exception:  # pragma: no cover - fallback path
    from dataclasses import dataclass, field

    def _build_dataclass():
        annotations = {}
        defaults = {}
        for key, val in _FIELDS.items():
            annotations[key] = type(val)
            defaults[key] = val

        @dataclass
        class Settings:  # type: ignore[no-redef]
            pass

        Settings.__annotations__ = annotations
        for key, val in defaults.items():
            setattr(Settings, key, field(default=val) if False else val)

        def _post_init(self):
            for env_name, field_name in _ENV_ALIASES.items():
                if env_name in os.environ:
                    setattr(self, field_name, os.environ[env_name])
            for key, default in _FIELDS.items():
                env_name = key.upper()
                if env_name in os.environ:
                    setattr(self, key, _coerce(default, os.environ[env_name]))

        Settings.__post_init__ = _post_init
        Settings.artifacts = property(lambda self: Path(self.artifacts_dir))
        Settings.cors_origin_list = property(
            lambda self: [o.strip() for o in self.cors_origins.split(",") if o.strip()]
        )
        Settings.is_production = property(lambda self: self.environment.lower() == "production")
        return dataclass(Settings)

    Settings = _build_dataclass()  # type: ignore[assignment]
    _USING_PYDANTIC = False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()  # type: ignore[call-arg]


USING_PYDANTIC = _USING_PYDANTIC
