"""FastAPI application factory.

Lifespan wires the process together once: create tables, bootstrap an admin if
the user table is empty, load the ranker warm, connect the cache, build the job
manager, and start tracing. Missing artifacts don't stop the app from booting —
it comes up "not ready" so an operator can trigger precompute and the LB simply
won't route to it until /health/ready passes.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

from api import errors
from api.metrics import render
from api.middleware import RequestContextMiddleware
from api.routers import admin, auth, developers, health, rank, search
from core.config import get_settings
from core.exceptions import ArtifactError
from core.logging import configure, get_logger
from core.security import hash_password
from core.tracing import setup_tracing
from db.base import dispose_engine, get_engine, init_models, session_scope
from db.models import ROLE_ADMIN
from db.repositories import UserRepository
from services.cache import make_cache
from services.jobs import JobManager
from services.ranker import RankerService
from services.ratelimit import RateLimiter

log = get_logger("api")


async def _bootstrap_admin() -> None:
    s = get_settings()
    from sqlalchemy.exc import IntegrityError

    try:
        async with session_scope() as session:
            repo = UserRepository(session)
            if await repo.count() == 0:
                await repo.create(
                    s.bootstrap_admin_email, hash_password(s.bootstrap_admin_password), ROLE_ADMIN
                )
                log.info("bootstrap.admin_created", email=s.bootstrap_admin_email)
    except IntegrityError:
        # Two API replicas can boot at the same instant against a fresh DB and
        # both pass the count()==0 check, then both insert the admin. The loser
        # hits a unique violation — which is fine, the admin exists either way.
        log.info("bootstrap.admin_exists_race")


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure()
    s = get_settings()
    log.info("startup.begin", env=s.environment, version=s.service_version)

    try:
        await init_models()
    except Exception as exc:
        # create_all is not atomic across replicas; if another replica created the
        # tables a microsecond earlier we'd see "already exists" — harmless.
        log.warning("startup.init_models_concurrent", reason=str(exc)[:120])
    await _bootstrap_admin()

    ranker = RankerService()
    try:
        ranker.load()
    except ArtifactError as exc:
        log.warning("startup.ranker_not_ready", reason=str(exc))

    cache = await make_cache(s.redis_url, enabled=s.cache_enabled)
    app.state.ranker = ranker
    app.state.cache = cache
    app.state.limiter = RateLimiter(cache, s.rate_limit_per_minute, s.rate_limit_enabled)
    app.state.jobs = JobManager(ranker, cache, max_concurrency=2)
    await app.state.jobs.start()  # start the queue worker

    # warm the evidence graph backend (Neo4j or in-memory). Best-effort: a graph
    # outage must not stop the API from booting or serving ranking.
    try:
        from graph.queries import get_backend

        app.state.graph = await get_backend()
    except Exception as exc:
        log.warning("startup.graph_unavailable", reason=str(exc)[:120])
        app.state.graph = None

    try:
        setup_tracing(app, get_engine())
    except Exception as exc:
        log.warning("startup.tracing_failed", reason=str(exc)[:120])

    log.info("startup.ready", ranker_ready=ranker.ready, cache=cache.backend)
    try:
        yield
    finally:
        log.info("shutdown.begin")
        await app.state.jobs.stop()  # drain in-flight + stop worker
        await cache.close()
        await dispose_engine()
        log.info("shutdown.done")


def create_app() -> FastAPI:
    s = get_settings()
    app = FastAPI(
        title="ProvenanceRank API",
        version=s.service_version,
        description="Production serving layer for the ProvenanceRank candidate ranker.",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=s.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-RateLimit-Remaining"],
    )
    app.add_middleware(RequestContextMiddleware)
    errors.register(app)

    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(rank.router)
    app.include_router(admin.router)
    app.include_router(developers.router)  # live ingestion layer
    app.include_router(search.router)  # natural-language evidence search

    @app.get("/", tags=["meta"])
    async def root() -> dict:
        return {
            "service": s.service_name,
            "version": s.service_version,
            "docs": "/docs",
            "health": "/health/ready",
        }

    if s.metrics_enabled:

        @app.get("/metrics", tags=["meta"])
        async def metrics() -> Response:
            data, content_type = render()
            return Response(content=data, media_type=content_type)

    return app


app = create_app()
