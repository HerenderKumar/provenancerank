"""Async SQLAlchemy engine + session.

One engine per process, created lazily from ``DATABASE_URL``. SQLite (dev/test)
and Postgres (prod) both work - pool sizing is only applied to real servers,
SQLite uses the default pool. ``get_session`` is the FastAPI dependency;
``session_scope`` is the same thing for background workers and scripts.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from core.config import get_settings
from core.logging import get_logger

log = get_logger("db")

_engine = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


class Base(DeclarativeBase):
    pass


def _build_engine():
    s = get_settings()
    url = s.database_url
    kwargs: dict = {"echo": s.db_echo, "pool_pre_ping": True, "future": True}
    if url.startswith("sqlite"):
        # WAL + busy_timeout so concurrent read (auth) and write (jobs) don't
        # trip "database is locked". Postgres gets MVCC for free; this is the
        # dev/test equivalent.
        kwargs["connect_args"] = {"timeout": 30}
    else:
        kwargs.update(pool_size=s.db_pool_size, max_overflow=s.db_max_overflow, pool_recycle=1800)
    engine = create_async_engine(url, **kwargs)
    if url.startswith("sqlite"):
        from sqlalchemy import event

        @event.listens_for(engine.sync_engine, "connect")
        def _sqlite_pragmas(dbapi_conn, _record):  # noqa: ANN001
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA busy_timeout=30000")
            cur.execute("PRAGMA synchronous=NORMAL")
            cur.close()

    return engine


def get_engine():
    global _engine
    if _engine is None:
        _engine = _build_engine()
        log.info("db.engine_created", url=get_settings().database_url.split("://")[0])
    return _engine


def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    global _sessionmaker
    if _sessionmaker is None:
        _sessionmaker = async_sessionmaker(
            get_engine(), expire_on_commit=False, class_=AsyncSession
        )
    return _sessionmaker


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency. Commits on success, rolls back on error."""
    async with get_sessionmaker()() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


@contextlib.asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    async with get_sessionmaker()() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_models() -> None:
    """Create tables if missing (dev/test convenience; prod uses Alembic)."""
    from db import models  # noqa: F401  (register mappers)

    async with get_engine().begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    log.info("db.models_ready")


async def dispose_engine() -> None:
    global _engine, _sessionmaker
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _sessionmaker = None
