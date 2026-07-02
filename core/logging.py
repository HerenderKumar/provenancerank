"""JSON logging to stdout.

Prefers structlog; if it isn't installed we fall back to a stdlib formatter
that emits the same JSON shape, so nothing else in the repo has to know which
one is active and we never resort to print(). log_duration() is the helper I
actually reach for everywhere: wrap a block, get one `<event>` line with
duration_ms when it exits.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
import time
from collections.abc import Iterator
from typing import Any

_CONFIGURED = False
_DEFAULT_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

try:  # pragma: no cover - exercised implicitly by whichever path is installed
    import structlog

    _HAVE_STRUCTLOG = True
except Exception:  # pragma: no cover
    structlog = None  # type: ignore
    _HAVE_STRUCTLOG = False


class _JsonFormatter(logging.Formatter):
    """Fallback stdlib formatter that renders records as compact JSON."""

    _RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()) | {
        "message",
        "asctime",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname.lower(),
            "logger": record.name,
            "event": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in self._RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure(level: str | None = None) -> None:
    """Configure structured logging once. Idempotent and thread-safe enough
    for our single-process pipeline."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    log_level = getattr(logging, (level or _DEFAULT_LEVEL), logging.INFO)

    if _HAVE_STRUCTLOG:
        logging.basicConfig(format="%(message)s", stream=sys.stdout, level=log_level)
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso", utc=True),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(log_level),
            logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
            cache_logger_on_first_use=True,
        )
    else:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_JsonFormatter())
        root = logging.getLogger()
        root.handlers[:] = [handler]
        root.setLevel(log_level)
    _CONFIGURED = True


class _StdlibBoundLogger:
    """Minimal structlog-compatible shim over stdlib logging.

    Supports ``logger.info("event", key=value)`` and ``logger.bind(...)``
    so call sites are identical regardless of whether structlog is present.
    """

    def __init__(self, name: str, context: dict[str, Any] | None = None):
        self._log = logging.getLogger(name)
        self._context = context or {}

    def bind(self, **kw: Any) -> _StdlibBoundLogger:
        merged = {**self._context, **kw}
        return _StdlibBoundLogger(self._log.name, merged)

    def _emit(self, level: int, event: str, **kw: Any) -> None:
        self._log.log(level, event, extra={**self._context, **kw})

    def debug(self, event: str, **kw: Any) -> None:
        self._emit(logging.DEBUG, event, **kw)

    def info(self, event: str, **kw: Any) -> None:
        self._emit(logging.INFO, event, **kw)

    def warning(self, event: str, **kw: Any) -> None:
        self._emit(logging.WARNING, event, **kw)

    def error(self, event: str, **kw: Any) -> None:
        self._emit(logging.ERROR, event, **kw)

    def exception(self, event: str, **kw: Any) -> None:
        self._log.exception(event, extra={**self._context, **kw})


def get_logger(name: str = "provenancerank", **initial_context: Any):
    """Return a structured logger bound with optional initial context."""
    configure()
    if _HAVE_STRUCTLOG:
        return structlog.get_logger(name).bind(**initial_context)
    return _StdlibBoundLogger(name, initial_context)


@contextlib.contextmanager
def log_duration(logger: Any, event: str, **context: Any) -> Iterator[dict]:
    """Time a block and emit ``<event>`` with ``duration_ms`` on exit.

    Yields a mutable dict so the caller can attach result metadata that is
    included in the final structured event::

        with log_duration(log, "feature_engineering.complete") as m:
            m["rows"] = len(df)
    """
    extra: dict[str, Any] = {}
    start = time.perf_counter()
    status = "ok"
    try:
        yield extra
    except Exception:
        status = "error"
        raise
    finally:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info(event, duration_ms=duration_ms, status=status, **context, **extra)
