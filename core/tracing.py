"""OpenTelemetry tracing — optional.

If the OTel packages are installed and otel_enabled is on, we export spans to
the configured OTLP endpoint (Jaeger in the compose stack) and auto-instrument
FastAPI + SQLAlchemy. If not, every helper here is a no-op so call sites stay
clean and nothing breaks when OTel isn't around. We also expose the current
trace_id so it can be stitched into the structured logs.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

from core.config import get_settings
from core.logging import get_logger

log = get_logger("core.tracing")

_TRACER = None
_ENABLED = False


def setup_tracing(app: Any = None, engine: Any = None) -> None:
    """Wire up tracing once at startup. Safe to call when OTel is absent."""
    global _TRACER, _ENABLED
    s = get_settings()
    if not s.otel_enabled:
        return
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider(
            resource=Resource.create(
                {
                    "service.name": s.service_name,
                    "service.version": s.service_version,
                    "deployment.environment": s.environment,
                }
            )
        )
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=s.otel_exporter_endpoint))
        )
        trace.set_tracer_provider(provider)
        _TRACER = trace.get_tracer(s.service_name)
        _ENABLED = True

        if app is not None:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

            FastAPIInstrumentor.instrument_app(app)
        if engine is not None:
            from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

            SQLAlchemyInstrumentor().instrument(engine=engine.sync_engine)
        log.info("tracing.enabled", endpoint=s.otel_exporter_endpoint)
    except Exception as exc:  # missing packages or bad endpoint -> stay no-op
        log.warning("tracing.setup_failed", reason=str(exc)[:160])


@contextlib.contextmanager
def span(name: str, **attributes: Any) -> Iterator[None]:
    """Start a child span (no-op if tracing is off)."""
    if not _ENABLED or _TRACER is None:
        yield
        return
    with _TRACER.start_as_current_span(name) as sp:
        for k, v in attributes.items():
            sp.set_attribute(k, v)
        yield


def current_trace_id() -> str | None:
    if not _ENABLED:
        return None
    try:
        from opentelemetry import trace

        ctx = trace.get_current_span().get_span_context()
        if ctx and ctx.trace_id:
            return format(ctx.trace_id, "032x")
    except Exception:
        return None
    return None
