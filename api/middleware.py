"""Per-request middleware: a request id, structured access logging, and the
HTTP Prometheus metrics. The metrics use the matched route template (not the raw
path) as a label so candidate ids in the URL don't blow up cardinality.
"""

from __future__ import annotations

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from api.metrics import HTTP_LATENCY, HTTP_REQUESTS
from core.logging import get_logger
from core.tracing import current_trace_id

log = get_logger("api.access")


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        request.state.request_id = request_id
        start = time.perf_counter()
        response = None
        try:
            response = await call_next(request)
            return response
        finally:
            duration = time.perf_counter() - start
            status = response.status_code if response is not None else 500
            route = request.scope.get("route")
            path = getattr(route, "path", request.url.path)
            method = request.method
            if response is not None:
                response.headers["X-Request-ID"] = request_id
            try:
                HTTP_REQUESTS.labels(method=method, path=path, status=str(status)).inc()
                HTTP_LATENCY.labels(method=method, path=path).observe(duration)
            except Exception:
                pass
            log.info(
                "http.request",
                method=method,
                path=request.url.path,
                status=status,
                duration_ms=round(duration * 1000, 2),
                request_id=request_id,
                trace_id=current_trace_id(),
            )
