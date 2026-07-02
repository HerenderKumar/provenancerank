"""Exception handlers — turn domain errors and validation failures into clean
JSON with the right status, instead of leaking tracebacks."""

from __future__ import annotations

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from core.exceptions import ProvenanceError
from core.logging import get_logger

log = get_logger("api.errors")


async def provenance_handler(request: Request, exc: ProvenanceError) -> JSONResponse:
    if exc.http_status >= 500:
        log.error("api.domain_error", code=exc.code, message=exc.message, path=request.url.path)
    return JSONResponse(status_code=exc.http_status, content=exc.to_dict())


async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_failed",
            "message": "request body failed validation",
            "detail": exc.errors(),
        },
    )


async def unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
    log.exception("api.unhandled", path=request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "message": "an unexpected error occurred",
        },
    )


def register(app) -> None:
    app.add_exception_handler(ProvenanceError, provenance_handler)
    app.add_exception_handler(RequestValidationError, validation_handler)
    app.add_exception_handler(Exception, unhandled_handler)
