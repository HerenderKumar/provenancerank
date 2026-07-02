"""Domain exceptions. Each one carries a stable machine `code` and the HTTP
status the API should answer with, so the error handler is a one-liner and
clients can switch on `code` instead of parsing prose.
"""

from __future__ import annotations

from typing import Any


class ProvenanceError(Exception):
    """Base for everything we raise on purpose."""

    code = "internal_error"
    http_status = 500

    def __init__(self, message: str | None = None, *, detail: Any = None):
        super().__init__(message or self.__class__.__name__)
        self.message = message or self.__class__.__name__
        self.detail = detail

    def to_dict(self) -> dict[str, Any]:
        body: dict[str, Any] = {"error": self.code, "message": self.message}
        if self.detail is not None:
            body["detail"] = self.detail
        return body


# ---- infrastructure / engine ----


class ArtifactError(ProvenanceError):
    """A precomputed artifact is missing or unreadable."""

    code = "artifact_unavailable"
    http_status = 503


class ArtifactChecksumError(ArtifactError):
    """Artifact on disk doesn't match the manifest hash — refuse to serve it."""

    code = "artifact_checksum_mismatch"
    http_status = 503


class RankingError(ProvenanceError):
    code = "ranking_failed"
    http_status = 500


class ServiceUnavailableError(ProvenanceError):
    code = "service_unavailable"
    http_status = 503


# ---- request / client ----


class BadRequestError(ProvenanceError):
    code = "bad_request"
    http_status = 400


class ValidationFailedError(ProvenanceError):
    code = "validation_failed"
    http_status = 422


class NotFoundError(ProvenanceError):
    code = "not_found"
    http_status = 404


class ConflictError(ProvenanceError):
    code = "conflict"
    http_status = 409


# ---- auth ----


class AuthenticationError(ProvenanceError):
    code = "unauthorized"
    http_status = 401


class PermissionDeniedError(ProvenanceError):
    code = "forbidden"
    http_status = 403


class RateLimitedError(ProvenanceError):
    code = "rate_limited"
    http_status = 429
