"""FastAPI dependencies: singletons from app.state, auth (API key OR JWT), RBAC,
and rate limiting. Routers just declare what they need."""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Depends, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import get_settings
from core.exceptions import AuthenticationError, PermissionDeniedError, RateLimitedError
from core.security import decode_token, hash_api_key
from db.base import get_session
from db.models import ROLE_ADMIN
from db.repositories import ApiKeyRepository, UserRepository
from services.cache import Cache
from services.jobs import JobManager
from services.ranker import RankerService
from services.ratelimit import RateLimiter


@dataclass
class AuthContext:
    subject: str  # user id (jwt) or api-key owner id
    role: str  # recruiter | admin
    via: str  # "jwt" | "api_key"
    email: str | None = None
    scopes: tuple[str, ...] = ()


# these come from app.state, populated once in the lifespan handler
def get_ranker(request: Request) -> RankerService:
    return request.app.state.ranker


def get_jobs(request: Request) -> JobManager:
    return request.app.state.jobs


def get_cache(request: Request) -> Cache:
    return request.app.state.cache


def get_limiter(request: Request) -> RateLimiter:
    return request.app.state.limiter


def client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# Accept either an API key (X-API-Key) or a JWT bearer token. API keys win if
# both are present - they're for machine clients.
async def authenticate(
    request: Request, session: AsyncSession = Depends(get_session)
) -> AuthContext:
    s = get_settings()
    raw_key = request.headers.get(s.api_key_header.lower()) or request.headers.get(s.api_key_header)
    if raw_key:
        key = await ApiKeyRepository(session).get_active_by_hash(hash_api_key(raw_key))
        if not key:
            raise AuthenticationError("invalid or revoked API key")
        await ApiKeyRepository(session).touch(key)
        owner = await UserRepository(session).get_by_id(key.owner_id)
        if not owner or not owner.is_active:
            raise AuthenticationError("API key owner inactive")
        return AuthContext(
            subject=owner.id,
            role=owner.role,
            via="api_key",
            email=owner.email,
            scopes=tuple(key.scopes.split(",")),
        )

    authz = request.headers.get("authorization", "")
    if authz.lower().startswith("bearer "):
        token = authz.split(" ", 1)[1].strip()
        data = decode_token(token, expected_type="access")
        return AuthContext(subject=data.sub, role=data.role, via="jwt", email=data.email)

    raise AuthenticationError("provide a Bearer token or API key")


def require_role(*roles: str):
    async def _checker(auth: AuthContext = Depends(authenticate)) -> AuthContext:
        if roles and auth.role not in roles:
            raise PermissionDeniedError(f"requires role: {', '.join(roles)}")
        return auth

    return _checker


require_admin = require_role(ROLE_ADMIN)


async def enforce_rate_limit(
    request: Request,
    response: Response,
    auth: AuthContext = Depends(authenticate),
    limiter: RateLimiter = Depends(get_limiter),
) -> AuthContext:
    identity = auth.subject or client_ip(request)
    decision = await limiter.check(identity)
    response.headers["X-RateLimit-Limit"] = str(decision.limit)
    response.headers["X-RateLimit-Remaining"] = str(decision.remaining)
    if not decision.allowed:
        raise RateLimitedError("rate limit exceeded", detail={"retry_after": decision.retry_after})
    return auth
