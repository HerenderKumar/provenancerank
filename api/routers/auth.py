"""Auth: password login -> JWTs, plus per-user API keys (shown once)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import AuthContext, authenticate, client_ip, require_admin
from api.schemas import (
    ApiKeyCreatedResponse,
    ApiKeyCreateRequest,
    ApiKeyOut,
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    UserOut,
)
from core.config import get_settings
from core.exceptions import AuthenticationError, ConflictError, NotFoundError
from core.security import (
    create_access_token,
    create_refresh_token,
    generate_api_key,
    hash_password,
    verify_password,
)
from db.base import get_session
from db.repositories import ApiKeyRepository, AuditRepository, UserRepository

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=TokenResponse)
async def login(
    body: LoginRequest, request: Request, session: AsyncSession = Depends(get_session)
) -> TokenResponse:
    user = await UserRepository(session).get_by_email(body.email)
    if not user or not verify_password(body.password, user.hashed_password):
        raise AuthenticationError("invalid email or password")
    if not user.is_active:
        raise AuthenticationError("account disabled")
    await AuditRepository(session).log("auth.login", actor_id=user.id, ip=client_ip(request))
    s = get_settings()
    return TokenResponse(
        access_token=create_access_token(user.id, user.role, user.email),
        refresh_token=create_refresh_token(user.id, user.role, user.email),
        expires_in=s.jwt_access_ttl_minutes * 60,
    )


@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
async def register(
    body: RegisterRequest,
    _: AuthContext = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
) -> UserOut:
    repo = UserRepository(session)
    if await repo.get_by_email(body.email):
        raise ConflictError("a user with that email already exists")
    user = await repo.create(body.email, hash_password(body.password), body.role)
    return UserOut(id=user.id, email=user.email, role=user.role, created_at=user.created_at)


@router.get("/me", response_model=UserOut)
async def me(
    auth: AuthContext = Depends(authenticate), session: AsyncSession = Depends(get_session)
) -> UserOut:
    user = await UserRepository(session).get_by_id(auth.subject)
    if not user:
        raise NotFoundError("user not found")
    return UserOut(id=user.id, email=user.email, role=user.role, created_at=user.created_at)


@router.post("/api-keys", response_model=ApiKeyCreatedResponse, status_code=status.HTTP_201_CREATED)
async def create_api_key(
    body: ApiKeyCreateRequest,
    auth: AuthContext = Depends(authenticate),
    session: AsyncSession = Depends(get_session),
) -> ApiKeyCreatedResponse:
    raw, key_hash, last4 = generate_api_key()
    key = await ApiKeyRepository(session).create(
        owner_id=auth.subject, key_hash=key_hash, name=body.name, last4=last4, scopes=body.scopes
    )
    await AuditRepository(session).log("apikey.created", actor_id=auth.subject, target=key.id)
    return ApiKeyCreatedResponse(
        id=key.id, name=key.name, api_key=raw, last4=last4, scopes=key.scopes
    )


@router.get("/api-keys", response_model=list[ApiKeyOut])
async def list_api_keys(
    auth: AuthContext = Depends(authenticate), session: AsyncSession = Depends(get_session)
) -> list[ApiKeyOut]:
    keys = await ApiKeyRepository(session).list_for_owner(auth.subject)
    return [
        ApiKeyOut(
            id=k.id,
            name=k.name,
            last4=k.last4,
            scopes=k.scopes,
            is_active=k.is_active,
            created_at=k.created_at,
            last_used_at=k.last_used_at,
        )
        for k in keys
    ]


@router.delete("/api-keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_api_key(
    key_id: str,
    auth: AuthContext = Depends(authenticate),
    session: AsyncSession = Depends(get_session),
) -> None:
    ok = await ApiKeyRepository(session).revoke(key_id, auth.subject)
    if not ok:
        raise NotFoundError("API key not found")
    await AuditRepository(session).log("apikey.revoked", actor_id=auth.subject, target=key_id)
