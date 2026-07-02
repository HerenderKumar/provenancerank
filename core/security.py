"""Passwords, JWTs and API keys.

Passwords use bcrypt directly (skipping passlib to dodge its bcrypt-4 version
shim). Access/refresh tokens are JWTs. API keys are shown once at creation and
only their SHA-256 is stored, so a DB leak doesn't hand over working keys.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import secrets
from dataclasses import dataclass

import bcrypt
import jwt

from core.config import get_settings
from core.exceptions import AuthenticationError

API_KEY_PREFIX = "pr_"


# --------------------------------------------------------------------------
# passwords
# --------------------------------------------------------------------------


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode()


def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
    except (ValueError, TypeError):
        return False


# --------------------------------------------------------------------------
# JWT
# --------------------------------------------------------------------------


@dataclass
class TokenData:
    sub: str  # user id
    role: str  # "recruiter" | "admin"
    type: str  # "access" | "refresh"
    email: str | None = None


def _encode(sub: str, role: str, email: str | None, token_type: str, ttl: dt.timedelta) -> str:
    s = get_settings()
    now = dt.datetime.now(dt.timezone.utc)
    payload = {
        "sub": sub,
        "role": role,
        "email": email,
        "type": token_type,
        "iat": int(now.timestamp()),
        "exp": int((now + ttl).timestamp()),
        "iss": s.service_name,
    }
    return jwt.encode(payload, s.jwt_secret, algorithm=s.jwt_algorithm)


def create_access_token(sub: str, role: str, email: str | None = None) -> str:
    s = get_settings()
    return _encode(sub, role, email, "access", dt.timedelta(minutes=s.jwt_access_ttl_minutes))


def create_refresh_token(sub: str, role: str, email: str | None = None) -> str:
    s = get_settings()
    return _encode(sub, role, email, "refresh", dt.timedelta(days=s.jwt_refresh_ttl_days))


def decode_token(token: str, expected_type: str | None = None) -> TokenData:
    s = get_settings()
    try:
        payload = jwt.decode(
            token, s.jwt_secret, algorithms=[s.jwt_algorithm], options={"require": ["exp", "sub"]}
        )
    except jwt.ExpiredSignatureError as exc:
        raise AuthenticationError("token expired") from exc
    except jwt.PyJWTError as exc:
        raise AuthenticationError("invalid token") from exc
    if expected_type and payload.get("type") != expected_type:
        raise AuthenticationError(f"expected a {expected_type} token")
    return TokenData(
        sub=payload["sub"],
        role=payload.get("role", "recruiter"),
        type=payload.get("type", "access"),
        email=payload.get("email"),
    )


# --------------------------------------------------------------------------
# API keys
# --------------------------------------------------------------------------


def generate_api_key() -> tuple[str, str, str]:
    """Return (plaintext, sha256_hash, last4). Show plaintext once, store hash."""
    raw = API_KEY_PREFIX + secrets.token_urlsafe(32)
    return raw, hash_api_key(raw), raw[-4:]


def hash_api_key(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def verify_api_key(raw: str, stored_hash: str) -> bool:
    return hmac.compare_digest(hash_api_key(raw), stored_hash)
