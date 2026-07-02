"""Fixed-window rate limiter backed by the cache (Redis incr, or in-memory).

Per-identity, per-minute. Cheap and good enough for an API edge; if you need
smoothing, the token-bucket variant lives one refactor away on the same incr
primitive.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from services.cache import Cache


@dataclass
class RateDecision:
    allowed: bool
    limit: int
    remaining: int
    retry_after: int


class RateLimiter:
    def __init__(self, cache: Cache, per_minute: int, enabled: bool = True):
        self.cache = cache
        self.per_minute = per_minute
        self.enabled = enabled

    async def check(self, identity: str) -> RateDecision:
        if not self.enabled:
            return RateDecision(True, self.per_minute, self.per_minute, 0)
        window = int(time.time() // 60)
        key = f"rl:{identity}:{window}"
        count = await self.cache.incr(key, ttl=60)
        remaining = max(self.per_minute - count, 0)
        allowed = count <= self.per_minute
        retry_after = 0 if allowed else 60 - int(time.time() % 60)
        return RateDecision(allowed, self.per_minute, remaining, retry_after)
