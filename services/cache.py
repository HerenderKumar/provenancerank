"""Cache abstraction with a Redis backend and an in-memory fallback.

The API never imports redis directly — it asks for a cache and gets whichever
backend is reachable. If Redis is down or not configured we transparently fall
back to an in-process dict so dev and tests don't need a running Redis, and a
Redis blip in prod degrades to local caching instead of erroring (the
availability-over-consistency call from ARCHITECTURE.md).
"""

from __future__ import annotations

import json
import time
from typing import Any, Protocol

from core.logging import get_logger

log = get_logger("services.cache")


class Cache(Protocol):
    async def get(self, key: str) -> Any | None: ...
    async def set(self, key: str, value: Any, ttl: int | None = None) -> None: ...
    async def incr(self, key: str, ttl: int) -> int: ...
    async def ping(self) -> bool: ...
    async def close(self) -> None: ...


class InMemoryCache:
    """Bounded in-process cache with TTL and a pluggable eviction policy.

    Redis (prod) does eviction for us (the compose file sets
    ``allkeys-lru``); this class mirrors that locally so the fallback behaves
    the same and stays memory-safe. Policies:
      * ``lru`` — evict least-recently-used (default; matches Redis allkeys-lru)
      * ``lfu`` — evict least-frequently-used (good when some keys are hot)
      * ``fifo`` — evict oldest inserted
    """

    backend = "memory"

    def __init__(self, capacity: int = 10_000, policy: str = "lru") -> None:
        from collections import OrderedDict

        self.capacity = capacity
        self.policy = policy
        self._data: OrderedDict[str, tuple[Any, float | None]] = OrderedDict()
        self._freq: dict[str, int] = {}
        self._counters: dict[str, tuple[int, float]] = {}
        self.evictions = 0

    async def get(self, key: str) -> Any | None:
        item = self._data.get(key)
        if not item:
            return None
        value, expire = item
        if expire is not None and expire < time.time():
            self._drop(key)
            return None
        if self.policy == "lru":
            self._data.move_to_end(key)
        self._freq[key] = self._freq.get(key, 0) + 1
        return value

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = (value, time.time() + ttl if ttl else None)
        self._freq[key] = self._freq.get(key, 0) + 1
        self._evict_if_needed()

    def _evict_if_needed(self) -> None:
        while len(self._data) > self.capacity:
            if self.policy == "lfu":
                victim = min(self._data, key=lambda k: self._freq.get(k, 0))
            else:  # lru / fifo both evict from the front of the OrderedDict
                victim = next(iter(self._data))
            self._drop(victim)
            self.evictions += 1

    def _drop(self, key: str) -> None:
        self._data.pop(key, None)
        self._freq.pop(key, None)

    async def incr(self, key: str, ttl: int) -> int:
        count, expire = self._counters.get(key, (0, 0.0))
        now = time.time()
        if expire < now:
            count, expire = 0, now + ttl
        count += 1
        self._counters[key] = (count, expire)
        return count

    async def ping(self) -> bool:
        return True

    async def close(self) -> None:
        self._data.clear()
        self._freq.clear()
        self._counters.clear()


class RedisCache:
    backend = "redis"

    def __init__(self, client: Any) -> None:
        self._r = client

    async def get(self, key: str) -> Any | None:
        raw = await self._r.get(key)
        return json.loads(raw) if raw is not None else None

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        await self._r.set(key, json.dumps(value, default=str), ex=ttl)

    async def incr(self, key: str, ttl: int) -> int:
        # atomic increment + set TTL on first hit, via pipeline
        async with self._r.pipeline(transaction=True) as pipe:
            await pipe.incr(key)
            await pipe.expire(key, ttl, nx=True)
            count, _ = await pipe.execute()
        return int(count)

    async def ping(self) -> bool:
        return bool(await self._r.ping())

    async def close(self) -> None:
        await self._r.aclose()


async def make_cache(redis_url: str, enabled: bool = True) -> Cache:
    """Return a Redis-backed cache if we can reach it, else in-memory."""
    if not enabled:
        log.info("cache.disabled_memory")
        return InMemoryCache()
    try:
        import redis.asyncio as aioredis

        client = aioredis.from_url(
            redis_url, encoding="utf-8", decode_responses=True, socket_connect_timeout=2
        )
        if await client.ping():
            log.info("cache.redis_connected", url=redis_url.split("@")[-1])
            return RedisCache(client)
    except Exception as exc:
        log.warning("cache.redis_unavailable_memory_fallback", reason=str(exc)[:140])
    return InMemoryCache()
