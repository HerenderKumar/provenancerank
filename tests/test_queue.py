"""Queue + worker + cache-eviction behaviour."""

from __future__ import annotations

import asyncio
import time

from services.cache import InMemoryCache
from services.queue import InMemoryQueue, QueueWorker


async def test_priority_then_fifo():
    q = InMemoryQueue()
    await q.enqueue({"n": 1}, priority=5)
    await q.enqueue({"n": 2}, priority=1)  # higher priority
    await q.enqueue({"n": 3}, priority=5)
    order = [(await q.dequeue()).payload["n"] for _ in range(3)]
    assert order == [2, 1, 3]  # priority 1 first, then FIFO among the 5s


async def test_dedup_drops_duplicate():
    q = InMemoryQueue()
    assert await q.enqueue({"x": 1}, dedup_key="same") is not None
    assert await q.enqueue({"x": 1}, dedup_key="same") is None  # duplicate dropped


async def test_poison_message_goes_to_dlq():
    q = InMemoryQueue()
    seen = []

    async def always_fails(msg):
        seen.append(msg.id)
        raise RuntimeError("boom")

    worker = QueueWorker(q, always_fails, concurrency=1)
    worker.start()
    await q.enqueue({"job": "bad"}, max_attempts=3)
    for _ in range(50):
        if len(q.dlq):
            break
        await asyncio.sleep(0.05)
    await worker.stop()
    assert q.stats()["dead_letter"] == 1
    assert len(seen) == 3  # tried max_attempts times before dead-lettering


async def test_worker_processes_success():
    q = InMemoryQueue()
    done = []

    async def handler(msg):
        done.append(msg.payload["v"])

    worker = QueueWorker(q, handler, concurrency=2)
    worker.start()
    for v in range(5):
        await q.enqueue({"v": v})
    for _ in range(50):
        if len(done) == 5:
            break
        await asyncio.sleep(0.05)
    await worker.stop()
    assert sorted(done) == [0, 1, 2, 3, 4]


async def test_cache_lru_eviction():
    c = InMemoryCache(capacity=3, policy="lru")
    for i in range(3):
        await c.set(f"k{i}", i)
    await c.get("k0")  # touch k0 so k1 becomes LRU
    await c.set("k3", 3)  # over capacity -> evict k1
    assert await c.get("k1") is None
    assert await c.get("k0") == 0
    assert await c.get("k3") == 3
    assert c.evictions == 1


async def test_cache_ttl_expiry():
    c = InMemoryCache()
    await c.set("temp", "v", ttl=60)
    # force the stored expiry into the past, then it should read as a miss
    val, _ = c._data["temp"]
    c._data["temp"] = (val, time.time() - 1)
    assert await c.get("temp") is None
