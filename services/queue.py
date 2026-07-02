"""Job message queue.

Decouples "a request came in" from "the ranking ran". The API enqueues a small
message and returns immediately (async processing); a worker pulls messages and
runs them. That's the seam where you'd later move execution onto a separate
worker fleet without touching the API.

Patterns made concrete here:
  * FIFO within a priority, lower number = higher priority (priority queue)
  * push (enqueue) vs pull (worker dequeues when it has capacity)
  * at-least-once + retries; after max_attempts a message is poison and goes to
    a dead-letter queue instead of looping forever
  * idempotent consumers (the handler checks job state) so a duplicate delivery
    doesn't double-process

Two backends: an in-process asyncio queue (dev/single-node) and Redis Streams
with consumer groups (prod, multi-node, survives restarts). Same interface.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field

from core.logging import get_logger

log = get_logger("services.queue")


@dataclass
class Message:
    payload: dict
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    priority: int = 5
    attempts: int = 0
    max_attempts: int = 3
    enqueued_at: float = field(default_factory=time.time)


@dataclass(order=True)
class _Prioritized:
    priority: int
    seq: int
    message: Message = field(compare=False)


class InMemoryQueue:
    name = "memory"

    def __init__(self) -> None:
        self._pq: asyncio.PriorityQueue[_Prioritized] = asyncio.PriorityQueue()
        self._seq = 0
        self.dlq: list[Message] = []
        self._seen: dict[str, float] = {}
        self.enqueued = 0

    async def enqueue(
        self,
        payload: dict,
        *,
        priority: int = 5,
        max_attempts: int = 3,
        dedup_key: str | None = None,
        dedup_ttl: int = 300,
    ) -> str | None:
        now = time.time()
        if dedup_key:
            # drop duplicates seen within the window (at-least-once safety net)
            self._seen = {k: t for k, t in self._seen.items() if now - t < dedup_ttl}
            if dedup_key in self._seen:
                log.info("queue.duplicate_dropped", dedup_key=dedup_key)
                return None
            self._seen[dedup_key] = now
        msg = Message(payload=payload, priority=priority, max_attempts=max_attempts)
        self._seq += 1
        await self._pq.put(_Prioritized(priority, self._seq, msg))
        self.enqueued += 1
        return msg.id

    async def dequeue(self, timeout: float | None = 1.0) -> Message | None:
        try:
            item = (
                await asyncio.wait_for(self._pq.get(), timeout) if timeout else await self._pq.get()
            )
            return item.message
        except asyncio.TimeoutError:
            return None

    async def retry(self, msg: Message) -> bool:
        """Requeue with backoff position; returns False if it went to the DLQ."""
        msg.attempts += 1
        if msg.attempts >= msg.max_attempts:
            self.dlq.append(msg)
            log.warning("queue.dead_lettered", msg_id=msg.id, attempts=msg.attempts)
            return False
        self._seq += 1
        await self._pq.put(_Prioritized(msg.priority, self._seq, msg))
        return True

    def stats(self) -> dict:
        return {
            "backend": self.name,
            "depth": self._pq.qsize(),
            "dead_letter": len(self.dlq),
            "enqueued_total": self.enqueued,
        }


class QueueWorker:
    """Pulls messages and hands them to a handler with bounded concurrency.
    A handler exception triggers retry/DLQ; success just drops the message."""

    def __init__(self, queue: InMemoryQueue, handler, concurrency: int = 2):
        self.queue = queue
        self.handler = handler
        self._sem = asyncio.Semaphore(concurrency)
        self._running = False
        self._loop_task: asyncio.Task | None = None
        self._inflight: set[asyncio.Task] = set()

    def start(self) -> None:
        self._running = True
        self._loop_task = asyncio.create_task(self._loop())
        log.info("queue.worker_started")

    async def _loop(self) -> None:
        while self._running:
            msg = await self.queue.dequeue(timeout=0.5)
            if msg is None:
                continue
            await self._sem.acquire()
            task = asyncio.create_task(self._process(msg))
            self._inflight.add(task)
            task.add_done_callback(lambda t: (self._inflight.discard(t), self._sem.release()))

    async def _process(self, msg: Message) -> None:
        try:
            await self.handler(msg)
        except Exception as exc:
            log.warning("queue.handler_error", msg_id=msg.id, error=str(exc)[:160])
            await self.queue.retry(msg)

    async def stop(self) -> None:
        self._running = False
        if self._loop_task:
            self._loop_task.cancel()
        if self._inflight:
            await asyncio.gather(*self._inflight, return_exceptions=True)
        log.info("queue.worker_stopped")


# Redis Streams backend — the multi-node, restart-surviving version of the above.


class RedisStreamQueue:
    """At-least-once delivery via a Redis Stream + consumer group. Messages that
    exceed max_attempts are XADDed to a dead-letter stream. Survives restarts
    and fans out across worker nodes — the multi-node version of the above."""

    name = "redis-streams"

    def __init__(
        self, client, stream: str = "pr:jobs", group: str = "rankers", consumer: str | None = None
    ):
        self._r = client
        self.stream = stream
        self.group = group
        self.consumer = consumer or f"worker-{uuid.uuid4().hex[:8]}"
        self.dlq_stream = f"{stream}:dlq"

    async def ensure_group(self) -> None:
        try:
            await self._r.xgroup_create(self.stream, self.group, id="0", mkstream=True)
        except Exception:
            pass  # group already exists

    async def enqueue(
        self,
        payload: dict,
        *,
        priority: int = 5,
        max_attempts: int = 3,
        dedup_key: str | None = None,
        dedup_ttl: int = 300,
    ) -> str | None:
        import json

        if dedup_key:
            ok = await self._r.set(f"dedup:{dedup_key}", "1", nx=True, ex=dedup_ttl)
            if not ok:
                return None
        return await self._r.xadd(
            self.stream,
            {
                "payload": json.dumps(payload),
                "priority": priority,
                "max_attempts": max_attempts,
                "attempts": 0,
            },
        )

    async def dequeue(self, timeout: float | None = 1.0) -> Message | None:
        import json

        block = int((timeout or 1) * 1000)
        resp = await self._r.xreadgroup(
            self.group, self.consumer, {self.stream: ">"}, count=1, block=block
        )
        if not resp:
            return None
        _stream, entries = resp[0]
        msg_id, fields = entries[0]
        return Message(
            payload=json.loads(fields["payload"]),
            id=msg_id,
            priority=int(fields.get("priority", 5)),
            attempts=int(fields.get("attempts", 0)),
            max_attempts=int(fields.get("max_attempts", 3)),
        )

    async def ack(self, msg: Message) -> None:
        await self._r.xack(self.stream, self.group, msg.id)
        await self._r.xdel(self.stream, msg.id)

    async def retry(self, msg: Message) -> bool:
        import json

        await self._r.xack(self.stream, self.group, msg.id)
        msg.attempts += 1
        if msg.attempts >= msg.max_attempts:
            await self._r.xadd(
                self.dlq_stream, {"payload": json.dumps(msg.payload), "attempts": msg.attempts}
            )
            return False
        await self._r.xadd(
            self.stream,
            {
                "payload": json.dumps(msg.payload),
                "priority": msg.priority,
                "max_attempts": msg.max_attempts,
                "attempts": msg.attempts,
            },
        )
        return True
