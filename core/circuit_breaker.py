"""A small async circuit breaker.

Wrap calls to a flaky dependency (Neo4j, GitHub, Gemini). After N consecutive
failures the breaker opens and short-circuits further calls for a cooldown —
the caller gets a fast CircuitOpen instead of piling up on a dead service. After
the cooldown it goes half-open and lets one probe through; success closes it,
failure re-opens it. This is what stops one broken service from cascading into
the rest of the system.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

from core.exceptions import ServiceUnavailableError
from core.logging import get_logger

log = get_logger("core.circuit")
T = TypeVar("T")

CLOSED, OPEN, HALF_OPEN = "closed", "open", "half_open"


class CircuitOpen(ServiceUnavailableError):
    code = "circuit_open"


class CircuitBreaker:
    def __init__(self, name: str, fail_max: int = 5, reset_timeout: float = 30.0):
        self.name = name
        self.fail_max = fail_max
        self.reset_timeout = reset_timeout
        self.state = CLOSED
        self.failures = 0
        self.opened_at = 0.0

    def _can_attempt(self) -> bool:
        if self.state == OPEN:
            if time.monotonic() - self.opened_at >= self.reset_timeout:
                self.state = HALF_OPEN
                return True
            return False
        return True

    def _on_success(self) -> None:
        self.failures = 0
        if self.state != CLOSED:
            log.info("circuit.closed", name=self.name)
        self.state = CLOSED

    def _on_failure(self) -> None:
        self.failures += 1
        if self.failures >= self.fail_max or self.state == HALF_OPEN:
            if self.state != OPEN:
                log.warning("circuit.opened", name=self.name, failures=self.failures)
            self.state = OPEN
            self.opened_at = time.monotonic()

    async def call(self, fn: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
        if not self._can_attempt():
            raise CircuitOpen(f"{self.name} circuit is open")
        try:
            result = await fn(*args, **kwargs)
        except Exception:
            self._on_failure()
            raise
        self._on_success()
        return result

    @property
    def healthy(self) -> bool:
        return self.state != OPEN
