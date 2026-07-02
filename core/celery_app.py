"""Celery app for the ingestion + intelligence workers.

Two queues so a GitHub rate-limit stall can't starve summarisation (bulkhead):
  - ingestion    : GitHub/SO fetch
  - intelligence : summarise + extract + index

Default is eager mode — tasks run inline in the calling process, no broker or
worker needed. That keeps dev + tests simple and means the API can fire a task
without a Celery cluster behind it. In prod you set CELERY_TASK_ALWAYS_EAGER=
false and run the worker/beat containers from docker-compose.
"""

from __future__ import annotations

from celery import Celery

from core.config import get_settings

s = get_settings()

celery_app = Celery(
    "provenancerank",
    broker=s.celery_broker_url,
    backend=s.celery_result_backend,
    include=["ingestion.tasks"],
)

celery_app.conf.update(
    task_always_eager=s.celery_task_always_eager,
    task_eager_propagates=True,
    task_acks_late=True,  # only ack after the task body succeeds
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,  # fair dispatch for long tasks
    task_default_queue="ingestion",
    task_routes={
        "ingestion.tasks.summarise_and_index": {"queue": "intelligence"},
        "ingestion.tasks.sync_developer_github": {"queue": "ingestion"},
        "ingestion.tasks.sync_all_developers": {"queue": "ingestion"},
        "ingestion.tasks.scan_dead_letters": {"queue": "ingestion"},
    },
    beat_schedule={
        "sync-all-developers": {
            "task": "ingestion.tasks.sync_all_developers",
            "schedule": s.sync_interval_hours * 3600.0,
        },
        "scan-dead-letters": {
            "task": "ingestion.tasks.scan_dead_letters",
            "schedule": 3600.0,  # hourly
        },
    },
    result_expires=3600,
)
