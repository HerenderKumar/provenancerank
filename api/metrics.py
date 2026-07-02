"""Prometheus metrics. Imported once; the middleware records HTTP stats and the
ranking service records domain stats. Exposed at GET /metrics.
"""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

HTTP_REQUESTS = Counter("http_requests_total", "HTTP requests", ["method", "path", "status"])
HTTP_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency",
    ["method", "path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
)
RANKING_REQUESTS = Counter("ranking_requests_total", "Ranking jobs run", ["source", "cache_hit"])
RANKING_LATENCY = Histogram(
    "ranking_latency_seconds",
    "Ranking compute latency",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120, 300),
)
GATE_EXCLUSIONS = Counter("gate_filter_exclusions_total", "Candidates excluded by gate", ["reason"])
HONEYPOTS = Counter("honeypot_detected_total", "Honeypots detected in a job")
CACHE_EVENTS = Counter("cache_events_total", "Cache hit/miss", ["result"])
JOBS_INFLIGHT = Gauge("ranking_jobs_inflight", "Ranking jobs currently running")


def record_ranking(source: str, cache_hit: bool, elapsed_ms: float, gate_stats: dict) -> None:
    RANKING_REQUESTS.labels(source=source, cache_hit=str(cache_hit).lower()).inc()
    RANKING_LATENCY.observe(elapsed_ms / 1000.0)
    HONEYPOTS.inc(int(gate_stats.get("honeypots", 0)))
    for reason, key in (
        ("all_consulting_career", "all_consulting_career"),
        ("keyword_stuffer", "keyword_stuffers"),
    ):
        if gate_stats.get(key):
            GATE_EXCLUSIONS.labels(reason=reason).inc(int(gate_stats[key]))


def render() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
