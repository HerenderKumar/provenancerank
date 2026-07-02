"""API integration tests against a temp SQLite DB + in-memory cache.

Ranking is exercised through the LIVE path (candidates posted in the request) so
these pass in CI without precomputed artifacts. The full fast-path is covered by
the engine tests + the live run in CI.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

DEV = Path(__file__).resolve().parent.parent / "data" / "dev_sample.jsonl"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'api.db'}")
    monkeypatch.setenv("BOOTSTRAP_ADMIN_EMAIL", "admin@test.local")
    monkeypatch.setenv("BOOTSTRAP_ADMIN_PASSWORD", "adminpass123")
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "1000")
    monkeypatch.setenv("CACHE_ENABLED", "false")

    from core.config import get_settings

    get_settings.cache_clear()
    import db.base as dbb

    dbb._engine = None
    dbb._sessionmaker = None

    from fastapi.testclient import TestClient

    from api.app import create_app

    with TestClient(create_app()) as c:
        yield c
    get_settings.cache_clear()


def _admin_headers(client):
    r = client.post("/auth/login", json={"email": "admin@test.local", "password": "adminpass123"})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def _load_sample(n=150):
    rows = []
    with open(DEV) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            rows.append(json.loads(line))
    return rows


def test_health_and_ready(client):
    assert client.get("/health").status_code == 200
    assert client.get("/health/ready").status_code in (200, 503)


def test_login_me_and_apikey(client):
    h = _admin_headers(client)
    me = client.get("/auth/me", headers=h).json()
    assert me["role"] == "admin"
    created = client.post("/auth/api-keys", json={"name": "k"}, headers=h)
    assert created.status_code == 201
    assert created.json()["api_key"].startswith("pr_")
    assert any(k["name"] == "k" for k in client.get("/auth/api-keys", headers=h).json())


def test_no_auth_is_rejected(client):
    assert client.post("/rank", json={"top_k": 5}).status_code == 401


def test_rbac_recruiter_cannot_register(client):
    h = _admin_headers(client)
    client.post(
        "/auth/register",
        json={"email": "r@test.local", "password": "recpass123", "role": "recruiter"},
        headers=h,
    )
    rt = client.post("/auth/login", json={"email": "r@test.local", "password": "recpass123"})
    rh = {"Authorization": f"Bearer {rt.json()['access_token']}"}
    resp = client.post(
        "/auth/register", json={"email": "x@test.local", "password": "xpass1234"}, headers=rh
    )
    assert resp.status_code == 403


@pytest.mark.skipif(not DEV.exists(), reason="dev_sample.jsonl not built")
def test_rank_job_live_path(client):
    h = _admin_headers(client)
    body = {"candidates": _load_sample(150), "top_k": 100}
    r = client.post("/rank", json=body, headers=h)
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    for _ in range(100):
        st = client.get(f"/rank/{job_id}", headers=h).json()
        if st["status"] in ("succeeded", "failed"):
            break
        time.sleep(0.2)
    assert st["status"] == "succeeded", st.get("error")
    res = client.get(f"/rank/{job_id}/results", headers=h).json()
    assert res["count"] == 100
    scores = [r["score"] for r in res["results"]]
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
    csv = client.get(f"/rank/{job_id}/submission.csv", headers=h)
    assert csv.text.splitlines()[0] == "candidate_id,rank,score,reasoning"


def test_metrics_exposed(client):
    assert "http_requests_total" in client.get("/metrics").text
