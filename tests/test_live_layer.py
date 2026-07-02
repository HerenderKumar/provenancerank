"""Live ingestion layer: confidence formula, summariser, skill extraction,
graph queries, NL routing, GitHub parsing."""

from __future__ import annotations

import datetime as dt

from graph.natural_language_query import natural_language_to_results
from graph.queries import GraphQuerier
from graph.schema import InMemoryGraphStore
from ingestion.github_client import GitHubClient
from intelligence.confidence_scorer import EvidenceRecord, compute_confidence
from intelligence.skill_extractor import canonicalise, extract_skills
from intelligence.summariser import summarise_artifact

NOW = dt.datetime.now(dt.timezone.utc)


def test_evidence_beats_resume_claim():
    commits = [EvidenceRecord("commit_diff", "kubernetes", NOW, 5) for _ in range(10)]
    assert compute_confidence(commits, "kubernetes", NOW) >= 0.95
    resume = [EvidenceRecord("resume_claim", "kubernetes", NOW, 3)]
    assert compute_confidence(resume, "kubernetes", NOW) < 0.05


def test_recency_decay():
    old = [EvidenceRecord("commit_diff", "go", NOW - dt.timedelta(days=730), 5)]
    new = [EvidenceRecord("commit_diff", "go", NOW, 5)]
    assert compute_confidence(new, "go", NOW) > compute_confidence(old, "go", NOW)


def test_skill_extractor_canonicalises():
    assert canonicalise("k8s") == "kubernetes"
    assert canonicalise("postgres") == "postgresql"
    skills = extract_skills("Built a service in Go using Kubernetes and Postgres")
    assert "kubernetes" in skills and "postgresql" in skills


async def test_summariser_complexity_and_signals():
    hard = await summarise_artifact(
        "commit_diff",
        "Fix race condition causing deadlock in the distributed scheduler under load in "
        "production; added mutex and retry. Deployed to users at scale.",
    )
    assert hard.complexity >= 4
    assert hard.production_signal is True
    assert hard.problem_category == "debugging"
    assert hard.content_hash  # cryptographic anchor present

    trivial = await summarise_artifact("commit_diff", "Fix a typo in the README")
    assert trivial.complexity == 1
    assert trivial.production_signal is False


async def _seed_store() -> InMemoryGraphStore:
    store = InMemoryGraphStore()
    # two high-complexity production concurrency commits + one weaker
    for i, cx in enumerate([5, 5, 3]):
        await store.upsert_skill_with_evidence(
            developer_id="dev1", skill_name="concurrency", artifact_id=f"a{i}",
            source_type="commit_diff", complexity=cx, confidence=0.3, artifact_date=NOW,
            production_signal=True, summary=f"fix {i}")
    return store


async def test_graph_find_by_skill_and_debuggers():
    store = await _seed_store()
    by_skill = await store.find_by_skill("concurrency", min_confidence=0.0)
    assert by_skill and by_skill[0]["developer_id"] == "dev1"
    assert by_skill[0]["evidence_count"] == 3

    debuggers = await store.find_production_debuggers("concurrency", min_complexity=4)
    assert debuggers and debuggers[0]["developer_id"] == "dev1"  # 2 complexity-5 artifacts


async def test_nl_query_routes_to_production_debuggers():
    store = await _seed_store()
    querier = GraphQuerier(store)
    res = await natural_language_to_results(
        "who debugged a race condition in production", querier)
    assert res["query_interpreted"]["require_production_signal"] is True
    assert res["candidates"] and res["candidates"][0]["developer_id"] == "dev1"


def test_github_parse_from_payload():
    payload = {
        "user": {
            "login": "octocat", "name": "Octo",
            "repositories": {"nodes": [{
                "name": "infra",
                "defaultBranchRef": {"target": {"history": {"nodes": [
                    {"oid": "abc", "message": "fix deadlock", "additions": 40,
                     "deletions": 5, "committedDate": "2026-01-01T00:00:00Z", "url": "u"}]}}},
                "pullRequests": {"nodes": [
                    {"title": "Add retrieval", "body": "vector search", "state": "MERGED",
                     "createdAt": "2026-02-01T00:00:00Z", "mergedAt": None, "url": "p",
                     "reviews": {"nodes": [{"body": "lgtm"}]}}]},
                "issues": {"nodes": []},
            }]},
        }
    }
    sig = GitHubClient.parse("octocat", payload)
    assert len(sig.commits) == 1 and len(sig.pull_requests) == 1
    assert sig.commits[0].content_hash and sig.commits[0].repo == "infra"
