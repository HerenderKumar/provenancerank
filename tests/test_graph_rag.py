"""Graph RAG: prove hybrid (skill + vector) retrieval beats skill-only, measured
on a gold set with the graph eval harness.

The point of the whole change is recall on *paraphrased* queries - questions whose
words don't map to a canonical skill tag, so the exact graph traversal returns
nothing, but whose meaning matches an evidence summary. We seed a small graph,
label which developers each query should return, and check the numbers move.

Runs offline: the vector index uses the NumPy hashing embedder (no model), so the
'semantic' match here is really lexical overlap - still enough to catch the
paraphrases. With sentence-transformers installed it becomes true semantic match.
"""

from __future__ import annotations

from datetime import datetime, timezone

from core.config import get_settings
from graph import evaluate as gev
from graph import vector_index
from graph.natural_language_query import natural_language_to_results
from graph.queries import GraphQuerier
from graph.schema import InMemoryGraphStore

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)

# developer -> canonical skill tag -> two evidence summaries
_SEED = [
    ("dev_k8s", "kubernetes", [
        "Operated kubernetes clusters and wrote helm charts for rollouts",
        "Migrated workloads onto kubernetes with zero downtime"]),
    ("dev_kafka", "kafka", [
        "Built a streaming consumer ingesting telemetry events in real time",
        "Scaled the streaming pipeline to a firehose of telemetry"]),
    ("dev_rust", "rust", [
        "Wrote a lock-free concurrent allocator in rust for low latency",
        "Cut tail latency with a lock-free ring buffer"]),
    ("dev_pg", "postgresql", [
        "Tuned postgres query plans and partitioned a huge events table",
        "Rewrote slow postgres joins and added covering indexes"]),
    ("dev_react", "react", [
        "Shipped a react dashboard with realtime websocket updates",
        "Built reusable react components and hooks"]),
]

# query -> the developer(s) that genuinely match it
GOLD = {
    # exact: the query maps to a canonical skill -> skill traversal finds it
    "kubernetes": {"dev_k8s"},
    "rust": {"dev_rust"},
    "postgresql": {"dev_pg"},
    # paraphrase: no canonical skill in the query -> only semantic match recalls it
    "telemetry event streaming": {"dev_kafka"},
    "lock-free low latency allocator": {"dev_rust"},
}


async def _seed() -> InMemoryGraphStore:
    store = InMemoryGraphStore()
    for dev, skill, summaries in _SEED:
        for i, summ in enumerate(summaries):
            await store.upsert_skill_with_evidence(
                developer_id=dev, skill_name=skill, artifact_id=f"{dev}_{i}",
                source_type="commit_diff", complexity=4, confidence=0.3,
                artifact_date=NOW, production_signal=True, summary=summ)
    return store


async def _ranked(query: str, querier: GraphQuerier) -> list[str]:
    res = await natural_language_to_results(query, querier)
    return [c["developer_id"] for c in res["candidates"]]


async def _run_all(querier: GraphQuerier) -> dict[str, list[str]]:
    return {q: await _ranked(q, querier) for q in GOLD}


async def test_hybrid_beats_skill_only_on_recall():
    store = await _seed()
    querier = GraphQuerier(store)
    s = get_settings()
    original = s.graph_hybrid_enabled
    try:
        vector_index.reset_index()
        s.graph_hybrid_enabled = False           # baseline: skill traversal only
        baseline = await _run_all(querier)

        vector_index.reset_index()
        s.graph_hybrid_enabled = True            # hybrid: + semantic vector search
        hybrid = await _run_all(querier)
    finally:
        s.graph_hybrid_enabled = original

    results = gev.compare({"baseline": baseline, "hybrid": hybrid}, GOLD)

    # hybrid never loses recall and wins overall (the paraphrases)
    assert results["hybrid"]["recall@5"] > results["baseline"]["recall@5"]
    assert results["hybrid"]["mrr"] >= results["baseline"]["mrr"]
    assert gev.lift(results["baseline"], results["hybrid"], "recall@5") > 0

    # the specific paraphrase the skill traversal cannot answer
    assert "dev_kafka" not in baseline["telemetry event streaming"]
    assert "dev_kafka" in hybrid["telemetry event streaming"]


async def test_exact_skill_queries_still_work_in_hybrid():
    store = await _seed()
    querier = GraphQuerier(store)
    vector_index.reset_index()
    res = await natural_language_to_results("kubernetes", querier)
    ids = [c["developer_id"] for c in res["candidates"]]
    assert ids and ids[0] == "dev_k8s"
    # the fused result is transparent about how it matched
    top = res["candidates"][0]
    assert top["match_via"] in {"skill", "both"}
    assert "retrieval" in res["query_interpreted"]


# --------------------------------------------------------------------------
# eval-harness unit tests
# --------------------------------------------------------------------------

def test_eval_metrics_math():
    retrieved = ["a", "x", "b", "y", "z"]
    relevant = {"a", "b", "c"}
    assert gev.precision_at_k(retrieved, relevant, 5) == 2 / 5
    assert gev.recall_at_k(retrieved, relevant, 5) == 2 / 3
    assert gev.reciprocal_rank(retrieved, relevant) == 1.0  # 'a' is first
    assert gev.hit_at_k(retrieved, relevant, 1) == 1.0


def test_eval_run_and_lift():
    gold = {"q1": {"a"}, "q2": {"b"}}
    weak = {"q1": ["x", "a"], "q2": ["y", "z"]}      # a at rank 2, b missing
    strong = {"q1": ["a"], "q2": ["b"]}              # both first
    res = gev.compare({"weak": weak, "strong": strong}, gold)
    assert res["strong"]["mrr"] > res["weak"]["mrr"]
    assert gev.lift(res["weak"], res["strong"], "recall@5") > 0


# --------------------------------------------------------------------------
# vector index unit test
# --------------------------------------------------------------------------

def test_vector_index_ranks_relevant_evidence_first():
    vector_index.reset_index()
    evidence = [
        {"developer_id": "d1", "summary": "lock-free concurrent allocator low latency",
         "source_type": "commit_diff", "skills": ["rust"]},
        {"developer_id": "d2", "summary": "react dashboard websocket updates",
         "source_type": "commit_diff", "skills": ["react"]},
    ]
    idx = vector_index.EvidenceVectorIndex().build(evidence)
    ranked = idx.developer_ranking("low latency lock-free allocator", top_k=5)
    assert ranked and ranked[0]["developer_id"] == "d1"
    assert ranked[0]["semantic_score"] > 0
