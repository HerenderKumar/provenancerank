"""Offline metrics for the evidence-graph search — so "more precise" is measured,
not asserted.

We don't have production ground truth, so (like the ranking side) we evaluate
against a small **gold set**: a list of natural-language queries each paired with
the developer ids that genuinely match. With that we can A/B any change —
skill-only vs. hybrid graph+vector retrieval, a new query parser, a different
confidence formula — and see whether precision/recall actually moved.

Metrics, all standard for retrieval:
  * precision@k — of the top k we returned, what fraction are relevant
  * recall@k    — of all relevant developers, what fraction we surfaced in top k
  * MRR         — 1 / rank of the first relevant hit (how high the first good one is)
  * hit@k       — did at least one relevant developer appear in the top k
"""

from __future__ import annotations

from collections.abc import Sequence


def precision_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    top = retrieved[:k]
    if not top:
        return 0.0
    return sum(1 for r in top if r in relevant) / len(top)


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top = set(retrieved[:k])
    return len(top & relevant) / len(relevant)


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    for i, r in enumerate(retrieved, start=1):
        if r in relevant:
            return 1.0 / i
    return 0.0


def hit_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    return 1.0 if set(retrieved[:k]) & relevant else 0.0


def evaluate_query(
    retrieved: Sequence[str], relevant: set[str], ks: Sequence[int] = (1, 3, 5, 10)
) -> dict[str, float]:
    """Metrics for one query's ranked developer ids against its relevant set."""
    out: dict[str, float] = {"mrr": round(reciprocal_rank(retrieved, relevant), 4)}
    for k in ks:
        out[f"precision@{k}"] = round(precision_at_k(retrieved, relevant, k), 4)
        out[f"recall@{k}"] = round(recall_at_k(retrieved, relevant, k), 4)
        out[f"hit@{k}"] = round(hit_at_k(retrieved, relevant, k), 4)
    return out


def aggregate(per_query: list[dict[str, float]]) -> dict[str, float]:
    """Mean of each metric across queries (the headline numbers)."""
    if not per_query:
        return {}
    keys = per_query[0].keys()
    return {k: round(sum(q[k] for q in per_query) / len(per_query), 4) for k in keys}


def evaluate_run(
    results_by_query: dict[str, Sequence[str]],
    gold: dict[str, set[str]],
    ks: Sequence[int] = (1, 3, 5, 10),
) -> dict[str, float]:
    """Aggregate metrics for one retrieval strategy over the whole gold set.

    ``results_by_query`` maps query -> ranked developer ids; ``gold`` maps the
    same query -> the set of relevant developer ids.
    """
    per_query = [
        evaluate_query(results_by_query.get(q, []), rel, ks) for q, rel in gold.items()
    ]
    return aggregate(per_query)


def compare(runs: dict[str, dict[str, Sequence[str]]], gold: dict[str, set[str]]) -> dict[str, dict]:
    """Evaluate several named strategies against the same gold set (for A/B)."""
    return {name: evaluate_run(results, gold) for name, results in runs.items()}


def lift(baseline: dict, candidate: dict, key: str = "recall@5") -> float:
    """Relative improvement of candidate over baseline on one metric (percent)."""
    b = baseline.get(key, 0.0)
    return round(100.0 * (candidate.get(key, 0.0) - b) / b, 2) if b else 0.0
