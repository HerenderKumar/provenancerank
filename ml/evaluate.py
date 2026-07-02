"""Offline ranking metrics — the same ones the hackathon grades on, so we can
measure every change instead of guessing.

    composite = 0.50*NDCG@10 + 0.30*NDCG@50 + 0.15*MAP + 0.05*P@10

Relevance is graded 0..5; "relevant" for MAP/P@k means tier >= 3 (the spec's
definition). We don't have the hidden ground truth, so we run this against the
pseudo-labels (ml/pseudo_labels.py) on a held-out slice — that's how we prove a
change actually helps rather than just feels better.
"""

from __future__ import annotations

import numpy as np

REL_THRESHOLD = 3  # tier 3+ counts as relevant (P@k, MAP)


def _gains(relevances: np.ndarray) -> np.ndarray:
    return (2.0**relevances) - 1.0


def dcg_at_k(relevances: np.ndarray, k: int) -> float:
    rel = np.asarray(relevances, dtype=float)[:k]
    if rel.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rel.size + 2))
    return float((_gains(rel) * discounts).sum())


def ndcg_at_k(ranked_relevances: np.ndarray, k: int) -> float:
    rel = np.asarray(ranked_relevances, dtype=float)
    ideal = np.sort(rel)[::-1]
    idcg = dcg_at_k(ideal, k)
    return dcg_at_k(rel, k) / idcg if idcg > 0 else 0.0


def average_precision(ranked_relevances: np.ndarray, threshold: int = REL_THRESHOLD) -> float:
    rel = (np.asarray(ranked_relevances) >= threshold).astype(float)
    if rel.sum() == 0:
        return 0.0
    hits = np.cumsum(rel)
    precision_at_i = hits / np.arange(1, rel.size + 1)
    return float((precision_at_i * rel).sum() / rel.sum())


def precision_at_k(ranked_relevances: np.ndarray, k: int, threshold: int = REL_THRESHOLD) -> float:
    rel = (np.asarray(ranked_relevances[:k]) >= threshold).astype(float)
    return float(rel.mean()) if rel.size else 0.0


def evaluate(ranked_ids: list[str], relevance: dict[str, float]) -> dict[str, float]:
    """Metrics for one ranked list given a {candidate_id: grade} label map."""
    rels = np.array([relevance.get(cid, 0.0) for cid in ranked_ids], dtype=float)
    ndcg10 = ndcg_at_k(rels, 10)
    ndcg50 = ndcg_at_k(rels, 50)
    mapv = average_precision(rels)
    p10 = precision_at_k(rels, 10)
    composite = 0.50 * ndcg10 + 0.30 * ndcg50 + 0.15 * mapv + 0.05 * p10
    return {
        "ndcg@10": round(ndcg10, 4),
        "ndcg@50": round(ndcg50, 4),
        "map": round(mapv, 4),
        "p@10": round(p10, 4),
        "composite": round(composite, 4),
    }


def compare(rankings: dict[str, list[str]], relevance: dict[str, float]) -> dict[str, dict]:
    """Evaluate several named rankings against the same labels (for A/B)."""
    return {name: evaluate(ids, relevance) for name, ids in rankings.items()}


def lift(baseline: dict, candidate: dict, key: str = "composite") -> float:
    """Relative improvement of candidate over baseline on a metric (%)."""
    b = baseline.get(key, 0.0)
    return round(100.0 * (candidate.get(key, 0.0) - b) / b, 2) if b else 0.0
