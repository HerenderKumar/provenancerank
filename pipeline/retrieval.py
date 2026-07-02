"""Hybrid retrieval: BM25 + dense cosine, fused with reciprocal rank fusion.

Neither half is enough on its own. BM25 nails exact jargon; cosine catches the
paraphrases ("recommendation engine" ~ "recommender system"). RRF lets them
vote without having to put their very different score scales on the same axis.
All vectorised — a JD query over 100K candidates is a fraction of a second.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def cosine_scores(jd_vec: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    """Cosine of the JD against every candidate. Vectors are normalised at
    embed time, but we re-normalise defensively (cheap, avoids NaNs)."""
    if embeddings.size == 0:
        return np.zeros(0, dtype=np.float32)
    jd = jd_vec.astype(np.float32).ravel()
    jn = np.linalg.norm(jd)
    if jn > 0:
        jd = jd / jn
    norms = np.linalg.norm(embeddings, axis=1)
    norms[norms == 0] = 1.0
    return (embeddings @ jd) / norms


def _ranks(scores: np.ndarray) -> np.ndarray:
    """Position of each item in descending-score order (0 = best)."""
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.int64)
    ranks[order] = np.arange(len(scores))
    return ranks


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def fuse_rrf(bm25: np.ndarray, cosine: np.ndarray, k: int = 60) -> np.ndarray:
    """RRF over the two score arrays -> normalised 0-1 retrieval score.

    A candidate only earns credit from a modality where its raw score is > 0,
    so the long tail of zero-match profiles doesn't pick up phantom rank credit
    from argsort tie ordering.
    """
    n = len(bm25)
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    rb, rc = _ranks(bm25), _ranks(cosine)
    contrib_bm = np.where(bm25 > 0, 1.0 / (k + 1 + rb), 0.0)
    contrib_cos = np.where(cosine > 0, 1.0 / (k + 1 + rc), 0.0)
    return _minmax(contrib_bm + contrib_cos).astype(np.float32)


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """Classic RRF over ranked id lists (kept for tests / readability)."""
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            scores[cid] += 1.0 / (k + rank + 1)
    return dict(scores)
