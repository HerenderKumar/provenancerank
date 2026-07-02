"""Cross-encoder reranker for the head of the ranking.

First-stage retrieval (BM25 + embeddings + GBDT) is recall-oriented: it's fast,
but it scores the query and each document *independently*, so it misses the
fine-grained interactions that decide the top of the list — "led" vs
"contributed to", "shipped a production LLM service" vs "took an LLM course". A
cross-encoder reads the JD and a profile together in a single transformer pass
and scores the pair. Far more precise, far too slow to run on 100K — so we only
rerank the top-K from stage one, which is exactly where NDCG@10/@50 is decided.

This runs *offline* in precompute: the rerank score is cached onto the feature
matrix and blended by the scorer, so rank.py stays no-network and < 5 min.

Degradation, in order: a real sentence-transformers ``CrossEncoder`` -> a lexical
scorer (JD-term overlap with IDF-ish weighting and length damping). The lexical
path needs nothing but numpy, so the stage always contributes signal even on a
bare machine; the strong model plugs in via requirements.txt.
"""

from __future__ import annotations

import math
import re
from functools import lru_cache

import numpy as np

from core.logging import get_logger, log_duration

log = get_logger("ml.reranker")

_TOKEN_RE = re.compile(r"[a-z0-9+#.]+")
_STOP = frozenset(
    "the a an and or of to in for with on at by is are be as we you our your they "
    "this that from will can must have has it its their who whom able strong".split()
)


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP and len(t) > 1]


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < 1e-12:
        return np.zeros_like(x, dtype=float)
    return (x - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Lexical fallback — a BM25-flavoured overlap of JD terms with each profile,
# scored only over the (small) head set so IDF reflects what's actually in play.
# ---------------------------------------------------------------------------

def _lexical_scores(jd_text: str, docs: list[str]) -> np.ndarray:
    jd_terms = set(_tokens(jd_text))
    if not jd_terms or not docs:
        return np.zeros(len(docs), dtype=float)

    tokenised = [_tokens(d) for d in docs]
    n = len(docs)
    # document frequency of each JD term within the head set
    df: dict[str, int] = dict.fromkeys(jd_terms, 0)
    for toks in tokenised:
        present = jd_terms.intersection(toks)
        for term in present:
            df[term] += 1
    idf = {t: math.log(1.0 + n / (1 + df[t])) for t in jd_terms}

    lengths = np.array([len(t) or 1 for t in tokenised], dtype=float)
    avg_len = float(lengths.mean()) or 1.0
    k1, b = 1.5, 0.75  # standard BM25 knobs

    scores = np.zeros(n, dtype=float)
    for i, toks in enumerate(tokenised):
        if not toks:
            continue
        counts: dict[str, int] = {}
        for tok in toks:
            if tok in jd_terms:
                counts[tok] = counts.get(tok, 0) + 1
        denom_norm = k1 * (1 - b + b * lengths[i] / avg_len)
        scores[i] = sum(
            idf[t] * (tf * (k1 + 1)) / (tf + denom_norm) for t, tf in counts.items()
        )
    return _minmax(scores)


# ---------------------------------------------------------------------------
# Cross-encoder (real model). Cached so we pay the load once per process.
# ---------------------------------------------------------------------------

@lru_cache(maxsize=2)
def _load_cross_encoder(model_name: str):
    """Return a loaded CrossEncoder, or None if unavailable (no package / no
    weights / offline). Cached because construction downloads + warms the model.
    Runs on the best device, optionally via ONNX/int8 — both fall back to plain
    torch, and only a genuine load failure returns None (-> lexical floor)."""
    try:
        from sentence_transformers import CrossEncoder

        from core.device import construct

        model, runtime = construct(CrossEncoder, model_name, max_length=512)
        log.info("reranker.cross_encoder_loaded", model=model_name, runtime=runtime)
        return model
    except Exception as exc:
        log.warning("reranker.cross_encoder_unavailable", reason=str(exc)[:140])
        return None


def _cross_encoder_scores(jd_text: str, docs: list[str], model_name: str) -> np.ndarray | None:
    model = _load_cross_encoder(model_name)
    if model is None:
        return None
    try:
        pairs = [(jd_text, d) for d in docs]
        raw = np.asarray(model.predict(pairs, batch_size=64), dtype=float)
        # logits -> bounded -> head-relative [0,1]
        return _minmax(1.0 / (1.0 + np.exp(-raw)))
    except Exception as exc:
        log.warning("reranker.predict_failed_fallback_lexical", reason=str(exc)[:140])
        return None


def rerank(
    jd_text: str,
    items: list[tuple[str, str]],
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> dict[str, float]:
    """Score the head set against the JD and return ``{candidate_id: score}`` in
    [0,1], normalised within the head. ``items`` is ``[(candidate_id, doc_text)]``.

    Tries the cross-encoder first; on any failure (or absence) falls back to the
    lexical scorer. Either way every input id gets a score, so the caller can
    cache a dense ``rerank_score`` column.
    """
    if not items:
        return {}
    ids = [cid for cid, _ in items]
    docs = [doc for _, doc in items]

    with log_duration(log, "reranker.run") as m:
        scores = _cross_encoder_scores(jd_text, docs, model_name)
        backend = "cross-encoder"
        if scores is None:
            scores = _lexical_scores(jd_text, docs)
            backend = "lexical"
        m["backend"] = backend
        m["items"] = len(items)
    return {cid: float(s) for cid, s in zip(ids, scores, strict=False)}
