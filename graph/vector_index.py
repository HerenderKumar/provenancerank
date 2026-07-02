"""Semantic (vector) index over evidence summaries - the second half of hybrid
graph retrieval.

The structured graph query is exact: it finds developers whose evidence is tagged
with a *canonical* skill. That's precise but brittle - ask for "real-time event
streaming" and, if nobody's evidence is tagged with the canonical skill `kafka`,
you get nothing, even though someone's commit summary literally says "streaming
Kafka consumer for live events". This index fixes that: it embeds every evidence
summary and lets a free-text query match on *meaning* (or, on the hashing
fallback, on lexical overlap), catching the paraphrases the skill traversal misses.

The two are then fused (see natural_language_query.py) so you keep the precision
of exact-skill matches and gain the recall of semantic matches.

Embeddings reuse the same `make_embedder` as the main ranker: sentence-transformers
when installed, a deterministic NumPy hashing embedder otherwise - so this runs
offline and in tests with no model.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from core.logging import get_logger, log_duration

log = get_logger("graph.vector_index")


@lru_cache(maxsize=1)
def _embedder():
    # cached process-wide: building the model once is the expensive part
    from pipeline.embedder import make_embedder

    return make_embedder(prefer_st=True)


def _evidence_text(ev: dict) -> str:
    """What we embed per evidence: the summary, its source, and the skills it
    demonstrates - so both the prose and the tags contribute to a match."""
    parts = [
        str(ev.get("summary", "")),
        str(ev.get("source_type", "")),
        " ".join(ev.get("skills", []) or []),
    ]
    return " ".join(p for p in parts if p).strip()


class EvidenceVectorIndex:
    """An in-memory embedding index over evidence. Cheap to build for the graph
    sizes we deal with; rebuilt when the evidence count changes."""

    def __init__(self) -> None:
        self._matrix: np.ndarray | None = None
        self._meta: list[dict] = []
        self.size = 0

    def build(self, evidence: list[dict]) -> EvidenceVectorIndex:
        texts, meta = [], []
        for ev in evidence:
            text = _evidence_text(ev)
            if not text:
                continue
            texts.append(text)
            meta.append(ev)
        with log_duration(log, "vector_index.build") as m:
            if texts:
                self._matrix = np.asarray(_embedder().encode(texts), dtype=np.float32)
            else:
                self._matrix = None
            self._meta = meta
            self.size = len(meta)
            m["n"] = self.size
        return self

    def search(self, query: str, top_k: int = 50) -> list[dict]:
        """Top evidence by cosine similarity to the query (vectors are already
        L2-normalised, so a dot product is the cosine)."""
        if self._matrix is None or not query.strip():
            return []
        q = np.asarray(_embedder().encode([query]), dtype=np.float32)[0]
        sims = self._matrix @ q
        if sims.size == 0:
            return []
        k = min(top_k, sims.shape[0])
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [{"score": float(sims[i]), "evidence": self._meta[i]} for i in idx]

    def developer_ranking(self, query: str, top_k: int = 50) -> list[dict]:
        """Aggregate evidence hits to developer level. A developer's semantic
        score is their best-matching piece of evidence; we keep the top matches
        as the trail to show the recruiter."""
        by_dev: dict[str, dict] = {}
        for hit in self.search(query, top_k=top_k):
            ev = hit["evidence"]
            dev_id = ev.get("developer_id")
            if dev_id is None:
                continue
            cur = by_dev.get(dev_id)
            if cur is None:
                by_dev[dev_id] = {
                    "developer_id": dev_id,
                    "semantic_score": round(hit["score"], 4),
                    "evidence": [ev],
                }
            else:
                cur["semantic_score"] = round(max(cur["semantic_score"], hit["score"]), 4)
                if len(cur["evidence"]) < 3:
                    cur["evidence"].append(ev)
        return sorted(by_dev.values(), key=lambda r: r["semantic_score"], reverse=True)


# A tiny module-level cache so we don't re-embed the whole graph on every query.
# Keyed by evidence count, which is a cheap, good-enough staleness signal for the
# demo/in-memory backend (a new ingestion bumps the count -> rebuild).
_cache: tuple[int, EvidenceVectorIndex] | None = None


def get_index(evidence: list[dict]) -> EvidenceVectorIndex:
    global _cache
    if _cache is not None and _cache[0] == len(evidence):
        return _cache[1]
    index = EvidenceVectorIndex().build(evidence)
    _cache = (len(evidence), index)
    return index


def reset_index() -> None:
    """Test hook / call after a large ingestion to force a rebuild."""
    global _cache
    _cache = None
