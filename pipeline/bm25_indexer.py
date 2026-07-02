"""BM25 keyword index over candidate text.

BM25 is the half of retrieval that catches exact technical terms the embeddings
smear together — "Qdrant", "NDCG", "LoRA", "OpenSearch". Uses rank-bm25 when
installed; otherwise a small inverted-index BM25 in numpy that scores only the
query terms (cheaper at query time and what we actually rely on in the sandbox).
Either object pickles to artifacts/bm25_index.pkl.

The inverted index builds in two streaming passes (document frequencies first,
then postings only for terms that survive the high-DF cut) so it never holds the
whole tokenized corpus in memory — that matters at 100K docs on a 16GB box.
"""

from __future__ import annotations

import math
import pickle
import re
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from core.logging import get_logger, log_duration

log = get_logger("pipeline.bm25_indexer")

_TOKEN = re.compile(r"[a-z0-9+#./-]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class InvertedBM25:
    """Okapi BM25 with an inverted index. Scoring only touches query terms, so a
    single JD query over 100K docs stays well under a second.

    Tokens appearing in more than ``max_df_ratio`` of docs are dropped (≈0 idf,
    pure bloat) and each doc is capped at ``max_tokens``.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n_docs = 0
        self.doc_len = np.zeros(0, dtype=np.float32)
        self.avgdl = 0.0
        self.postings: dict[str, list[tuple[int, int]]] = {}
        self.idf: dict[str, float] = {}

    @classmethod
    def from_corpus(
        cls,
        corpus: Sequence[str],
        tok: Callable[[str], list[str]] = tokenize,
        k1: float = 1.5,
        b: float = 0.75,
        max_df_ratio: float = 0.5,
        max_tokens: int = 200,
    ) -> InvertedBM25:
        self = cls(k1, b)
        n = len(corpus)
        self.n_docs = n
        self.doc_len = np.zeros(n, dtype=np.float32)

        # pass 1 — document frequencies + lengths (no postings held yet)
        df: dict[str, int] = defaultdict(int)
        for i, text in enumerate(corpus):
            toks = tok(text)[:max_tokens] if max_tokens else tok(text)
            self.doc_len[i] = len(toks)
            for t in set(toks):
                df[t] += 1
        self.avgdl = float(self.doc_len.mean()) if n else 0.0
        cutoff = max_df_ratio * n
        keep = {t for t, c in df.items() if c <= cutoff}

        # pass 2 — postings, but only for the terms we kept
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, text in enumerate(corpus):
            toks = tok(text)[:max_tokens] if max_tokens else tok(text)
            tf: dict[str, int] = defaultdict(int)
            for t in toks:
                if t in keep:
                    tf[t] += 1
            for t, f in tf.items():
                postings[t].append((i, f))
        self.postings = dict(postings)
        self.idf = {t: math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5)) for t in keep}
        return self

    def get_scores(self, query_tokens: Sequence[str]) -> np.ndarray:
        scores = np.zeros(self.n_docs, dtype=np.float32)
        denom_len = self.k1 * (1 - self.b + self.b * self.doc_len / (self.avgdl or 1))
        for t in set(query_tokens):
            posting = self.postings.get(t)
            if not posting:
                continue
            idf = self.idf.get(t, 0.0)
            for doc_idx, f in posting:
                scores[doc_idx] += idf * (f * (self.k1 + 1)) / (f + denom_len[doc_idx])
        return scores


def build_index(corpus: Sequence[str], backend: str = "auto"):
    """backend: "auto" (rank-bm25 if importable), "rank-bm25", or "inverted".

    The inverted backend is lighter on memory, which matters when indexing 100K
    docs on a 16GB box alongside the embeddings.
    """
    with log_duration(log, "bm25.build") as m:
        if backend in ("auto", "rank-bm25"):
            try:
                from rank_bm25 import BM25Okapi

                index = BM25Okapi([tokenize(t) for t in corpus])
                chosen = "rank-bm25"
            except Exception as exc:
                if backend == "rank-bm25":
                    log.warning("bm25.fallback", reason=str(exc)[:120])
                index = InvertedBM25.from_corpus(corpus)
                chosen = "inverted-numpy"
        else:
            index = InvertedBM25.from_corpus(corpus)
            chosen = "inverted-numpy"
        m["docs"] = len(corpus)
        m["backend"] = chosen
    return index


def query_scores(index, text: str) -> np.ndarray:
    return np.asarray(index.get_scores(tokenize(text)), dtype=np.float32)


def save_index(index, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_index(path: str | Path):
    with open(path, "rb") as f:
        return pickle.load(f)
