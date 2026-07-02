"""Candidate / JD embeddings.

Default backend is sentence-transformers (all-MiniLM-L6-v2, 384-dim, CPU). If
it can't be loaded - not installed, or no cached model and we're not allowed to
download - we fall back to a hashing embedder built on numpy alone. Same dim
either way, so the rest of the pipeline doesn't care which one ran.

The backend choice is written to artifacts/embedding_meta.json at precompute
time. rank.py reads it and rebuilds the *same* embedder so the JD query vector
lives in the same space as the stored candidate vectors - and it does this with
no network (a cached ST model loads fine offline; otherwise hashing).
"""

from __future__ import annotations

import json
import re
import zlib
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from core.config import get_settings
from core.logging import get_logger, log_duration

log = get_logger("pipeline.embedder")

_TOKEN = re.compile(r"[a-z0-9+#./-]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class HashingEmbedder:
    """Feature-hashing embedder. Each token gets a (bucket, sign) from a crc32
    hash; we tf-weight with bincount and L2-normalise. Cosine of two of these
    tracks shared vocabulary - coarser than a real model but deterministic,
    offline, and quick enough to embed 100K docs in a few seconds when torch
    isn't around. Token hashes are cached, so a big corpus only pays the hash
    cost once per distinct word.
    """

    backend = "hashing"

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.name = f"hashing-{dim}"
        self._cache: dict[str, tuple[int, float]] = {}

    def _hi(self, token: str) -> tuple[int, float]:
        v = self._cache.get(token)
        if v is None:
            h = zlib.crc32(token.encode())
            v = (h % self.dim, 1.0 if (h >> 20) & 1 else -1.0)
            self._cache[token] = v
        return v

    def encode(
        self, texts: Sequence[str], batch_size: int = 1024, show_progress: bool = False
    ) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            toks = _tokenize(text)
            if not toks:
                continue
            idxs = np.empty(len(toks), dtype=np.int64)
            sgns = np.empty(len(toks), dtype=np.float32)
            for j, t in enumerate(toks):
                idx, sign = self._hi(t)
                idxs[j] = idx
                sgns[j] = sign
            vec = np.bincount(idxs, weights=sgns, minlength=self.dim)[: self.dim]
            norm = np.sqrt((vec * vec).sum())
            out[i] = vec / norm if norm > 0 else vec
        return out


class SentenceTransformerEmbedder:
    backend = "sentence-transformers"

    def __init__(self, model_name: str, dim: int, allow_onnx: bool = True):
        from sentence_transformers import SentenceTransformer

        from core.device import construct

        # walks onnx(-int8) -> torch on the best device; raises only if even
        # plain torch can't load, which make_embedder turns into the hashing floor
        self.model, self.runtime = construct(
            SentenceTransformer, model_name, allow_onnx=allow_onnx
        )
        self.name = model_name
        self.dim = self.model.get_sentence_embedding_dimension() or dim

    def encode(
        self, texts: Sequence[str], batch_size: int = 512, show_progress: bool = False
    ) -> np.ndarray:
        vecs = self.model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )
        return vecs.astype(np.float32)


def make_embedder(prefer_st: bool = True):
    """Pick the best embedder we can actually run right now."""
    s = get_settings()
    if prefer_st:
        try:
            emb = SentenceTransformerEmbedder(s.embedding_model, s.embedding_dim)
            log.info("embedder.ready", backend=emb.backend, model=emb.name, dim=emb.dim)
            return emb
        except Exception as exc:
            # offline sandbox / no torch - totally fine, hashing it is.
            log.warning("embedder.st_unavailable", reason=str(exc)[:140])
    emb = HashingEmbedder(s.embedding_dim)
    log.info("embedder.ready", backend=emb.backend, dim=emb.dim)
    return emb


def embed_corpus(texts: Iterable[str], prefer_st: bool = True) -> tuple[np.ndarray, dict]:
    texts = list(texts)
    emb = make_embedder(prefer_st=prefer_st)
    with log_duration(log, "embedder.encode_corpus") as m:
        vecs = emb.encode(texts, batch_size=get_settings().embedding_batch_size, show_progress=True)
        m["n"] = len(texts)
        m["backend"] = emb.backend
    meta = {"backend": emb.backend, "model": emb.name, "dim": int(emb.dim)}
    return vecs, meta


def save_meta(meta: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_embedder_from_meta(path: str | Path):
    """Rebuild the embedder that precompute used (offline). Falls back to
    hashing if the ST model can't be loaded without a network."""
    meta = json.loads(Path(path).read_text(encoding="utf-8"))
    if meta.get("backend") == "sentence-transformers":
        try:
            import os

            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            # rank.py is the no-network path: device is fine, ONNX (which may
            # export/download on load) is not - so disable it here.
            return SentenceTransformerEmbedder(meta["model"], meta["dim"], allow_onnx=False)
        except Exception as exc:
            log.warning("embedder.meta_st_unavailable", reason=str(exc)[:140])
    return HashingEmbedder(int(meta.get("dim", 384)))
