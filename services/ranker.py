"""RankerService - the engine, loaded once and kept warm.

The CLI (`rank.py`) loads artifacts per invocation; a service can't afford that,
so this loads the feature matrix, embeddings, BM25 index and model a single time
at startup and answers many requests against them. Requests are read-only over
the shared matrix (scores are computed into fresh arrays, never written back),
so it's safe to call concurrently.

Two request shapes:
  * default JD over the full precomputed pool        -> fast path, cached retrieval
  * a custom JD, or a small ad-hoc candidate sample  -> recompute what's needed
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from core.config import get_settings
from core.exceptions import ArtifactError, RankingError
from core.logging import get_logger, log_duration
from core.tracing import span

log = get_logger("services.ranker")


@dataclass
class RankResult:
    rows: list[dict]  # candidate_id, rank, score, reasoning
    source: str  # fast_path | live
    candidates_count: int
    top_k: int
    jd_hash: str
    elapsed_ms: float
    gate_stats: dict = field(default_factory=dict)
    coverage: dict = field(default_factory=dict)
    cache_hit: bool = False


class RankerService:
    def __init__(self, artifacts_dir: str | None = None):
        s = get_settings()
        self.art = Path(artifacts_dir or s.artifacts_dir)
        self.matrix: pd.DataFrame | None = None
        self.candidate_ids: list[str] = []
        self.embeddings = None
        self.bm25 = None
        self.jd_spec = None
        self.embedding_meta_path = self.art / "embedding_meta.json"
        self.model_path = s.model_path
        self.default_jd = ""
        self.ready = False
        self.meta: dict = {}

    def load(self) -> RankerService:
        from pipeline.jd_deconstructor import load_job_spec

        fm = self.art / "feature_matrix.pkl"
        if not fm.exists():
            raise ArtifactError(f"feature matrix missing at {fm}; run precompute.py")
        with log_duration(log, "ranker.load") as m:
            self.matrix = pd.read_pickle(fm)
            self._validate_matrix()
            ids_file = self.art / "candidate_ids.txt"
            self.candidate_ids = (
                ids_file.read_text().split("\n")
                if ids_file.exists()
                else list(self.matrix["candidate_id"])
            )
            spec_file = self.art / "jd_spec.json"
            self.jd_spec = load_job_spec(spec_file) if spec_file.exists() else None
            self._load_retrieval_artifacts()
            jd_path = Path(get_settings().jd_file)
            self.default_jd = jd_path.read_text(encoding="utf-8") if jd_path.exists() else ""
            man = self.art / "manifest.json"
            self.meta = json.loads(man.read_text()) if man.exists() else {}
            m["n"] = len(self.matrix)
        self.ready = True
        log.info(
            "ranker.ready",
            candidates=len(self.matrix),
            has_embeddings=self.embeddings is not None,
            has_bm25=self.bm25 is not None,
        )
        return self

    def _validate_matrix(self) -> None:
        from core.constants import NUMERIC_FEATURE_COLUMNS

        missing = [
            c
            for c in ("candidate_id", "is_honeypot", *NUMERIC_FEATURE_COLUMNS)
            if c not in self.matrix.columns
        ]
        if missing:
            raise ArtifactError("feature matrix is missing columns", detail=missing[:8])
        man = self.art / "manifest.json"
        if man.exists():
            n = json.loads(man.read_text()).get("n_candidates")
            if n is not None and n != len(self.matrix):
                raise ArtifactError(
                    "manifest n_candidates != feature matrix rows",
                    detail={"manifest": n, "matrix": len(self.matrix)},
                )

    def _load_retrieval_artifacts(self) -> None:
        emb = self.art / "embeddings.npy"
        idx = self.art / "bm25_index.pkl"
        try:
            if emb.exists():
                self.embeddings = np.load(emb, mmap_mode="r")  # mmap: don't pin 150MB
        except Exception as exc:
            log.warning("ranker.embeddings_skipped", reason=str(exc)[:120])
        try:
            if idx.exists():
                from pipeline.bm25_indexer import load_index

                self.bm25 = load_index(idx)
        except Exception as exc:
            log.warning("ranker.bm25_skipped", reason=str(exc)[:120])

    def health(self) -> dict:
        return {
            "ready": self.ready,
            "candidates": len(self.matrix) if self.matrix is not None else 0,
            "has_embeddings": self.embeddings is not None,
            "has_bm25": self.bm25 is not None,
            "jd_source": (self.jd_spec.source if self.jd_spec else "none"),
            "embedding_backend": self.meta.get("embedding_meta", {}).get("backend"),
            "built_at": self.meta.get("built_at"),
        }

    def rank(
        self,
        jd_text: str | None = None,
        candidate_ids: list[str] | None = None,
        candidates: list[dict] | None = None,
        top_k: int | None = None,
    ) -> RankResult:
        top_k = top_k or get_settings().top_k
        if candidates is not None:
            return self._rank_live(candidates, jd_text, top_k)
        if not self.ready:
            raise ArtifactError("ranker not loaded")
        return self._rank_fast(jd_text, candidate_ids, top_k)

    def _rank_fast(
        self, jd_text: str | None, candidate_ids: list[str] | None, top_k: int
    ) -> RankResult:
        t0 = time.perf_counter()
        custom_jd = bool(jd_text and jd_text.strip() and jd_text.strip() != self.default_jd.strip())
        jd = jd_text if custom_jd else self.default_jd

        if candidate_ids:
            mask = self.matrix["candidate_id"].isin(set(candidate_ids))
            df = self.matrix.loc[mask].reset_index(drop=True)
            if df.empty:
                raise RankingError("none of the requested candidate_ids are known")
        else:
            df = self.matrix

        with span("rank.retrieval"):
            retrieval = self._retrieval(df, jd, custom_jd)
        result = self._finalize(df, retrieval, top_k, source="fast_path", jd_text=jd)
        result.elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        return result

    def _rank_live(self, candidates: list[dict], jd_text: str | None, top_k: int) -> RankResult:
        from pipeline.bm25_indexer import build_index, query_scores
        from pipeline.embedder import embed_corpus, make_embedder
        from pipeline.feature_engineering import build_feature_row, frame_from_rows
        from pipeline.gate_filter import apply_gates
        from pipeline.jd_deconstructor import deconstruct_jd
        from pipeline.loader import corpus_text
        from pipeline.retrieval import cosine_scores, fuse_rrf

        t0 = time.perf_counter()
        df = frame_from_rows([build_feature_row(c) for c in candidates])
        df = apply_gates(df, self.jd_spec or deconstruct_jd())
        corpus = [corpus_text(c) for c in candidates]
        jd = jd_text or self.default_jd or (corpus[0] if corpus else "")
        try:
            emb, _ = embed_corpus(corpus, prefer_st=False)
            jd_vec = make_embedder(prefer_st=False).encode([jd])[0]
            retrieval = fuse_rrf(query_scores(build_index(corpus), jd), cosine_scores(jd_vec, emb))
        except Exception as exc:
            log.warning("ranker.live_retrieval_failed", reason=str(exc)[:120])
            retrieval = np.zeros(len(df), dtype=np.float32)
        result = self._finalize(df, retrieval, top_k, source="live", jd_text=jd)
        result.elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        return result

    def _retrieval(self, df: pd.DataFrame, jd: str, custom_jd: bool) -> np.ndarray:
        from pipeline.retrieval import cosine_scores, fuse_rrf

        # default JD + cached column = the fast, exact path
        if not custom_jd and "retrieval_score" in df.columns:
            return df["retrieval_score"].to_numpy(dtype=np.float32)
        if self.embeddings is None and self.bm25 is None:
            if "retrieval_score" in df.columns:
                return df["retrieval_score"].to_numpy(dtype=np.float32)
            return np.zeros(len(df), dtype=np.float32)

        # recompute against the (possibly custom) JD over the full pool, then map
        from pipeline.bm25_indexer import query_scores
        from pipeline.embedder import load_embedder_from_meta

        pos = {cid: i for i, cid in enumerate(self.candidate_ids)}
        n = len(self.candidate_ids)
        cos = bm = None
        if self.embeddings is not None:
            jd_vec = load_embedder_from_meta(self.embedding_meta_path).encode([jd])[0]
            cos = cosine_scores(jd_vec, np.asarray(self.embeddings))
        if self.bm25 is not None:
            bm = query_scores(self.bm25, jd)
        cos = cos if cos is not None else np.zeros(n, dtype=np.float32)
        bm = bm if bm is not None else np.zeros(n, dtype=np.float32)
        full = fuse_rrf(bm, cos)
        return np.array(
            [full[pos[c]] if c in pos else 0.0 for c in df["candidate_id"]], dtype=np.float32
        )

    def _finalize(
        self, df: pd.DataFrame, retrieval: np.ndarray, top_k: int, source: str, jd_text: str
    ) -> RankResult:
        from ml.predictor import predict_fit
        from output.reasoning_generator import generate_all
        from output.submission_writer import build_submission
        from pipeline.loop_agent import run_loop
        from pipeline.scorer import merge_final_scores

        try:
            ml_fit = predict_fit(df, self.model_path)
            final = merge_final_scores(df, ml_fit, retrieval)
        except Exception as exc:
            raise RankingError(f"scoring failed: {exc}") from exc

        scored = df.assign(final_score=final.values)
        try:
            _, state = run_loop(scored, target=top_k)
            coverage = {k: round(v, 3) for k, v in state.coverage_scores.items()}
        except Exception:
            coverage = {}

        pool = scored[scored["is_honeypot"] != 1]
        ranked = (
            pool.sort_values(["final_score", "candidate_id"], ascending=[False, True])
            .head(top_k)
            .reset_index(drop=True)
        )
        ranked["reasoning"] = generate_all(ranked.to_dict("records"))
        sub = build_submission(ranked, top_k=top_k)
        rows = sub.to_dict("records")

        gate_stats = self._gate_stats(df)
        return RankResult(
            rows=rows,
            source=source,
            candidates_count=len(df),
            top_k=top_k,
            jd_hash=hashlib.sha256((jd_text or "").encode()).hexdigest()[:16],
            elapsed_ms=0.0,
            gate_stats=gate_stats,
            coverage=coverage,
        )

    @staticmethod
    def _gate_stats(df: pd.DataFrame) -> dict:
        stats = {"total": len(df)}
        if "gate_passed" in df.columns:
            stats["passed"] = int(df["gate_passed"].sum())
            stats["excluded"] = int((~df["gate_passed"]).sum())
        stats["honeypots"] = int(df["is_honeypot"].sum())
        stats["keyword_stuffers"] = int(
            df.get("keyword_stuffer_flag", pd.Series(dtype=float)).sum()
        )
        return stats
