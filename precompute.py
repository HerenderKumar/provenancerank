#!/usr/bin/env python3
"""Phase 1 — offline. Builds everything rank.py reads. Can be slow (embedding
100K candidates is the bottleneck); that's fine, it runs once.

    python precompute.py --candidates ./candidates.jsonl --jd ./data/job_description.txt

Idempotent: we hash the candidates file and skip work whose inputs haven't
changed when --resume is passed. Artifacts written to ./artifacts:
    feature_matrix.pkl  embeddings.npy  bm25_index.pkl  jd_spec.json
    embedding_meta.json  candidate_ids.txt  manifest.json
    ml/models/fit_scorer_v1.pkl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from core import artifact_cache as cache
from core.config import get_settings
from core.logging import configure, get_logger, log_duration
from ml.pseudo_labels import generate_pseudo_labels, heuristic_pseudo_labels
from ml.reranker import rerank
from ml.trainer import create_proxy_labels, train_ranker
from pipeline.bm25_indexer import build_index, query_scores, save_index
from pipeline.embedder import embed_corpus, make_embedder, save_meta
from pipeline.feature_engineering import build_feature_row, frame_from_rows
from pipeline.gate_filter import apply_gates, write_rejection_registry
from pipeline.jd_deconstructor import deconstruct_jd, save_job_spec
from pipeline.loader import corpus_text, count_lines, file_sha256, iter_candidates, rerank_text
from pipeline.retrieval import cosine_scores, fuse_rrf
from pipeline.tournament import run_tournament

log = get_logger("precompute")


def _tqdm(iterable, total, desc):
    try:
        from tqdm import tqdm

        return tqdm(iterable, total=total, desc=desc, unit="cand")
    except Exception:
        return iterable


def _collect_head_candidates(df, path, sample_size, limit):
    """Pull the raw dicts for the most promising rows so Gemini can grade them.

    Memory-bounded: we only ever hold ~sample_size dicts, found by streaming the
    file once. Only called when an API key is configured (otherwise the labels
    are pure heuristic and no dicts are needed)."""
    labels = heuristic_pseudo_labels(df)
    top = labels.sort_values(ascending=False).head(sample_size).index
    wanted = set(df.loc[top, "candidate_id"])
    found = []
    for i, c in enumerate(iter_candidates(path)):
        if limit and i >= limit:
            break
        if c.get("candidate_id") in wanted:
            found.append(c)
            if len(found) >= len(wanted):
                break
    return found


def _provisional_order(df):
    """A cheap pointwise ranking (proxy + retrieval) used only to pick which rows
    enter the rerank/tournament head. Gated and honeypot rows are pushed to the
    bottom so they can never occupy the head."""
    prov = create_proxy_labels(df).to_numpy(dtype=float)
    if "retrieval_score" in df.columns:
        prov = 0.6 * prov + 0.4 * df["retrieval_score"].to_numpy(dtype=float)
    masked = np.zeros(len(df), dtype=bool)
    if "gate_passed" in df.columns:
        masked |= ~df["gate_passed"].to_numpy(dtype=bool)
    masked |= df["is_honeypot"].to_numpy() == 1
    prov = np.where(masked, -np.inf, prov)
    order = np.argsort(prov)[::-1]
    valid = int(np.isfinite(prov).sum())
    return order, valid


def _build_accuracy_layer(df, rerank_corpus, jd_text, args, s, art, data_hash, jd_hash) -> dict:
    """Offline accuracy stack, all cached into the matrix:

      1. graded pseudo-labels -> LambdaMART ranker (saved to model_path);
      2. cross-encoder rerank of the head -> ``rerank_score`` column;
      3. Bradley-Terry tournament of the very top -> ``tournament_score`` column.

    Each stage degrades independently — a failure in one is logged and skipped,
    never aborting the precompute — and rank.py only ever reads the columns. The
    model and the (expensive) rerank pass are content-cached, so an unchanged
    re-run reuses them instead of recomputing.
    """
    stats = {"ranker_backend": "n/a", "rerank_backend": "off", "tournament_n": 0}

    # 1. pseudo-labels -> ranker. Gemini grades a head sample only if configured.
    use_llm_labels = bool(s.google_api_key and s.pseudo_label_llm_enabled)
    label_src = f"gemini:{jd_hash}" if use_llm_labels else "heuristic"
    model_key = cache.key_of(data_hash, label_src, s.ranker_neg_ratio, s.ranker_min_train_rows)
    if args.resume and cache.is_fresh(s.model_path, model_key):
        log.info("cache.hit", artifact="fit_scorer (ranker)")
        stats["ranker_backend"] = "cached"
    else:
        try:
            sample = None
            if jd_text and use_llm_labels:
                log.info("accuracy.llm_label_grading_enabled", sample=s.pseudo_label_sample)
                sample = _collect_head_candidates(
                    df, args.candidates, s.pseudo_label_sample, args.limit
                )
            labels = generate_pseudo_labels(
                df, sample, jd_text, sample_size=s.pseudo_label_sample
            )
            artifact = train_ranker(df, labels, s.model_path)
            cache.stamp(s.model_path, model_key)
            stats["ranker_backend"] = artifact.get("backend", "?")
        except Exception:
            log.exception("accuracy.ranker_failed_keeping_formula")

    # default columns so the scorer always finds them (0 == no head signal)
    df["rerank_score"] = 0.0
    df["tournament_score"] = 0.0
    if not jd_text:
        return stats

    order, valid = _provisional_order(df)
    ids_arr = df["candidate_id"].to_numpy()

    # 2. cross-encoder rerank over the head (cached: the cross-encoder pass is the
    #    second-biggest cost once the real model is installed)
    if s.reranker_enabled and valid > 1:
        try:
            k = min(int(s.rerank_top_k), valid)
            rr_key = cache.key_of(data_hash, jd_hash, s.reranker_model, k)
            col = cache.load_npy(art / "rerank_scores.npy", rr_key) if args.resume else None
            if col is None:
                head = order[:k]
                items = [(ids_arr[p], rerank_corpus[p]) for p in head]
                scores = rerank(jd_text, items, model_name=s.reranker_model)
                col = np.zeros(len(df), dtype=float)
                for p in head:
                    col[p] = scores.get(ids_arr[p], 0.0)
                cache.save_npy(art / "rerank_scores.npy", rr_key, col)
                stats["rerank_backend"] = "applied"
            else:
                stats["rerank_backend"] = "cached"
            df["rerank_score"] = col
            stats["rerank_k"] = k
        except Exception:
            log.exception("accuracy.rerank_failed_skipping")

    # 3. Bradley-Terry tournament over the very top (uses rerank_score as a signal)
    if s.tournament_enabled and valid > 1:
        try:
            n = min(int(s.tournament_top_n), valid)
            top = order[:n]
            strengths = run_tournament(df.iloc[top])
            col = np.zeros(len(df), dtype=float)
            for p in top:
                col[p] = strengths.get(ids_arr[p], 0.0)
            df["tournament_score"] = col
            stats["tournament_n"] = n
        except Exception:
            log.exception("accuracy.tournament_failed_skipping")

    # 4. LLM head-scoring (offline, cached) — the smartest signal, head only.
    if s.llm_head_scoring_enabled and valid > 1:
        try:
            from ml.llm_scorer import score_head

            k = min(int(s.llm_head_top_k), valid)
            llm_key = cache.key_of(data_hash, jd_hash, s.glm_model, s.ollama_model, k, "llm")
            col = cache.load_npy(art / "llm_scores.npy", llm_key) if args.resume else None
            if col is None:
                head = order[:k]
                items = [(ids_arr[p], rerank_corpus[p]) for p in head]
                scores = score_head(jd_text, items)
                col = np.zeros(len(df), dtype=float)
                for p in head:
                    col[p] = scores.get(ids_arr[p], 0.0)
                if (col > 0).any():  # only persist/use a real signal
                    cache.save_npy(art / "llm_scores.npy", llm_key, col)
                stats["llm_scored"] = int((col > 0).sum())
            else:
                stats["llm_scored"] = "cached"
            if (col > 0).any():
                df["llm_fit_score"] = col  # absent => scorer adds nothing
        except Exception:
            log.exception("accuracy.llm_scoring_failed_skipping")

    return stats


def main(argv: list[str] | None = None) -> int:
    configure()
    s = get_settings()
    ap = argparse.ArgumentParser(description="ProvenanceRank offline precompute")
    ap.add_argument("--candidates", default=s.candidates_file)
    ap.add_argument("--jd", default=s.jd_file)
    ap.add_argument("--artifacts", default=s.artifacts_dir)
    ap.add_argument(
        "--resume", action="store_true", help="skip if artifacts already match the candidates hash"
    )
    ap.add_argument(
        "--no-st",
        action="store_true",
        help="force the hashing embedder (skip sentence-transformers)",
    )
    ap.add_argument("--bm25-backend", default="auto", choices=["auto", "rank-bm25", "inverted"])
    ap.add_argument(
        "--limit", type=int, default=0, help="only process the first N candidates (dev only)"
    )
    ap.add_argument(
        "--no-accuracy",
        action="store_true",
        help="skip the rerank/tournament accuracy layer (proxy regressor only)",
    )
    args = ap.parse_args(argv)

    art = Path(args.artifacts)
    art.mkdir(parents=True, exist_ok=True)
    manifest_path = art / "manifest.json"

    t0 = time.perf_counter()
    data_hash = file_sha256(args.candidates)
    jd_text = Path(args.jd).read_text(encoding="utf-8") if Path(args.jd).exists() else None
    jd_hash = hashlib.sha256(jd_text.encode()).hexdigest()[:16] if jd_text else "no-jd"

    # whole-run skip only when *both* the candidates and the JD are unchanged.
    # If just the JD changed, we still proceed — the per-stage cache reuses the
    # embeddings/BM25 and only the JD-dependent work reruns.
    # Only skip when the inputs match AND the final artifact is actually present.
    # A leftover manifest from a partial/failed run (e.g. the disk filled mid-build)
    # must NOT trick us into skipping — that leaves the API with no feature matrix.
    matrix_exists = (art / "feature_matrix.pkl").exists()
    if args.resume and manifest_path.exists() and matrix_exists:
        prev = json.loads(manifest_path.read_text())
        unchanged = (
            prev.get("candidates_sha256") == data_hash
            and prev.get("jd_sha256") == jd_hash
            and not args.limit
        )
        if unchanged:
            log.info("precompute.resume_skip", reason="inputs unchanged")
            return 0
    elif args.resume and manifest_path.exists() and not matrix_exists:
        log.warning("precompute.resume_ignored", reason="feature_matrix.pkl missing; rebuilding")

    # ---- JD spec (the only place an LLM could be called; falls back) ----
    spec = deconstruct_jd(jd_text)
    save_job_spec(spec, art / "jd_spec.json")

    # ---- single streaming pass: features + corpus + ids ----
    total = count_lines(args.candidates)
    if args.limit:
        total = min(total, args.limit)
    rows, corpus, rerank_corpus, ids = [], [], [], []
    with log_duration(log, "precompute.feature_pass") as m:
        for i, c in enumerate(_tqdm(iter_candidates(args.candidates), total, "features")):
            if args.limit and i >= args.limit:
                break
            rows.append(build_feature_row(c))
            corpus.append(corpus_text(c))  # full doc for BM25 + embeddings
            rerank_corpus.append(rerank_text(c))  # tight doc for the cross-encoder
            ids.append(c["candidate_id"])
        m["rows"] = len(rows)
    df = frame_from_rows(rows)
    df = apply_gates(df, spec)
    write_rejection_registry(df, art / "excluded_candidates.csv")
    log.info(
        "precompute.features_done",
        rows=len(df),
        honeypots=int(df.is_honeypot.sum()),
        keyword_stuffers=int(df.keyword_stuffer_flag.sum()),
        gate_passed=int(df.gate_passed.sum()),
    )

    # ---- embeddings (cached: depend only on the candidates + backend, so a JD
    #      swap reuses all 100K). Compute the JD cosine, then free the matrix. ----
    emb_backend = "hash" if args.no_st else "st"
    emb_key = cache.key_of(data_hash, emb_backend, s.embedding_model, s.embedding_dim)
    embeddings = cache.load_npy(art / "embeddings.npy", emb_key) if args.resume else None
    if embeddings is None:
        embeddings, emb_meta = embed_corpus(corpus, prefer_st=not args.no_st)
        cache.save_npy(art / "embeddings.npy", emb_key, embeddings)
        save_meta(emb_meta, art / "embedding_meta.json")
    else:
        emb_meta = json.loads((art / "embedding_meta.json").read_text())

    cos = None
    if jd_text:
        cos = cache.load_npy(art / "jd_cosine.npy", cache.key_of(emb_key, jd_hash)) \
            if args.resume else None
        if cos is None:
            jd_vec = make_embedder(prefer_st=not args.no_st).encode([jd_text])[0]
            cos = cosine_scores(jd_vec, embeddings)
            cache.save_npy(art / "jd_cosine.npy", cache.key_of(emb_key, jd_hash), cos)
    del embeddings

    # ---- bm25 index; score the JD query, then drop the index + full corpus.
    #      The reranker uses the tight rerank_corpus, not this, so corpus is done. ----
    index = build_index(corpus, backend=args.bm25_backend)
    save_index(index, art / "bm25_index.pkl")
    bm = query_scores(index, jd_text) if jd_text else None
    del index, corpus

    if cos is not None:
        df["retrieval_score"] = fuse_rrf(bm, cos)

    # ---- accuracy layer (offline): pseudo-labels -> LambdaMART, then a cross-
    #      encoder rerank + Bradley-Terry tournament over the head. All cached
    #      into the matrix; rank.py just reads the columns. ----
    if args.no_accuracy:
        from ml.trainer import train_fit_scorer

        train_fit_scorer(df, s.model_path)  # baseline path: proxy regressor only
        acc_stats = {"accuracy_layer": "disabled"}
    else:
        with log_duration(log, "precompute.accuracy_layer") as am:
            acc_stats = _build_accuracy_layer(
                df, rerank_corpus, jd_text, args, s, art, data_hash, jd_hash
            )
            am.update(acc_stats)
    del rerank_corpus

    # ---- persist matrix + ids ----
    df.to_pickle(art / "feature_matrix.pkl")
    (art / "candidate_ids.txt").write_text("\n".join(ids), encoding="utf-8")

    manifest = {
        "candidates_sha256": data_hash,
        "jd_sha256": jd_hash,
        "n_candidates": len(df),
        "embedding_meta": emb_meta,
        "jd_source": spec.source,
        "accuracy": acc_stats,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_sec": round(time.perf_counter() - t0, 1),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info(
        "precompute.done", **{k: manifest[k] for k in ("n_candidates", "jd_source", "elapsed_sec")}
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
