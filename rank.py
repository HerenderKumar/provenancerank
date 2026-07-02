#!/usr/bin/env python3
"""Phase 2 — produce submission.csv. No network, CPU only, < 5 min for 100K.

    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Two ways in:
  * fast path  — precomputed artifacts exist and cover the candidates -> read the
                 feature matrix, re-run retrieval over the cached index/embeddings,
                 predict, merge. This is the Stage-3 reproduction path.
  * live path  — no artifacts (or a fresh small sample, e.g. the sandbox upload)
                 -> compute features + an ephemeral index for just those
                 candidates. Slower per candidate but trivial at sandbox sizes.

Every artifact load is wrapped so a missing/corrupt one degrades instead of
crashing: no model -> proxy formula; no embeddings -> BM25 only; no index ->
cosine only; neither -> fit + behavioural still rank.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from core.config import get_settings
from core.logging import configure, get_logger, log_duration
from output.reasoning_generator import generate_all
from output.submission_writer import build_submission, validate, write_submission
from pipeline.loop_agent import run_loop
from pipeline.scorer import merge_final_scores

log = get_logger("rank")


# --------------------------------------------------------------------------
# retrieval (re-run at ranking time; falls back as artifacts are missing)
# --------------------------------------------------------------------------


def _retrieval_scores(ids: list[str], art: Path, df: pd.DataFrame) -> np.ndarray:
    """Retrieval score per id, recomputed from cached artifacts when present."""
    from pipeline.retrieval import cosine_scores, fuse_rrf

    jd_path = Path(get_settings().jd_file)
    jd_text = jd_path.read_text(encoding="utf-8") if jd_path.exists() else ""
    order_file = art / "candidate_ids.txt"
    emb_file = art / "embeddings.npy"
    idx_file = art / "bm25_index.pkl"

    if order_file.exists() and (emb_file.exists() or idx_file.exists()) and jd_text:
        full_ids = order_file.read_text(encoding="utf-8").split("\n")
        pos = {cid: i for i, cid in enumerate(full_ids)}
        cos = bm = None
        try:
            if emb_file.exists():
                from pipeline.embedder import load_embedder_from_meta

                emb = np.load(emb_file)
                jd_vec = load_embedder_from_meta(art / "embedding_meta.json").encode([jd_text])[0]
                cos = cosine_scores(jd_vec, emb)
        except Exception as exc:
            log.warning("rank.cosine_skipped", reason=str(exc)[:120])
        try:
            if idx_file.exists():
                from pipeline.bm25_indexer import load_index, query_scores

                bm = query_scores(load_index(idx_file), jd_text)
        except Exception as exc:
            log.warning("rank.bm25_skipped", reason=str(exc)[:120])

        if cos is None and bm is None:
            pass  # fall through to cached column
        else:
            n = len(full_ids)
            cos = cos if cos is not None else np.zeros(n, dtype=np.float32)
            bm = bm if bm is not None else np.zeros(n, dtype=np.float32)
            full_rrf = fuse_rrf(bm, cos)
            return np.array([full_rrf[pos[c]] if c in pos else 0.0 for c in ids], dtype=np.float32)

    if "retrieval_score" in df.columns:
        log.info("rank.retrieval_from_cache")
        return df["retrieval_score"].to_numpy(dtype=np.float32)
    log.warning("rank.retrieval_unavailable_zeroed")
    return np.zeros(len(ids), dtype=np.float32)


# --------------------------------------------------------------------------
# scoring + output (shared by both paths)
# --------------------------------------------------------------------------


def _finalise(
    df: pd.DataFrame, retrieval: np.ndarray, out_path: str, top_k: int
) -> tuple[Path, list[str]]:
    from ml.predictor import predict_fit

    ml_fit = predict_fit(df, get_settings().model_path)
    df = df.reset_index(drop=True)
    df["ml_fit"] = ml_fit
    df["retrieval"] = retrieval
    df["final_score"] = merge_final_scores(df, ml_fit, retrieval).values

    # loop is informational (coverage report); doesn't change the final order
    try:
        _, state = run_loop(df, target=top_k)
        log.info(
            "rank.loop_coverage",
            exit=state.exit_reason,
            coverage={k: round(v, 2) for k, v in state.coverage_scores.items()},
        )
    except Exception as exc:
        log.warning("rank.loop_skipped", reason=str(exc)[:120])

    # honeypots can never be a top pick, regardless of score
    pool = df[df["is_honeypot"] != 1].copy()
    ranked = (
        pool.sort_values(["final_score", "candidate_id"], ascending=[False, True])
        .head(top_k)
        .reset_index(drop=True)
    )
    ranked["reasoning"] = generate_all(ranked.to_dict("records"))

    submission = build_submission(ranked, top_k=top_k)
    path = write_submission(submission, out_path)
    errors = validate(path)
    return path, errors


# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------


def _fast_path(candidate_ids: list[str], art: Path, out: str, top_k: int):
    df = pd.read_pickle(art / "feature_matrix.pkl")
    known = set(df["candidate_id"])
    if not set(candidate_ids).issubset(known):
        return None  # caller falls back to live path
    df = df[df["candidate_id"].isin(set(candidate_ids))].reset_index(drop=True)
    retrieval = _retrieval_scores(df["candidate_id"].tolist(), art, df)
    return _finalise(df, retrieval, out, top_k)


def _live_path(candidates: list[dict], out: str, top_k: int):
    from pipeline.bm25_indexer import build_index, query_scores
    from pipeline.embedder import embed_corpus, make_embedder
    from pipeline.feature_engineering import build_feature_row, frame_from_rows
    from pipeline.gate_filter import apply_gates
    from pipeline.jd_deconstructor import deconstruct_jd
    from pipeline.retrieval import cosine_scores, fuse_rrf

    df = frame_from_rows([build_feature_row(c) for c in candidates])
    df = apply_gates(df, deconstruct_jd())
    corpus = [_corpus(c) for c in candidates]
    jd_path = Path(get_settings().jd_file)
    jd_text = jd_path.read_text(encoding="utf-8") if jd_path.exists() else " ".join(corpus[:1])
    try:
        emb, _ = embed_corpus(corpus, prefer_st=False)
        jd_vec = make_embedder(prefer_st=False).encode([jd_text])[0]
        retrieval = fuse_rrf(query_scores(build_index(corpus), jd_text), cosine_scores(jd_vec, emb))
    except Exception as exc:
        log.warning("rank.live_retrieval_failed", reason=str(exc)[:120])
        retrieval = np.zeros(len(df), dtype=np.float32)
    return _finalise(df, retrieval, out, top_k)


def _corpus(c: dict) -> str:
    from pipeline.loader import corpus_text

    return corpus_text(c)


# --------------------------------------------------------------------------
# entry
# --------------------------------------------------------------------------


def run(
    candidates_file: str, out: str, artifacts: str | None = None, top_k: int | None = None
) -> int:
    s = get_settings()
    art = Path(artifacts or s.artifacts_dir)
    top_k = top_k or s.top_k
    t0 = time.perf_counter()

    from pipeline.loader import iter_candidates

    ids = [c["candidate_id"] for c in iter_candidates(candidates_file)]
    log.info("rank.start", candidates=len(ids), artifacts=str(art))

    result = None
    if (art / "feature_matrix.pkl").exists():
        with log_duration(log, "rank.fast_path"):
            result = _fast_path(ids, art, out, top_k)
    if result is None:
        log.info("rank.live_path", reason="no_artifacts_or_new_candidates")
        cands = list(iter_candidates(candidates_file))
        with log_duration(log, "rank.live_path_run"):
            result = _live_path(cands, out, top_k)

    path, errors = result
    elapsed = round(time.perf_counter() - t0, 1)
    if errors:
        for e in errors:
            log.error("rank.submission_invalid", detail=e)
        print(f"FAILED: submission invalid ({len(errors)} issues). See logs.", file=sys.stderr)
        return 1

    _print_summary(path, elapsed, top_k)
    return 0


def _print_summary(path: Path, elapsed: float, top_k: int) -> None:
    df = pd.read_csv(path)
    top = df.head(5)
    log.info(
        "ranking.complete",
        out=str(path),
        rows=len(df),
        elapsed_sec=elapsed,
        top5=list(zip(top.candidate_id, top.score.round(4), strict=False)),
    )
    # human-readable tail so a person running it sees something useful
    lines = [f"\nProvenanceRank -> {path}  ({len(df)} rows, {elapsed}s)", "Top 5:"]
    for _, r in top.iterrows():
        lines.append(
            f"  #{int(r['rank']):>2}  {r['candidate_id']}  {r['score']:.4f}  "
            f"{str(r['reasoning'])[:90]}"
        )
    print("\n".join(lines))


def smoke_test(candidates_file: str | None = None, n: int = 150) -> int:
    """CI smoke: rank a small live sample end-to-end, assert valid + fast."""
    configure()
    s = get_settings()
    src = candidates_file or s.candidates_file
    if not Path(src).exists():
        for fallback in ("data/smoke_candidates.jsonl", "data/dev_sample.jsonl"):
            cand = Path(__file__).parent / fallback
            if cand.exists():
                src = str(cand)
                break
    from pipeline.loader import iter_candidates

    cands = []
    for i, c in enumerate(iter_candidates(src)):
        if i >= n:
            break
        cands.append(c)
    out = str(Path(__file__).parent / "artifacts" / "smoke_submission.csv")
    t0 = time.perf_counter()
    path, errors = _live_path(cands, out, top_k=min(100, len(cands)))
    dt = time.perf_counter() - t0
    assert not errors, f"smoke submission invalid: {errors}"
    assert dt < 30, f"smoke too slow: {dt:.1f}s"
    log.info("smoke_test.ok", seconds=round(dt, 2), candidates=len(cands))
    return 0


def main(argv: list[str] | None = None) -> int:
    configure()
    s = get_settings()
    ap = argparse.ArgumentParser(description="ProvenanceRank — produce submission.csv")
    ap.add_argument("--candidates", default=s.candidates_file)
    ap.add_argument("--out", default="./submission.csv")
    ap.add_argument("--artifacts", default=s.artifacts_dir)
    ap.add_argument("--top-k", type=int, default=s.top_k)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    if args.smoke:
        return smoke_test(args.candidates)
    return run(args.candidates, args.out, args.artifacts, args.top_k)


if __name__ == "__main__":
    raise SystemExit(main())
