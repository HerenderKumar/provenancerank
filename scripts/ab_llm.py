#!/usr/bin/env python3
"""A/B the LLM head-scoring contribution - one command, prints the metric delta.

First build the matrix WITH the LLM column (precompute caches llm_fit_score):

    GLM_API_KEY=... LLM_HEAD_SCORING_ENABLED=true \
      python precompute.py --candidates ./candidates.jsonl --jd ./data/job_description.txt

Then:

    python scripts/ab_llm.py

It computes two rankings from that ONE matrix - with and without `llm_fit_score`
(everything else identical) - scores both against the graded pseudo-labels, prints
the metric delta + head churn, and writes `submission_base.csv` / `submission_llm.csv`
so you can eyeball the actual top names.

Caveat printed at the end: the pseudo-labels are a proxy, so treat the number as
directional and sanity-check the top-10 by eye (the real test is the leaderboard).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import get_settings  # noqa: E402
from core.logging import configure  # noqa: E402
from ml.evaluate import compare, lift  # noqa: E402
from ml.predictor import predict_fit  # noqa: E402
from ml.pseudo_labels import heuristic_pseudo_labels  # noqa: E402
from output.reasoning_generator import generate_all  # noqa: E402
from output.submission_writer import build_submission, write_submission  # noqa: E402
from pipeline.scorer import merge_final_scores  # noqa: E402


def _rank_and_write(df, scores, pool, k, out_path) -> list[str]:
    """Top-k by score (honeypots excluded), with reasons, written to a CSV.
    Mirrors rank.py's final step so the CSVs match a real submission."""
    d = df.copy()
    d["final_score"] = np.asarray(scores, dtype=float)
    ranked = (
        d[pool]
        .sort_values(["final_score", "candidate_id"], ascending=[False, True])
        .head(k)
        .reset_index(drop=True)
    )
    ranked["reasoning"] = generate_all(ranked.to_dict("records"))
    write_submission(build_submission(ranked, top_k=k), out_path)
    return ranked["candidate_id"].tolist()


def main(argv=None) -> int:
    configure()
    s = get_settings()
    ap = argparse.ArgumentParser(description="A/B the LLM head-scoring lift")
    ap.add_argument("--artifacts", default=s.artifacts_dir)
    ap.add_argument("--top-k", type=int, default=100)
    args = ap.parse_args(argv)

    mp = Path(args.artifacts) / "feature_matrix.pkl"
    if not mp.exists():
        print(f"feature matrix missing at {mp} - run precompute.py first.")
        return 1
    df = pd.read_pickle(mp).reset_index(drop=True)
    if "llm_fit_score" not in df.columns:
        print(
            "No `llm_fit_score` in the matrix. Re-run precompute with an LLM reachable:\n"
            "  GLM_API_KEY=... LLM_HEAD_SCORING_ENABLED=true python precompute.py "
            "--candidates ./candidates.jsonl --jd ./data/job_description.txt"
        )
        return 1

    labels = heuristic_pseudo_labels(df)
    rel = {c: float(g) for c, g in zip(df["candidate_id"], labels, strict=False)}
    ml = predict_fit(df, s.model_path)
    retr = (
        df["retrieval_score"].to_numpy() if "retrieval_score" in df.columns else np.zeros(len(df))
    )
    pool = df["is_honeypot"] != 1
    k = args.top_k

    base_scores = merge_final_scores(df.drop(columns=["llm_fit_score"]), ml, retr).values
    llm_scores = merge_final_scores(df, ml, retr).values
    base_ids = _rank_and_write(df, base_scores, pool, k, "submission_base.csv")
    llm_ids = _rank_and_write(df, llm_scores, pool, k, "submission_llm.csv")

    res = compare({"without_llm": base_ids, "with_llm": llm_ids}, rel)
    print("\n=== LLM head-scoring A/B (scored vs graded pseudo-labels) ===")
    for name in ("without_llm", "with_llm"):
        m = res[name]
        print(
            f"  {name:12}  ndcg@10={m['ndcg@10']:.4f}  ndcg@50={m['ndcg@50']:.4f}  "
            f"map={m['map']:.4f}  p@10={m['p@10']:.4f}  composite={m['composite']:.4f}"
        )
    print(
        f"\n  composite lift: {lift(res['without_llm'], res['with_llm']):+.2f}%"
        f"   ndcg@10 lift: {lift(res['without_llm'], res['with_llm'], 'ndcg@10'):+.2f}%"
    )
    moved = sum(1 for a, b in zip(base_ids[:20], llm_ids[:20], strict=False) if a != b)
    print(
        f"  top-20 reordered: {moved}/20   "
        f"top-{k} membership overlap: {len(set(base_ids) & set(llm_ids))}/{k}"
    )
    print("\n  Wrote submission_base.csv and submission_llm.csv - compare the top names by eye.")
    print(
        "  NOTE: scored against heuristic pseudo-labels (a proxy, not ground truth) - treat the\n"
        "  delta as directional; the real test is the hidden leaderboard, so trust your eyes too."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
