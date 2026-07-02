"""Accuracy layer: eval harness, graded pseudo-labels, cross-encoder rerank
(lexical fallback), the Bradley-Terry tournament, and the scorer head-blend.

These run with only numpy/pandas installed — the ranker falls back to the proxy
formula and the reranker to its lexical scorer — so they pin the *plumbing and
the maths*, not the heavy models (which plug in on the deployment box)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml import evaluate as ev
from ml.pseudo_labels import generate_pseudo_labels, heuristic_pseudo_labels
from ml.reranker import _lexical_scores, rerank
from ml.trainer import train_ranker
from pipeline.feature_engineering import build_feature_row, frame_from_rows
from pipeline.scorer import merge_final_scores
from pipeline.tournament import run_tournament
from tests.conftest import make_candidate

# --------------------------------------------------------------------------
# eval harness
# --------------------------------------------------------------------------

def test_ndcg_perfect_and_reversed():
    ideal = ["a", "b", "c", "d"]
    rel = {"a": 5, "b": 4, "c": 3, "d": 0}
    assert ev.evaluate(ideal, rel)["ndcg@10"] == 1.0
    worse = ev.evaluate(list(reversed(ideal)), rel)["ndcg@10"]
    assert worse < 1.0


def test_map_and_precision_threshold():
    # only tier >= 3 counts as relevant; relevants at ranks 1 and 3 of a 10-list
    rel = {f"c{i}": 0 for i in range(10)}
    rel["c0"] = 5
    rel["c2"] = 4
    ranked = [f"c{i}" for i in range(10)]
    m = ev.evaluate(ranked, rel)
    assert m["p@10"] == 0.2  # 2 relevant across the 10 slots
    assert abs(m["map"] - (1.0 + 2.0 / 3.0) / 2.0) < 1e-3


def test_lift_is_relative_percent():
    base = {"composite": 0.50}
    cand = {"composite": 0.55}
    assert ev.lift(base, cand) == 10.0
    assert ev.lift({"composite": 0.0}, cand) == 0.0  # guards divide-by-zero


# --------------------------------------------------------------------------
# graded pseudo-labels
# --------------------------------------------------------------------------

def _toy_matrix(n=400, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "candidate_id": [f"C{i:05d}" for i in range(n)],
            "jd_fit_score": rng.random(n),
            "retrieval_skill_count": rng.integers(0, 8, n),
            "jd_skill_match_count": rng.integers(0, 6, n),
            "assessment_mean_score": rng.random(n) * 100,
            "has_production_keywords": rng.integers(0, 2, n),
            "product_company_ratio": rng.random(n),
            "behavioural_composite": rng.random(n),
            "evidence_confidence_bonus": np.zeros(n),
            "gate_passed": rng.random(n) > 0.2,
            "is_honeypot": (rng.random(n) < 0.05).astype(int),
        }
    )
    return df


def test_pseudo_labels_span_all_tiers_and_zero_honeypots():
    df = _toy_matrix()
    labels = generate_pseudo_labels(df, None, "Senior AI Engineer LLM RAG", sample_size=50)
    # graded, not collapsed to {0,1}
    assert set(labels.unique()).issuperset({0, 3}) and labels.max() == 5
    # honeypots are never positive
    assert (labels[df["is_honeypot"] == 1] == 0).all()
    # gate failures are never positive
    assert (labels[~df["gate_passed"]] == 0).all()


def test_tier_distribution_is_top_heavy():
    df = _toy_matrix()
    labels = heuristic_pseudo_labels(df)
    counts = {t: int((labels == t).sum()) for t in range(6)}
    assert counts[5] <= counts[4] <= counts[3]  # a few elite, more mediocre


# --------------------------------------------------------------------------
# ranker artifact (formula fallback here; same shape predict_fit expects)
# --------------------------------------------------------------------------

def test_train_ranker_artifact_shape(tmp_path):
    # needs the full 40-column feature contract -> build from real candidates
    df = _scored_frame()
    labels = heuristic_pseudo_labels(df)
    art = train_ranker(df, labels, tmp_path / "rank.pkl")
    assert art["kind"] == "ranker"
    assert art["features"] and "model" in art
    assert (tmp_path / "rank.pkl").exists()


# --------------------------------------------------------------------------
# cross-encoder reranker (lexical fallback)
# --------------------------------------------------------------------------

def test_lexical_reranker_prefers_relevant_doc():
    jd = "senior ai engineer building production llm and rag vector search systems"
    items = [
        ("match", "Built production LLM and RAG vector search ranking systems in Python."),
        ("noise", "Experienced retail store manager handling inventory and staffing."),
    ]
    scores = rerank(jd, items)
    assert set(scores) == {"match", "noise"}
    assert all(0.0 <= v <= 1.0 for v in scores.values())
    assert scores["match"] > scores["noise"]


def test_lexical_scores_empty_inputs():
    assert list(_lexical_scores("", ["x"])) == [0.0]
    assert list(_lexical_scores("jd", [])) == []


# --------------------------------------------------------------------------
# Bradley-Terry tournament
# --------------------------------------------------------------------------

def test_tournament_dominant_candidate_wins():
    # one row dominates every criterion -> must get the top strength
    df = pd.DataFrame(
        {
            "candidate_id": ["dom", "mid", "low"],
            "jd_fit_score": [0.9, 0.5, 0.2],
            "rerank_score": [0.9, 0.5, 0.2],
            "retrieval_skill_count": [9, 5, 1],
            "jd_skill_match_count": [6, 3, 0],
            "assessment_mean_score": [95, 60, 30],
            "evidence_confidence_bonus": [0.1, 0.0, 0.0],
            "has_production_keywords": [1, 1, 0],
            "product_company_ratio": [0.9, 0.5, 0.1],
            "behavioural_composite": [0.9, 0.5, 0.2],
        }
    )
    strengths = run_tournament(df)
    assert strengths["dom"] == max(strengths.values())
    assert strengths["dom"] == 1.0 and strengths["low"] == 0.0
    assert all(0.0 <= v <= 1.0 for v in strengths.values())


def test_tournament_resolves_a_cycle_without_crashing():
    # rock-paper-scissors across two criteria — no Condorcet winner
    df = pd.DataFrame(
        {
            "candidate_id": ["a", "b", "c"],
            "jd_fit_score": [3, 2, 1],
            "rerank_score": [1, 3, 2],
            "retrieval_skill_count": [2, 1, 3],
        }
    )
    strengths = run_tournament(df)
    assert len(strengths) == 3
    assert all(0.0 <= v <= 1.0 for v in strengths.values())


def test_tournament_degenerate_inputs():
    one = pd.DataFrame({"candidate_id": ["solo"], "jd_fit_score": [0.5]})
    assert run_tournament(one) == {"solo": 0.0}


# --------------------------------------------------------------------------
# scorer head-blend: no-op when absent, bounded + reorders when present
# --------------------------------------------------------------------------

def _scored_frame():
    rows = [
        build_feature_row(make_candidate(cid=f"CAND_{i:07d}", title="ML Engineer"))
        for i in range(1, 9)
    ]
    df = frame_from_rows(rows)
    df["gate_passed"] = True
    return df


def test_head_blend_is_noop_when_columns_absent():
    df = _scored_frame()
    ml = np.linspace(0.2, 0.9, len(df))
    retr = np.linspace(0.3, 0.8, len(df))
    before = merge_final_scores(df, ml, retr)
    # adding all-zero head columns must not change the ordering or saturate
    df2 = df.copy()
    df2["rerank_score"] = 0.0
    df2["tournament_score"] = 0.0
    after = merge_final_scores(df2, ml, retr)
    assert (before.rank().values == after.rank().values).all()
    assert after.between(0, 1).all()


def test_head_blend_can_reorder_and_stays_bounded():
    df = _scored_frame()
    ml = np.full(len(df), 0.5)
    retr = np.full(len(df), 0.5)
    flat = merge_final_scores(df, ml, retr)
    # a strong head signal on the last row should lift it above the tie
    df["rerank_score"] = 0.0
    df["tournament_score"] = 0.0
    df.iloc[-1, df.columns.get_loc("rerank_score")] = 1.0
    df.iloc[-1, df.columns.get_loc("tournament_score")] = 1.0
    blended = merge_final_scores(df, ml, retr)
    assert blended.between(0, 1).all()
    assert blended.iloc[-1] == blended.max()
    assert blended.iloc[-1] > flat.iloc[-1]
