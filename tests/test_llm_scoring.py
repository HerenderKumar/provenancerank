"""LLM head-scoring: the scorer (`score_head`) batches + normalises, and the
final-score merge blends `llm_fit_score` into the head. The LLM call is mocked, so
no provider is needed."""

from __future__ import annotations

import re

import numpy as np

from ml import llm_scorer
from pipeline.feature_engineering import build_feature_row, frame_from_rows
from pipeline.scorer import merge_final_scores
from tests.conftest import make_candidate


def test_score_head_batches_and_normalises(monkeypatch):
    calls = {"n": 0}

    def fake_complete_json(prompt, timeout=None):
        calls["n"] += 1
        cids = re.findall(r"\[(CAND_\d+)\]", prompt)   # echo a score per candidate
        return dict.fromkeys(cids, 90)

    monkeypatch.setattr(llm_scorer, "complete_json", fake_complete_json)

    items = [(f"CAND_{i:07d}", f"profile text {i}") for i in range(45)]
    out = llm_scorer.score_head("the JD", items, batch_size=20)

    assert calls["n"] == 3                      # 45 candidates / 20 per call -> 3 batches
    assert len(out) == 45
    assert all(0.0 <= v <= 1.0 for v in out.values())
    assert out["CAND_0000000"] == 0.9           # 90/100 normalised


def test_score_head_noop_without_provider(monkeypatch):
    monkeypatch.setattr(llm_scorer, "complete_json", lambda *a, **k: None)
    out = llm_scorer.score_head("jd", [("CAND_0000001", "x")])
    assert out == {}                            # nothing reachable -> empty -> scorer no-op


def _scored_frame(n=8):
    rows = [build_feature_row(make_candidate(cid=f"CAND_{i:07d}")) for i in range(1, n + 1)]
    df = frame_from_rows(rows)
    df["gate_passed"] = True
    return df


def test_head_blend_includes_llm_fit_score():
    df = _scored_frame()
    ml = np.full(len(df), 0.5)
    retr = np.full(len(df), 0.5)
    flat = merge_final_scores(df, ml, retr)        # no llm column yet

    df["llm_fit_score"] = 0.0
    df.iloc[-1, df.columns.get_loc("llm_fit_score")] = 1.0  # LLM loves the last one
    blended = merge_final_scores(df, ml, retr)

    assert blended.between(0, 1).all()
    assert blended.iloc[-1] == blended.max()       # the LLM-favoured row rises to the top
    assert blended.iloc[-1] > flat.iloc[-1]
