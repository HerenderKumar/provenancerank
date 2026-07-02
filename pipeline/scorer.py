"""Final-score merge — shared by rank.py and precompute so the formula lives in
exactly one place.

    core   = 0.40*ml_fit + 0.35*retrieval + 0.25*behavioural
    base   = (1 - w_head)*core + w_head_blend(rerank, tournament)   # head only
    final  = base * location_factor * signal_modifier
             + evidence_confidence_bonus                            # live add-on

Then anything gated out or flagged honeypot is pinned to 0 so it can't reach the
top 100. Weights come from config; the location factor is a gentle nudge (the
JD is India-first but "case by case" for relocators) and the signal modifier is
the availability/engagement down-weight.

Two add-ons fold in here, both designed to be invisible when their inputs are
absent so the graded submission.csv is unchanged with the extras switched off:

  * the cross-encoder rerank + Bradley-Terry tournament reorder the *head* of the
    list. They're folded in as a convex blend (we shave the same weight off the
    pointwise core), so head scores stay in [0,1] instead of saturating at 1 and
    losing their ordering. Outside the head both columns are 0; absent entirely,
    the core keeps its full weight.
  * the evidence bonus is how the live ingestion layer surfaces verified
    developers above identical static-only ones — an additive term that's 0 for
    every candidate without graph-backed evidence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.config import get_settings
from pipeline.signal_scorer import signal_modifier


def _head_blend(df: pd.DataFrame, s) -> tuple[pd.Series, float] | None:
    """Weighted head signals (cross-encoder rerank, Bradley-Terry tournament, and
    the offline LLM fit score) plus the total weight to shave off the pointwise
    core. None when none of the columns is present (every add-on off)."""
    weights = {
        "rerank_score": float(getattr(s, "rerank_weight", 0.15)),
        "tournament_score": float(getattr(s, "tournament_weight", 0.10)),
        "llm_fit_score": float(getattr(s, "llm_score_weight", 0.15)),
    }
    present = {c: w for c, w in weights.items() if c in df.columns and w > 0}
    if not present:
        return None

    w_head = sum(present.values())
    signal = pd.Series(0.0, index=df.index)
    for col, w in present.items():
        signal = signal + w * pd.to_numeric(df[col], errors="coerce").fillna(0.0).clip(0, 1)
    return signal, w_head


def merge_final_scores(df: pd.DataFrame, ml_fit: np.ndarray, retrieval: np.ndarray) -> pd.Series:
    s = get_settings()
    ml_fit = pd.Series(np.asarray(ml_fit, dtype=float), index=df.index).clip(0, 1)
    retrieval = pd.Series(np.asarray(retrieval, dtype=float), index=df.index).clip(0, 1)
    behavioural = df["behavioural_composite"].clip(0, 1)

    base = (
        s.ml_score_weight * ml_fit
        + s.retrieval_score_weight * retrieval
        + s.signal_score_weight * behavioural
    )
    # accuracy add-ons reorder the head as a convex blend (no saturation).
    head = _head_blend(df, s)
    if head is not None:
        signal, w_head = head
        base = (1.0 - w_head) * base + signal

    location_factor = 0.85 + 0.15 * df["location_fit"].clip(0, 1)
    final = base * location_factor * signal_modifier(df)

    # live-layer add-on: verified developers float above static-only twins.
    # Absent the live layer this column is all zeros and changes nothing.
    if "evidence_confidence_bonus" in df.columns:
        final = final + pd.to_numeric(df["evidence_confidence_bonus"], errors="coerce").fillna(0.0)

    if "gate_passed" in df.columns:
        final = final.where(df["gate_passed"], 0.0)
    final = final.where(df["is_honeypot"] != 1, 0.0)
    return final.clip(0, 1)
