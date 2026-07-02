"""Graded relevance labels (tier 0..5) for learning-to-rank.

The dataset ships no hire/reject labels, so the GBDT was fitting a hand-written
proxy — which caps accuracy. Here we build *graded* labels two ways:

  * Gemini grades a sample of candidates 0..5 against the JD (offline, one-time,
    cached). This is the real signal — the model learns the actual relevance
    definition, not our formula.
  * Without an API key we fall back to a graded heuristic: a richer composite
    than the GBDT proxy, bucketed into tiers by percentile so the label
    distribution looks like the real one ("a few great matches, lots of noise").

Either way the output is an integer 0..5 Series aligned to the feature matrix,
ready for an NDCG-objective ranker.
"""

from __future__ import annotations

import json

import pandas as pd

from core.config import get_settings
from core.logging import get_logger, log_duration

log = get_logger("ml.pseudo_labels")

# tier shares, best-first: ~1% are tier 5, then 4, 8, 16, 30, rest tier 0.
_TIER_QUANTILES = [(5, 0.99), (4, 0.97), (3, 0.90), (2, 0.75), (1, 0.50)]


def _norm(s: pd.Series) -> pd.Series:
    lo, hi = float(s.min()), float(s.max())
    return (s - lo) / (hi - lo) if hi - lo > 1e-9 else pd.Series(0.0, index=s.index)


def _heuristic_relevance(df: pd.DataFrame) -> pd.Series:
    """A discriminative 0..1 relevance, richer than the GBDT proxy."""
    rel = (
        0.34 * df["jd_fit_score"]
        + 0.22 * _norm(df["retrieval_skill_count"])
        + 0.12 * _norm(df["assessment_mean_score"])
        + 0.12 * df["has_production_keywords"]
        + 0.10 * df["product_company_ratio"]
        + 0.10 * df["behavioural_composite"].clip(0, 1)
    )
    if "gate_passed" in df.columns:
        rel = rel.where(df["gate_passed"], 0.0)
    rel = rel.where(df["is_honeypot"] != 1, 0.0)
    return rel.clip(0, 1)


def _bucket_to_tiers(relevance: pd.Series) -> pd.Series:
    """Map a continuous relevance to integer tiers 0..5 by percentile.

    Assign from the lowest cutoff up so higher tiers overwrite lower ones for the
    very top rows — otherwise the last (lowest) band would clobber the elite tiers
    and everything collapses to tier 1.
    """
    tiers = pd.Series(0, index=relevance.index, dtype=int)
    positive = relevance[relevance > 0]
    if positive.empty:
        return tiers
    for tier, q in sorted(_TIER_QUANTILES, key=lambda tq: tq[1]):
        cut = positive.quantile(q)
        tiers[relevance >= cut] = tier
    return tiers


def heuristic_pseudo_labels(df: pd.DataFrame) -> pd.Series:
    return _bucket_to_tiers(_heuristic_relevance(df))


# ---------------------------------------------------------------------------
# Gemini grading (offline, one-time, cached). Falls back to the heuristic.
# ---------------------------------------------------------------------------

def _llm_grade(candidates: list[dict], jd_text: str) -> dict[str, int] | None:
    """Grade the sample 0..5 via the configured LLM (local Ollama or Gemini).

    Each batch is bounded by a timeout and logged, so a slow provider can't hang
    precompute silently. Returns None when no provider is reachable (or the first
    batch fails), so the caller keeps the heuristic labels."""
    from core.llm import complete_json

    timeout = int(getattr(get_settings(), "pseudo_label_llm_timeout", 30))
    grades: dict[str, int] = {}
    n_batches = (len(candidates) + 19) // 20
    for i, batch_start in enumerate(range(0, len(candidates), 20), start=1):
        batch = candidates[batch_start : batch_start + 20]
        prompt = (
            "Grade each candidate 0-5 for fit to the JD (5=ideal hire, "
            "3=relevant, 0=irrelevant/impossible). Return STRICT JSON "
            '{candidate_id: grade}. JD:\n' + jd_text[:4000] + "\n\nCANDIDATES:\n"
            + json.dumps(batch)[:12000]
        )
        data = complete_json(prompt, timeout=timeout)
        if not data:
            return grades or None
        for cid, g in data.items():
            try:
                grades[cid] = max(0, min(int(g), 5))
            except (ValueError, TypeError):
                continue
        log.info("pseudo_labels.llm_progress", batch=i, of=n_batches, graded=len(grades))
    return grades


def generate_pseudo_labels(df: pd.DataFrame, candidates: list[dict] | None = None,
                           jd_text: str | None = None, sample_size: int = 1500) -> pd.Series:
    """Graded 0..5 labels aligned to df. An LLM (Ollama/Gemini) grades a sample
    when enabled, heuristic for the rest — so every row gets a tier."""
    with log_duration(log, "pseudo_labels.generate") as m:
        labels = heuristic_pseudo_labels(df)
        source = "heuristic"
        _s = get_settings()
        if candidates and jd_text and _s.pseudo_label_llm_enabled:
            # grade the most promising slice (where label precision matters most)
            top_idx = labels.sort_values(ascending=False).head(sample_size).index
            id_to_cand = {c["candidate_id"]: c for c in candidates}
            sample = [id_to_cand[c] for c in df.loc[top_idx, "candidate_id"] if c in id_to_cand]
            graded = _llm_grade(sample, jd_text)
            if graded:
                source = "llm+heuristic"
                cid_to_idx = {c: i for i, c in zip(df.index, df["candidate_id"], strict=False)}
                for cid, g in graded.items():
                    if cid in cid_to_idx:
                        labels[cid_to_idx[cid]] = g
        m["source"] = source
        m["distribution"] = {int(t): int((labels == t).sum()) for t in range(6)}
    return labels.astype(int)
