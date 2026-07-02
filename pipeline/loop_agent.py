"""Lightweight plan/retrieve/reflect loop. No LLM, no network - it just reads
the precomputed feature matrix and relaxes a fit threshold until the shortlist
pool has enough candidates covering each JD dimension (retrieval, LLM,
production evidence, product-company, availability).

The point isn't to "find" candidates - scoring does that. It's to guarantee the
pool we hand to final scoring isn't accidentally starved of, say, anyone with
real retrieval experience, and to produce the coverage numbers the UI shows.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from core.config import get_settings
from core.logging import get_logger

log = get_logger("pipeline.loop_agent")

# Each JD dimension -> predicate over a candidate row (vectorised below).
COVERAGE_DIMS = {
    "retrieval_experience": lambda d: d["retrieval_skill_count"] > 0,
    "llm_experience": lambda d: d["llm_skill_count"] > 0,
    "production_evidence": lambda d: d["has_production_keywords"] > 0,
    "product_company": lambda d: d["product_company_ratio"] > 0.5,
    "available": lambda d: d["availability_score"] > 0.3,
}


@dataclass
class LoopState:
    candidates_pool: list[str]
    coverage_scores: dict[str, float] = field(default_factory=dict)
    gaps: list[str] = field(default_factory=list)
    iteration: int = 0
    max_iterations: int = 4
    confidence_threshold: float = 0.80
    gap_threshold: float = 0.60
    threshold_relaxation: float = 0.0
    exit_reason: str = ""
    history: list[dict] = field(default_factory=list)


def _coverage(pool_df: pd.DataFrame, target: int) -> dict[str, float]:
    out = {}
    for name, pred in COVERAGE_DIMS.items():
        covering = int(pred(pool_df).sum())
        out[name] = min(covering / max(target, 1), 1.0)
    return out


def plan(state: LoopState) -> list[str]:
    """Which dimensions are short."""
    return [d for d, c in state.coverage_scores.items() if c < state.gap_threshold]


def retrieve(state: LoopState, feature_df: pd.DataFrame, base_fit: float) -> list[str]:
    """Widen the pool by relaxing the fit threshold. Anyone gate-passed with real
    retrieval/LLM signal is always allowed in regardless of threshold."""
    min_fit = max(base_fit - state.threshold_relaxation, 0.0)
    passed = feature_df["gate_passed"] if "gate_passed" in feature_df else True
    keep = passed & (
        (feature_df["jd_fit_score"] >= min_fit)
        | (feature_df["retrieval_skill_count"] > 0)
        | (feature_df["llm_skill_count"] > 0)
    )
    return feature_df.loc[keep, "candidate_id"].tolist()


def reflect(
    state: LoopState, pool_ids: list[str], feature_df: pd.DataFrame, target: int
) -> LoopState:
    pool_df = feature_df[feature_df["candidate_id"].isin(pool_ids)]
    state.candidates_pool = pool_ids
    state.coverage_scores = _coverage(pool_df, target)
    state.gaps = plan(state)
    state.history.append(
        {
            "iteration": state.iteration,
            "pool_size": len(pool_ids),
            "min_coverage": round(min(state.coverage_scores.values(), default=0), 3),
            "gaps": list(state.gaps),
        }
    )
    return state


def should_continue(state: LoopState) -> bool:
    if state.iteration >= state.max_iterations:
        state.exit_reason = "max_iterations"
        return False
    if min(state.coverage_scores.values(), default=0) >= state.confidence_threshold:
        state.exit_reason = "confident"
        return False
    return True


def run_loop(
    feature_df: pd.DataFrame, target: int | None = None, base_fit: float = 0.6
) -> tuple[list[str], LoopState]:
    s = get_settings()
    target = target or s.top_k
    state = LoopState(
        candidates_pool=[],
        max_iterations=s.max_loop_iterations,
        confidence_threshold=s.confidence_threshold,
        gap_threshold=s.coverage_gap_threshold,
    )
    # iteration 0: tight pool
    pool = retrieve(state, feature_df, base_fit)
    state = reflect(state, pool, feature_df, target)
    while should_continue(state):
        state.iteration += 1
        state.threshold_relaxation += 0.15  # widen each pass
        pool = retrieve(state, feature_df, base_fit)
        state = reflect(state, pool, feature_df, target)

    log.info(
        "loop_agent.done",
        iterations=state.iteration,
        exit_reason=state.exit_reason,
        pool_size=len(state.candidates_pool),
        coverage=state.coverage_scores,
    )
    return state.candidates_pool, state
