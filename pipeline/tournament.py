"""Pairwise tournament re-ordering of the head — our own ranking algorithm.

Pointwise scoring (hand every candidate a number, sort) has a blind spot: the
numbers come from features on different scales, and one noisy column can drag a
great candidate down. Recruiters don't think pointwise — they think "between
these two, who's better?". So for the very top of the list we run an actual
round-robin tournament:

  * every candidate plays every other once;
  * a match is judged on several JD-relevant criteria (fit, proven skills,
    assessments, production signal, behaviour, live evidence). A beats B on a
    criterion by being *ordinally* better, and the match result is the weighted
    share of criteria A won — a soft 0..1 outcome, not just win/lose;
  * a round-robin contains cycles (A>B>C>A), so we resolve the whole thing with a
    Bradley-Terry model fit by MM iteration (Hunter, 2004): the maximum-
    likelihood global strength that best explains every pairwise result.

Because only ordinal comparisons matter, it's immune to feature-scale quirks, and
it surfaces the genuinely dominant candidates instead of whoever spiked a single
column. Runs offline over the top-N; the strength is cached as ``tournament_score``
and blended by the scorer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.logging import get_logger, log_duration

log = get_logger("pipeline.tournament")

# Criterion column -> weight. Only columns actually present in the matrix are
# used (rerank_score / evidence appear once their stages have run), and the kept
# weights are renormalised — so this degrades cleanly if a stage was skipped.
_CRITERIA: dict[str, float] = {
    "jd_fit_score": 0.26,
    "rerank_score": 0.20,
    "retrieval_skill_count": 0.14,
    "jd_skill_match_count": 0.12,
    "assessment_mean_score": 0.12,
    "evidence_confidence_bonus": 0.05,
    "has_production_keywords": 0.04,
    "product_company_ratio": 0.04,
    "behavioural_composite": 0.03,
}


def _soft_outcomes(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """W[i, j] = weighted share of criteria on which i beats j (ties split).

    ``values`` is (N, C); ``weights`` is (C,) summing to 1. Vectorised per
    criterion via broadcasting — N is small (top-N), so the N*N matrices are cheap.
    """
    n = values.shape[0]
    wins = np.zeros((n, n), dtype=float)
    for c in range(values.shape[1]):
        col = values[:, c][:, None]  # (N, 1)
        greater = (col > col.T).astype(float)
        ties = (col == col.T).astype(float)
        np.fill_diagonal(ties, 0.0)
        wins += weights[c] * (greater + 0.5 * ties)
    np.fill_diagonal(wins, 0.0)
    return wins


def _bradley_terry(wins: np.ndarray, iters: int = 200, tol: float = 1e-10) -> np.ndarray:
    """MLE Bradley-Terry strengths via Hunter's MM update.

        p_i <- W_i / Σ_{j!=i} n_ij / (p_i + p_j)

    Every pair plays once, so n_ij = 1 off-diagonal. Normalise to sum 1 each step
    to stop the scale drifting; converges monotonically in likelihood.
    """
    n = wins.shape[0]
    win_totals = wins.sum(axis=1)
    p = np.full(n, 1.0 / n, dtype=float)
    for _ in range(iters):
        pij = p[:, None] + p[None, :]
        np.fill_diagonal(pij, 1.0)  # placeholder; diagonal is masked out below
        inv = 1.0 / pij
        np.fill_diagonal(inv, 0.0)
        denom = inv.sum(axis=1)
        new_p = np.where(denom > 0, win_totals / denom, 0.0)
        total = new_p.sum()
        if total > 0:
            new_p /= total
        if np.max(np.abs(new_p - p)) < tol:
            p = new_p
            break
        p = new_p
    return p


def run_tournament(df_head: pd.DataFrame, id_col: str = "candidate_id") -> dict[str, float]:
    """Bradley-Terry strength in [0,1] for each row of the head set.

    Fewer than two rows, or no usable criteria, yields all-zeros (a no-op the
    scorer simply adds nothing for).
    """
    cols = [c for c in _CRITERIA if c in df_head.columns]
    ids = list(df_head[id_col])
    if len(df_head) < 2 or not cols:
        return dict.fromkeys(ids, 0.0)

    weights = np.array([_CRITERIA[c] for c in cols], dtype=float)
    weights /= weights.sum()
    values = df_head[cols].to_numpy(dtype=float)

    with log_duration(log, "tournament.run") as m:
        wins = _soft_outcomes(values, weights)
        # Pull every match a touch toward a draw (a Beta(a,a)-style prior) so the
        # MLE stays finite even for a candidate that dominates or loses every
        # criterion against everyone.
        n = wins.shape[0]
        eps = 0.02
        draw = 0.5 * (np.ones((n, n)) - np.eye(n))
        wins = (1.0 - eps) * wins + eps * draw
        strengths = _bradley_terry(wins)
        m["n"] = n
        m["criteria"] = len(cols)

    lo, hi = float(strengths.min()), float(strengths.max())
    norm = (strengths - lo) / (hi - lo) if hi - lo > 1e-12 else np.zeros_like(strengths)
    return {cid: float(v) for cid, v in zip(ids, norm, strict=False)}
