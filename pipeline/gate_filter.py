"""Hard gate filter - runs at ranking time, vectorised, no LLM, no network.

Only HARD disqualifiers exclude a candidate here (honeypot, all-consulting
career, keyword stuffer, dead-and-unresponsive). Notice period, location and
wrong-specialisation are deliberately *not* hard gates: the JD says those
candidates stay in scope, just "the bar gets higher", so they're handled as
score penalties later. We record them as soft_flags for auditing.

Excluded candidates are dumped to artifacts/excluded_candidates.csv with their
reasons, so a weird top-100 can always be traced back to who got dropped and
why (this isn't part of the submission).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from core.logging import get_logger
from pipeline.jd_deconstructor import JobSpec

log = get_logger("pipeline.gate_filter")


def apply_gates(df: pd.DataFrame, jd_spec: JobSpec | None = None) -> pd.DataFrame:
    df = df.copy()
    n = len(df)
    gate_passed = pd.Series(True, index=df.index)
    failures = [[] for _ in range(n)]
    soft = [[] for _ in range(n)]

    def hard(mask: pd.Series, reason: str) -> None:
        nonlocal gate_passed
        mask = mask.fillna(False)
        gate_passed &= ~mask
        for i in df.index[mask]:
            failures[df.index.get_loc(i)].append(reason)

    def flag(mask: pd.Series, reason: str) -> None:
        mask = mask.fillna(False)
        for i in df.index[mask]:
            soft[df.index.get_loc(i)].append(reason)

    # hard gates: these actually drop the candidate
    hard(df["is_honeypot"] == 1, "honeypot")
    hard(df["all_consulting_career"] == 1, "all_consulting_career")
    hard(df["keyword_stuffer_flag"] == 1, "keyword_stuffer")

    rr = pd.to_numeric(df.get("recruiter_response_rate", 0), errors="coerce").fillna(0)
    inactive = (df["days_since_active"] > 180) & (rr < 0.15)
    hard(inactive, "inactive_unresponsive")

    # soft flags: recorded for the audit trail; scoring handles them, not the gate
    flag(df["notice_period_days"] > 90, "notice_gt_90")
    flag(df["title_chaser_flag"] == 1, "title_chaser")
    flag(df["wrong_specialisation_penalty"] > 0.5, "cv_speech_no_nlp")
    flag(df["location_fit"] < 0.5, "outside_india_focus")
    flag((df["days_since_active"] > 180) & ~inactive, "stale_but_responsive")

    df["gate_passed"] = gate_passed.values
    df["gate_failures"] = failures
    df["soft_flags"] = soft

    excluded = int((~gate_passed).sum())
    breakdown: dict[str, int] = {}
    for fl in failures:
        for r in fl:
            breakdown[r] = breakdown.get(r, 0) + 1
    log.info(
        "gate_filter.complete",
        total=n,
        passed=int(gate_passed.sum()),
        excluded=excluded,
        breakdown=breakdown,
    )
    return df


def write_rejection_registry(df: pd.DataFrame, path: str | Path) -> int:
    """Dead-letter style audit file for everyone we dropped."""
    rejected = df[~df["gate_passed"]].copy()
    if rejected.empty:
        return 0
    out = rejected[["candidate_id", "current_title", "gate_failures"]].copy()
    out["gate_failures"] = out["gate_failures"].apply(lambda x: "|".join(x))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    log.info("gate_filter.registry_written", path=str(path), rows=len(out))
    return len(out)
