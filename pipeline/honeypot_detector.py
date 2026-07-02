"""Catch the ~80 honeypot profiles (impossible career maths -> ground-truth
tier 0). Ranking them costs us, and >10% in the top-100 is an instant DQ, so
we flag them during feature engineering and force their score to zero.

Rules are deliberately conservative — each one is an actual impossibility, not
just a "weak" profile. I'd rather miss a few honeypots (scoring buries them
anyway) than wrongly drop a real candidate. Every flag carries a reason string
so exclusions show up in the rejection registry.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

# Reference "today" for the dataset snapshot (deterministic, not wall-clock, so
# the detector is reproducible regardless of when precompute runs).
DATASET_TODAY = dt.date(2026, 6, 19)
_MONTH = 30.44


def _parse_date(value: Any) -> dt.date | None:
    if not value:
        return None
    try:
        return dt.date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _career_span_months(career: list[dict]) -> float:
    """Calendar months from earliest start to latest end across all roles."""
    starts, ends = [], []
    for r in career:
        s = _parse_date(r.get("start_date"))
        if s:
            starts.append(s)
        e = _parse_date(r.get("end_date")) or DATASET_TODAY
        ends.append(e)
    if not starts or not ends:
        return 0.0
    return max((max(ends) - min(starts)).days / _MONTH, 0.0)


def detect_honeypot(candidate: dict) -> tuple[bool, list[str]]:
    """Return ``(is_honeypot, reasons)`` for a single candidate."""
    flags: list[str] = []
    career = candidate.get("career_history", []) or []
    skills = candidate.get("skills", []) or []
    profile = candidate.get("profile", {}) or {}
    signals = candidate.get("redrob_signals", {}) or {}

    total_career_months = sum(int(r.get("duration_months", 0) or 0) for r in career)
    yoe = float(profile.get("years_of_experience", 0) or 0)

    # Rule 1: implausibly long tenure at a tiny company (8+ yrs at a 1-50 co).
    for r in career:
        if r.get("company_size") in ("1-10", "11-50"):
            dur = int(r.get("duration_months", 0) or 0)
            if dur > 96:
                flags.append(f"implausible_tenure:{r.get('company', '?')}:{dur}mo")

    # Rule 2: per-role duration_months contradicts its own start/end dates.
    for r in career:
        s = _parse_date(r.get("start_date"))
        e = _parse_date(r.get("end_date")) or DATASET_TODAY
        if s and e:
            calendar = (e - s).days / _MONTH
            dur = int(r.get("duration_months", 0) or 0)
            if dur > calendar + 9:  # ~9 month tolerance for rounding/overlap
                flags.append(
                    f"duration_exceeds_calendar:{r.get('company', '?')}:{dur}>{calendar:.0f}mo"
                )

    # Rule 3: >= 3 "expert" skills with zero months of usage.
    expert_zero = [
        s
        for s in skills
        if s.get("proficiency") == "expert" and int(s.get("duration_months", 0) or 0) == 0
    ]
    if len(expert_zero) >= 3:
        flags.append(f"expert_zero_duration:{len(expert_zero)}_skills")

    # Rule 4: stated years_of_experience far exceeds summed career history.
    # (A skill-duration-vs-career rule was evaluated and rejected: in this
    # dataset skill durations are sampled independently of tenure and produced
    # ~660 false positives, so it is intentionally NOT used.)
    if total_career_months > 0 and yoe > (total_career_months / 12.0) + 3.0:
        flags.append(f"yoe_exceeds_career_history:{yoe:.1f}>{total_career_months / 12:.1f}yr")

    # Rule 5: physically impossible platform activity (defensive guard; the
    # released data caps at 24/30d, so this protects against future drift).
    apps = int(signals.get("applications_submitted_30d", 0) or 0)
    if apps > 50:
        flags.append(f"impossible_application_volume:{apps}")

    # NOTE: "many AI keywords on a non-technical career" is the *keyword
    # stuffer* trap, not an impossible profile. It is handled with high recall
    # by ``keyword_stuffer_flag`` in feature_engineering (a heavy penalty),
    # NOT here, because those profiles are possible — just mismatched.
    return (len(flags) > 0, flags)


def is_honeypot(candidate: dict) -> bool:
    return detect_honeypot(candidate)[0]
