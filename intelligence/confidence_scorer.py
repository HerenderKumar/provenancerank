"""Confidence = how much we trust a skill claim, given the evidence behind it.

This is the whole point of the live layer. A skill proven by 20 recent
production commit diffs should crush the same skill typed into a resume. The
formula encodes three things: where the evidence came from (a commit diff beats
a self-claim), how hard it was (a race-condition fix beats a typo), and how
recent it is (skills rot - exponential decay, 1-year half-life).

    score(skill) = clamp( Σ  source_weight · (complexity/5) · 2^(-age_days/365)
                          ----------------------------------------------------- , 0, 1)
                                              10

The /10 normaliser means ~10 perfect, recent commit diffs == full confidence.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

# how much each kind of evidence is worth. self-reported claims barely count.
SOURCE_WEIGHTS: dict[str, float] = {
    "commit_diff": 1.0,
    "pr_review": 0.9,
    "issue_thread": 0.7,
    "so_answer": 0.6,
    "assessment_score": 0.5,
    "resume_claim": 0.2,
}

PROFICIENCY_WEIGHTS: dict[str, float] = {
    "expert": 1.0,
    "advanced": 0.8,
    "intermediate": 0.5,
    "beginner": 0.25,
}

HALF_LIFE_DAYS = 365.0
NORMALISER = 10.0


@dataclass
class EvidenceRecord:
    source_type: str
    skill_name: str
    artifact_date: dt.datetime
    complexity_score: int  # 1-5
    source_url: str = ""
    content_hash: str = ""


def _as_utc(d: dt.datetime) -> dt.datetime:
    return d if d.tzinfo else d.replace(tzinfo=dt.timezone.utc)


def _decay(age_days: float, half_life: float = HALF_LIFE_DAYS) -> float:
    return 2.0 ** (-max(age_days, 0.0) / half_life)


def compute_confidence(
    evidence_list: list[EvidenceRecord],
    skill_name: str,
    as_of: dt.datetime | None = None,
    *,
    half_life: float = HALF_LIFE_DAYS,
    normaliser: float = NORMALISER,
) -> float:
    """Confidence in [0,1] for one skill, from its evidence as of a given date."""
    now = _as_utc(as_of or dt.datetime.now(dt.timezone.utc))
    target = skill_name.strip().lower()

    total = 0.0
    for ev in evidence_list:
        if ev.skill_name.strip().lower() != target:
            continue
        weight = SOURCE_WEIGHTS.get(ev.source_type, 0.2)
        complexity = max(1, min(int(ev.complexity_score or 1), 5)) / 5.0
        age = (now - _as_utc(ev.artifact_date)).total_seconds() / 86400.0
        total += weight * complexity * _decay(age, half_life)

    return max(0.0, min(total / normaliser, 1.0))


def confidence_by_skill(
    evidence_list: list[EvidenceRecord], as_of: dt.datetime | None = None
) -> dict[str, float]:
    """Confidence for every distinct skill present in the evidence."""
    skills = {ev.skill_name.strip().lower() for ev in evidence_list}
    return {s: compute_confidence(evidence_list, s, as_of) for s in skills}
