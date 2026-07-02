# This is the file that decides ranking quality, so the thresholds here were
# tuned against the real candidates.jsonl, not guessed. The output DataFrame is
# pickled once (artifacts/feature_matrix.pkl) and only *read* at ranking time —
# that's what keeps the 5-minute budget realistic.
#
# What the JD actually rewards, after reading between the lines:
#   - retrieval / IR / ranking skills are the signal that matters most
#   - AI keywords on a non-technical title = the stuffer trap -> kill it
#   - CV/speech depth with no NLP/IR = "wrong specialisation" -> penalise
#   - behavioural signals decide whether a great profile is even reachable
"""Per-candidate feature vector (40 numeric features + a few display columns)."""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from core.config import get_settings
from core.constants import (
    CONSULTING_FIRMS,
    CORE_ML_SKILLS,
    CV_SPEECH_ROBOTICS_SKILLS,
    JD_CORE_SKILLS,
    LLM_SKILLS,
    NON_RELEVANT_SKILLS,
    NON_TECHNICAL_TITLE_TOKENS,
    NUMERIC_FEATURE_COLUMNS,
    PREFERRED_LOCATIONS,
    PRODUCTION_KEYWORDS,
    PROFICIENCY_WEIGHTS,
    RETRIEVAL_SKILLS,
    TECHNICAL_TITLE_TOKENS,
)
from core.logging import get_logger, log_duration
from pipeline.honeypot_detector import DATASET_TODAY, detect_honeypot

log = get_logger("pipeline.feature_engineering")

_TITLE_LEVELS: list[tuple[tuple[str, ...], int]] = [
    (("director", "vp", "vice president", "head of", "chief"), 7),
    (("manager", "lead manager"), 6),
    (("principal", "architect", "distinguished"), 5),
    (("staff", "lead", "tech lead"), 4),
    (("senior", "sr.", "sr "), 3),
    (("intern", "trainee"), 0),
    (("junior", "jr.", "associate", "graduate"), 1),
]


def _parse_date(value: Any) -> dt.date | None:
    if not value:
        return None
    try:
        return dt.date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _title_level(title: str) -> int:
    """Ordinal seniority of a title (0 intern .. 7 director). Default mid=2."""
    t = title.lower()
    for tokens, level in _TITLE_LEVELS:
        if any(tok in t for tok in tokens):
            return level
    return 2  # plain "Engineer" / "Developer" => mid


def _is_consulting(company: str) -> bool:
    c = company.lower()
    return any(firm in c for firm in CONSULTING_FIRMS)


def _is_technical_title(title: str) -> bool:
    t = title.lower()
    if any(tok in t for tok in NON_TECHNICAL_TITLE_TOKENS):
        return False
    return any(tok in t for tok in TECHNICAL_TITLE_TOKENS)


def _title_relevance_factor(title: str) -> float:
    """1.0 AI/ML/data, 0.55 other technical, 0.0 non-technical, 0.3 unknown."""
    t = title.lower()
    if any(tok in t for tok in NON_TECHNICAL_TITLE_TOKENS):
        return 0.0
    if any(
        tok in t
        for tok in (
            "ml",
            "ai",
            "machine learning",
            "data scientist",
            "applied scientist",
            "nlp",
            "research",
            "data engineer",
        )
    ):
        return 1.0
    if any(tok in t for tok in TECHNICAL_TITLE_TOKENS):
        return 0.55
    return 0.3


# ---------------------------------------------------------------------------
# Career
# ---------------------------------------------------------------------------


def compute_career_features(candidate: dict) -> dict:
    career = candidate.get("career_history", []) or []
    profile = candidate.get("profile", {}) or {}
    roles = sorted(career, key=lambda r: _parse_date(r.get("start_date")) or dt.date(1900, 1, 1))
    durations = [int(r.get("duration_months", 0) or 0) for r in roles]
    yoe = float(profile.get("years_of_experience", 0) or 0)

    # scope progression: ordinal level change across the career, squashed to 0-1.
    if len(roles) >= 2:
        first_lvl = _title_level(roles[0].get("title", ""))
        last_lvl = _title_level(roles[-1].get("title", ""))
        scope = float(np.clip(0.5 + 0.1 * (last_lvl - first_lvl), 0.0, 1.0))
    else:
        scope = 0.5

    # career gaps within the last 60 months (>=6 month gap between roles).
    gap = False
    for prev, cur in zip(roles, roles[1:], strict=False):
        prev_end = _parse_date(prev.get("end_date")) or DATASET_TODAY
        cur_start = _parse_date(cur.get("start_date"))
        if cur_start and (cur_start - prev_end).days > 183:
            if (DATASET_TODAY - cur_start).days < 60 * 30:
                gap = True
                break

    non_consulting = [r for r in roles if not _is_consulting(r.get("company", ""))]
    product_ratio = (len(non_consulting) / len(roles)) if roles else 0.0

    # current role recency
    current = next((r for r in roles if r.get("is_current")), roles[-1] if roles else None)
    if current and _parse_date(current.get("start_date")):
        last_recency = max((DATASET_TODAY - _parse_date(current["start_date"])).days / 30.44, 0.0)
    else:
        last_recency = 0.0

    # title chaser: 3+ companies in last 4 years, each tenure < 18 months
    recent = [
        r
        for r in roles
        if (_parse_date(r.get("start_date")) or dt.date(1900, 1, 1)) >= dt.date(2022, 6, 19)
    ]
    short_recent = [r for r in recent if int(r.get("duration_months", 0) or 0) < 18]
    distinct_recent_companies = len({r.get("company", "") for r in short_recent})
    title_chaser = distinct_recent_companies >= 3

    descriptions = " ".join(str(r.get("description", "")) for r in roles).lower()
    avg_desc_words = (
        float(np.mean([len(str(r.get("description", "")).split()) for r in roles]))
        if roles
        else 0.0
    )
    has_prod_kw = any(kw in descriptions for kw in PRODUCTION_KEYWORDS)

    unique_titles = len({r.get("title", "") for r in roles}) or 1
    return {
        "role_count": len(roles),
        "total_experience_years": yoe,
        "avg_tenure_months": float(np.mean(durations)) if durations else 0.0,
        "scope_progression_score": scope,
        "has_recent_career_gap": float(gap),
        "product_company_ratio": product_ratio,
        "all_consulting_career": float(bool(roles) and len(non_consulting) == 0),
        "last_role_start_recency_months": last_recency,
        "avg_title_tenure_months": (sum(durations) / unique_titles) if durations else 0.0,
        "title_chaser_flag": float(title_chaser),
        "avg_description_words": avg_desc_words,
        "has_production_keywords": float(has_prod_kw),
        "_technical_role_count": len([r for r in roles if _is_technical_title(r.get("title", ""))]),
    }


# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------


def compute_skill_features(candidate: dict, career: dict) -> dict:
    skills = candidate.get("skills", []) or []
    signals = candidate.get("redrob_signals", {}) or {}
    assessments = signals.get("skill_assessment_scores", {}) or {}
    profile = candidate.get("profile", {}) or {}

    names = [str(s.get("name", "")).lower() for s in skills]
    matched = [s for s in skills if str(s.get("name", "")).lower() in JD_CORE_SKILLS]
    retrieval = [n for n in names if n in RETRIEVAL_SKILLS]
    llm = [n for n in names if n in LLM_SKILLS]
    core_ml = [n for n in names if n in CORE_ML_SKILLS]
    cv = [n for n in names if n in CV_SPEECH_ROBOTICS_SKILLS]
    non_relevant = [n for n in names if n in NON_RELEVANT_SKILLS]

    specialised = len(retrieval) + len(llm) + len(core_ml) + len(cv)
    cv_dominance = (len(cv) / specialised) if specialised else 0.0

    # trust-weighted depth over matched JD skills
    trust = 0.0
    for s in matched:
        pw = PROFICIENCY_WEIGHTS.get(str(s.get("proficiency", "")).lower(), 0.3)
        endo = int(s.get("endorsements", 0) or 0)
        dur = int(s.get("duration_months", 0) or 0)
        trust += pw * math.log1p(endo) * math.sqrt(max(dur, 0) / 12.0)

    # assessment evidence (verified on-platform) restricted to JD-relevant skills
    relevant_assessments = [v for k, v in assessments.items() if str(k).lower() in JD_CORE_SKILLS]
    assessment_jd_relevance = (
        float(np.mean(relevant_assessments)) / 100.0 if relevant_assessments else 0.0
    )

    max_dur = max((int(s.get("duration_months", 0) or 0) for s in matched), default=0)

    # keyword stuffer (high recall): AI skills on a non-technical current title
    # with zero technical roles ever — the HR-Manager-with-9-AI-skills trap.
    nontech_current = _title_relevance_factor(profile.get("current_title", "")) == 0.0
    no_tech_roles = career["_technical_role_count"] == 0
    keyword_stuffer = bool(len(matched) >= 3 and nontech_current and no_tech_roles) or bool(
        len(matched) >= 8
        and np.mean([int(s.get("endorsements", 0) or 0) for s in matched] or [0]) < 5
        and len(non_relevant) > len(matched)
        and no_tech_roles
    )

    return {
        "jd_skill_match_count": len(matched),
        "jd_skill_match_ratio": len(matched) / max(len(skills), 1),
        "retrieval_skill_count": len(retrieval),
        "llm_skill_count": len(llm),
        "cv_speech_dominance": cv_dominance,
        "non_relevant_skill_count": len(non_relevant),
        "skill_trust_score": trust,
        "keyword_stuffer_flag": float(keyword_stuffer),
        "assessment_verified_skills": len(assessments),
        "assessment_mean_score": (
            float(np.mean(list(assessments.values()))) if assessments else 0.0
        ),
        "top_assessment_score": (float(max(assessments.values())) if assessments else 0.0),
        "max_jd_skill_duration_months": max_dur,
        "recent_jd_skill_flag": float(
            any(int(s.get("duration_months", 0) or 0) >= 12 for s in matched)
        ),
        "_assessment_jd_relevance": assessment_jd_relevance,
        "_top_jd_skills": [
            s.get("name", "")
            for s in sorted(
                matched,
                key=lambda s: (
                    PROFICIENCY_WEIGHTS.get(str(s.get("proficiency", "")).lower(), 0)
                    * (1 + int(s.get("endorsements", 0) or 0))
                ),
                reverse=True,
            )
        ][:3],
        "_top_retrieval_skills": [
            s.get("name", "") for s in skills if str(s.get("name", "")).lower() in RETRIEVAL_SKILLS
        ][:3],
    }


# ---------------------------------------------------------------------------
# Behavioural signals
# ---------------------------------------------------------------------------


def compute_signal_features(candidate: dict) -> dict:
    s = get_settings()
    sig = candidate.get("redrob_signals", {}) or {}
    last_active = _parse_date(sig.get("last_active_date"))
    days_since = (DATASET_TODAY - last_active).days if last_active else 365
    recency = math.exp(-max(days_since, 0) / s.recency_half_life_days)

    notice = int(sig.get("notice_period_days", 0) or 0)
    notice_factor = 1.0 if notice <= 30 else 0.7 if notice <= 60 else 0.4 if notice <= 90 else 0.1
    availability = (1.0 if sig.get("open_to_work_flag") else 0.3) * notice_factor * recency

    rr = float(sig.get("recruiter_response_rate", 0) or 0)
    icr = float(sig.get("interview_completion_rate", 0) or 0)
    pcs = float(sig.get("profile_completeness_score", 0) or 0)
    saved = int(sig.get("saved_by_recruiters_30d", 0) or 0)
    art = float(sig.get("avg_response_time_hours", 0) or 0)
    engagement = (
        0.40 * rr
        + 0.20 * icr
        + 0.20 * min(pcs / 100.0, 1.0)
        + 0.10 * min(saved / 10.0, 1.0)
        + 0.10 * (1.0 / (1.0 + art / 24.0))
    )

    trust = (
        (0.3 if sig.get("verified_email") else 0.0)
        + (0.3 if sig.get("verified_phone") else 0.0)
        + (0.2 if sig.get("linkedin_connected") else 0.0)
        + 0.2 * min(pcs / 100.0, 1.0)
    )

    gh_raw = float(sig.get("github_activity_score", -1) or -1)
    github = gh_raw if gh_raw >= 0 else 0.0
    oar = float(sig.get("offer_acceptance_rate", -1) or -1)

    behavioural = (
        s.availability_weight * availability
        + s.engagement_weight * engagement
        + s.trust_weight * trust
        + s.github_weight * (github / 100.0)
    )

    return {
        "days_since_active": float(days_since),
        "recency_score": recency,
        "availability_score": availability,
        "engagement_score": engagement,
        "trust_score": trust,
        "github_activity_score": github,
        "github_linked": float(gh_raw >= 0),
        "notice_period_days": float(notice),
        "open_to_work": float(bool(sig.get("open_to_work_flag"))),
        "offer_acceptance_rate": oar if oar >= 0 else 0.5,
        "behavioural_composite": behavioural,
        "_recruiter_response_rate": rr,
    }


# ---------------------------------------------------------------------------
# JD-fit (competence) + location + wrong-specialisation
# ---------------------------------------------------------------------------


def _location_fit(candidate: dict) -> float:
    profile = candidate.get("profile", {}) or {}
    sig = candidate.get("redrob_signals", {}) or {}
    loc = str(profile.get("location", "")).lower()
    country = str(profile.get("country", "")).lower()
    in_india = "india" in country or any(m in loc for m in PREFERRED_LOCATIONS)
    if any(m in loc for m in PREFERRED_LOCATIONS):
        return 1.0
    if in_india:
        return 0.7
    if sig.get("willing_to_relocate"):
        return 0.35
    return 0.1


def compute_jd_fit_features(candidate: dict, career: dict, skill: dict) -> dict:
    profile = candidate.get("profile", {}) or {}
    names = {str(s.get("name", "")).lower() for s in candidate.get("skills", []) or []}
    has_nlp_ir = (
        skill["retrieval_skill_count"] > 0
        or skill["llm_skill_count"] > 0
        or "nlp" in names
        or "natural language processing" in names
    )
    wrong_spec = skill["cv_speech_dominance"] * (0.0 if has_nlp_ir else 1.0)
    location_fit = _location_fit(candidate)
    title_factor = _title_relevance_factor(profile.get("current_title", ""))

    base = (
        0.30 * min(skill["retrieval_skill_count"] / 3.0, 1.0)
        + 0.15 * min(skill["jd_skill_match_count"] / 6.0, 1.0)
        + 0.12 * math.tanh(skill["skill_trust_score"] / 4.0)
        + 0.12 * skill["_assessment_jd_relevance"]
        + 0.10 * career["product_company_ratio"]
        + 0.08 * career["has_production_keywords"]
        + 0.08 * career["scope_progression_score"]
        + 0.05 * min(skill["llm_skill_count"] / 3.0, 1.0)
    )
    fit = base * (0.4 + 0.6 * title_factor)
    fit *= 1.0 - 0.7 * wrong_spec
    fit *= 1.0 - 0.6 * skill["keyword_stuffer_flag"]
    fit *= 1.0 - 0.5 * career["all_consulting_career"]
    fit = float(np.clip(fit, 0.0, 1.0))
    return {
        "jd_fit_score": fit,
        "location_fit": location_fit,
        "wrong_specialisation_penalty": wrong_spec,
    }


# ---------------------------------------------------------------------------
# Row assembly
# ---------------------------------------------------------------------------


def build_feature_row(candidate: dict) -> dict:
    career = compute_career_features(candidate)
    skill = compute_skill_features(candidate, career)
    signal = compute_signal_features(candidate)
    jd_fit = compute_jd_fit_features(candidate, career, skill)
    is_hp, hp_reasons = detect_honeypot(candidate)

    profile = candidate.get("profile", {}) or {}
    sig = candidate.get("redrob_signals", {}) or {}
    row: dict[str, Any] = {"candidate_id": candidate["candidate_id"]}
    row.update({k: v for k, v in career.items() if not k.startswith("_")})
    row.update({k: v for k, v in skill.items() if not k.startswith("_")})
    row.update({k: v for k, v in signal.items() if not k.startswith("_")})
    row.update(jd_fit)
    row["is_honeypot"] = float(is_hp)

    # object columns for reasoning / display (not part of the ML contract)
    row["honeypot_reasons"] = hp_reasons
    row["current_title"] = profile.get("current_title", "")
    row["current_company"] = profile.get("current_company", "")
    row["location"] = profile.get("location", "")
    row["country"] = profile.get("country", "")
    row["anonymized_name"] = profile.get("anonymized_name", "")
    row["top_jd_skills"] = skill["_top_jd_skills"]
    row["top_retrieval_skills"] = skill["_top_retrieval_skills"]
    row["recruiter_response_rate"] = signal["_recruiter_response_rate"]
    row["preferred_work_mode"] = sig.get("preferred_work_mode", "")
    row["willing_to_relocate"] = bool(sig.get("willing_to_relocate"))
    # live-evidence columns — 0 unless a developer profile is linked (see
    # apply_live_signals). Present here so the scorer can always read them.
    row["has_live_evidence"] = 0.0
    row["evidence_confidence_bonus"] = 0.0
    return row


# ---------------------------------------------------------------------------
# Live ingestion bridge — verified evidence overrides self-reported confidence.
# These run only for candidates that have a linked developer profile; for the
# static dataset they're never invoked, so the submission is unchanged.
# ---------------------------------------------------------------------------

RESUME_CLAIM_CONFIDENCE = 0.2  # a self-reported skill is weak evidence


def get_skill_confidence_with_live_override(
    candidate: dict, skill_name: str, live_confidence: dict[str, float] | None = None
) -> float:
    """Graph confidence for the skill if we have live evidence, else the static
    resume-claim confidence. ``live_confidence`` is a {skill: confidence} map
    resolved from the graph ahead of time (keeps this function sync)."""
    if live_confidence:
        hit = live_confidence.get(skill_name.lower())
        if hit is not None:
            return float(hit)
    return RESUME_CLAIM_CONFIDENCE


def apply_live_signals(row: dict, live_confidence: dict[str, float] | None) -> dict:
    """Mutate a feature row with live evidence: lift skill_trust_score to the
    mean verified confidence and grant the evidence bonus. No-op without a map."""
    if not live_confidence:
        return row
    jd_skills = [s.lower() for s in (row.get("top_jd_skills") or [])]
    confidences = [
        get_skill_confidence_with_live_override({}, s, live_confidence) for s in jd_skills
    ]
    confidences = [c for c in confidences if c > RESUME_CLAIM_CONFIDENCE]
    if confidences:
        row["skill_trust_score"] = float(np.mean(confidences)) * 5.0  # back to raw scale
        row["has_live_evidence"] = 1.0
        row["evidence_confidence_bonus"] = float(get_settings().evidence_confidence_bonus)
    return row


def frame_from_rows(rows: list[dict]) -> pd.DataFrame:
    """Turn feature-row dicts into a DataFrame with the numeric contract coerced.
    Kept separate so precompute can build rows + corpus in one streaming pass."""
    df = pd.DataFrame(rows)
    for col in NUMERIC_FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


def build_feature_matrix(candidates: Iterable[dict]) -> pd.DataFrame:
    """Build the full feature DataFrame (one row per candidate)."""
    with log_duration(log, "feature_engineering.complete") as m:
        df = frame_from_rows([build_feature_row(c) for c in candidates])
        m["rows"] = len(df)
        m["honeypots"] = int(df["is_honeypot"].sum())
        m["keyword_stuffers"] = int(df["keyword_stuffer_flag"].sum())
    return df
