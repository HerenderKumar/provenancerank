"""Behavioural-signal scoring at ranking time (vectorised over the matrix).

feature_engineering already stored the per-candidate behavioural_composite; this
module recomputes it from config weights (so you can retune weights without a
full precompute) and turns the "is this person actually reachable" signals into
a multiplier we apply to fit. The multiplier has a floor - a strong candidate
who's been a bit quiet shouldn't be zeroed, just nudged down.
"""

from __future__ import annotations

import pandas as pd

from core.config import get_settings


def recompute_behavioural_composite(df: pd.DataFrame) -> pd.Series:
    s = get_settings()
    gh = pd.to_numeric(df["github_activity_score"], errors="coerce").fillna(0) / 100.0
    return (
        s.availability_weight * df["availability_score"]
        + s.engagement_weight * df["engagement_score"]
        + s.trust_weight * df["trust_score"]
        + s.github_weight * gh
    ).clip(0, 1)


def availability_multiplier(df: pd.DataFrame, floor: float = 0.45) -> pd.Series:
    """0.45-1.0 multiplier from availability (open-to-work, notice, recency).

    This is the JD's "perfect on paper but hasn't logged in for 6 months isn't
    actually available" rule made concrete.
    """
    avail = df["availability_score"].clip(0, 1)
    return (floor + (1 - floor) * avail).clip(floor, 1.0)


def engagement_multiplier(df: pd.DataFrame, floor: float = 0.6) -> pd.Series:
    eng = df["engagement_score"].clip(0, 1)
    return (floor + (1 - floor) * eng).clip(floor, 1.0)


def signal_modifier(df: pd.DataFrame) -> pd.Series:
    """Combined behavioural modifier used to scale fit in the final merge."""
    return (availability_multiplier(df) * engagement_multiplier(df)).clip(0.3, 1.0)
