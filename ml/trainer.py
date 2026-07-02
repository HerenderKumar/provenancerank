"""Train the fit scorer.

There are no hire/reject labels in the data, so we build a proxy relevance
target from the features we trust most and fit a gradient-boosted regressor to
it. Why bother, if we already have the formula? Because the tree model picks up
interactions the linear proxy can't - e.g. "retrieval skills only count when the
title is actually technical" - and it smooths the final ordering.

Model pick is whatever's installed: XGBoost -> LightGBM -> sklearn HistGBR ->
plain formula (predictor falls back to the proxy directly). Same feature
contract regardless, so nothing downstream changes.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from core.config import get_settings
from core.constants import NUMERIC_FEATURE_COLUMNS
from core.logging import get_logger, log_duration

log = get_logger("ml.trainer")


def _norm(s: pd.Series) -> pd.Series:
    lo, hi = float(s.min()), float(s.max())
    if hi - lo < 1e-12:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - lo) / (hi - lo)


def create_proxy_labels(df: pd.DataFrame) -> pd.Series:
    """Proxy relevance in [0,1]. Gate failures and honeypots are pinned to 0 so
    the model never learns to like them."""
    label = (
        0.45 * df["jd_fit_score"]
        + 0.20 * _norm(df["retrieval_skill_count"])
        + 0.15 * _norm(df["jd_skill_match_count"])
        + 0.10 * df["product_company_ratio"]
        + 0.10 * df["behavioural_composite"].clip(0, 1)
    )
    label = label.clip(0, 1)
    if "gate_passed" in df.columns:
        label = label.where(df["gate_passed"], 0.0)
    label = label.where(df["is_honeypot"] != 1, 0.0)
    return label


def _make_regressor():
    """First ML lib that imports wins (Strategy pattern)."""
    try:
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42,
            objective="reg:squarederror",
        ), "xgboost"
    except Exception:
        pass
    try:
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            n_estimators=400,
            max_depth=-1,
            num_leaves=48,
            learning_rate=0.05,
            subsample=0.85,
            random_state=42,
            n_jobs=-1,
        ), "lightgbm"
    except Exception:
        pass
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(
            max_iter=400,
            max_depth=6,
            learning_rate=0.05,
            random_state=42,
        ), "sklearn-histgbr"
    except Exception:
        return None, "formula"


def train_fit_scorer(df: pd.DataFrame, out_path: str | Path) -> dict:
    y = create_proxy_labels(df)
    X = df[list(NUMERIC_FEATURE_COLUMNS)].astype(np.float32)
    model, backend = _make_regressor()

    with log_duration(log, "ml.train") as m:
        if model is not None:
            model.fit(X, y)
            in_sample = float(np.corrcoef(model.predict(X), y)[0, 1])
        else:
            in_sample = 1.0  # formula == its own prediction
        m["backend"] = backend
        m["rows"] = len(df)
        m["label_mean"] = round(float(y.mean()), 4)
        m["in_sample_corr"] = round(in_sample, 4)

    artifact = {
        "backend": backend,
        "model": model,
        "features": list(NUMERIC_FEATURE_COLUMNS),
        "label_mean": float(y.mean()),
        "kind": "regressor",
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, out_path)
    log.info("ml.model_saved", path=str(out_path), backend=backend)
    return artifact


# ---------------------------------------------------------------------------
# Learning-to-rank head (LambdaMART). Trains on graded 0..5 pseudo-labels with
# an NDCG objective - optimising the metric the contest grades directly, rather
# than regressing a proxy. Same artifact shape, so predict_fit is unchanged.
# ---------------------------------------------------------------------------

def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson corr that degrades to 0.0 on a constant vector (no variance)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _make_ranker():
    """LambdaMART if XGBoost/LightGBM is present, else None (caller falls back)."""
    try:
        from xgboost import XGBRanker

        return XGBRanker(
            objective="rank:ndcg",
            tree_method="hist",  # histogram split-finding - the fast path on CPU
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42,
        ), "xgboost-ranker"
    except Exception:
        pass
    try:
        from lightgbm import LGBMRanker

        return LGBMRanker(
            objective="lambdarank",
            n_estimators=500,
            num_leaves=63,
            learning_rate=0.05,
            subsample=0.85,
            random_state=42,
            n_jobs=-1,
        ), "lightgbm-ranker"
    except Exception:
        return None, "none"


def _synthetic_groups(n: int, n_groups: int, seed: int = 42):
    """Split rows into balanced random query groups for the listwise objective.

    Every candidate is scored against the *same* JD, so semantically there is one
    query. Feeding one giant group is valid but slow and prone to overfitting the
    pairwise structure; splitting into many balanced mini-rankings is cheaper and
    acts as a mild regulariser. Returns (row order, per-group sizes) - the ranker
    wants rows grouped contiguously.
    """
    rng = np.random.default_rng(seed)
    assignments = rng.integers(0, n_groups, size=n)
    order = np.argsort(assignments, kind="stable")
    sizes = np.bincount(assignments, minlength=n_groups)
    return order, sizes[sizes > 0]


def _downsample_for_training(df: pd.DataFrame, labels: pd.Series):
    """Keep every positive (label > 0) and a sample of the negatives.

    The label distribution is mostly tier-0 noise (and honeypots/gated rows are
    pinned to 0), so training on all 100K wastes time learning from rows we'll
    zero out anyway. Keeping all signal-bearing rows + a bounded negative sample
    is standard for learning-to-rank and barely moves quality, for a real speedup.
    """
    s = get_settings()
    y = labels.to_numpy()
    pos = np.flatnonzero(y > 0)
    neg = np.flatnonzero(y <= 0)
    if pos.size == 0 or neg.size == 0:
        return df, labels  # nothing to balance against
    budget = max(s.ranker_neg_ratio * pos.size, s.ranker_min_train_rows - pos.size)
    if neg.size > budget:
        rng = np.random.default_rng(s.random_seed)
        neg = rng.choice(neg, size=int(budget), replace=False)
    keep = np.sort(np.concatenate([pos, neg]))
    return df.iloc[keep].reset_index(drop=True), labels.iloc[keep].reset_index(drop=True)


def _fit_ranker(model, X: pd.DataFrame, y: pd.Series, sizes, s) -> bool:
    """Fit a grouped ranker, using early stopping on a held-out tail of groups
    when there are enough of them. Returns True on success; on any failure the
    caller rebuilds a clean model and does a plain fit (no early-stopping state).
    """
    rounds = int(getattr(s, "ranker_early_stop_rounds", 0) or 0)
    if rounds > 0 and len(sizes) >= 4 and hasattr(model, "set_params"):
        n_val = max(1, len(sizes) // 7)  # ~15% of groups for validation
        tr_sizes, va_sizes = sizes[:-n_val], sizes[-n_val:]
        split = int(sum(tr_sizes))
        try:
            model.set_params(early_stopping_rounds=rounds, eval_metric="ndcg@10")
            model.fit(
                X.iloc[:split], y.iloc[:split], group=list(tr_sizes),
                eval_set=[(X.iloc[split:], y.iloc[split:])], eval_group=[list(va_sizes)],
                verbose=False,
            )
            return True
        except Exception as exc:
            log.warning("ml.early_stop_unavailable", reason=str(exc)[:120])
            return False
    model.fit(X, y, group=list(sizes))
    return True


def train_ranker(
    df: pd.DataFrame,
    labels: pd.Series,
    out_path: str | Path,
    n_groups: int = 24,
) -> dict:
    """Train the LambdaMART ranking head on graded labels.

    Degrades in three steps so it always produces a usable artifact:
    XGBRanker/LGBMRanker (NDCG objective) -> proxy regressor on the same labels
    -> formula. predict_fit min-maxes the output, so unbounded rank scores are
    fine and nothing downstream changes.
    """
    # full matrix is kept for the honesty check; the model trains on a balanced
    # subset (all positives + sampled negatives) for speed.
    X_full = df[list(NUMERIC_FEATURE_COLUMNS)].astype(np.float32).reset_index(drop=True)
    y_full = labels.reset_index(drop=True).astype(float)
    df_t, y_t = _downsample_for_training(df, labels)
    X = df_t[list(NUMERIC_FEATURE_COLUMNS)].astype(np.float32).reset_index(drop=True)
    y = y_t.reset_index(drop=True).astype(float)
    model, backend = _make_ranker()

    with log_duration(log, "ml.train_ranker") as m:
        if model is not None:
            order, sizes = _synthetic_groups(len(X), n_groups)
            X_grouped = X.iloc[order].reset_index(drop=True)
            y_grouped = y.iloc[order].reset_index(drop=True)
            try:
                if not _fit_ranker(model, X_grouped, y_grouped, sizes, get_settings()):
                    model, backend = _make_ranker()  # fresh, no early-stop state
                    model.fit(X_grouped, y_grouped, group=list(sizes))
            except Exception as exc:
                # a ranker lib is present but rejected the fit - don't crash the
                # whole precompute, drop to the regressor on the graded labels.
                log.warning("ml.ranker_fit_failed_fallback", error=str(exc)[:140])
                model, backend = _make_regressor()
                if model is not None:
                    model.fit(X, y)
            in_sample = _safe_corr(model.predict(X_full), y_full) if model is not None else 1.0
        else:
            # no ranker lib at all - regress the graded labels, then formula
            model, backend = _make_regressor()
            if model is not None:
                model.fit(X, y)
                in_sample = _safe_corr(model.predict(X_full), y_full)
            else:
                backend = "formula"
                in_sample = 1.0
        m["backend"] = backend
        m["rows"] = len(df)
        m["train_rows"] = len(df_t)
        m["groups"] = n_groups
        m["label_mean"] = round(float(y_full.mean()), 4)
        m["in_sample_corr"] = round(in_sample, 4)

    artifact = {
        "backend": backend,
        "model": model,
        "features": list(NUMERIC_FEATURE_COLUMNS),
        "label_mean": float(y_full.mean()),
        "kind": "ranker",
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, out_path)
    log.info("ml.ranker_saved", path=str(out_path), backend=backend)
    return artifact
