"""Load the fit scorer and predict. Sub-second for 100K on CPU.

If the model file is missing or won't load, we don't crash — we fall back to the
proxy formula directly. Worst case the ranker is still a sensible scorer.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from core.constants import NUMERIC_FEATURE_COLUMNS
from core.logging import get_logger
from ml.trainer import create_proxy_labels

log = get_logger("ml.predictor")


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def predict_fit(df: pd.DataFrame, model_path: str | Path) -> np.ndarray:
    """Return normalised fit scores in [0,1] aligned to df rows."""
    path = Path(model_path)
    if not path.exists():
        log.warning("predictor.no_model_fallback_formula", path=str(path))
        return _minmax(create_proxy_labels(df).to_numpy())

    try:
        artifact = joblib.load(path)
    except Exception as exc:
        log.warning("predictor.load_failed_fallback_formula", error=str(exc)[:140])
        return _minmax(create_proxy_labels(df).to_numpy())

    if artifact.get("backend") == "formula" or artifact.get("model") is None:
        return _minmax(create_proxy_labels(df).to_numpy())

    features = artifact.get("features", list(NUMERIC_FEATURE_COLUMNS))
    X = df[features].astype(np.float32)
    try:
        raw = np.asarray(artifact["model"].predict(X), dtype=np.float32)
    except Exception as exc:
        log.warning("predictor.predict_failed_fallback_formula", error=str(exc)[:140])
        return _minmax(create_proxy_labels(df).to_numpy())
    return _minmax(raw)
