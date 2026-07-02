"""Performance levers: device selection, the focused rerank text, the training
downsample, and the content-hash cache. None of these change *which* features
run — they make the same work cheaper — so the tests assert behaviour is
preserved, not that a particular backend was chosen."""

from __future__ import annotations

import numpy as np
import pandas as pd

from core import artifact_cache as cache
from core.device import best_device, transformer_attempts
from ml.trainer import _downsample_for_training
from pipeline.loader import rerank_text
from tests.conftest import make_candidate

# --------------------------------------------------------------------------
# device selection
# --------------------------------------------------------------------------

def test_best_device_is_known_value():
    assert best_device() in {"cpu", "cuda", "mps"}


def test_transformer_attempts_always_end_in_torch():
    attempts = transformer_attempts("any-model", allow_onnx=True)
    assert attempts[-1][0] == "torch"
    # with ONNX off by default, torch is the only attempt
    assert [label for label, _ in attempts] == ["torch"]
    assert "device" in attempts[-1][1]


# --------------------------------------------------------------------------
# focused rerank text
# --------------------------------------------------------------------------

def test_rerank_text_is_focused_and_bounded():
    c = make_candidate(title="Senior AI Engineer")
    full = " ".join(
        str(c["profile"][k]) for k in ("headline", "summary", "current_title")
    )
    focused = rerank_text(c)
    assert "Senior AI Engineer" in focused
    # a tight projection: shorter than dumping the whole profile blob
    assert 0 < len(focused) <= len(full) + 400
    # top endorsed skill survives the projection
    assert "Python" in focused


def test_rerank_text_survives_sparse_candidate():
    sparse = {"candidate_id": "X", "profile": {"current_title": "Data Scientist"}}
    assert "Data Scientist" in rerank_text(sparse)


# --------------------------------------------------------------------------
# training downsample
# --------------------------------------------------------------------------

def test_downsample_keeps_all_positives():
    # big enough that negatives exceed the min-train floor and get sampled down
    n = 60000
    rng = np.random.default_rng(0)
    labels = pd.Series(np.where(rng.random(n) < 0.05, rng.integers(1, 6, n), 0))
    df = pd.DataFrame({"candidate_id": [f"C{i}" for i in range(n)], "x": rng.random(n)})
    df_t, y_t = _downsample_for_training(df, labels)
    n_pos = int((labels > 0).sum())
    # every positive is retained...
    assert int((y_t > 0).sum()) == n_pos
    # ...and the set shrank (negatives were sampled down)
    assert len(df_t) < n


def test_downsample_noop_when_no_negatives():
    labels = pd.Series([3, 4, 5])
    df = pd.DataFrame({"candidate_id": ["a", "b", "c"], "x": [1.0, 2.0, 3.0]})
    df_t, y_t = _downsample_for_training(df, labels)
    assert len(df_t) == 3


# --------------------------------------------------------------------------
# content-hash cache
# --------------------------------------------------------------------------

def test_cache_roundtrip_and_staleness(tmp_path):
    art = tmp_path / "vec.npy"
    arr = np.arange(12, dtype=float)
    key = cache.key_of("candidates-v1", "st", 384)

    assert cache.load_npy(art, key) is None  # cold miss
    cache.save_npy(art, key, arr)
    np.testing.assert_array_equal(cache.load_npy(art, key), arr)  # hit

    # a different input key must be treated as stale, not silently reused
    assert cache.load_npy(art, cache.key_of("candidates-v2", "st", 384)) is None


def test_key_of_is_order_sensitive():
    assert cache.key_of("a", "b") != cache.key_of("b", "a")
    assert cache.key_of("a", "b") == cache.key_of("a", "b")
