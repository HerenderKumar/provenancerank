"""End-to-end smoke: the live path must produce a valid submission, fast."""

import json
from pathlib import Path

import pytest

import rank
from output.submission_writer import validate

DEV = Path(__file__).resolve().parent.parent / "data" / "dev_sample.jsonl"


@pytest.mark.skipif(not DEV.exists(), reason="dev_sample.jsonl not built")
def test_live_path_valid_submission(tmp_path):
    cands = []
    with open(DEV) as f:
        for i, line in enumerate(f):
            if i >= 150:
                break
            cands.append(json.loads(line))
    out = tmp_path / "team_smoke.csv"
    path, errors = rank._live_path(cands, str(out), top_k=100)
    assert errors == [], errors
    assert validate(path) == []


@pytest.mark.skipif(not DEV.exists(), reason="dev_sample.jsonl not built")
def test_no_honeypots_in_live_top100(tmp_path):
    import pandas as pd

    from pipeline.feature_engineering import build_feature_matrix
    from pipeline.loader import iter_candidates

    cands = list(iter_candidates(str(DEV)))[:400]
    out = tmp_path / "team_smoke2.csv"
    rank._live_path(cands, str(out), top_k=100)
    sub = pd.read_csv(out)
    fm = build_feature_matrix(cands)[["candidate_id", "is_honeypot"]]
    merged = sub.merge(fm, on="candidate_id", how="left")
    assert int(merged.is_honeypot.sum()) == 0
