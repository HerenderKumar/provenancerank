import pandas as pd

from output.submission_writer import build_submission, validate, write_submission


def _fake_scored(n=120):
    # deliberately messy: unsorted, duplicate scores, ids out of order
    rows = []
    for i in range(n):
        rows.append(
            {
                "candidate_id": f"CAND_{i:07d}",
                "final_score": round(1.0 - (i // 3) * 0.01, 4),  # forces score ties
                "reasoning": f"Candidate {i} - retrieval/IR depth, at product companies.",
            }
        )
    return pd.DataFrame(rows).sample(frac=1, random_state=1).reset_index(drop=True)


def test_official_validator_passes(tmp_path):
    sub = build_submission(_fake_scored(), top_k=100)
    path = write_submission(sub, tmp_path / "team_test.csv")
    errors = validate(path)
    assert errors == [], errors


def test_exactly_100_rows_and_unique_ranks(tmp_path):
    sub = build_submission(_fake_scored(), top_k=100)
    assert len(sub) == 100
    assert sorted(sub["rank"]) == list(range(1, 101))
    assert sub["candidate_id"].nunique() == 100


def test_scores_non_increasing_and_tiebreak(tmp_path):
    sub = build_submission(_fake_scored(), top_k=100).reset_index(drop=True)
    scores = sub["score"].tolist()
    ids = sub["candidate_id"].tolist()
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
    # equal adjacent scores must be candidate_id ascending
    for i in range(len(scores) - 1):
        if scores[i] == scores[i + 1]:
            assert ids[i] < ids[i + 1]


def test_reasoning_with_commas_is_quoted(tmp_path):
    df = pd.DataFrame(
        [
            {
                "candidate_id": f"CAND_{i:07d}",
                "final_score": 0.9 - i * 0.001,
                "reasoning": "ML engineer, retrieval, ranking; strong fit, but 90-day notice.",
            }
            for i in range(100)
        ]
    )
    sub = build_submission(df, top_k=100)
    path = write_submission(sub, tmp_path / "team_q.csv")
    # if quoting were wrong the validator would see != 4 columns
    assert validate(path) == []
