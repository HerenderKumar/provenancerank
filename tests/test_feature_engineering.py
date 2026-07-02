from core.constants import NUMERIC_FEATURE_COLUMNS
from pipeline.feature_engineering import build_feature_matrix, build_feature_row


def test_all_numeric_columns_present(good_candidate):
    row = build_feature_row(good_candidate)
    for col in NUMERIC_FEATURE_COLUMNS:
        assert col in row, f"missing feature {col}"
        assert isinstance(row[col], (int, float))


def test_good_candidate_beats_stuffer(good_candidate, keyword_stuffer):
    df = build_feature_matrix([good_candidate, keyword_stuffer])
    g = df[df.candidate_id == good_candidate["candidate_id"]].iloc[0]
    k = df[df.candidate_id == keyword_stuffer["candidate_id"]].iloc[0]
    assert g.jd_fit_score > k.jd_fit_score
    assert k.keyword_stuffer_flag == 1
    assert g.keyword_stuffer_flag == 0


def test_retrieval_skills_counted(good_candidate):
    row = build_feature_row(good_candidate)
    assert row["retrieval_skill_count"] >= 1
    assert 0.0 <= row["jd_fit_score"] <= 1.0


def test_consulting_only_flagged(consulting_only):
    row = build_feature_row(consulting_only)
    assert row["all_consulting_career"] == 1.0
    assert row["product_company_ratio"] == 0.0


def test_matrix_numeric_coercion(sample_candidates_path):
    import json

    cands = json.load(open(sample_candidates_path))
    df = build_feature_matrix(cands)
    assert len(df) == len(cands)
    # the numeric contract must be fully numeric (no NaN/object leakage)
    assert df[list(NUMERIC_FEATURE_COLUMNS)].isna().sum().sum() == 0
    assert df[list(NUMERIC_FEATURE_COLUMNS)].select_dtypes("object").empty
