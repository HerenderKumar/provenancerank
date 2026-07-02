from pipeline.feature_engineering import build_feature_matrix
from pipeline.gate_filter import apply_gates


def test_gates_exclude_the_bad_ones(
    good_candidate, keyword_stuffer, honeypot_expert_zero, consulting_only
):
    df = build_feature_matrix(
        [good_candidate, keyword_stuffer, honeypot_expert_zero, consulting_only]
    )
    df = apply_gates(df)
    passed = dict(zip(df.candidate_id, df.gate_passed))
    assert passed[good_candidate["candidate_id"]] is True or passed[good_candidate["candidate_id"]]
    assert not passed[keyword_stuffer["candidate_id"]]
    assert not passed[honeypot_expert_zero["candidate_id"]]
    assert not passed[consulting_only["candidate_id"]]


def test_failure_reasons_recorded(consulting_only):
    df = apply_gates(build_feature_matrix([consulting_only]))
    reasons = df.iloc[0]["gate_failures"]
    assert "all_consulting_career" in reasons


def test_notice_period_is_soft_not_hard(good_candidate):
    # a long notice period should NOT exclude (JD keeps them in scope)
    good_candidate["redrob_signals"]["notice_period_days"] = 120
    df = apply_gates(build_feature_matrix([good_candidate]))
    assert df.iloc[0]["gate_passed"]
    assert "notice_gt_90" in df.iloc[0]["soft_flags"]
