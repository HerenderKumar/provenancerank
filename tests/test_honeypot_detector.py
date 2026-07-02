from pipeline.honeypot_detector import detect_honeypot
from tests.conftest import make_candidate


def test_clean_candidate_not_flagged(good_candidate):
    flagged, reasons = detect_honeypot(good_candidate)
    assert not flagged, reasons


def test_expert_zero_duration(honeypot_expert_zero):
    flagged, reasons = detect_honeypot(honeypot_expert_zero)
    assert flagged
    assert any("expert_zero_duration" in r for r in reasons)


def test_tiny_company_long_tenure(honeypot_tiny_long):
    flagged, reasons = detect_honeypot(honeypot_tiny_long)
    assert flagged
    assert any("implausible_tenure" in r for r in reasons)


def test_duration_exceeds_calendar():
    c = make_candidate(
        cid="CAND_0000010",
        career=[
            {
                "company": "Foo",
                "title": "ML Engineer",
                "start_date": "2023-01-01",
                "end_date": "2024-01-01",
                "duration_months": 60,
                "is_current": False,
                "industry": "Internet",
                "company_size": "201-500",
                "description": "x",
            }
        ],
    )
    flagged, reasons = detect_honeypot(c)
    assert flagged
    assert any("duration_exceeds_calendar" in r for r in reasons)


def test_yoe_exceeds_history():
    c = make_candidate(
        cid="CAND_0000011",
        yoe=25.0,
        career=[
            {
                "company": "Foo",
                "title": "ML Engineer",
                "start_date": "2022-01-01",
                "end_date": None,
                "duration_months": 24,
                "is_current": True,
                "industry": "Internet",
                "company_size": "201-500",
                "description": "x",
            }
        ],
    )
    flagged, reasons = detect_honeypot(c)
    assert flagged
    assert any("yoe_exceeds_career_history" in r for r in reasons)
