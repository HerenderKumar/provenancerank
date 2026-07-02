"""Shared fixtures + a candidate factory that emits schema-valid profiles.

Building candidates by hand (rather than only leaning on the real file) lets the
tests pin specific behaviours: this exact profile is a honeypot, that one is a
keyword stuffer, etc.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def make_candidate(
    cid="CAND_0000001",
    title="Machine Learning Engineer",
    yoe=6.0,
    skills=None,
    career=None,
    signals=None,
    company_size="201-500",
):
    """A reasonable default candidate; pass overrides to shape edge cases."""
    skills = (
        skills
        if skills is not None
        else [
            {
                "name": "Vector Search",
                "proficiency": "advanced",
                "endorsements": 20,
                "duration_months": 36,
            },
            {
                "name": "Embeddings",
                "proficiency": "expert",
                "endorsements": 30,
                "duration_months": 30,
            },
            {"name": "Python", "proficiency": "expert", "endorsements": 40, "duration_months": 72},
        ]
    )
    career = (
        career
        if career is not None
        else [
            {
                "company": "Swiggy",
                "title": "ML Engineer",
                "start_date": "2021-01-01",
                "end_date": None,
                "duration_months": 40,
                "is_current": True,
                "industry": "Internet",
                "company_size": company_size,
                "description": "Built production retrieval and ranking systems serving users at scale; "
                "shipped embedding-based search with A/B tested NDCG improvements.",
            },
        ]
    )
    sig = {
        "profile_completeness_score": 90.0,
        "signup_date": "2020-01-01",
        "last_active_date": "2026-06-10",
        "open_to_work_flag": True,
        "profile_views_received_30d": 40,
        "applications_submitted_30d": 5,
        "recruiter_response_rate": 0.8,
        "avg_response_time_hours": 6.0,
        "skill_assessment_scores": {"Vector Search": 82.0, "NLP": 78.0},
        "connection_count": 500,
        "endorsements_received": 200,
        "notice_period_days": 30,
        "expected_salary_range_inr_lpa": {"min": 30.0, "max": 45.0},
        "preferred_work_mode": "hybrid",
        "willing_to_relocate": True,
        "github_activity_score": 70.0,
        "search_appearance_30d": 30,
        "saved_by_recruiters_30d": 8,
        "interview_completion_rate": 0.9,
        "offer_acceptance_rate": 0.5,
        "verified_email": True,
        "verified_phone": True,
        "linkedin_connected": True,
    }
    if signals:
        sig.update(signals)
    return {
        "candidate_id": cid,
        "profile": {
            "anonymized_name": "Candidate X",
            "headline": f"{title} | retrieval & ranking",
            "summary": "Applied ML engineer focused on retrieval, ranking and recommendation.",
            "location": "Bengaluru, Karnataka",
            "country": "India",
            "years_of_experience": yoe,
            "current_title": title,
            "current_company": career[0]["company"],
            "current_company_size": company_size,
            "current_industry": "Internet",
        },
        "career_history": career,
        "education": [
            {
                "institution": "IIT",
                "degree": "B.Tech",
                "field_of_study": "CS",
                "start_year": 2012,
                "end_year": 2016,
                "grade": "8.5",
                "tier": "tier_1",
            }
        ],
        "skills": skills,
        "certifications": [],
        "languages": [{"language": "English", "proficiency": "professional"}],
        "redrob_signals": sig,
    }


@pytest.fixture
def good_candidate():
    return make_candidate()


@pytest.fixture
def keyword_stuffer():
    # HR Manager whose skills are stuffed with AI keywords, no technical roles.
    return make_candidate(
        cid="CAND_0000002",
        title="HR Manager",
        yoe=7.0,
        skills=[
            {"name": "LLMs", "proficiency": "advanced", "endorsements": 1, "duration_months": 3},
            {
                "name": "Vector Search",
                "proficiency": "advanced",
                "endorsements": 0,
                "duration_months": 2,
            },
            {"name": "Pinecone", "proficiency": "expert", "endorsements": 1, "duration_months": 1},
            {
                "name": "LangChain",
                "proficiency": "advanced",
                "endorsements": 0,
                "duration_months": 2,
            },
            {
                "name": "Recruitment",
                "proficiency": "expert",
                "endorsements": 50,
                "duration_months": 84,
            },
            {"name": "Excel", "proficiency": "expert", "endorsements": 40, "duration_months": 84},
        ],
        career=[
            {
                "company": "ABC Corp",
                "title": "HR Manager",
                "start_date": "2018-01-01",
                "end_date": None,
                "duration_months": 96,
                "is_current": True,
                "industry": "Services",
                "company_size": "1001-5000",
                "description": "Managed recruitment and HR operations.",
            }
        ],
    )


@pytest.fixture
def honeypot_expert_zero():
    # >=3 expert skills with 0 months used -> impossible.
    return make_candidate(
        cid="CAND_0000003",
        title="AI Engineer",
        yoe=6.0,
        skills=[
            {
                "name": "Vector Search",
                "proficiency": "expert",
                "endorsements": 5,
                "duration_months": 0,
            },
            {"name": "LoRA", "proficiency": "expert", "endorsements": 5, "duration_months": 0},
            {"name": "QLoRA", "proficiency": "expert", "endorsements": 5, "duration_months": 0},
            {"name": "Pinecone", "proficiency": "expert", "endorsements": 5, "duration_months": 0},
        ],
    )


@pytest.fixture
def honeypot_tiny_long():
    # 11 years at a 1-10 person company.
    return make_candidate(
        cid="CAND_0000004",
        title="ML Engineer",
        company_size="1-10",
        career=[
            {
                "company": "TinyCo",
                "title": "ML Engineer",
                "start_date": "2015-01-01",
                "end_date": None,
                "duration_months": 132,
                "is_current": True,
                "industry": "Internet",
                "company_size": "1-10",
                "description": "Built systems.",
            }
        ],
    )


@pytest.fixture
def consulting_only():
    return make_candidate(
        cid="CAND_0000005",
        title="Consultant",
        yoe=8.0,
        career=[
            {
                "company": "Infosys",
                "title": "Technology Analyst",
                "start_date": "2016-01-01",
                "end_date": "2020-01-01",
                "duration_months": 48,
                "is_current": False,
                "industry": "IT Services",
                "company_size": "10001+",
                "description": "Delivery work.",
            },
            {
                "company": "TCS",
                "title": "Consultant",
                "start_date": "2020-01-01",
                "end_date": None,
                "duration_months": 60,
                "is_current": True,
                "industry": "IT Services",
                "company_size": "10001+",
                "description": "Client projects.",
            },
        ],
    )


@pytest.fixture
def sample_candidates_path():
    return str(Path(__file__).resolve().parent.parent / "data" / "sample_candidates.json")
