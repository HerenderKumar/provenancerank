from output.reasoning_generator import generate_all, generate_reasoning


def _row(cid, title, skills, **kw):
    base = dict(
        candidate_id=cid,
        current_title=title,
        total_experience_years=6.0,
        top_jd_skills=skills,
        top_retrieval_skills=skills,
        product_company_ratio=0.9,
        retrieval_skill_count=len(skills),
        scope_progression_score=0.7,
        github_activity_score=60,
        recruiter_response_rate=0.8,
        notice_period_days=30,
        days_since_active=10,
        location_fit=1.0,
        wrong_specialisation_penalty=0.0,
        location="Pune, Maharashtra",
    )
    base.update(kw)
    return base


def test_non_empty_and_specific():
    r = generate_reasoning(_row("CAND_0000001", "ML Engineer", ["Vector Search", "Embeddings"]), 1)
    assert r.strip().endswith(".")
    assert "ML Engineer" in r
    assert "Vector Search" in r


def test_only_mentions_real_skills():
    skills = ["Qdrant", "Embeddings"]
    r = generate_reasoning(_row("CAND_0000002", "AI Engineer", skills), 3)
    # nothing outside the candidate's skills should be named
    for forbidden in ["Pinecone", "LoRA", "TensorFlow"]:
        assert forbidden not in r


def test_rank_consistency_concern_for_low_rank():
    r = generate_reasoning(
        _row(
            "CAND_0000003",
            "Data Scientist",
            ["Embeddings"],
            notice_period_days=120,
            days_since_active=200,
            location_fit=0.1,
            location="Berlin, Germany",
        ),
        95,
    )
    assert "Watch-out" in r  # a rank-95 candidate should surface a concern


def test_reasonings_are_unique():
    rows = [
        _row("CAND_0000001", "ML Engineer", ["Vector Search", "Embeddings"]),
        _row("CAND_0000002", "AI Engineer", ["Qdrant", "BM25"]),
        _row("CAND_0000003", "NLP Engineer", ["Semantic Search", "Information Retrieval"]),
    ]
    out = generate_all(rows)
    assert len(set(out)) == len(out)
