"""Recruiter types English, we run a graph query, we return candidates with the
exact artifacts that back them. The thing LinkedIn can't do.

"show me who debugged a race condition in production" parses to
{skills:[concurrency], min_complexity:5, require_production:true} and routes to
the production-debuggers traversal. Gemini does the parse when a key is set;
otherwise a keyword heuristic handles the common shapes.
"""

from __future__ import annotations

import asyncio

from core.config import get_settings
from core.logging import get_logger
from graph.queries import GraphQuerier
from graph.vector_index import get_index
from intelligence.skill_extractor import extract_skills

log = get_logger("graph.nlq")

_PRODUCTION_HINTS = (
    "production",
    "deployed",
    "shipped",
    "real users",
    "at scale",
    "incident",
    "live",
    "users",
)
_DEBUG_HINTS = (
    "debug",
    "race condition",
    "deadlock",
    "fixed",
    "bug",
    "crash",
    "memory leak",
    "concurren",
    "hard problem",
    "complex",
)


def _heuristic_intent(query: str) -> dict:
    q = query.lower()
    skills = extract_skills(query)
    if not skills and "race condition" in q:
        skills = ["concurrency"]
    production = any(h in q for h in _PRODUCTION_HINTS)
    hard = any(h in q for h in _DEBUG_HINTS)
    return {
        "skill_names": skills,
        "min_confidence": 0.5,
        "min_complexity": 4 if hard else 1,
        "require_production_signal": production,
        "source_types": ["commit_diff", "pr_review", "issue_thread"],
        "limit": 10,
    }


async def _llm_intent(query: str) -> dict | None:
    """Parse the query into a structured intent via the configured LLM
    (local Ollama / Gemini). None when no provider is reachable -> heuristic."""
    from core.llm import complete_json

    prompt = (
        "Parse this recruiter query into STRICT JSON with keys: skill_names "
        "(list of canonical tech skills), min_confidence (0-1), min_complexity (1-5), "
        "require_production_signal (bool), source_types (subset of commit_diff,pr_review,"
        "issue_thread,so_answer), limit (int). Query: " + query
    )
    data = await asyncio.to_thread(complete_json, prompt)
    if not data:
        return None
    data["skill_names"] = extract_skills(data.get("skill_names", [])) or extract_skills(query)
    return data


def _rrf(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal Rank Fusion of several ranked id lists — the same trick the main
    ranker uses for BM25 + embeddings. Combines *ranks*, so the two retrievers'
    incomparable scores (graph confidence vs. cosine similarity) don't need to be
    on the same scale."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, dev_id in enumerate(ranking, start=1):
            scores[dev_id] = scores.get(dev_id, 0.0) + 1.0 / (k + rank)
    return scores


async def _structured_rows(query: str, intent: dict, querier: GraphQuerier) -> dict[str, dict]:
    """Exact skill-traversal retrieval (precise). Keyed by developer, best
    confidence wins across the query's skills."""
    skills = intent.get("skill_names") or []
    production = intent.get("require_production_signal") or intent.get("min_complexity", 1) >= 4
    merged: dict[str, dict] = {}
    for skill in skills:
        if production:
            rows = await querier.find_production_debuggers(
                skill, min_complexity=max(intent.get("min_complexity", 4), 4),
                limit=intent.get("limit", 10),
            )
            if not rows:  # broaden rather than show nothing
                intent["relaxed"] = True
                rows = await querier.find_by_skill(skill, min_confidence=0.0,
                                                   limit=intent.get("limit", 20))
        else:
            rows = await querier.find_by_skill(
                skill, min_confidence=intent.get("min_confidence", 0.5),
                limit=intent.get("limit", 20),
            )
        for r in rows:
            cur = merged.get(r["developer_id"])
            if cur is None or r["confidence"] > cur["confidence"]:
                merged[r["developer_id"]] = r
    return merged


async def _semantic_rows(query: str, intent: dict, querier: GraphQuerier) -> dict[str, dict]:
    """Vector retrieval over evidence summaries (recall). Catches paraphrases the
    canonical-skill traversal misses. Empty when the backend can't enumerate
    evidence, so the result is simply skill-only."""
    if not get_settings().graph_hybrid_enabled:
        return {}
    evidence = await querier.all_evidence()
    if not evidence:
        return {}
    index = get_index(evidence)
    rows = index.developer_ranking(query, top_k=get_settings().graph_vector_top_k)
    return {r["developer_id"]: r for r in rows}


async def natural_language_to_results(query: str, querier: GraphQuerier) -> dict:
    intent = await _llm_intent(query) or _heuristic_intent(query)

    structured = await _structured_rows(query, intent, querier)
    semantic = await _semantic_rows(query, intent, querier)

    # Fuse the two retrievers by rank (RRF). With no semantic side this reduces to
    # the structured order, so the change is invisible when hybrid is off.
    fused = _rrf(
        [list(structured.keys()), list(semantic.keys())],
        k=get_settings().graph_rrf_k,
    )
    order = sorted(fused, key=lambda d: fused[d], reverse=True)

    candidates = []
    for dev_id in order:
        s_row, v_row = structured.get(dev_id), semantic.get(dev_id)
        if s_row and v_row:
            via = "both"
        elif s_row:
            via = "skill"
        else:
            via = "semantic"
        evidence = (s_row or {}).get("evidence") or (v_row or {}).get("evidence") or []
        candidates.append(
            {
                "developer_id": dev_id,
                "github_username": (s_row or {}).get("github_username"),
                "display_name": (s_row or {}).get("display_name"),
                "confidence": (s_row or {}).get("confidence", 0.0),
                "semantic_score": (v_row or {}).get("semantic_score", 0.0),
                "evidence_count": (s_row or {}).get("evidence_count", len(evidence)),
                "evidence": evidence[:3],
                "match_via": via,
                "rrf_score": round(fused[dev_id], 5),
            }
        )

    intent["retrieval"] = {"skill_hits": len(structured), "semantic_hits": len(semantic)}
    return {
        "query_interpreted": intent,
        "candidates": candidates[: intent.get("limit", 10)],
    }
