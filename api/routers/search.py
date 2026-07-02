"""Search over the evidence graph. The natural-language endpoint is the "aha":
ask in English, get developers back with the exact artifacts that prove it.

If the graph backend is Neo4j and its circuit breaker is open, CircuitOpen maps
to a clean 503 — the recruiter sees "graph unavailable", the rest of the API
(ranking, auth) keeps working. That's the isolation guarantee in action.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from api.deps import AuthContext, authenticate
from api.schemas import NLSearchRequest, NLSearchResponse
from core.logging import get_logger
from graph.natural_language_query import natural_language_to_results
from graph.queries import GraphQuerier, get_backend

log = get_logger("api.search")
router = APIRouter(prefix="/search", tags=["search"])


@router.post("/natural-language", response_model=NLSearchResponse)
async def natural_language(
    body: NLSearchRequest, _: AuthContext = Depends(authenticate)
) -> NLSearchResponse:
    querier = GraphQuerier(await get_backend())
    result = await natural_language_to_results(body.query, querier)
    log.info("search.nl", query=body.query[:80], hits=len(result["candidates"]))
    return NLSearchResponse(**result)


@router.get("/by-skill/{skill_name}")
async def by_skill(
    skill_name: str,
    min_confidence: float = Query(0.6, ge=0.0, le=1.0),
    limit: int = Query(20, ge=1, le=100),
    _: AuthContext = Depends(authenticate),
) -> dict:
    querier = GraphQuerier(await get_backend())
    rows = await querier.find_by_skill(skill_name.lower(), min_confidence, limit)
    return {"skill": skill_name, "count": len(rows), "candidates": rows}
