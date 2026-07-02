"""GraphWriter / GraphQuerier over a pluggable backend.

Backend is Neo4j in prod, in-memory otherwise. Every Neo4j call goes through a
circuit breaker so a flaky graph DB degrades to fast failures instead of hanging
the API — and the factory falls back to in-memory if Neo4j can't be reached at
startup, so a dead graph never takes down ranking.
"""

from __future__ import annotations

from core.circuit_breaker import CircuitBreaker
from core.config import get_settings
from core.logging import get_logger
from graph.schema import WORK_SOURCES, InMemoryGraphStore

log = get_logger("graph.queries")


class Neo4jBackend:
    name = "neo4j"

    def __init__(self, uri: str, user: str, password: str):
        from neo4j import AsyncGraphDatabase

        self._driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
        self._cb = CircuitBreaker("neo4j", fail_max=5, reset_timeout=30)

    async def ping(self) -> bool:
        async def _ping():
            async with self._driver.session() as s:
                await s.run("RETURN 1")
            return True

        return await self._cb.call(_ping)

    async def _write(self, cypher: str, **params):
        async def _run():
            async with self._driver.session() as s:
                await s.run(cypher, **params)

        return await self._cb.call(_run)

    async def _read(self, cypher: str, **params) -> list[dict]:
        async def _run():
            async with self._driver.session() as s:
                res = await s.run(cypher, **params)
                return [r.data() async for r in res]

        return await self._cb.call(_run)

    async def upsert_developer(self, developer_id, github_username=None, display_name=None):
        await self._write(
            "MERGE (d:Developer {id:$id}) "
            "ON CREATE SET d.created_at=timestamp() "
            "SET d.github_username=coalesce($u, d.github_username), "
            "    d.display_name=coalesce($n, d.display_name), d.last_updated=timestamp()",
            id=developer_id,
            u=github_username,
            n=display_name,
        )

    async def upsert_skill_with_evidence(
        self,
        developer_id,
        skill_name,
        artifact_id,
        source_type,
        complexity,
        confidence,
        artifact_date,
        *,
        summary="",
        url="",
        production_signal=False,
        skill_category="general",
    ):
        await self._write(
            "MERGE (d:Developer {id:$dev}) "
            "MERGE (s:Skill {name:$skill}) ON CREATE SET s.category=$cat "
            "MERGE (a:Artifact {id:$art}) "
            "SET a.source_type=$src, a.complexity=$cx, a.production_signal=$prod, "
            "    a.summary=$summary, a.url=$url, a.date=$date "
            "MERGE (d)-[:PRODUCED]->(a) "
            "MERGE (a)-[:DEMONSTRATES]->(s) "
            "MERGE (d)-[h:HAS_SKILL]->(s) "
            "ON CREATE SET h.confidence=$conf, h.evidence_count=1, h.last_seen=$date "
            "ON MATCH SET h.confidence=CASE WHEN h.confidence+$conf>1 THEN 1 ELSE h.confidence+$conf END, "
            "             h.evidence_count=h.evidence_count+1, "
            "             h.last_seen=CASE WHEN $date>h.last_seen THEN $date ELSE h.last_seen END",
            dev=developer_id,
            skill=skill_name,
            art=artifact_id,
            src=source_type,
            cx=int(complexity),
            prod=bool(production_signal),
            summary=summary,
            url=url,
            date=artifact_date.isoformat() if artifact_date else None,
            conf=float(confidence),
            cat=skill_category,
        )

    async def find_by_skill(self, skill_name, min_confidence=0.6, limit=20):
        return await self._read(
            "MATCH (d:Developer)-[h:HAS_SKILL]->(s:Skill {name:$skill}) "
            "WHERE h.confidence >= $minc "
            "OPTIONAL MATCH (d)-[:PRODUCED]->(a:Artifact)-[:DEMONSTRATES]->(s) "
            "WITH d, h, collect(a)[..3] AS evidence "
            "RETURN d.id AS developer_id, d.github_username AS github_username, "
            "       d.display_name AS display_name, h.confidence AS confidence, "
            "       h.evidence_count AS evidence_count, evidence "
            "ORDER BY h.confidence DESC, h.evidence_count DESC LIMIT $limit",
            skill=skill_name,
            minc=min_confidence,
            limit=limit,
        )

    async def find_production_debuggers(self, skill_name, min_complexity=4, limit=10):
        return await self._read(
            "MATCH (d:Developer)-[:PRODUCED]->(a:Artifact)-[:DEMONSTRATES]->(s:Skill {name:$skill}) "
            "WHERE a.source_type IN $sources AND a.complexity >= $minx "
            "WITH d, collect(a) AS arts WHERE size(arts) >= 2 "
            "OPTIONAL MATCH (d)-[h:HAS_SKILL]->(s:Skill {name:$skill}) "
            "RETURN d.id AS developer_id, d.github_username AS github_username, "
            "       d.display_name AS display_name, coalesce(h.confidence,0.0) AS confidence, "
            "       size(arts) AS evidence_count, arts[..3] AS evidence "
            "ORDER BY evidence_count DESC, confidence DESC LIMIT $limit",
            skill=skill_name,
            sources=list(WORK_SOURCES),
            minx=min_complexity,
            limit=limit,
        )

    async def developer_subgraph(self, developer_id):
        rows = await self._read(
            "MATCH (d:Developer {id:$id}) "
            "OPTIONAL MATCH (d)-[h:HAS_SKILL]->(s:Skill) "
            "OPTIONAL MATCH (d)-[:PRODUCED]->(a:Artifact) "
            "RETURN d AS developer, collect(DISTINCT {skill:s.name, confidence:h.confidence}) AS skills, "
            "collect(DISTINCT a) AS artifacts",
            id=developer_id,
        )
        return (
            rows[0] if rows else {"developer": {"id": developer_id}, "skills": [], "artifacts": []}
        )

    async def all_evidence(self):
        return await self._read(
            "MATCH (d:Developer)-[:PRODUCED]->(a:Artifact)-[:DEMONSTRATES]->(s:Skill) "
            "RETURN a.id AS id, d.id AS developer_id, a.source_type AS source_type, "
            "       a.summary AS summary, a.complexity AS complexity, "
            "       a.production_signal AS production_signal, collect(s.name) AS skills"
        )

    @property
    def healthy(self) -> bool:
        return self._cb.healthy


# thin write/read facades so callers don't depend on the backend type
class GraphWriter:
    def __init__(self, backend):
        self.b = backend

    async def upsert_developer(self, developer_id, github_username=None, display_name=None):
        await self.b.upsert_developer(developer_id, github_username, display_name)

    async def upsert_skill_with_evidence(self, **kw):
        await self.b.upsert_skill_with_evidence(**kw)


class GraphQuerier:
    def __init__(self, backend):
        self.b = backend

    async def find_by_skill(self, skill_name, min_confidence=0.6, limit=20):
        return await self.b.find_by_skill(skill_name, min_confidence, limit)

    async def find_production_debuggers(self, skill_name, min_complexity=4, limit=10):
        return await self.b.find_production_debuggers(skill_name, min_complexity, limit)

    async def developer_subgraph(self, developer_id):
        return await self.b.developer_subgraph(developer_id)

    async def skill_confidence(self, developer_id: str, skill_name: str) -> float | None:
        """Used by the ranking override — current graph confidence for one skill."""
        try:
            rows = await self.b.find_by_skill(skill_name, min_confidence=0.0, limit=10000)
        except Exception:
            return None
        for r in rows:
            if r["developer_id"] == developer_id:
                return float(r["confidence"])
        return None

    async def all_evidence(self) -> list[dict]:
        """All evidence for the semantic index. Returns [] if the backend can't
        enumerate it, so hybrid retrieval simply degrades to skill-only."""
        fn = getattr(self.b, "all_evidence", None)
        if fn is None:
            return []
        try:
            return await fn()
        except Exception as exc:
            log.warning("graph.all_evidence_failed", reason=str(exc)[:120])
            return []


_backend = None


async def get_backend():
    """Process-wide graph backend. Neo4j when enabled + reachable, else in-memory.
    Shared by ingestion (writes) and the API (reads) in a single process."""
    global _backend
    if _backend is not None:
        return _backend
    s = get_settings()
    if s.neo4j_enabled:
        try:
            be = Neo4jBackend(s.neo4j_uri, s.neo4j_user, s.neo4j_password)
            await be.ping()
            log.info("graph.backend_ready", backend=be.name)
            _backend = be
            return _backend
        except Exception as exc:
            log.warning("graph.neo4j_unavailable_inmemory_fallback", reason=str(exc)[:140])
    _backend = InMemoryGraphStore()
    log.info("graph.backend_ready", backend=_backend.name)
    return _backend


def reset_backend() -> None:
    """Test hook."""
    global _backend
    _backend = None
