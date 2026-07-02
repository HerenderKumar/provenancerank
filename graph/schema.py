"""Knowledge-graph schema + an in-memory backend.

Graph shape (same nodes/edges whether we're on Neo4j or in-memory):

    (Developer)-[:HAS_SKILL {confidence, evidence_count, last_seen}]->(Skill)
    (Developer)-[:PRODUCED]->(Artifact)-[:DEMONSTRATES]->(Skill)

The in-memory backend is the default — it makes the whole feature run and test
without a Neo4j server, and it's the fallback if Neo4j is down. It implements
the same async interface the Neo4j backend does, so callers don't care which is
behind them. The "production debugger" query is the one that justifies a graph:
it's a 3-hop traversal (Developer→Artifact→Skill with predicates) that's ugly in
SQL and natural here.
"""

from __future__ import annotations

import datetime as dt

# canonical labels / relationship types (shared with the Cypher backend)
DEVELOPER, SKILL, ARTIFACT = "Developer", "Skill", "Artifact"
HAS_SKILL, PRODUCED, DEMONSTRATES = "HAS_SKILL", "PRODUCED", "DEMONSTRATES"

# evidence types that count as "real production work" (not just SO answers)
WORK_SOURCES = ("commit_diff", "pr_review", "issue_thread")


class InMemoryGraphStore:
    name = "in-memory"

    def __init__(self) -> None:
        self.developers: dict[str, dict] = {}
        self.skills: dict[str, dict] = {}
        self.artifacts: dict[str, dict] = {}
        self.has_skill: dict[tuple[str, str], dict] = {}
        self.produced: set[tuple[str, str]] = set()
        self.demonstrates: set[tuple[str, str]] = set()

    async def ping(self) -> bool:
        return True

    async def upsert_developer(
        self, developer_id: str, github_username: str | None = None, display_name: str | None = None
    ) -> None:
        d = self.developers.setdefault(developer_id, {"id": developer_id})
        if github_username:
            d["github_username"] = github_username
        if display_name:
            d["display_name"] = display_name

    async def upsert_skill_with_evidence(
        self,
        developer_id: str,
        skill_name: str,
        artifact_id: str,
        source_type: str,
        complexity: int,
        confidence: float,
        artifact_date: dt.datetime | None,
        *,
        summary: str = "",
        url: str = "",
        production_signal: bool = False,
        skill_category: str = "general",
    ) -> None:
        await self.upsert_developer(developer_id)
        self.skills.setdefault(skill_name, {"name": skill_name, "category": skill_category})
        self.artifacts[artifact_id] = {
            "id": artifact_id,
            "source_type": source_type,
            "summary": summary,
            "complexity": int(complexity),
            "production_signal": bool(production_signal),
            "url": url,
            "date": artifact_date.isoformat() if artifact_date else None,
            "developer_id": developer_id,
        }
        # `confidence` is this artifact's contribution; confidence accumulates
        # (and clamps at 1.0) so more proven work => higher trust.
        edge = self.has_skill.get((developer_id, skill_name))
        if edge is None:
            self.has_skill[(developer_id, skill_name)] = {
                "confidence": min(confidence, 1.0),
                "evidence_count": 1,
                "last_seen": artifact_date.isoformat() if artifact_date else None,
            }
        else:
            edge["confidence"] = min(edge["confidence"] + confidence, 1.0)
            edge["evidence_count"] += 1
            iso = artifact_date.isoformat() if artifact_date else None
            if iso and (edge["last_seen"] is None or iso > edge["last_seen"]):
                edge["last_seen"] = iso
        self.produced.add((developer_id, artifact_id))
        self.demonstrates.add((artifact_id, skill_name))

    def _evidence_for(self, developer_id: str, skill_name: str) -> list[dict]:
        out = []
        for art_id, sk in self.demonstrates:
            if sk != skill_name:
                continue
            art = self.artifacts.get(art_id)
            if art and art["developer_id"] == developer_id:
                out.append(art)
        return sorted(out, key=lambda a: a.get("date") or "", reverse=True)

    async def find_by_skill(
        self, skill_name: str, min_confidence: float = 0.6, limit: int = 20
    ) -> list[dict]:
        rows = []
        for (dev_id, sk), edge in self.has_skill.items():
            if sk != skill_name or edge["confidence"] < min_confidence:
                continue
            dev = self.developers.get(dev_id, {})
            rows.append(
                {
                    "developer_id": dev_id,
                    "github_username": dev.get("github_username"),
                    "display_name": dev.get("display_name"),
                    "confidence": round(edge["confidence"], 4),
                    "evidence_count": edge["evidence_count"],
                    "evidence": self._evidence_for(dev_id, skill_name)[:3],
                }
            )
        rows.sort(key=lambda r: (r["confidence"], r["evidence_count"]), reverse=True)
        return rows[:limit]

    async def find_production_debuggers(
        self, skill_name: str, min_complexity: int = 4, limit: int = 10
    ) -> list[dict]:
        by_dev: dict[str, list[dict]] = {}
        for art_id, sk in self.demonstrates:
            if sk != skill_name:
                continue
            art = self.artifacts.get(art_id)
            if art and art["source_type"] in WORK_SOURCES and art["complexity"] >= min_complexity:
                by_dev.setdefault(art["developer_id"], []).append(art)
        rows = []
        for dev_id, arts in by_dev.items():
            if len(arts) < 2:  # needs >= 2 high-complexity work artifacts
                continue
            dev = self.developers.get(dev_id, {})
            edge = self.has_skill.get((dev_id, skill_name), {})
            rows.append(
                {
                    "developer_id": dev_id,
                    "github_username": dev.get("github_username"),
                    "display_name": dev.get("display_name"),
                    "confidence": round(edge.get("confidence", 0.0), 4),
                    "evidence_count": len(arts),
                    "evidence": sorted(arts, key=lambda a: a["complexity"], reverse=True)[:3],
                }
            )
        rows.sort(key=lambda r: (r["evidence_count"], r["confidence"]), reverse=True)
        return rows[:limit]

    async def all_evidence(self) -> list[dict]:
        """Every artifact joined with the skills it demonstrates — the corpus the
        semantic (vector) index embeds for hybrid retrieval."""
        skills_by_art: dict[str, list[str]] = {}
        for art_id, sk in self.demonstrates:
            skills_by_art.setdefault(art_id, []).append(sk)
        return [{**art, "skills": skills_by_art.get(art_id, [])} for art_id, art in self.artifacts.items()]

    async def developer_subgraph(self, developer_id: str) -> dict:
        skills = [
            {"skill": sk, **edge}
            for (dev, sk), edge in self.has_skill.items()
            if dev == developer_id
        ]
        artifacts = [a for a in self.artifacts.values() if a["developer_id"] == developer_id]
        return {
            "developer": self.developers.get(developer_id, {"id": developer_id}),
            "skills": sorted(skills, key=lambda s: s["confidence"], reverse=True),
            "artifacts": sorted(artifacts, key=lambda a: a.get("date") or "", reverse=True),
        }
