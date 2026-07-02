"""JD deconstruction - the ONLY place an LLM may be called, and only offline.

Produces a structured ``JobSpec`` saved to ``artifacts/jd_spec.json``. At
ranking time the gate filter and scorers read this JSON; rank.py never imports
this module, guaranteeing zero network during ranking.

Reliability pattern - Circuit Breaker:
    The single Gemini Flash call is wrapped with bounded retries. After the
    breaker opens (repeated failures) OR when no API key is configured, we fall
    back to a hand-derived ``JobSpec`` built from a careful reading of the real
    JD. Ranking therefore never blocks on an external API, and the system is
    fully functional with zero credentials.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from core.constants import JD_CORE_SKILLS
from core.logging import get_logger

log = get_logger("pipeline.jd_deconstructor")


@dataclass
class Gate:
    """A hard disqualifier mapped to a precomputed feature key so the runtime
    gate filter is a pure vectorised comparison (no parsing, no LLM)."""

    description: str
    feature_key: str
    operator: str  # "eq" | "lt" | "gt" | "gte" | "lte" | "bool" | "and"
    value: Any


@dataclass
class WeightedReq:
    description: str
    weight: float
    skill_tags: list[str]


@dataclass
class JobSpec:
    hard_gates: list[Gate]
    weighted_requirements: list[WeightedReq]
    latent_requirements: list[str]
    jd_core_skills: list[str]
    anti_patterns: list[str]
    boilerplate_removed: list[str]
    source: str = "fallback"  # "gemini" | "fallback"

    def to_json(self) -> dict:
        return {
            "hard_gates": [asdict(g) for g in self.hard_gates],
            "weighted_requirements": [asdict(w) for w in self.weighted_requirements],
            "latent_requirements": self.latent_requirements,
            "jd_core_skills": self.jd_core_skills,
            "anti_patterns": self.anti_patterns,
            "boilerplate_removed": self.boilerplate_removed,
            "source": self.source,
        }

    @classmethod
    def from_json(cls, data: dict) -> JobSpec:
        return cls(
            hard_gates=[Gate(**g) for g in data.get("hard_gates", [])],
            weighted_requirements=[WeightedReq(**w) for w in data.get("weighted_requirements", [])],
            latent_requirements=data.get("latent_requirements", []),
            jd_core_skills=data.get("jd_core_skills", []),
            anti_patterns=data.get("anti_patterns", []),
            boilerplate_removed=data.get("boilerplate_removed", []),
            source=data.get("source", "fallback"),
        )


# ---------------------------------------------------------------------------
# Hand-derived fallback JobSpec (from the actual Senior AI Engineer JD).
# These hard gates intentionally encode only HIGH-CONFIDENCE disqualifiers;
# "softer" disqualifiers (notice period, wrong specialisation) are applied as
# score penalties rather than hard exclusions because the JD explicitly keeps
# those candidates "in scope, the bar gets higher".
# ---------------------------------------------------------------------------


def fallback_job_spec() -> JobSpec:
    return JobSpec(
        hard_gates=[
            Gate(
                "Honeypot / impossible profile (ground-truth tier 0)", "is_honeypot", "bool", True
            ),
            Gate(
                "Entire career at pure consulting/services firms, no product co",
                "all_consulting_career",
                "bool",
                True,
            ),
            Gate(
                "AI keyword stuffer on a non-technical career (framework trap)",
                "keyword_stuffer_flag",
                "bool",
                True,
            ),
            Gate(
                "Inactive & unresponsive (>180d inactive AND response_rate<0.15)",
                "inactive_unresponsive",
                "bool",
                True,
            ),
        ],
        weighted_requirements=[
            WeightedReq(
                "Production embeddings-based retrieval deployed to users",
                1.0,
                [
                    "embeddings",
                    "sentence transformers",
                    "retrieval",
                    "vector search",
                    "semantic search",
                ],
            ),
            WeightedReq(
                "Vector DB / hybrid search operational experience",
                1.0,
                [
                    "pinecone",
                    "qdrant",
                    "weaviate",
                    "milvus",
                    "faiss",
                    "elasticsearch",
                    "opensearch",
                    "hybrid search",
                    "bm25",
                ],
            ),
            WeightedReq(
                "Evaluation frameworks for ranking (NDCG/MRR/MAP, A/B)",
                0.9,
                ["ndcg", "mrr", "map", "learning to rank", "a/b testing"],
            ),
            WeightedReq("Strong Python / code quality", 0.6, ["python"]),
            WeightedReq(
                "Prior HR-tech / recruiting / marketplace experience",
                0.9,
                ["recommendation systems", "search relevance"],
            ),
            WeightedReq(
                "LLM fine-tuning (LoRA/QLoRA/PEFT)",
                0.8,
                ["lora", "qlora", "peft", "fine-tuning llms"],
            ),
            WeightedReq(
                "Learning-to-rank (XGBoost/neural LTR)",
                0.7,
                ["learning to rank", "xgboost", "lightgbm", "ltr"],
            ),
            WeightedReq(
                "Distributed systems / large-scale inference",
                0.6,
                ["distributed systems", "mlops", "model serving"],
            ),
            WeightedReq(
                "Open-source AI/ML contributions",
                0.7,
                ["pytorch", "transformers", "hugging face transformers"],
            ),
        ],
        latent_requirements=[
            "Product-company experience over pure research",
            "Ships fast (tilt toward shipper, not researcher)",
            "Pre-LLM-era ML production experience (understood retrieval before it was fashionable)",
            "Based in / willing to relocate to Pune or Noida (India, no visa sponsorship)",
            "Actively in the job market (recent platform activity, responsive)",
            "Has shipped at least one end-to-end ranking/search/recsys system at scale",
        ],
        jd_core_skills=sorted(JD_CORE_SKILLS),
        anti_patterns=[
            "Title-chaser: company switches every ~1.5 years for title bumps",
            "Framework enthusiast: LangChain/prompt demos without systems depth",
            "Pure consulting-firm career with no product exposure",
            "Primary expertise CV/speech/robotics without NLP/IR",
            "'Open to learning AI' with no pre-LLM ML production history",
            "Pure research background with zero production deployment",
            "Moved to architecture/tech-lead; no production code in 18 months",
            "Perfect-on-paper but inactive (>6mo) with low recruiter response",
        ],
        boilerplate_removed=[
            "vibe check",
            "we disagree openly",
            "async-first and write a lot",
            "move fast and break things",
            "the JD changes every six months",
        ],
        source="fallback",
    )


# ---------------------------------------------------------------------------
# Optional LLM path (local Ollama / Gemini, via core.llm) - enriches the
# deterministic fallback. Never required; absent it, the rules-based spec stands.
# ---------------------------------------------------------------------------


def _build_prompt(jd_text: str) -> str:
    return (
        "You are parsing a job description into a strict JSON JobSpec. "
        "Return ONLY JSON with keys: latent_requirements (array of strings), "
        "anti_patterns (array of strings), boilerplate_removed (array of strings). "
        "Focus on the gap between what the JD says and what it means.\n\nJD:\n" + jd_text
    )


def _merge_llm_into_fallback(data: dict) -> JobSpec:
    """Keep deterministic hard gates / weighted reqs from the fallback; let the
    LLM enrich only the free-text latent/anti-pattern fields."""
    spec = fallback_job_spec()
    if isinstance(data.get("latent_requirements"), list):
        spec.latent_requirements = data["latent_requirements"]
    if isinstance(data.get("anti_patterns"), list):
        spec.anti_patterns = data["anti_patterns"]
    if isinstance(data.get("boilerplate_removed"), list):
        spec.boilerplate_removed = data["boilerplate_removed"]
    return spec


def deconstruct_jd(jd_text: str | None = None) -> JobSpec:
    """Return a JobSpec, letting an LLM (local Ollama or Gemini) enrich the
    deterministic fallback when one is reachable, else the rules-based spec."""
    if not jd_text:
        log.info("jd_deconstruct.fallback_engaged", reason="no_jd_text")
        return fallback_job_spec()
    data = None
    try:
        from core.llm import complete_json

        data = complete_json(_build_prompt(jd_text))
    except Exception as exc:
        log.warning("jd_deconstruct.llm_error", reason=str(exc)[:160])
    if data:
        spec = _merge_llm_into_fallback(data)
        spec.source = "llm"
        log.info("jd_deconstruct.llm_ok")
        return spec
    log.info("jd_deconstruct.fallback_engaged", reason="no_llm")
    return fallback_job_spec()


def save_job_spec(spec: JobSpec, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(spec.to_json(), f, indent=2)
    log.info(
        "jd_spec.saved",
        path=str(path),
        source=spec.source,
        gates=len(spec.hard_gates),
        weighted=len(spec.weighted_requirements),
    )


def load_job_spec(path: str | Path) -> JobSpec:
    with open(path, encoding="utf-8") as f:
        return JobSpec.from_json(json.load(f))
