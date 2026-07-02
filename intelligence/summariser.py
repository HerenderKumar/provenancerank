"""Turn a raw artifact (commit diff, PR review, issue thread, SO answer) into a
structured, queryable SummaryResult.

Gemini Flash in JSON mode does the real work when a key is set. Without one we
fall back to a deterministic heuristic so the whole pipeline still runs offline
(and so tests don't need a network). Either way the content_hash is the SHA-256
of the raw content — the cryptographic anchor that makes the evidence tamper-
evident and the indexing step idempotent.
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, field
from typing import Literal

from core.logging import get_logger
from intelligence.skill_extractor import extract_skills

log = get_logger("intelligence.summariser")

ArtifactType = Literal["commit_diff", "pr_review", "issue_thread", "so_answer"]
MAX_CHARS = 8000

_PRODUCTION_TOKENS = (
    "production",
    "deployed",
    "users",
    "scale",
    "latency",
    "incident",
    "monitoring",
    "a/b",
    "experiment",
    "throughput",
    "p99",
    "sla",
    "rollout",
)
_HARD_TOKENS = (
    "race condition",
    "deadlock",
    "distributed",
    "consensus",
    "throughput",
    "latency",
    "scale",
    "optimi",
    "memory leak",
    "concurren",
    "sharding",
)
_TRIVIAL_TOKENS = ("typo", "readme", "comment", "lint", "formatting", "rename", "bump")
_COLLAB_TOKENS = ("review", "discuss", "feedback", "pair", "teammate", "approve", "comment")


@dataclass
class SummaryResult:
    what_was_done: str
    skills_demonstrated: list[str]
    complexity: int  # 1-5
    problem_category: str  # debugging|architecture|feature|infrastructure|refactoring|review
    production_signal: bool
    collaboration_signal: bool
    content_hash: str
    raw_chars: int = 0
    source: str = "heuristic"  # heuristic | gemini
    skill_confidences: dict[str, float] = field(default_factory=dict)


def content_hash(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _classify(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ("race condition", "deadlock", "bug", "fix", "crash", "regression")):
        return "debugging"
    if any(k in t for k in ("architecture", "design", "rfc", "consensus", "schema")):
        return "architecture"
    if any(k in t for k in ("deploy", "infra", "terraform", "kubernetes", "ci/cd", "pipeline")):
        return "infrastructure"
    if any(k in t for k in ("refactor", "cleanup", "rename", "migrate")):
        return "refactoring"
    if any(k in t for k in ("review", "lgtm", "approve")):
        return "review"
    return "feature"


def _complexity(text: str) -> int:
    t = text.lower()
    if any(k in t for k in _TRIVIAL_TOKENS) and len(t) < 400:
        return 1
    score = 2
    score += sum(1 for k in _HARD_TOKENS if k in t)
    if len(t) > 3000:
        score += 1
    return max(1, min(score, 5))


def _heuristic(artifact_type: str, raw: str) -> SummaryResult:
    text = raw[:MAX_CHARS]
    low = text.lower()
    skills = extract_skills(text)
    first = text.strip().splitlines()[0] if text.strip() else ""
    summary = (first[:160] or f"{artifact_type.replace('_', ' ')} with no description").strip()
    return SummaryResult(
        what_was_done=summary,
        skills_demonstrated=skills,
        complexity=_complexity(text),
        problem_category=_classify(text),
        production_signal=any(tok in low for tok in _PRODUCTION_TOKENS),
        collaboration_signal=artifact_type in ("pr_review", "issue_thread")
        or any(tok in low for tok in _COLLAB_TOKENS),
        content_hash=content_hash(raw),
        raw_chars=len(raw),
        source="heuristic",
    )


def _prompt(artifact_type: str, raw: str) -> str:
    return (
        "You analyse a software work artifact and return STRICT JSON only.\n"
        "Rules:\n"
        "- skills_demonstrated: list ONLY skills clearly demonstrated, not merely mentioned.\n"
        "- complexity 1-5: 5 = race-condition debugging / distributed design / scale "
        "optimisation; 1 = typo or README edit.\n"
        "- production_signal true if it mentions production, deployed users, scale, latency, "
        "incident, monitoring, A/B, or experiment.\n"
        "- problem_category: one of debugging|architecture|feature|infrastructure|refactoring|review.\n"
        "Return keys: what_was_done (1-2 sentences), skills_demonstrated, complexity, "
        "problem_category, production_signal, collaboration_signal.\n\n"
        f"ARTIFACT TYPE: {artifact_type}\nCONTENT:\n{raw[:MAX_CHARS]}"
    )


async def summarise_artifact(artifact_type: ArtifactType, raw_content: str) -> SummaryResult:
    chash = content_hash(raw_content)
    from core.llm import complete_json

    # complete_json is sync (httpx/genai); run it off the event loop.
    data = await asyncio.to_thread(complete_json, _prompt(artifact_type, raw_content))
    if not data:
        return _heuristic(artifact_type, raw_content)
    try:
        skills = extract_skills(data.get("skills_demonstrated", [])) or extract_skills(raw_content)
        return SummaryResult(
            what_was_done=str(data.get("what_was_done", ""))[:300],
            skills_demonstrated=skills,
            complexity=max(1, min(int(data.get("complexity", 2)), 5)),
            problem_category=str(data.get("problem_category", "feature")),
            production_signal=bool(data.get("production_signal", False)),
            collaboration_signal=bool(data.get("collaboration_signal", False)),
            content_hash=chash,
            raw_chars=len(raw_content),
            source="llm",
        )
    except Exception as exc:
        log.warning("summariser.llm_fallback", reason=str(exc)[:140])
        return _heuristic(artifact_type, raw_content)
