"""Streaming candidate loader.

Reads ``candidates.jsonl`` in bounded batches so the 100K-record, ~480 MB file
never lands in memory all at once (functional requirement #1 / DO-NOT #2).
Malformed lines are skipped and logged rather than crashing the run - a single
bad record must not abort a 100K-candidate job (resilience requirement).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from core.logging import get_logger

log = get_logger("pipeline.loader")


def file_sha256(path: str | Path, _chunk: int = 1 << 20) -> str:
    """Content hash used to invalidate precomputed artifacts when the dataset
    changes (idempotent pre-computation pattern)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(_chunk), b""):
            h.update(block)
    return h.hexdigest()


def count_lines(path: str | Path) -> int:
    """Fast newline count for progress bars without parsing JSON."""
    total = 0
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            total += block.count(b"\n")
    return total


def iter_candidates(path: str | Path) -> Iterator[dict]:
    """Yield one parsed candidate dict per line, skipping malformed rows."""
    bad = 0
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                bad += 1
                if bad <= 10:
                    log.warning("loader.bad_line", lineno=lineno, error=str(exc)[:120])
    if bad:
        log.warning("loader.bad_lines_total", count=bad)


def stream_batches(path: str | Path, batch_size: int = 1000) -> Iterator[list[dict]]:
    """Yield lists of at most ``batch_size`` candidates."""
    batch: list[dict] = []
    for cand in iter_candidates(path):
        batch.append(cand)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


# ---------------------------------------------------------------------------
# Text projection used by the BM25 index and the embedder. Defined once so the
# offline corpus and any online query share an identical text contract.
# ---------------------------------------------------------------------------


def corpus_text(candidate: dict) -> str:
    """Flatten a candidate into a single searchable document."""
    profile = candidate.get("profile", {}) or {}
    parts: list[str] = [
        str(profile.get("headline", "")),
        str(profile.get("summary", "")),
        str(profile.get("current_title", "")),
        str(profile.get("current_industry", "")),
    ]
    for role in candidate.get("career_history", []) or []:
        parts.append(str(role.get("title", "")))
        parts.append(str(role.get("description", "")))
    for skill in candidate.get("skills", []) or []:
        parts.append(str(skill.get("name", "")))
    return " ".join(p for p in parts if p).strip()


def rerank_text(candidate: dict) -> str:
    """A tight projection for the cross-encoder: title, headline, the most recent
    role, and the top endorsed skills - not the whole career history.

    The reranker reads the JD and this together in one transformer pass, and
    attention cost grows with sequence length, so a focused ~100-token document is
    both faster and a cleaner relevance signal than the full concatenated blob
    (which buries the signal under a decade of older roles)."""
    profile = candidate.get("profile", {}) or {}
    parts: list[str] = [
        str(profile.get("current_title", "")),
        str(profile.get("headline", "")),
    ]
    roles = candidate.get("career_history") or []
    if roles:
        recent = roles[0]
        parts.append(str(recent.get("title", "")))
        parts.append(str(recent.get("description", ""))[:300])
    skills = candidate.get("skills") or []
    top = sorted(skills, key=lambda sk: sk.get("endorsements", 0) or 0, reverse=True)[:10]
    parts.append(" ".join(str(sk.get("name", "")) for sk in top))
    return " ".join(p for p in parts if p).strip()


def safe_get(candidate: dict, *keys: str, default=None):
    """Nested ``dict.get`` that tolerates missing intermediate keys."""
    node = candidate
    for key in keys:
        if not isinstance(node, dict):
            return default
        node = node.get(key)
        if node is None:
            return default
    return node


def load_all(path: str | Path) -> list[dict]:
    """Materialise every candidate (used by tests and small samples only)."""
    return list(iter_candidates(path))


def candidate_ids(candidates: Iterable[dict]) -> list[str]:
    return [c["candidate_id"] for c in candidates]
