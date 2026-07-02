"""Content-addressed cache for precompute stages.

``--resume`` already skips *everything* when the candidates file is unchanged.
This is finer-grained: each expensive stage records a key derived from its own
inputs, so a re-run recomputes only what actually changed. The headline case:
swap the JD and we reuse the 100K embeddings (and the BM25 index) untouched -
only retrieval, rerank and the tournament rerun.

A stage writes its artifact plus a sidecar ``<artifact>.key`` holding the hash of
its inputs. On the next run it loads the artifact iff the file exists, the key
file exists, and the recomputed key matches. Anything else (missing file, stale
key, unreadable artifact) is a clean miss that triggers recompute - so a bad
cache can never produce a wrong result, only a slower run.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from core.logging import get_logger

log = get_logger("core.artifact_cache")


def key_of(*parts: object) -> str:
    """Stable short hash of the given inputs (order-sensitive)."""
    h = hashlib.sha256()
    for part in parts:
        h.update(repr(part).encode("utf-8"))
    return h.hexdigest()[:16]


def _key_path(artifact: Path) -> Path:
    return artifact.with_suffix(artifact.suffix + ".key")


def is_fresh(artifact: str | Path, key: str) -> bool:
    """True when the artifact exists and its sidecar key matches ``key``."""
    artifact = Path(artifact)
    kp = _key_path(artifact)
    return artifact.exists() and kp.exists() and kp.read_text().strip() == key


def stamp(artifact: str | Path, key: str) -> None:
    """Record the input key next to an artifact that was just written."""
    _key_path(Path(artifact)).write_text(key)


def load_npy(artifact: str | Path, key: str) -> np.ndarray | None:
    """Return the cached array if fresh, else None (a clean miss)."""
    artifact = Path(artifact)
    if not is_fresh(artifact, key):
        return None
    try:
        arr = np.load(artifact)
        log.info("cache.hit", artifact=artifact.name)
        return arr
    except Exception as exc:
        log.warning("cache.unreadable_recomputing", artifact=artifact.name, reason=str(exc)[:120])
        return None


def save_npy(artifact: str | Path, key: str, arr: np.ndarray) -> None:
    artifact = Path(artifact)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    np.save(artifact, arr)
    stamp(artifact, key)


def load_json(artifact: str | Path, key: str) -> dict | None:
    artifact = Path(artifact)
    if not is_fresh(artifact, key):
        return None
    try:
        return json.loads(artifact.read_text())
    except Exception:
        return None


def save_json(artifact: str | Path, key: str, payload: dict) -> None:
    artifact = Path(artifact)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps(payload, indent=2))
    stamp(artifact, key)
