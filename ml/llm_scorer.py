"""LLM head-scoring — the smartest ranking signal, computed offline.

The GBDT, retrieval and behavioural signals are fast but shallow; they can't read
a career narrative and tell "shipped a production RAG service at scale" from "took
an LLM course". An LLM can. So in precompute we let the LLM (GLM via core.llm)
read each *top-of-the-list* candidate against the JD and score fit 0-100.

We only score the **head** (top ~250) — that's where NDCG@10/@50 is decided —
batch the candidates per call to keep it cheap (~a dozen calls, not 100K), cache
the result, and blend it into the final score like the cross-encoder rerank.
rank.py never calls the LLM; it reads the cached column, so the no-network / <5min
contract holds. If no LLM is reachable, `score_head` returns {} and the scorer
simply adds nothing.
"""

from __future__ import annotations

from core.config import get_settings
from core.llm import complete_json
from core.logging import get_logger, log_duration

log = get_logger("ml.llm_scorer")


def _prompt(jd_text: str, batch: list[tuple[str, str]]) -> str:
    lines = [f"[{cid}] {text[:500]}" for cid, text in batch]
    return (
        "You are screening candidates for the job below. Score EACH candidate 0-100 "
        "for fit (100 = ideal hire, 50 = plausible, 0 = irrelevant). Reward real, "
        "demonstrated, production-grade experience; discount buzzword-stuffing and "
        "courses-without-shipping. Return STRICT JSON mapping each candidate id to an "
        "integer score, e.g. {\"CAND_0000001\": 82}.\n\n"
        "JOB:\n" + jd_text[:3000] + "\n\nCANDIDATES:\n" + "\n".join(lines)
    )


def score_head(
    jd_text: str, items: list[tuple[str, str]], batch_size: int | None = None
) -> dict[str, float]:
    """Score a head set against the JD via the configured LLM.

    ``items`` is ``[(candidate_id, profile_text)]``; returns ``{candidate_id:
    score}`` in [0,1]. Stops early (returning what it has, possibly empty) the
    first time the LLM is unreachable, so a missing provider degrades to a no-op.
    """
    if not items:
        return {}
    s = get_settings()
    batch_size = batch_size or int(getattr(s, "llm_score_batch", 20))
    timeout = float(getattr(s, "llm_timeout", 60))
    out: dict[str, float] = {}

    with log_duration(log, "llm_scorer.run") as m:
        n_batches = (len(items) + batch_size - 1) // batch_size
        for bi, start in enumerate(range(0, len(items), batch_size), start=1):
            batch = items[start : start + batch_size]
            data = complete_json(_prompt(jd_text, batch), timeout=timeout)
            if not data:
                break  # no provider / failure — keep whatever we have
            for cid, sc in data.items():
                try:
                    out[cid] = max(0.0, min(float(sc) / 100.0, 1.0))
                except (ValueError, TypeError):
                    continue
            log.info("llm_scorer.progress", batch=bi, of=n_batches, scored=len(out))
        m["scored"] = len(out)
        m["items"] = len(items)
    return out
