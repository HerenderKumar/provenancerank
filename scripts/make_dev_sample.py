#!/usr/bin/env python3
"""Build a small, *stratified* development sample from the full dataset.

The full ``candidates.jsonl`` is 100K records; iterating it for every code
change is slow. This script streams it once and writes a representative subset
that deliberately over-samples the interesting strata (strong AI fits, keyword
stuffers, honeypot-like profiles, consulting-only careers, inactive accounts)
on top of a uniform random base, so local tests exercise the hard cases.

Usage:
    python scripts/make_dev_sample.py --candidates ./candidates.jsonl \
        --out ./data/dev_sample.jsonl --size 1500
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.constants import (  # noqa: E402
    CONSULTING_FIRMS,
    JD_CORE_SKILLS,
    NON_TECHNICAL_TITLE_TOKENS,
    RETRIEVAL_SKILLS,
)
from core.logging import get_logger  # noqa: E402
from pipeline.loader import iter_candidates  # noqa: E402

log = get_logger("scripts.make_dev_sample")
TODAY = dt.date(2026, 6, 19)


def _classify(c: dict) -> str:
    title = c.get("profile", {}).get("current_title", "").lower()
    skills = [s.get("name", "").lower() for s in c.get("skills", [])]
    ai_n = sum(1 for s in skills if s in JD_CORE_SKILLS)
    retr_n = sum(1 for s in skills if s in RETRIEVAL_SKILLS)
    nontech = any(t in title for t in NON_TECHNICAL_TITLE_TOKENS)
    companies = [r.get("company", "").lower() for r in c.get("career_history", [])]
    consulting_only = bool(companies) and all(
        any(f in comp for f in CONSULTING_FIRMS) for comp in companies
    )
    # cheap honeypot smell: expert skills with 0 duration, or long tenure at tiny co
    expert_zero = sum(
        1
        for s in c.get("skills", [])
        if s.get("proficiency") == "expert" and s.get("duration_months", 0) == 0
    )
    tiny_long = any(
        r.get("company_size") in ("1-10", "11-50") and r.get("duration_months", 0) > 96
        for r in c.get("career_history", [])
    )
    if expert_zero >= 3 or tiny_long:
        return "honeypot_like"
    if ai_n >= 4 and nontech:
        return "keyword_stuffer"
    if retr_n >= 2 and not nontech:
        return "strong_ai"
    if consulting_only:
        return "consulting_only"
    return "base"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="./candidates.jsonl")
    ap.add_argument("--out", default="./data/dev_sample.jsonl")
    ap.add_argument("--size", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    # Quotas per interesting stratum; remainder filled with a uniform sample.
    quotas = {
        "honeypot_like": max(40, args.size // 20),
        "keyword_stuffer": args.size // 6,
        "strong_ai": args.size // 4,
        "consulting_only": args.size // 10,
    }
    buckets: dict[str, list[dict]] = {k: [] for k in quotas}
    base: list[dict] = []
    base_target = args.size

    seen = 0
    for c in iter_candidates(args.candidates):
        seen += 1
        strat = _classify(c)
        if strat in buckets and len(buckets[strat]) < quotas[strat]:
            buckets[strat].append(c)
        # reservoir sample for the uniform base
        if len(base) < base_target:
            base.append(c)
        else:
            j = random.randint(0, seen - 1)
            if j < base_target:
                base[j] = c

    chosen: dict[str, dict] = {}
    for items in buckets.values():
        for c in items:
            chosen[c["candidate_id"]] = c
    for c in base:
        if len(chosen) >= args.size:
            break
        chosen[c["candidate_id"]] = c

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for c in chosen.values():
            f.write(json.dumps(c) + "\n")

    log.info(
        "dev_sample.written",
        path=str(out),
        scanned=seen,
        written=len(chosen),
        strata={k: len(v) for k, v in buckets.items()},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
