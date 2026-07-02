# Architecture

This document is honest about what's built and what's planned. Everything in
"The engine" below is implemented and tested in this repo. The serving layer
(API, cache, tracing, Docker) is designed but not in this pass - it's listed at
the end so the boundary is clear.

## Why two phases

The submission rule is the architecture: the ranking step gets **no network,
CPU only, ≤16 GB, ≤5 minutes**. Anything expensive - LLM calls, embedding 100K
profiles, building a BM25 index - has to happen *before* that clock starts. So:

```
                         PHASE 1  (offline, slow, runs once)
 candidates.jsonl ─┬─► feature engineering ─► feature_matrix.pkl
                   ├─► JD deconstruction (LLM optional) ─► jd_spec.json
                   ├─► sentence-transformers / hashing ─► embeddings.npy
                   ├─► BM25 index ─► bm25_index.pkl
                   └─► train fit scorer ─► ml/models/fit_scorer_v1.pkl
                                                     │
                         PHASE 2  (online, < 5 min, no network)
 candidates.jsonl ─► read artifacts ─► gate filter ─► hybrid retrieval ─►
                     ML fit predict ─► merge ─► loop coverage check ─►
                     top 100 ─► reasoning ─► submission.csv ─► validate
```

This is candidate-generation-then-ranking, the same shape most large retrieval
systems use, for the same reason: precompute the heavy stuff, keep the request
path cheap.

## The engine - patterns in use

**Two-phase pipeline.** `precompute.py` vs `rank.py`. Hard boundary: `rank.py`'s
import graph never touches `google-generativeai`, `sentence-transformers` is only
loaded to embed one short JD string (from local cache), and there are no HTTP
clients anywhere on the ranking path.

**Materialised view.** `artifacts/feature_matrix.pkl` is a precomputed DataFrame,
one row per candidate, 40 numeric features + a few display columns. At ranking
time features are read, never recomputed. `core/constants.NUMERIC_FEATURE_COLUMNS`
is the contract between feature engineering and the model.

**Hybrid search + scatter-gather + RRF.** `pipeline/retrieval.py` runs BM25
(exact jargon: "Qdrant", "NDCG", "LoRA") and dense cosine (paraphrase: "recsys"
≈ "recommendation system") independently, then fuses by Reciprocal Rank Fusion
so neither score scale dominates. A candidate only earns RRF credit from a
modality where its raw score is > 0, which keeps the zero-match tail from
picking up phantom rank credit.

**Strategy pattern (model selection).** `ml/trainer.py` picks the first of
XGBoost -> LightGBM -> sklearn HistGBR that imports, and falls back to the proxy
formula if none do. The feature contract and the predictor are identical
regardless of which one ran.

**Proxy labelling.** There are no hire/reject labels, so we build a proxy
relevance target from the features we trust and fit a regressor to it. The tree
model exists to capture interactions the linear proxy misses (e.g. "retrieval
skills only count when the title is actually technical"), not to invent signal.

**Guard-clause honeypot filter.** `pipeline/honeypot_detector.py` runs during
feature engineering and pins impossible profiles to `is_honeypot`. The gate
filter drops them before any scoring, and the final selection excludes them
outright - honeypot rate in the top 100 is 0 by construction.

**Idempotent precompute.** `precompute.py` writes the SHA-256 of
`candidates.jsonl` into `artifacts/manifest.json`; with `--resume` it skips work
when the inputs haven't changed. Re-running produces identical artifacts.

**Circuit breaker.** The one optional LLM call (JD deconstruction) is wrapped in
bounded retries; on exhaustion - or with no API key - it falls back to a
hand-derived `JobSpec`. The system is fully functional with zero credentials.

**Graceful degradation.** Missing model -> proxy formula. Missing embeddings ->
BM25 only. Missing index -> cosine only. Missing both -> fit + behavioural still
rank. The ranker never crashes on a missing artifact.

**Dead-letter / rejection registry.** Everyone the gate filter drops is written
to `artifacts/excluded_candidates.csv` with reasons, so a surprising top 100 can
always be traced back to who was excluded and why.

**Structured logging.** `core/logging.py` emits one JSON object per event
(structlog, or a stdlib fallback with the same shape). `log_duration` times each
stage. No `print()` in library code; the two entry points print a single
human-readable summary.

**Reproducibility.** Seeded RNG, deterministic final sort (score desc, then
candidate_id asc - which also satisfies the validator's tie-break rule), and
hash-pinned artifacts. Same inputs -> same `submission.csv`.

## The agentic loop

`pipeline/loop_agent.py` is a plan->retrieve->reflect loop with **no LLM**. It
relaxes a fit threshold across ≤4 iterations until the candidate pool has enough
people covering each JD dimension (retrieval, LLM, production evidence,
product-company, availability), and reports the coverage. It doesn't change the
final ordering - scoring does that - it's a guardrail against handing scoring a
pool that's accidentally missing a whole dimension, plus the coverage numbers
the UI shows.

## CAP trade-off

Two pieces of state, two different choices:

- **Feature matrix on disk - consistency over availability.** If an artifact is
  missing or unreadable, the relevant stage degrades or the run refuses rather
  than ranking on half-written features. A correct answer late beats a wrong
  answer now.
- **Ranking cache (serving layer) - availability over consistency.** When the
  API cache is added, a cache partition should fall back to recomputing rather
  than serving stale results; a 30-second recompute beats a wrong cached list.

## Planned - serving layer (not in this pass)

Designed but intentionally deferred to keep this PR to the engine: a FastAPI
sandbox (`POST /rank`, SSE progress, `/health`), Redis read-through cache keyed
by JD hash (CQRS read/write split), Prometheus metrics + OpenTelemetry traces,
an Nginx-fronted two-replica deployment with the precompute container isolated
from the API (bulkhead), and a React recruiter UI. These don't affect how
`submission.csv` is produced; they're how you'd operate it as a service.
