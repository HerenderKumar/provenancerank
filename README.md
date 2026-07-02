# ProvenanceRank

A candidate ranking engine for the Redrob "Intelligent Candidate Discovery &
Ranking" challenge (Track 1). It reads the 100K-profile `candidates.jsonl`,
scores every candidate against the Senior AI Engineer JD, and writes a top-100
`submission.csv` with a short, honest reason per pick.

The whole thing is built around one hard constraint from the spec: **the step
that produces the CSV gets no network, CPU only, ≤16 GB, ≤5 minutes.** So the
work is split in two — a slow offline precompute that builds artifacts once, and
a fast online ranker that only reads them.

```
precompute.py   (offline, can be slow)   ->  artifacts/
rank.py         (online, < 5 min, no net) ->  submission.csv
```

## Reproduce the submission

```bash
pip install -r requirements.txt

# 1. one-time, offline: features + embeddings + BM25 index + fit model
python precompute.py --candidates ./candidates.jsonl --jd ./data/job_description.txt

# 2. the graded step — reads artifacts only, no network
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

`rank.py` runs the official `validate_submission.py` on its own output and exits
non-zero if anything is off, so a green run is already a valid submission.

> Put the released `candidates.jsonl` at the repo root (or pass `--candidates`).
> It's gitignored because it's ~480 MB.

## What actually decides the ranking

The JD is the kind that warns you about itself: *"the right answer involves the
gap between what the JD says and what it means."* The dataset backs that up with
traps. So the scoring leans on a few opinions:

- **Retrieval / IR / ranking experience is the signal.** Vector search, hybrid
  search, embeddings, learning-to-rank, recommendation systems — named in the
  profile and, better, verified by on-platform assessment scores.
- **A non-technical title with AI keywords is a trap, not a candidate.** An "HR
  Manager" with nine AI skills and no technical role ever is flagged as a
  keyword stuffer and gated out. The sample submission ranks exactly this person
  #1 to show you the trap.
- **CV / speech / robotics without any NLP/IR is the wrong specialisation** the
  JD calls out, and gets penalised.
- **Behavioural signals decide reachability.** Someone perfect on paper who
  hasn't logged in for six months with a 5% recruiter response rate isn't
  actually available — they get down-weighted, hard.
- **Honeypots never make the top 100.** ~80 profiles have impossible career
  maths (8 years at a 3-year-old company, "expert" in 10 skills used 0 months).
  They're detected up front and pinned to score 0. Honeypot rate in the top 100
  is **0**.

Final score per candidate:

```
final = (0.40·ml_fit + 0.35·retrieval + 0.25·behavioural)
        · location_factor · availability_modifier
```

where `ml_fit` is a gradient-boosted model trained on a proxy relevance label
(there are no hire/reject labels in the data), `retrieval` is RRF-fused BM25 +
dense cosine, and `behavioural` blends availability, engagement and trust.

## Layout

```
core/         config (pydantic), JSON logging (structlog), constants/taxonomies
pipeline/     loader, feature_engineering, honeypot_detector, jd_deconstructor,
              gate_filter, embedder, bm25_indexer, retrieval, signal_scorer,
              loop_agent, scorer
ml/           trainer (proxy labels + XGBoost/LightGBM/sklearn), predictor
output/       reasoning_generator, submission_writer (+ official validator hook)
rank.py       Phase 2 entry point  (fast path + live fallback for the sandbox)
precompute.py Phase 1 entry point  (idempotent, hashed, --resume)
tests/        pytest suite (run: python -m pytest)
```

## Notes on the moving parts

**Embeddings.** Default is `sentence-transformers` (all-MiniLM-L6-v2, 384-dim,
CPU). If it isn't installed or can't be downloaded, the embedder falls back to a
deterministic hashing embedder (numpy only) so the pipeline still runs fully
offline. The backend used is recorded in `artifacts/embedding_meta.json` and
`rank.py` rebuilds the same one to embed the JD query — no network either way.

**ML model.** `ml/trainer.py` uses whichever of XGBoost / LightGBM / sklearn is
installed; if none are, `predict` falls back to the proxy formula directly.
Nothing downstream changes.

**Graceful degradation.** Missing model → formula. Missing embeddings → BM25
only. Missing index → cosine only. The ranker always produces output.

**Two ways to run `rank.py`:** with precomputed artifacts it takes the fast path
(read the feature matrix, re-run retrieval over the cached index). With no
artifacts — e.g. a small sample uploaded to the sandbox — it computes everything
live for just those candidates. Same code produces the CSV both ways.

## Live signal ingestion layer (the differentiator)

The static ranker scores whoever is in `candidates.jsonl` using self-reported
skills. The live layer builds *verifiable* profiles for developers who connect
their GitHub, and those richer profiles feed the **same** ranking pipeline with
higher-confidence, evidence-weighted skills.

The idea in one line: a skill proven by 20 production commit diffs scores ~1.0;
the same skill claimed on a resume scores 0.2. Evidence beats assertions.

```
ingestion/    GitHub GraphQL client, resume parser, Celery tasks (fan-out)
intelligence/ confidence scorer, skill extractor, artifact summariser (Gemini→heuristic)
graph/        Neo4j knowledge graph (Developer-[:HAS_SKILL]->Skill, [:PRODUCED]->Artifact)
              + an in-memory backend + natural-language → Cypher query
```

How it works:
1. `POST /developers/connect-github {github_username}` kicks off a background
   sync. A worker fetches commits/PRs/issues, summarises each into structured
   evidence (what was done, skills, complexity 1–5, production signal), and
   writes it to Postgres + the graph. Every artifact carries a **SHA-256
   content_hash** (cryptographic anchor + idempotency).
2. Confidence accumulates per skill with recency decay (1-year half-life).
3. `POST /search/natural-language {query}` — ask *"who debugged a race condition
   in production"* and get developers back **with the exact commits/PRs that
   prove it**. Intent is parsed by Gemini (heuristic fallback) and routed to a
   graph traversal SQL can't express cleanly.
4. In ranking, a verified candidate gets a small additive `evidence_confidence_
   bonus` so they float above an identical static-only candidate.

**This does not change the graded submission.** The bonus is 0 for any candidate
without linked evidence, so `rank.py` produces exactly the same `submission.csv`.

Everything degrades: **Gemini → heuristic summariser**, **spaCy → taxonomy NER**,
**Neo4j → in-memory graph**, **Celery → eager inline tasks** — so it runs and
tests with no external services, and is production-real in Docker (the
`worker-ingestion`, `worker-intelligence`, `beat`, and `neo4j` services).

## Measured on the released 100K (this machine)

| step | time | peak RAM |
|---|---|---|
| precompute (hashing embedder, inverted BM25) | ~28 s | ~2.0 GB |
| `rank.py` (fast path, full 100K) | ~2.3 s | ~1.3 GB |

With `sentence-transformers` the embedding pass in precompute is the slow part
(tens of minutes on CPU) — but that's offline and outside the 5-minute budget.
The ranking step stays a few seconds either way.

## Tests

```bash
python -m pytest        # unit + format + end-to-end smoke
python rank.py --smoke  # CI gate: live-rank 150 candidates, assert valid + < 30s
```
