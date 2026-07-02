# Running ProvenanceRank on your laptop (macOS)

Three ways, smallest first. You only need **Path A** to produce the graded
`submission.csv`. Path B runs the full app (API + UI) with no Docker. Path C is
the whole production stack.

Prereqs: **Python 3.11+**, and for the UI **Node 18+**. `python3 --version`.

---

## 0. One-time setup

```bash
cd /Users/herender/Desktop/iCode/LinkedIn_finder/provenancerank

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Full deps (includes sentence-transformers/torch - large, best quality):
pip install -r requirements.txt

# ...OR lean + fast (skip the heavy optional libs; code falls back gracefully):
# grep -viE "sentence-transformers|google-generativeai" requirements.txt > req.lean.txt && pip install -r req.lean.txt
```

Put the dataset at the repo root (it's gitignored because it's ~480 MB). It's
inside the zip the organisers gave you:

```bash
# adjust the source path to wherever you unzipped the challenge bundle
cp "/Users/herender/Desktop/iCode/LinkedIn_finder/.../candidates.jsonl" ./candidates.jsonl
# or symlink it:
# ln -s /absolute/path/to/candidates.jsonl ./candidates.jsonl
```

---

## Path A - produce submission.csv (the graded output)

```bash
# 1. one-time, offline: features + embeddings + BM25 + the fit model + the
#    accuracy layer (graded-label LambdaMART, cross-encoder rerank, tournament)
python precompute.py --candidates ./candidates.jsonl --jd ./data/job_description.txt
#   (add --no-st --bm25-backend inverted to skip torch and run in ~30s)
#   (add --no-accuracy to ship the original proxy-regressor behaviour instead)

# 2. the graded step - no network, < 5 min
python rank.py --candidates ./candidates.jsonl --out ./submission.csv

# 3. verify with the official validator
python validate_submission.py submission.csv      # -> "Submission is valid."
```

The accuracy layer runs automatically in step 1 and is described in
[`ACCURACY.md`](ACCURACY.md). It degrades to a pure-numpy floor without the heavy
libraries; install `xgboost` + `sentence-transformers` (both in
`requirements.txt`) to switch on the real LambdaMART ranker and cross-encoder.

Run the tests too if you want proof it all works:

```bash
python -m pytest -q          # 42 tests
```

---

## Path B - run the full app locally (no Docker)

The API auto-falls back to **SQLite + in-memory cache/graph/queue**, so you need
**nothing else installed** - no Postgres, Redis, or Neo4j.

```bash
# Terminal 1 - API (sqlite file is created automatically)
source .venv/bin/activate
uvicorn api.main:app --reload --port 8000
#   open http://localhost:8000/docs   (Swagger UI for every endpoint)
```

```bash
# Terminal 2 - the React console
cd frontend
npm install
npm run dev
#   open http://localhost:5173
```

Log in with the bootstrap admin (created on first boot):

- email: `admin@provenancerank.local`
- password: `changeme-admin`

Then: **Dashboard -> Run ranking** (needs the artifacts from Path A step 1), and
**Evidence Search** to try the live layer. To try GitHub ingestion for real, set
`GITHUB_TOKEN` before starting uvicorn:

```bash
export GITHUB_TOKEN=ghp_xxx            # optional; without it, ingestion just no-ops
export GOOGLE_API_KEY=AIza_xxx         # optional; enables Gemini summaries, else heuristic
```

---

## Path C - the full production stack (Docker)

Needs **Docker Desktop**. Brings up Postgres + Redis + Neo4j + 2 API replicas +
nginx + Celery workers + Prometheus + Grafana + Jaeger + the UI (14 services).

```bash
# candidates.jsonl must be at the repo root (precompute container reads it)
docker compose up --build
```

Then open:

| URL | what |
|---|---|
| http://localhost:3001 | recruiter UI |
| http://localhost:8080/docs | API (via nginx load balancer) |
| http://localhost:7474 | Neo4j browser (neo4j / provenancerank) |
| http://localhost:3000 | Grafana (anonymous) |
| http://localhost:16686 | Jaeger traces |

Stop with `docker compose down` (add `-v` to wipe data volumes).

---

## Gotchas

- **`candidates.jsonl` missing** -> Path A errors with "feature matrix missing";
  make sure you copied it to the repo root.
- **torch install is slow/big** -> use the lean install and `precompute --no-st`.
- **Apple Silicon** -> everything works; torch has arm64 wheels.
- **Port already in use** -> change `--port` (API) or vite's port in
  `frontend/vite.config.ts`.
- **Reset local DB** -> `rm provenancerank.db*` (Path B re-creates it).
```
