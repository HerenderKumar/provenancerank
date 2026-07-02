# Accuracy layer

Three stages sit on top of the base retrieval + GBDT ranker. All of them run
**offline in `precompute.py`** and cache their output into `feature_matrix.pkl`,
so `rank.py` stays exactly what it was: no network, CPU-only, well under five
minutes. Every stage degrades to a pure-numpy floor when the heavy libraries
aren't installed, so the pipeline runs anywhere; the strong models switch on the
moment `pip install -r requirements.txt` brings in xgboost and
sentence-transformers.

If you'd rather ship the original behaviour, `precompute.py --no-accuracy` skips
the whole thing and trains the old proxy regressor instead. The released
`submission.csv` is byte-for-byte unchanged when the layer is off - the scorer
treats the new columns as a no-op when they're absent.

## 1. Graded pseudo-labels -> LambdaMART  (`ml/pseudo_labels.py`, `ml/trainer.py`)

The dataset has no hire/reject labels, so the original model regressed a
hand-written proxy - which caps how good it can get. Instead we now build
*graded* relevance labels (tiers 0-5: a few ideal hires, more "relevant", lots of
noise) and train an **NDCG-objective ranker** (`XGBRanker(objective="rank:ndcg")`,
i.e. LambdaMART) on them. That optimises the exact metric the contest grades
instead of a proxy correlated with it.

Labels come from a richer composite than the GBDT proxy, bucketed into tiers by
percentile. If a `GOOGLE_API_KEY` is set, Gemini grades the most promising
~1,500 candidates 0-5 against the JD and those grades override the heuristic for
that slice - the model then learns the real definition of relevance, not ours.
No key, no problem: the heuristic tiers stand in.

Falls back to the proxy regressor, then to the formula, if no ranker library is
present. `predict_fit` min-maxes whatever it loads, so nothing downstream changes.

## 2. Cross-encoder rerank of the head  (`ml/reranker.py`)

First-stage retrieval scores the JD and each profile *independently*, so it
misses the interactions that decide the very top of the list ("shipped a
production LLM service" vs "took an LLM course"). A cross-encoder reads the JD
and a profile **together** in one transformer pass and scores the pair - far more
precise, far too slow for 100K. So we rerank only the top ~300 from stage one,
which is exactly where NDCG@10/@50 is won, and cache the result as `rerank_score`.

Real model: `cross-encoder/ms-marco-MiniLM-L-6-v2`. Floor (no
sentence-transformers): a BM25-flavoured lexical overlap of JD terms against each
profile - still a real, useful signal, just blunter.

## 3. Bradley-Terry tournament  (`pipeline/tournament.py`) - our differentiator

Pointwise scoring has a blind spot: the numbers come from features on different
scales, and one noisy column can sink a great candidate. Recruiters don't think
pointwise - they think "between these two, who's better?". So for the top ~60 we
run an actual **round-robin tournament**:

- every candidate plays every other once;
- a match is judged on several JD-relevant criteria (fit, proven skills,
  assessments, production signal, behaviour, live evidence, the rerank score). A
  beats B on a criterion by being *ordinally* better; the match result is the
  weighted share of criteria A won - a soft 0-1 outcome, not just win/lose;
- a round-robin has cycles (A>B>C>A), so we resolve the whole thing with a
  **Bradley-Terry model fit by MM iteration** (Hunter, 2004): the maximum-
  likelihood global strength that best explains every pairwise result.

Because only ordinal comparisons matter, it's immune to feature-scale quirks and
surfaces genuinely dominant candidates instead of whoever spiked one column. The
strength is cached as `tournament_score`.

## How they fold into the final score  (`pipeline/scorer.py`)

The rerank and tournament signals reorder the **head** as a convex blend - we
shave the same total weight off the pointwise core so head scores stay in [0,1]
and keep their ordering instead of saturating at 1:

```
core  = 0.40*ml_fit + 0.35*retrieval + 0.25*behavioural
base  = (1 - w_head)*core + w_rerank*rerank_score + w_tour*tournament_score
final = base * location_factor * signal_modifier + evidence_bonus
```

`w_rerank = 0.15`, `w_tour = 0.10` (config). Outside the head both columns are 0;
absent entirely, the core keeps its full weight and the formula is the original.
Honeypots and gated candidates are still pinned to 0 - none of this can float a
bad candidate into the top 100.

## Measuring it  (`ml/evaluate.py`)

The eval harness implements the contest metric exactly -
`0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10` - so every change is
measured, not guessed. We don't have the hidden ground truth, so we score against
the graded pseudo-labels: useful for A/B'ing a change and confirming the head
actually moves, but optimistic in absolute terms (the labels are partly our own).

On the full 100K, switching the layer on reorders 17 of the top 20 and swaps ~10
of the top 100 - real movement at the head while the submission stays valid. The
size of the win depends on the models: with only numpy/pandas (formula ranker +
lexical reranker) the gain is a floor; install xgboost + sentence-transformers
and the graded-label LambdaMART and the real cross-encoder do the heavy lifting.

```bash
# baseline vs full, end to end
python precompute.py --candidates ./candidates.jsonl --jd ./data/job_description.txt --no-accuracy
python rank.py --candidates ./candidates.jsonl --out ./submission_base.csv

python precompute.py --candidates ./candidates.jsonl --jd ./data/job_description.txt
python rank.py --candidates ./candidates.jsonl --out ./submission_full.csv
```

## Keeping it fast (all in `precompute`; `rank.py` is untouched at ~2.3s)

Every feature above runs offline, so the graded step never gets slower. These
levers keep the *offline* build fast without dropping or weakening any feature -
they make the same computation cheaper or run it on better hardware:

- **Device auto-detect** (`core/device.py`, `compute_device=auto`). The embedder
  and cross-encoder run on Apple MPS / CUDA when present, CPU otherwise -
  identical output, typically 3-8x faster. No flag needed.
- **ONNX + int8** (`use_onnx=true`, needs `optimum[onnxruntime]`). Loads the
  transformers through ONNX Runtime with quantised weights: ~2-4x on CPU, <1%
  quality change. Walks onnx-int8 -> onnx -> plain torch, so it never silently
  drops to the hashing/lexical floor. Off in `rank.py` (that path stays
  no-network).
- **Leaner LambdaMART** (`core/config.py`: `ranker_neg_ratio`,
  `ranker_min_train_rows`, `ranker_early_stop_rounds`). Trains on all positives +
  a bounded negative sample, `tree_method="hist"`, early-stops once NDCG
  plateaus. Same or better quality, fewer rows and trees. (Benefit scales with
  label sparsity - generous heuristic labels trim less; sparse Gemini labels trim
  a lot.)
- **Focused rerank text** (`pipeline.loader.rerank_text`). The cross-encoder sees
  a tight title + recent-role + top-skills projection (~100 tokens) instead of
  the full profile blob - fewer tokens per pair *and* a cleaner signal.
- **Per-stage content cache** (`core/artifact_cache.py`, `--resume`). Each stage
  records a hash of its inputs. Same candidates + same JD -> whole-run skip
  (~0.8s). **Change only the JD -> the 100K embeddings, the BM25 index and the
  trained model are reused** (`cache.hit`); just retrieval + rerank + tournament
  rerun. A stale or unreadable cache is always a clean miss, never a wrong result.

Measured on the 100K set (hashing embedder floor): cold build 27s -> unchanged
re-run 0.8s -> JD-swap 18s. With the real sentence-transformers embedder the
JD-swap saving is far larger, because the reused step is the multi-minute 100K
embedding pass.
