# Eval Runner: Retrieval-Metrics Scoring — Design

Status: Approved for planning
Date: 2026-08-04

## 1. Problem

The frozen 50-question benchmark (`data/eval/questions.json`, committed) and its
supporting machinery — `load_questions`, `resolve_gold`, `assert_frozen_and_get_hash`,
all in `src/fda_device_rag/eval/` — exist but nothing consumes them yet. Architecture
doc §6 specifies the metrics this benchmark exists to produce (Hit Rate@5 primary, MRR
secondary, dense/BM25/hybrid ablation, stratified by source type, corpus-size context)
but the scoring script itself was explicitly out of scope for the eval-harness
implementation plan (design doc `2026-08-03-eval-harness-design.md` §9: "the
eval-runner script's retrieval-scoring logic itself... is its own implementation
unit"). This document specifies that unit.

**Explicit scope decision:** this design covers retrieval metrics only. Architecture
doc §6 also specifies a generation-faithfulness spot-check (15-20 questions, manually
graded on whether a generated answer stayed faithful to retrieved content) — that
requires an actual generated answer to grade, and no generation/answering component
exists in the codebase yet (the project's task list stops at the retrieval pipeline;
citation grounding, which presumably includes answer generation, is a later,
unstarted phase per the architecture doc's roadmap). Deferred to that phase, not
scaffolded here — this project has repeatedly found that mechanisms designed before
real evidence exists (the section-detection heuristic, firm-name retrievability, the
furniture rule) needed real implementation to find their actual failure modes; a
grading rubric designed before a single real generated answer exists would need
redoing once real LLM output is in hand.

## 2. Architecture

`scripts/run_eval.py` — a thin CLI script, matching the existing
`pull_corpus.py`/`build_index.py`/`query.py`/`sample_eval_candidates.py` convention —
loads the frozen questions and the live retrieval stack (the same `BM25Index`,
`ChromaStore`, `Embedder`, `HybridRetriever` the deployed system uses, not a parallel
copy), runs all 50 questions through three ablation legs (dense-only, BM25-only,
hybrid), and writes `data/eval/results.json` plus a human-readable stdout summary.

All metric math — Hit Rate@5, MRR, Wilson confidence intervals — lives in a new,
independently testable module, `src/fda_device_rag/eval/metrics.py`, as pure
functions with no retrieval dependency (they operate on ranks and hit/miss booleans,
not on live index queries), so they're unit-testable without loading the embedding
model or the index.

## 3. Freeze Enforcement (first step, not an afterthought)

Before loading anything else, the script calls
`assert_frozen_and_get_hash(Path("data/eval/questions.json"))`. If the file has
uncommitted changes or isn't tracked in `HEAD`, the script exits with a clear error —
no partial run, no silent scoring against an unfrozen file. The returned git blob
hash is stamped into both `results.json` and the stdout summary, giving every run a
permanent, verifiable link back to the exact frozen question set it scored against
(per eval-harness design doc §7: a clean working tree alone can't distinguish a
genuine one-time freeze from edit-rerun-recommit-rerun; the hash can).

## 4. Retrieval: Three Legs, One Real Algorithm Each

For each question, three rankings are computed:

- **Dense-only:** `ChromaStore.query(embedder.embed_query(q), top_k=CORPUS_SIZE)` —
  legitimate to query at full-corpus size since this is an ablation baseline with no
  deployed truncation to match.
- **BM25-only:** `BM25Index.query(q, top_k=CORPUS_SIZE)` — same reasoning.
- **Hybrid:** `HybridRetriever.retrieve(q, top_k=CORPUS_SIZE)`.

**Verified, not assumed, that the hybrid call is safe:** `HybridRetriever`'s
`DENSE_TOP_N`/`BM25_TOP_N` (both 20) are fixed module constants, decoupled from the
`top_k` argument passed to `.retrieve()` — that argument only slices the
already-fused result list (`fused[:top_k]`), which is naturally capped at ≤40 entries
(the deduplicated union of two real top-20 legs) regardless of what `top_k` is.
Empirically confirmed against the live index: `retrieve(top_k=1848)` returned exactly
37 results, matching precisely `len(dense_top_20 ∪ bm25_top_20)` for a real query.
Requesting a large `top_k` therefore does not expand either leg's real candidate
pool — it only stops truncating the return value to 5, letting MRR see the full
real-production-reachable ranking. A gold chunk outside both legs' true top-20 is
genuinely absent from `fused` and correctly scores as a miss, exactly matching what a
real user gets from production hybrid retrieval. `CORPUS_SIZE` is read from the
loaded `BM25Index` at run time (currently 1848), not hardcoded.

## 5. Metrics

**Hit Rate@5** (per leg × per stratum × pooled): `hits / n`, where a hit is any gold
chunk ID appearing in the leg's first 5 ranked results, and `n` excludes resolution
errors for that stratum (§6). Reported as `"X/N (P%)"` plus a Wilson 95% confidence
interval — computed **per stratum, not just pooled**. Architecture doc §6's fixed
"~±14pp margin at 95% CI" figure is only valid for the pooled n=50; stratified
reporting means n=13/13/12/12 per source type, each with a wider true margin than the
aggregate — a single hand-written prose caveat would be actively wrong the moment
someone reads a per-stratum breakdown. Computing the interval per-metric is the only
way every printed number is honest on its own.

**Wilson score interval**, hand-rolled closed-form (no new dependency — the project
has no scipy/statsmodels, and Wilson score is a simple formula not worth pulling one
in for):

```
center = (p + z²/(2n)) / (1 + z²/n)
margin = z * sqrt(p(1-p)/n + z²/(4n²)) / (1 + z²/n)
CI = [center - margin, center + margin]
```
with `z = 1.96` (95%), `p = hits/n`, `n` = stratum or pooled valid-question count.

**MRR** (per leg × per stratum × pooled): mean of `1/rank_of_first_gold_hit`,
computed over the **full ranking** (not capped at top-5) — using the standard MRR
convention (reciprocal rank of the *first*, best-ranked, relevant item; consistent
with Hit Rate@5's own "any match counts" semantics, not a different "average across
all gold chunks found" metric that would no longer actually be MRR). `0` if no gold
chunk appears anywhere in the leg's full ranking. No confidence interval — MRR is a
mean of reciprocal ranks, not a binomial proportion, so Wilson doesn't apply and no
separate uncertainty machinery is built for it. A one-line note in the output states
MRR's precision isn't separately quantified, so it doesn't silently read as equally
precise just because it's printed next to a Hit-Rate-with-CI figure.

## 6. Resolution Failures

Caught per-question via `GoldResolutionError` from `resolve_gold()`. A resolution
failure is **never** represented in `per_question` — not as an entry with null ranks,
which could later be misread as "retrieval found nothing" rather than "this question
couldn't be tested." Resolution failures live exclusively in a separate
`resolution_errors` list (question ID + error message), and are excluded from both
that question's stratum's `n` and the pooled `n`. Denominators are reported
explicitly wherever this matters — e.g. `"47/48 valid questions (2 resolution
errors)"` — never a percentage silently computed over a silently shrunk n, per
eval-harness design doc §5.

## 7. Corpus Size Context

Per source type, both document count and chunk count — architecture doc §6 requires
both, not chunk count alone, since "12 chunks from 4 documents" and "12 chunks from 1
document" are different retrieval-difficulty signals chunk count alone can't
distinguish. Computed at run time from the loaded `BM25Index` (never hardcoded):

- `guidance`/`ifu`: document count = number of distinct `document_title` values
  among chunks with that `source_type`; chunk count = total such chunks.
- `recall`/`maude`: document count = chunk count, directly (the count of chunks with
  that `source_type`) — **not** computed via distinct `document_title`, and this
  distinction is load-bearing, not stylistic. `ChunkMetadata.document_title` for
  these two source types holds `product_description`/`device_name` text (see
  `recall_to_chunk`/`event_to_chunk`), not a unique per-record key, and different
  real records can share it. Verified against the live corpus at two severities:
  recall records `Z-0006-2022` and `Z-0007-2022` are the same product recalled in
  two separate actions and share one `document_title` string exactly (a narrow,
  one-pair collision — 499 distinct titles across 500 recall records); MAUDE is far
  more collision-prone by construction, since its `document_title` is a device's
  *generic name*, which many independent adverse-event reports about the same
  device model legitimately share — only 86 distinct `document_title` values across
  500 MAUDE records on the current pull. Computing "distinct `document_title` count"
  uniformly across all four source types would silently misreport recall as 499
  (instead of the true 500) and MAUDE as 86 (instead of the true 500) — the MAUDE
  case isn't a rounding error, it would report the corpus as 17% of its real record
  count. `recall_to_chunk`/`event_to_chunk` each produce exactly one chunk per
  record with no splitting, so "chunk count" is already the correct, exact document
  count for these two types — no distinct-value computation needed or wanted. This
  equality is structural, not missing data, and is stated as such in the output so a
  reader doesn't mistake identical numbers for a bug.

## 8. Gold-Set-Size Outliers

`gold_set_size = len(gold_chunk_ids)` logged per question (already required by
eval-harness design doc §5). Outliers — computed dynamically from the loaded
questions file each run, not hardcoded to today's known values — are flagged
explicitly in the output. For the currently-frozen set this surfaces `guidance-12`
(4 chunks) and `ifu-01` (5 chunks), but the script doesn't assume these specific IDs;
it computes the distribution and flags what's actually unusual each run.

## 9. `results.json` Schema

Gitignored and overwritten on routine runs, consistent with other regenerable
artifacts (`data/chroma/`, `data/bm25_index.pkl`). Committing a specific run's
`results.json` — when a number is going into the README or reported externally — is
a deliberate manual step, the same way freezing `questions.json` was a deliberate act,
not automatic. No automatic run-history versioning is built for a need that hasn't
come up yet.

```json
{
  "frozen_file_hash": "881b2cb3e53c6ec04e85e5da7f128670bb42d63b",
  "run_date": "2026-08-04",
  "corpus_size": {
    "recall": {"documents": 500, "chunks": 500},
    "maude": {"documents": 500, "chunks": 500},
    "guidance": {"documents": 4, "chunks": 456},
    "ifu": {"documents": 3, "chunks": 392}
  },
  "gold_set_size_outliers": [
    {"question_id": "guidance-12", "gold_set_size": 4},
    {"question_id": "ifu-01", "gold_set_size": 5}
  ],
  "resolution_errors": [
    {"question_id": "...", "error": "..."}
  ],
  "per_question": [
    {
      "question_id": "recall-01",
      "source_type": "recall",
      "gold_set_size": 1,
      "dense": {"hit_at_5": true, "rank": 2},
      "bm25": {"hit_at_5": true, "rank": 1},
      "hybrid": {"hit_at_5": true, "rank": 1}
    }
  ],
  "metrics": {
    "pooled": {
      "dense": {"n": 50, "hit_rate_5": {"hits": 38, "pct": 76.0, "wilson_ci_95": [62.6, 85.7]}, "mrr": 0.71},
      "bm25": {"...": "..."},
      "hybrid": {"...": "..."}
    },
    "by_source_type": {
      "recall": {"dense": {"...": "..."}, "bm25": {"...": "..."}, "hybrid": {"...": "..."}},
      "maude": {"...": "..."},
      "guidance": {"...": "..."},
      "ifu": {"...": "..."}
    }
  },
  "mrr_precision_note": "MRR has no confidence interval computed -- it is a mean of reciprocal ranks, not a binomial proportion, so the Wilson interval used for Hit Rate@5 does not apply. Treat MRR differences with the same caution as an unquantified statistic, not as precise as the adjacent Hit Rate@5 figures."
}
```

Each `{...}` metrics block: `{"n": N, "hit_rate_5": {"hits": N, "pct": F,
"wilson_ci_95": [lo, hi]}, "mrr": F}`.

## 10. Stdout Summary

Human-readable rendering of the same data: corpus size context first (so a reader has
scale before seeing numbers), resolution errors (if any) called out prominently,
gold-set-size outliers noted, then a per-source-type table of Hit Rate@5 (with CI)
and MRR for all three legs, then the pooled table, then the MRR precision note and
the frozen-file hash.

## 11. Scope

**In scope:** `src/fda_device_rag/eval/metrics.py` (Wilson interval, Hit Rate@5, MRR
— pure functions), `scripts/run_eval.py` (CLI orchestration: freeze check, retrieval
calls across three legs, corpus-size computation, results assembly, JSON + stdout
output), `data/eval/results.json` (gitignored).

**Out of scope, explicitly:** the generation-faithfulness spot-check (§1 — deferred
to the citation-grounding phase). Any change to `HybridRetriever`, `BM25Index`,
`ChromaStore`, or `Embedder` — this design consumes them exactly as they exist today,
verified rather than assumed (§4). README authoring — the script produces numbers and
a committed `results.json` when a specific run is worth citing; writing prose around
those numbers is a separate, manual editorial step.

## 12. Testing

New tests, in `tests/eval/test_metrics.py`:

- Wilson interval: known reference values (e.g. hand-computed or cross-checked
  against a published Wilson-interval example) at a few `(hits, n)` pairs, including
  edge cases `hits=0` and `hits=n`.
- Hit Rate@5: hit when a gold ID is within the first 5 ranked results; miss when
  present only beyond rank 5; miss when absent entirely; correct behavior with a
  multi-chunk gold set (hit if *any* gold ID is in the top 5).
- MRR: reciprocal rank of the first gold hit in a full ranking, using the
  best-ranked gold ID when a multi-chunk gold set has multiple matches at different
  ranks; `0` when no gold ID appears anywhere in the ranking.
- Resolution-error exclusion: a stratum's `n` and the pooled `n` correctly exclude
  resolution-errored questions, and no null-rank entry for them appears in
  `per_question`.

`scripts/run_eval.py` itself gets no dedicated unit tests beyond what `metrics.py`
already covers, matching the existing convention (`sample_eval_candidates.py` has
none either) — its only untested logic is file I/O and orchestration, verified by
actually running it against the real corpus during implementation, not by a mocked
unit test.
