# Eval Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the retrieval-metrics scoring pipeline (Hit Rate@5, MRR, dense/BM25/
hybrid ablation, stratified reporting) that scores the frozen 50-question benchmark
against the live retrieval stack.

**Architecture:** A new, independently testable pure-function module
(`src/fda_device_rag/eval/metrics.py`) holds all metric math with no retrieval
dependency. A thin CLI script (`scripts/run_eval.py`) wires that module to the
frozen questions and the existing, unmodified retrieval stack (`BM25Index`,
`ChromaStore`, `Embedder`, `HybridRetriever`), writing `data/eval/results.json` plus
a human-readable stdout summary.

**Tech Stack:** Python 3.11, pytest, existing project dependencies only. No new
third-party packages — the Wilson score interval is a closed-form formula
implemented directly (the project has no scipy/statsmodels and this design doesn't
introduce one for a single formula).

## Global Constraints

- Wilson 95% confidence interval computed **per stratum, not just pooled** — the
  architecture doc's fixed "~±14pp at n=50" figure is only valid for the pooled
  aggregate; stratified n=13/13/12/12 needs its own, wider interval per stratum.
- MRR gets **no confidence interval** — it's a mean of reciprocal ranks, not a
  binomial proportion. A one-line note in every run's output states this explicitly.
- MRR's reciprocal rank uses the **rank of the first (best-ranked) gold chunk**,
  computed over the **full ranking** (not capped at top-5) — standard MRR
  convention, consistent with Hit Rate@5's own "any match counts" semantics.
- Resolution-errored questions **never** appear in `per_question` (not even with
  null ranks) — they live only in a separate `resolution_errors` list, excluded from
  both that question's stratum's `n` and the pooled `n`.
- `HybridRetriever.retrieve(query, top_k=CORPUS_SIZE)` is safe to call at full
  corpus size: `DENSE_TOP_N`/`BM25_TOP_N` (both 20) are fixed module constants in
  `hybrid_retriever.py`, decoupled from the `top_k` argument, which only slices the
  already-fused (≤40-entry) result list. Verified empirically against the live
  index: `retrieve(top_k=1848)` returned exactly 37 results, matching precisely
  `len(dense_top_20 ∪ bm25_top_20)` for a real query.
- Document-count computation for `recall`/`maude` must use **chunk count directly**,
  never distinct `ChunkMetadata.document_title` — that field holds
  `product_description`/`device_name` text, not a per-record key. Verified on the
  live corpus: `document_title` has only 499 distinct values across 500 recall
  records, and only **86** distinct values across 500 MAUDE records (MAUDE's
  `document_title` is a device generic name, shared across many independent
  reports). Using distinct-`document_title` count for MAUDE would misreport its
  corpus size as 86 records — 17% of the true count.
- For `guidance`/`ifu`, document count = distinct `document_title` count among
  chunks of that source type (these values are genuinely per-document, unlike
  recall/MAUDE's).
- No changes to `HybridRetriever`, `BM25Index`, `ChromaStore`, or `Embedder` — this
  plan consumes them exactly as they exist today. `BM25Index` exposes only `.query()`
  publicly; accessing `._chunks` directly for corpus-size computation is necessary
  and matches the access pattern already used elsewhere in this project's own
  validation work.
- `data/eval/results.json` is gitignored and overwritten on routine runs — committing
  a specific run's results is a deliberate manual step, not automatic.
- `scripts/run_eval.py` gets no dedicated unit tests, matching the existing
  `scripts/sample_eval_candidates.py` convention — its only untested logic is file
  I/O and orchestration, verified by actually running it against the real corpus.

---

## File Structure

- `src/fda_device_rag/eval/metrics.py` — Wilson interval, rank-finding primitives,
  aggregate summary functions. Pure functions, no I/O, no retrieval dependency.
- `tests/eval/test_metrics.py` — tests for the above.
- `scripts/run_eval.py` — CLI orchestration: freeze check, load questions + retrieval
  stack, score all questions across three legs, aggregate, write
  `data/eval/results.json`, print stdout summary.
- `.gitignore` — add `data/eval/results.json`.

---

### Task 1: Wilson Score Interval

**Files:**
- Create: `src/fda_device_rag/eval/metrics.py`
- Test: `tests/eval/test_metrics.py`

**Interfaces:**
- Produces: `wilson_interval(hits: int, n: int, z: float = 1.96) -> tuple[float, float]`
  — returns `(lower_pct, upper_pct)` as percentages in `[0, 100]`. Consumed by
  `summarize_hit_rate` (Task 3).

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_metrics.py
import math

from fda_device_rag.eval.metrics import wilson_interval


def test_wilson_interval_matches_reference_values():
    cases = [
        ((38, 50), (62.5871, 85.7027)),
        ((0, 50), (0.0000, 7.1350)),
        ((50, 50), (92.8650, 100.0000)),
        ((25, 50), (36.6443, 63.3557)),
        ((1, 1), (20.6543, 100.0000)),
        ((0, 1), (0.0000, 79.3457)),
    ]
    for (hits, n), (expected_lo, expected_hi) in cases:
        lo, hi = wilson_interval(hits, n)
        assert lo == pytest.approx(expected_lo, abs=1e-3)
        assert hi == pytest.approx(expected_hi, abs=1e-3)


def test_wilson_interval_bounds_stay_within_0_and_100():
    lo, hi = wilson_interval(0, 50)
    assert lo >= 0.0
    lo, hi = wilson_interval(50, 50)
    assert hi <= 100.0


def test_wilson_interval_zero_n_returns_zero_zero():
    assert wilson_interval(0, 0) == (0.0, 0.0)
```

Add `import pytest` at the top of the file (needed for `pytest.approx`).

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/eval/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fda_device_rag.eval.metrics'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/fda_device_rag/eval/metrics.py
import math

WILSON_Z_95 = 1.96


def wilson_interval(hits: int, n: int, z: float = WILSON_Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion, returned as
    (lower_pct, upper_pct) percentages in [0, 100]. Used instead of a
    normal-approximation interval because it stays well-behaved at small n
    and at p near 0 or 1 -- both regimes this project's per-stratum
    (n=12-13) Hit Rate@5 figures actually fall into.
    """
    if n == 0:
        return (0.0, 0.0)
    p = hits / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    lo = max(0.0, center - margin) * 100
    hi = min(1.0, center + margin) * 100
    return (lo, hi)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/eval/test_metrics.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/fda_device_rag/eval/metrics.py tests/eval/test_metrics.py
git commit -m "feat: add Wilson score interval for eval-runner metrics"
```

---

### Task 2: Rank-Finding Primitives

**Files:**
- Modify: `src/fda_device_rag/eval/metrics.py` (add alongside Task 1's `wilson_interval`)
- Modify: `tests/eval/test_metrics.py`

**Interfaces:**
- Produces: `rank_of_first_gold_hit(ranked_ids: list[str], gold_ids: list[str]) -> int | None`
  (1-indexed rank of the first `ranked_ids` entry that appears in `gold_ids`, or
  `None` if none do), `hit_at_k(rank: int | None, k: int = 5) -> bool`,
  `reciprocal_rank(rank: int | None) -> float`. All three consumed by
  `summarize_hit_rate`/`summarize_mrr` (Task 3) and directly by `scripts/run_eval.py`
  (Task 4).

- [ ] **Step 1: Write the failing test**

Append to `tests/eval/test_metrics.py`:

```python
from fda_device_rag.eval.metrics import rank_of_first_gold_hit, hit_at_k, reciprocal_rank


def test_rank_of_first_gold_hit_returns_one_indexed_rank():
    ranked = ["a", "b", "c", "d"]
    assert rank_of_first_gold_hit(ranked, ["c"]) == 3


def test_rank_of_first_gold_hit_returns_earliest_match_with_multi_chunk_gold():
    ranked = ["a", "b", "c", "d"]
    # "d" is gold too, but "b" (rank 2) is the best-ranked gold hit
    assert rank_of_first_gold_hit(ranked, ["d", "b"]) == 2


def test_rank_of_first_gold_hit_returns_none_when_absent():
    ranked = ["a", "b", "c"]
    assert rank_of_first_gold_hit(ranked, ["z"]) is None


def test_hit_at_k_true_within_k():
    assert hit_at_k(5, k=5) is True
    assert hit_at_k(1, k=5) is True


def test_hit_at_k_false_beyond_k():
    assert hit_at_k(6, k=5) is False


def test_hit_at_k_false_when_rank_is_none():
    assert hit_at_k(None, k=5) is False


def test_reciprocal_rank_of_a_rank():
    assert reciprocal_rank(4) == pytest.approx(0.25)


def test_reciprocal_rank_of_none_is_zero():
    assert reciprocal_rank(None) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/eval/test_metrics.py -v`
Expected: FAIL with `ImportError: cannot import name 'rank_of_first_gold_hit'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/fda_device_rag/eval/metrics.py`:

```python
def rank_of_first_gold_hit(ranked_ids: list[str], gold_ids: list[str]) -> int | None:
    """1-indexed rank of the first entry in ranked_ids that appears in
    gold_ids, or None if no entry does. When gold_ids has multiple chunk
    ids (a multi-chunk gold set), this returns the best (lowest) rank among
    them -- standard MRR convention, matching Hit Rate@5's own "any match
    counts" semantics rather than averaging across all gold chunks found.
    """
    gold_set = set(gold_ids)
    for rank, chunk_id in enumerate(ranked_ids, start=1):
        if chunk_id in gold_set:
            return rank
    return None


def hit_at_k(rank: int | None, k: int = 5) -> bool:
    return rank is not None and rank <= k


def reciprocal_rank(rank: int | None) -> float:
    return 1.0 / rank if rank is not None else 0.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/eval/test_metrics.py -v`
Expected: PASS (11 tests total)

- [ ] **Step 5: Commit**

```bash
git add src/fda_device_rag/eval/metrics.py tests/eval/test_metrics.py
git commit -m "feat: add rank-finding primitives for eval-runner metrics"
```

---

### Task 3: Aggregate Summary Functions

**Files:**
- Modify: `src/fda_device_rag/eval/metrics.py` (add alongside Tasks 1-2)
- Modify: `tests/eval/test_metrics.py`

**Interfaces:**
- Consumes: `wilson_interval` (Task 1); `hit_at_k`, `reciprocal_rank` (Task 2).
- Produces: `summarize_hit_rate(ranks: list[int | None], k: int = 5) -> dict` (keys:
  `hits`, `n`, `pct`, `wilson_ci_95`), `summarize_mrr(ranks: list[int | None]) -> float`.
  Both consumed by `scripts/run_eval.py` (Task 4).

- [ ] **Step 1: Write the failing test**

Append to `tests/eval/test_metrics.py`:

```python
from fda_device_rag.eval.metrics import summarize_hit_rate, summarize_mrr


def test_summarize_hit_rate_counts_hits_within_k():
    # ranks: hit, hit, miss (beyond k), miss (absent)
    ranks = [1, 5, 6, None]
    summary = summarize_hit_rate(ranks, k=5)
    assert summary["hits"] == 2
    assert summary["n"] == 4
    assert summary["pct"] == pytest.approx(50.0)
    lo, hi = summary["wilson_ci_95"]
    assert 0.0 <= lo <= summary["pct"] <= hi <= 100.0


def test_summarize_hit_rate_empty_list_is_zero_over_zero():
    summary = summarize_hit_rate([], k=5)
    assert summary["hits"] == 0
    assert summary["n"] == 0
    assert summary["pct"] == 0.0
    assert summary["wilson_ci_95"] == (0.0, 0.0)


def test_summarize_mrr_averages_reciprocal_ranks():
    # 1/1, 1/4, 0 (absent) -> mean = (1 + 0.25 + 0) / 3
    ranks = [1, 4, None]
    assert summarize_mrr(ranks) == pytest.approx((1.0 + 0.25 + 0.0) / 3)


def test_summarize_mrr_empty_list_is_zero():
    assert summarize_mrr([]) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/eval/test_metrics.py -v`
Expected: FAIL with `ImportError: cannot import name 'summarize_hit_rate'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/fda_device_rag/eval/metrics.py`:

```python
def summarize_hit_rate(ranks: list[int | None], k: int = 5) -> dict:
    n = len(ranks)
    hits = sum(1 for r in ranks if hit_at_k(r, k))
    pct = 100.0 * hits / n if n else 0.0
    ci = wilson_interval(hits, n)
    return {"hits": hits, "n": n, "pct": pct, "wilson_ci_95": ci}


def summarize_mrr(ranks: list[int | None]) -> float:
    if not ranks:
        return 0.0
    return sum(reciprocal_rank(r) for r in ranks) / len(ranks)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/eval/test_metrics.py -v`
Expected: PASS (15 tests total)

- [ ] **Step 5: Commit**

```bash
git add src/fda_device_rag/eval/metrics.py tests/eval/test_metrics.py
git commit -m "feat: add aggregate hit-rate and MRR summary functions"
```

---

### Task 4: Eval-Runner CLI Script

**Files:**
- Create: `scripts/run_eval.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: `wilson_interval`, `rank_of_first_gold_hit`, `hit_at_k`, `reciprocal_rank`,
  `summarize_hit_rate`, `summarize_mrr` (Tasks 1-3, all in
  `fda_device_rag.eval.metrics`); `load_questions` (from
  `fda_device_rag.eval.questions`); `resolve_gold`, `GoldResolutionError` (from
  `fda_device_rag.eval.gold_resolution`); `assert_frozen_and_get_hash`,
  `FrozenFileError` (from `fda_device_rag.eval.freeze_check`); `Embedder` (from
  `fda_device_rag.embedding.embedder`); `ChromaStore` (from
  `fda_device_rag.store.chroma_store`); `BM25Index` (from
  `fda_device_rag.retrieval.bm25_index`, loaded via `pickle`);
  `HybridRetriever` (from `fda_device_rag.retrieval.hybrid_retriever`). All existing,
  unmodified.
- Produces: `data/eval/results.json` on disk — the script that ends the "in scope"
  list from the design doc's §11.

This is a thin CLI wrapper with all its real logic already tested in Tasks 1-3 —
matching the existing `scripts/pull_corpus.py`/`build_index.py`/
`sample_eval_candidates.py` convention where only extractable pure-logic helpers get
dedicated unit tests, not the orchestration script itself.

- [ ] **Step 1: Add the results file to `.gitignore`**

Read the current `.gitignore`, then add one line so run outputs never get committed
by accident:

```
.worktrees/
__pycache__/
*.pyc
.venv/
venv/
data/raw/
data/chroma/
data/manifest.csv
data/bm25_index.pkl
data/eval/candidates.json
data/eval/results.json
*.egg-info/
.pytest_cache/
```

- [ ] **Step 2: Write the script**

```python
# scripts/run_eval.py
"""Scores the frozen benchmark (data/eval/questions.json) against the live
retrieval stack -- Hit Rate@5, MRR, and a dense-only/BM25-only/hybrid
ablation, stratified by source type. Refuses to run if the frozen file has
uncommitted changes (per docs/superpowers/specs/2026-08-03-eval-harness-design.md
section 7's freeze-enforcement guarantee).

Writes data/eval/results.json (gitignored -- overwritten on routine runs;
commit a specific run's results only when a number is actually being cited)
and prints a human-readable summary to stdout.

Usage: python scripts/run_eval.py
"""
import datetime
import json
import pickle
import sys
from pathlib import Path

from fda_device_rag.embedding.embedder import Embedder
from fda_device_rag.eval.freeze_check import FrozenFileError, assert_frozen_and_get_hash
from fda_device_rag.eval.gold_resolution import GoldResolutionError, resolve_gold
from fda_device_rag.eval.metrics import (
    hit_at_k,
    rank_of_first_gold_hit,
    summarize_hit_rate,
    summarize_mrr,
)
from fda_device_rag.eval.questions import load_questions
from fda_device_rag.retrieval.bm25_index import BM25Index
from fda_device_rag.retrieval.hybrid_retriever import HybridRetriever
from fda_device_rag.store.chroma_store import ChromaStore

DATA_DIR = Path("data/raw")
CHROMA_DIR = Path("data/chroma")
BM25_INDEX_PATH = Path("data/bm25_index.pkl")
QUESTIONS_PATH = Path("data/eval/questions.json")
RESULTS_PATH = Path("data/eval/results.json")

# Frozen file's benchmark-level source_type -> corpus's ChunkMetadata.source_type.
# Two different vocabularies for two different purposes -- see
# src/fda_device_rag/eval/gold_resolution.py's resolve_gold docstring.
CORPUS_SOURCE_TYPES = {
    "recall": "recall",
    "maude": "maude_event",
    "guidance": "guidance_pdf",
    "ifu": "ifu_pdf",
}
SOURCE_TYPES = ("recall", "maude", "guidance", "ifu")
LEGS = ("dense", "bm25", "hybrid")
GOLD_SET_SIZE_OUTLIER_THRESHOLD = 2


def _corpus_size(bm25_index: BM25Index) -> dict:
    """Per source type: document count and chunk count, computed at run
    time from the loaded index. recall/maude document count = chunk count
    directly (each record produces exactly one chunk, and their
    document_title field holds shared descriptive text, not a per-record
    key -- see this plan's Global Constraints). guidance/ifu document count
    = distinct document_title count (genuinely per-document there).
    """
    chunks_by_corpus_type: dict[str, list] = {}
    for chunk in bm25_index._chunks:
        chunks_by_corpus_type.setdefault(chunk.metadata.source_type, []).append(chunk)

    sizes = {}
    for benchmark_type, corpus_type in CORPUS_SOURCE_TYPES.items():
        type_chunks = chunks_by_corpus_type.get(corpus_type, [])
        chunk_count = len(type_chunks)
        if benchmark_type in ("recall", "maude"):
            document_count = chunk_count
        else:
            document_count = len({c.metadata.document_title for c in type_chunks})
        sizes[benchmark_type] = {"documents": document_count, "chunks": chunk_count}
    return sizes


def _score_question(question, bm25_index, dense_store, hybrid_retriever, embedder, corpus_total: int) -> dict:
    """Raises GoldResolutionError if the question's gold locator can't be
    resolved -- the caller must catch this and record it as a resolution
    error, never as a per_question entry with null ranks."""
    gold_ids = resolve_gold(question)

    query_embedding = embedder.embed_query(question.question)
    dense_results = dense_store.query(query_embedding, top_k=corpus_total)
    bm25_results = bm25_index.query(question.question, top_k=corpus_total)
    hybrid_results = hybrid_retriever.retrieve(question.question, top_k=corpus_total)

    dense_rank = rank_of_first_gold_hit([r.id for r in dense_results], gold_ids)
    bm25_rank = rank_of_first_gold_hit([r.id for r in bm25_results], gold_ids)
    hybrid_rank = rank_of_first_gold_hit([r.id for r in hybrid_results], gold_ids)

    return {
        "question_id": question.question_id,
        "source_type": question.source_type,
        "gold_set_size": len(gold_ids),
        "dense": {"hit_at_5": hit_at_k(dense_rank), "rank": dense_rank},
        "bm25": {"hit_at_5": hit_at_k(bm25_rank), "rank": bm25_rank},
        "hybrid": {"hit_at_5": hit_at_k(hybrid_rank), "rank": hybrid_rank},
    }


def _aggregate(per_question_subset: list[dict]) -> dict:
    result = {}
    for leg in LEGS:
        ranks = [q[leg]["rank"] for q in per_question_subset]
        result[leg] = {
            "n": len(ranks),
            "hit_rate_5": summarize_hit_rate(ranks),
            "mrr": summarize_mrr(ranks),
        }
    return result


MRR_PRECISION_NOTE = (
    "MRR has no confidence interval computed -- it is a mean of reciprocal ranks, "
    "not a binomial proportion, so the Wilson interval used for Hit Rate@5 does not "
    "apply. Treat MRR differences with the same caution as an unquantified "
    "statistic, not as precise as the adjacent Hit Rate@5 figures."
)


def _render_summary(results: dict) -> str:
    lines = []
    lines.append("=== Corpus Size ===")
    for source_type in SOURCE_TYPES:
        sizes = results["corpus_size"][source_type]
        lines.append(f"  {source_type}: {sizes['documents']} documents, {sizes['chunks']} chunks")

    if results["resolution_errors"]:
        lines.append("")
        lines.append(f"=== Resolution Errors ({len(results['resolution_errors'])}) ===")
        for err in results["resolution_errors"]:
            lines.append(f"  {err['question_id']}: {err['error']}")

    if results["gold_set_size_outliers"]:
        lines.append("")
        lines.append("=== Gold-Set-Size Outliers ===")
        for outlier in results["gold_set_size_outliers"]:
            lines.append(f"  {outlier['question_id']}: {outlier['gold_set_size']} gold chunks")

    total_questions = len(results["per_question"]) + len(results["resolution_errors"])
    valid_count = len(results["per_question"])
    lines.append("")
    lines.append(
        f"=== Results ({valid_count}/{total_questions} valid questions, "
        f"{len(results['resolution_errors'])} resolution errors) ==="
    )

    scopes = [("Pooled", results["metrics"]["pooled"])] + [
        (source_type.capitalize(), results["metrics"]["by_source_type"][source_type])
        for source_type in SOURCE_TYPES
    ]
    for scope_name, scope_data in scopes:
        lines.append("")
        lines.append(f"--- {scope_name} ---")
        for leg in LEGS:
            leg_data = scope_data[leg]
            hr = leg_data["hit_rate_5"]
            ci_lo, ci_hi = hr["wilson_ci_95"]
            lines.append(
                f"  {leg:8s}  Hit Rate@5: {hr['hits']}/{hr['n']} ({hr['pct']:.1f}%, "
                f"95% CI [{ci_lo:.1f}, {ci_hi:.1f}])  MRR: {leg_data['mrr']:.3f}"
            )

    lines.append("")
    lines.append(MRR_PRECISION_NOTE)
    lines.append("")
    lines.append(f"Frozen file hash: {results['frozen_file_hash']}")

    return "\n".join(lines)


def main() -> None:
    try:
        frozen_hash = assert_frozen_and_get_hash(QUESTIONS_PATH)
    except FrozenFileError as e:
        print(f"Cannot run eval: {e}")
        sys.exit(1)

    if not BM25_INDEX_PATH.exists() or not CHROMA_DIR.exists():
        print(f"Index not found ({CHROMA_DIR}, {BM25_INDEX_PATH}) -- run "
              f"scripts/pull_corpus.py then scripts/build_index.py first.")
        sys.exit(1)

    questions = load_questions(QUESTIONS_PATH)

    bm25_index = pickle.loads(BM25_INDEX_PATH.read_bytes())
    dense_store = ChromaStore(persist_dir=str(CHROMA_DIR))
    embedder = Embedder()
    hybrid_retriever = HybridRetriever(dense_store=dense_store, bm25_index=bm25_index, embedder=embedder)

    corpus_size = _corpus_size(bm25_index)
    corpus_total = len(bm25_index._chunks)

    per_question = []
    resolution_errors = []
    for question in questions:
        try:
            per_question.append(
                _score_question(question, bm25_index, dense_store, hybrid_retriever, embedder, corpus_total)
            )
        except GoldResolutionError as e:
            resolution_errors.append({"question_id": question.question_id, "error": str(e)})

    gold_set_size_outliers = [
        {"question_id": pq["question_id"], "gold_set_size": pq["gold_set_size"]}
        for pq in per_question
        if pq["gold_set_size"] > GOLD_SET_SIZE_OUTLIER_THRESHOLD
    ]

    by_source_type = {
        source_type: _aggregate([pq for pq in per_question if pq["source_type"] == source_type])
        for source_type in SOURCE_TYPES
    }
    pooled = _aggregate(per_question)

    results = {
        "frozen_file_hash": frozen_hash,
        "run_date": datetime.date.today().isoformat(),
        "corpus_size": corpus_size,
        "gold_set_size_outliers": gold_set_size_outliers,
        "resolution_errors": resolution_errors,
        "per_question": per_question,
        "metrics": {"pooled": pooled, "by_source_type": by_source_type},
        "mrr_precision_note": MRR_PRECISION_NOTE,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))

    print(_render_summary(results))
    print(f"\nFull results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the script against the real corpus to verify it works**

Run: `python scripts/run_eval.py`

Expected: prints the corpus-size block, a resolution-errors block only if any exist
(should be none, since all 50 frozen questions were already verified to resolve
cleanly before freezing), a results header showing `50/50 valid questions, 0
resolution errors`, per-stratum and pooled tables of Hit Rate@5 (with Wilson CI) and
MRR for all three legs, the MRR precision note, and the frozen file hash. Confirm
`data/eval/results.json` was written and is valid JSON (`python -c "import json;
json.load(open('data/eval/results.json'))"` should succeed without error).

- [ ] **Step 4: Verify the results file is gitignored**

Run: `git status --porcelain data/eval/`
Expected: no output for `results.json` (an ignored file produces no status line; if
it appears as `??`, the `.gitignore` edit in Step 1 didn't take effect).

- [ ] **Step 5: Commit**

```bash
git add scripts/run_eval.py .gitignore
git commit -m "feat: add eval-runner CLI script (Hit Rate@5, MRR, ablation)"
```

---

## Self-Review

**Spec coverage:** design doc §3 (freeze enforcement first) → Task 4 `main()`'s first
action. §4 (three legs, verified-safe hybrid `top_k`) → Task 4 `_score_question`.
§5 (Hit Rate@5 with per-stratum Wilson CI; MRR over full ranking, no CI) → Tasks 1-3
(metrics module) consumed by Task 4's `_aggregate`. §6 (resolution failures never in
`per_question`) → Task 4's try/except around `_score_question`, structurally separate
`resolution_errors` list. §7 (corpus size, correct per-type document-count method) →
Task 4's `_corpus_size`. §8 (gold-set-size outliers, computed dynamically) → Task 4's
`gold_set_size_outliers` list comprehension over whatever `per_question` actually
contains, no hardcoded question IDs. §9 (`results.json` schema) → Task 4's `results`
dict, matches the design doc's schema field-for-field. §10 (stdout summary) → Task
4's `_render_summary`. §11 scope → file list matches exactly; no changes to
`HybridRetriever`/`BM25Index`/`ChromaStore`/`Embedder` anywhere in this plan. §12
testing → every required test case (Wilson reference values incl. edges, Hit Rate@5
hit/miss/multi-chunk-gold, MRR first-best-ranked-hit and zero-when-absent,
resolution-error exclusion) is covered by Tasks 1-3's tests; Task 4 correctly has no
dedicated unit tests per the design's explicit convention.

**Placeholder scan:** no TBD/TODO/"add error handling"-style steps — every step has
complete, runnable code.

**Type consistency:** `Question` (from `fda_device_rag.eval.questions`, already
built) is used identically across Task 4's functions (`.question_id`, `.source_type`,
`.question`). `rank_of_first_gold_hit`/`hit_at_k`/`reciprocal_rank`'s signatures
(Task 2) are used identically in `summarize_hit_rate`/`summarize_mrr` (Task 3) and
directly in `_score_question` (Task 4). The `per_question` dict shape produced by
`_score_question` (Task 4) matches exactly what `_aggregate` and `_render_summary`
(also Task 4) expect to read from it — same keys, same nesting, checked line-by-line
against the design doc's §9 schema.

---

Plan complete and saved to `docs/superpowers/plans/2026-08-04-eval-runner.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
