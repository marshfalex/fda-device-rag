# FDA Device RAG

[![CI](https://github.com/marshfalex/fda-device-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/marshfalex/fda-device-rag/actions/workflows/ci.yml)

A Q&A system over public FDA medical-device documentation — device recalls,
adverse event reports (MAUDE), FDA guidance documents, and manufacturer
Instructions for Use (IFUs) — with citation grounding, so every answer can be
traced back to the source record or document section it came from.

This is a portfolio project. The design rationale for each decision is written
up in [`docs/superpowers/specs/2026-07-28-fda-device-rag-architecture-design.md`](docs/superpowers/specs/2026-07-28-fda-device-rag-architecture-design.md).

> **Status: Phases 1-3 (retrieval pipeline + accuracy benchmark + citation
> grounding).** What exists today is corpus ingestion, chunking, local
> embedding, hybrid retrieval, a leakage-safe retrieval-accuracy benchmark,
> and Ollama-backed grounded answer generation with citation extraction
> (`scripts/ask.py`). The deployed demo and CI are future phases and are
> **not** implemented yet.

## Corpus

| Source | What it provides | Chunking unit |
|---|---|---|
| openFDA `device/recall` | `reason_for_recall` + `action` + `product_description` narrative | one recall record = one document |
| openFDA `device/event` (MAUDE) | `mdr_text[].text` narrative complaint descriptions | one adverse event report = one document |
| FDA guidance documents (PDF) | long-form, section-structured regulatory text | structure-aware (section, then recursive split) |
| Manufacturer IFUs (PDF) | long-form, section-structured technical/procedural text | structure-aware |

PDF sources are a **hand-curated, fixed list** of URLs (see `GUIDANCE_PDF_URLS`
/ `IFU_PDF_URLS` in `scripts/pull_corpus.py`) — chosen and verified by hand, not
scraped, so every document in the corpus is traceable and defensible. These
lists ship empty; populate them before running the pull to include PDF sources.

openFDA `device/510k` was evaluated and **excluded**: verified live against the
API, it returns structured metadata only, with no narrative summary text to
retrieve. See §2 of the design doc.

**Reproducibility.** `pull_corpus.py` uses fixed, explicit date-range queries
(`event_date_posted:[2015-01-01 TO 2024-12-31]` for recalls,
`date_received:[20150101 TO 20241231]` for MAUDE events) and paginates to a hard
cap of 500 records per endpoint. The exact queries and record counts are printed
at the end of every run. The same query and date range will match the same
underlying records (subject to FDA updating historical data), and the record
count is capped consistently; exact record-level ordering across repeated pulls
isn't pinned without an explicit `sort` parameter, which is a known follow-up
(the pager dedups across pages by natural identifier so a mid-pull ordering
shift can't produce duplicate ids). The recorded pull date is what "as of"
refers to, since FDA can add records to a historical range after the fact.
Embeddings are computed locally (`BAAI/bge-small-en-v1.5` via
`sentence-transformers`), so re-running the pipeline requires no API key and
costs nothing.

**Corpus stats:** run the pipeline below to populate `data/manifest.csv`; counts
will be logged by `pull_corpus.py` on completion.

## Setup

Requires Python 3.11+.

```bash
pip install -e ".[dev]"
```

## Running the pipeline

```bash
python scripts/pull_corpus.py              # fetch corpus -> data/raw/ + data/manifest.csv
python scripts/build_index.py              # chunk, embed, build Chroma + BM25 indices
python scripts/query.py "Which infusion pumps were recalled for battery failure?"
python scripts/ask.py "Which infusion pumps were recalled for battery failure?"   # retrieve + generate a grounded, cited answer (requires Ollama running with llama3 pulled)
```

- `pull_corpus.py` writes raw JSON/PDFs to `data/raw/`, logs every source to
  `data/manifest.csv`, and records the pull date in `data/raw/pull_date.txt`.
- `build_index.py` reads that date back into each chunk's `retrieved_date`
  metadata (used later for citations), embeds every chunk locally, and writes
  the Chroma collection to `data/chroma/` and the BM25 index to
  `data/bm25_index.pkl`. It upserts, so re-running after a corpus change
  refreshes existing entries rather than leaving stale vectors behind.
- `query.py` loads both indices and prints the top 5 hybrid-retrieval results.
- `ask.py` retrieves the top-5 chunks via the same hybrid retriever, generates
  a grounded answer with llama3 via a local Ollama server, and prints the
  answer alongside two separately labeled citation lists: **"context
  provided"** (the full top-5, always shown) and **"cited by the model"**
  (only the bracket-numbered passages the model actually referenced in its
  answer — falls back to showing all context, honestly labeled, if the model
  skips the bracket format). Requires Ollama running locally with `llama3`
  pulled (`ollama pull llama3`); exits with a specific message if it isn't.

## Retrieval design

Hybrid retrieval over two legs, fused with **reciprocal rank fusion** (k=60):

- **Dense:** local `sentence-transformers` embeddings in Chroma, cosine similarity.
- **Sparse:** BM25 (`rank_bm25`) over the same chunks — keeps exact-match recall
  for device model numbers and regulatory terms that embeddings tend to blur.

Because the two legs' raw scores are on incomparable scales, `HybridRetriever`
returns the **RRF fused score**, not the leg-native score.

## Retrieval-accuracy benchmark

A frozen, hand-authored 50-question benchmark (`data/eval/questions.json`) —
each question grounded in real source content, gold answer resolved by
locator rather than free text, frozen before any retrieval scoring ran
against it to avoid leakage. Design rationale:
[`docs/superpowers/specs/2026-08-03-eval-harness-design.md`](docs/superpowers/specs/2026-08-03-eval-harness-design.md)
and
[`docs/superpowers/specs/2026-08-04-eval-runner-design.md`](docs/superpowers/specs/2026-08-04-eval-runner-design.md).

Run it yourself:

```bash
python scripts/run_eval.py     # scores all 50 questions -> data/eval/results.json (gitignored)
```

**Corpus size at benchmark time:**

| Source | Documents | Chunks |
|---|---|---|
| Recall | 500 | 500 |
| MAUDE | 500 | 500 |
| Guidance (PDF) | 4 | 456 |
| IFU (PDF) | 3 | 392 |

Recall/MAUDE documents equal chunks by construction (one record → one
chunk); Guidance/IFU document counts are distinct-document counts, which is
why they diverge from chunk counts there. The Guidance/IFU chunk counts
still carry a known residual noise source from chunking: on 2 of the 7
PDF documents (`188844`, `153781`), a repeated boilerplate section label
(`Contains Nonbinding Recommendations`) accounts for **40-41%** of that
document's indexed chunks — a repeated running-footer/label artifact, not
duplicated real content (see
[`docs/superpowers/specs/2026-07-29-section-detection-title-case-fix-design.md`](docs/superpowers/specs/2026-07-29-section-detection-title-case-fix-design.md)
§4 for why this wasn't suppressed: doing so would have merged genuinely
distinct worked examples elsewhere in the same corpus). The other 5
documents' repeated-label share is 5-12%. Raw chunk counts above should be
read with this in mind — they overstate unique content in guidance/IFU by
roughly this margin on the two affected documents.

**Pooled results (n=50 per leg):**

| Leg | Hit Rate@5 | 95% CI | MRR |
|---|---|---|---|
| Dense-only | 38/50 (76.0%) | [62.6, 85.7] | 0.638 |
| BM25-only | 39/50 (78.0%) | [64.8, 87.2] | 0.630 |
| Hybrid (RRF) | 43/50 (86.0%) | [73.8, 93.0] | 0.657 |

**Per source type:**

| Stratum (n) | Dense | BM25 | Hybrid |
|---|---|---|---|
| Recall (13) | 84.6% [57.8, 95.7] | 92.3% [66.7, 98.6] | 100.0% [77.2, 100.0] |
| MAUDE (13) | 84.6% [57.8, 95.7] | 84.6% [57.8, 95.7] | 100.0% [77.2, 100.0] |
| Guidance (12) | 83.3% [55.2, 95.3] | 75.0% [46.8, 91.1] | 91.7% [64.6, 98.5] |
| IFU (12) | 50.0% [25.4, 74.6] | 58.3% [32.0, 80.7] | 50.0% [25.4, 74.6] |

**Reading these results:** on this benchmark, hybrid retrieval had the
highest pooled point estimate (86.0% Hit Rate@5 vs. 78.0% BM25-only and
76.0% dense-only), and led the point estimate in 3 of 4 strata (recall,
MAUDE, guidance) while tying dense and trailing BM25 in the IFU stratum.
However, at n=50 pooled — and more so at n=12-13 per stratum — the 95%
Wilson intervals for all three legs overlap substantially at every level of
aggregation. The data does not support a statistically significant claim
that hybrid outperforms either single-leg baseline; it supports a
directional signal only, consistent with the point estimates but well
within the noise band this sample size produces. A larger benchmark (or a
paired/bootstrap significance test across the same 50 questions, rather
than comparing independent CIs, since all three legs score identical
questions and are not independent samples — the comparison used here is
conservative in that respect) would be needed before "hybrid wins" could be
stated as a settled result.

MRR has no confidence interval — it's a mean of reciprocal ranks over the
full ranking, not a binomial proportion, so it's reported alongside Hit
Rate@5 as a directional signal only, not with the same statistical
precision.

**Future work, if this benchmark is revisited:** run a paired significance
test (e.g. McNemar's test or a bootstrap over per-question hit/miss pairs)
across the three legs on the same 50 questions, rather than relying on
independent-CI overlap as a proxy — the current comparison is a conservative
lower bound on significance, not a rejection of a real effect.

**Manual faithfulness spot-check.** Per the architecture doc's §6, a 15-20
question subset is hand-graded pass/fail on whether the generated answer
relied only on retrieved content (no automated grading/LLM-judge for v1).

```bash
python scripts/spot_check_transcripts.py    # -> data/eval/spot_check_transcripts.md (gitignored)
```

Selects 4 questions per source type (16 total) from the frozen 50 via a
seeded, stratified draw (`--seed` to override; default seed is fixed and
documented in the transcript's header for reproducibility), runs each
through the same retrieve-then-generate pipeline as `ask.py`, and writes a
markdown transcript with a blank verdict line per question for a human
grader to fill in.

## Tests

```bash
pytest
```

The operator scripts (`pull_corpus.py`, `build_index.py`, `query.py`, `ask.py`)
are otherwise deliberately untested — they are thin network/IO wrappers over
the library code in `src/fda_device_rag/`, which is where most test coverage
lives. The one exception is `pull_corpus.py`'s `_fetch_paginated` cross-page
dedup logic (`tests/test_pull_corpus.py`), which is plain enough logic to be
worth pinning directly. `spot_check_transcripts.py` follows the same
untested-CLI reasoning for its batch/generation loop, but its
`select_spot_check_questions` selection function is plain enough logic to
have its own unit tests (`tests/eval/test_spot_check_transcripts.py`), so its
coverage is partial rather than fully absent.

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 1 | Retrieval pipeline (ingest → chunk → embed → hybrid retrieve) | Done |
| 2 | Leakage-safe retrieval-accuracy benchmark | Done |
| 3 | Citation grounding / answer generation | Done |
| 4 | Deployed demo | Planned |
| 5 | CI | Planned |
