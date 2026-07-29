# FDA Device RAG

A Q&A system over public FDA medical-device documentation — device recalls,
adverse event reports (MAUDE), FDA guidance documents, and manufacturer
Instructions for Use (IFUs) — with citation grounding, so every answer can be
traced back to the source record or document section it came from.

This is a portfolio project. The design rationale for each decision is written
up in [`docs/superpowers/specs/2026-07-28-fda-device-rag-architecture-design.md`](docs/superpowers/specs/2026-07-28-fda-device-rag-architecture-design.md).

> **Status: Phase 1 (retrieval pipeline) only.** What exists today is corpus
> ingestion, chunking, local embedding, and hybrid retrieval. Citation
> grounding, the retrieval-accuracy benchmark, and the deployed demo are
> future phases and are **not** implemented yet.

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
```

- `pull_corpus.py` writes raw JSON/PDFs to `data/raw/`, logs every source to
  `data/manifest.csv`, and records the pull date in `data/raw/pull_date.txt`.
- `build_index.py` reads that date back into each chunk's `retrieved_date`
  metadata (used later for citations), embeds every chunk locally, and writes
  the Chroma collection to `data/chroma/` and the BM25 index to
  `data/bm25_index.pkl`. It upserts, so re-running after a corpus change
  refreshes existing entries rather than leaving stale vectors behind.
- `query.py` loads both indices and prints the top 5 hybrid-retrieval results.

## Retrieval design

Hybrid retrieval over two legs, fused with **reciprocal rank fusion** (k=60):

- **Dense:** local `sentence-transformers` embeddings in Chroma, cosine similarity.
- **Sparse:** BM25 (`rank_bm25`) over the same chunks — keeps exact-match recall
  for device model numbers and regulatory terms that embeddings tend to blur.

Because the two legs' raw scores are on incomparable scales, `HybridRetriever`
returns the **RRF fused score**, not the leg-native score.

## Tests

```bash
pytest
```

The operator scripts (`pull_corpus.py`, `build_index.py`, `query.py`) are
otherwise deliberately untested — they are thin network/IO wrappers over the
library code in `src/fda_device_rag/`, which is where most test coverage
lives. The one exception is `pull_corpus.py`'s `_fetch_paginated` cross-page
dedup logic (`tests/test_pull_corpus.py`), which is plain enough logic to be
worth pinning directly.

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 1 | Retrieval pipeline (ingest → chunk → embed → hybrid retrieve) | Done |
| 2 | Leakage-safe retrieval-accuracy benchmark | Planned |
| 3 | Citation grounding / answer generation | Planned |
| 4 | Deployed demo | Planned |
| 5 | CI | Planned |
