# FDA Device RAG — Architecture Design

Status: Approved for planning
Date: 2026-07-28

## 1. Purpose & Framing

A Q&A system over public FDA medical-device documentation (recalls, adverse
event reports, FDA guidance documents, manufacturer IFUs), with citation
grounding and a measured, leakage-safe retrieval-accuracy benchmark. This is a
portfolio project targeting AI/ML internship applications (Optum, Medtronic,
Thomson Reuters and similar). The bar for rigor is a companion Retail Demand
Forecasting project (leakage-safe features, walk-forward CV, tuned model vs.
baseline, dedicated test suite, CI). This project must match that bar: every
design decision below needs to be defensible verbally in an interview, not
just functional in a demo.

Core scope, in shipping order: (1) working retrieval pipeline, (2) honest
retrieval-accuracy benchmark, (3) citation grounding, (4) deployed demo,
(5) tests + CI. Agentic fallback (vector search → web search on low
confidence) is a stretch goal only and must not block core scope.

## 2. Corpus & Ingestion

**Sources (final, after live-API verification):**

| Source | What it provides | Chunking unit |
|---|---|---|
| openFDA `device/recall` | `reason_for_recall` + `action` + `product_description` narrative text | one recall record = one document |
| openFDA `device/event` (MAUDE) | `mdr_text[].text` narrative complaint descriptions | one adverse event report = one document (concatenated if multiple `mdr_text` entries) |
| FDA public guidance documents (PDF) | long-form, section-structured regulatory text | structure-aware (see §3) |
| Manufacturer IFUs (PDF) | long-form, section-structured technical/procedural text | structure-aware (see §3) |
| openFDA `device/510k` | structured metadata only (device name, product code, decision date) | not retrievable text — kept as metadata for potential future filtering |

**510(k) note:** verified live against the API before designing chunking —
the `device/510k` endpoint returns metadata only; narrative 510(k) summary
text is not present in the API response (would require separately scraping
PDFs from `accessdata.fda.gov`, which was explicitly descoped). This is
documented as a corpus-scope decision made from evidence, not assumption.

**Ingestion pipeline:** `scripts/pull_corpus.py`:
1. Hits openFDA `recall` and `event` endpoints with a fixed query/date range, paginates results.
2. Downloads a hand-curated, fixed list of guidance/IFU PDF URLs (not scraped — each one traceable and defensible).
3. Writes everything to `data/raw/`, logging source, URL, retrieval date, and doc type to a manifest CSV.

This manifest backs the README's first-paragraph line: "Corpus: N public FDA
device guidance docs + M IFUs, sourced from [openFDA/manufacturer sites],
last pulled [date]."

## 3. Chunking & Embedding

**Structured openFDA records (recall, MAUDE):** no chunking. One record = one
document, built via light field-templating, e.g.:
```
Recall reason: {reason_for_recall}
Action taken: {action}
Product: {product_description}
```
These records are already short and complete; splitting them would only
fragment a single coherent narrative.

**Long-form PDFs (guidance docs, IFUs):** structure-aware, two-tier chunking:
1. Split by detected section headers first (e.g. "Indications for Use",
   "Contraindications", "Warnings", numbered regulatory sections). This is
   the primary boundary because these are real regulatory documents with
   meaningful structure — a citation naming a section ("Contraindications,
   p.4") is more useful than an arbitrary text window.
2. Within a section, if it exceeds the target chunk size, apply a
   recursive character/token splitter with overlap so no single chunk
   exceeds the embedding model's context limit.

Concrete defaults (tunable during implementation via the benchmark, not
re-litigated here): target chunk size ~400 tokens, ~15% (≈60 token) overlap
within an oversized section.

**Embedding model:** local, open-source, via `sentence-transformers` (e.g.
`BAAI/bge-small-en-v1.5`). Chosen over an API embedding model for full
reproducibility (no API key required to rerun the pipeline) and zero
marginal cost. Similarity metric: cosine similarity (standard for
sentence-transformers' normalized embeddings; Chroma's default HNSW space
is configured accordingly).

**Chunk metadata schema** (attached to every chunk, used later for
citations): `source_type`, `source_url`, `document_title`, `section_name`
(PDFs only), `record_id` (recall number / MAUDE report number, where
applicable), `retrieved_date`.

## 4. Vector Store & Retrieval

**Vector store:** ChromaDB, persistent local collection, cosine similarity space.

**Retrieval strategy: hybrid (BM25 + dense), not pure dense.** Pure dense
vector search misses exact-match tokens common in this domain (product
codes, k-numbers, device model names). Design:
- Dense: top-20 via Chroma cosine similarity over `bge-small` embeddings.
- Sparse: top-20 via BM25 (`rank_bm25`, pure Python, no new infra), tokenized
  with lowercase + light punctuation stripping only — no aggressive
  stemming, so codes like `K123456` or product codes stay matchable verbatim.
- Merge via Reciprocal Rank Fusion, k=60.
- Take fused top-5 as the context passed to generation.

This hybrid approach is the documented differentiator over a bare
vector-search demo, and is verified (not assumed) via the eval harness's
per-strategy ablation (§6).

## 5. Generation & Citation Grounding

**LLM: llama3 (local, via Ollama), selected on measured latency, not a
leaderboard.** gemma4:12b scores better on published instruction-following/
hallucination benchmarks, but the deciding factor was a real latency pass on
target hardware: llama3 measured 11.4s cold load (one-time) then 3.1s
average per grounded answer once warm (4 runs, range 2.8–3.3s, tested with a
~1800-token context prompt matching real RAG-call size) — comfortably
clearing the <20s live-demo bar. The first pass (run while a second model was
downloading in the background, contending for disk/CPU/bandwidth) had
inflated both figures to ~4 min cold load and 18.1s warm average; this is the
clean re-measurement with no competing processes. That "measured with nothing
else running" condition is documented as part of the methodology itself, not
a footnote — it's what makes the number trustworthy. This measured-latency
methodology is documented in the README as the actual selection criterion.

**Operational note (demo-day):** Ollama unloads an idle model from memory
after 5 minutes by default, which would re-trigger the (now minor, ~11s)
cold load mid-demo. `keep_alive` and a warm-up request are still good
practice regardless — the 5-minute idle-unload behavior is real — but this
is a minor smoothing step, not mitigation of a scary multi-minute risk.
Mitigation: pass `"keep_alive": "30m"` in the API call (or set
`OLLAMA_KEEP_ALIVE`), and send a throwaway warm-up request about a minute
before demoing. Documented in deployment/demo instructions, not just in code.

**Grounding strategy:**
- Strict system prompt: answer only from provided numbered context passages,
  no outside knowledge, explicit refusal string when context is insufficient.
- Inline bracket citation (e.g. `[1]`, or `[1][3]` for multi-source
  sentences), parsed deterministically out of the generated text via regex
  afterward — not a self-reported structured citation list. Rationale: an
  8B model can fabricate a JSON citations array exactly as easily as it can
  fabricate an answer; regex-parsing bracket numbers it already wrote inline
  is citation-by-what-it-did, not citation-by-what-it-claims, and is a much
  narrower failure surface.
- Two distinct concepts, always labeled separately (never collapsed into one
  list) in both the UI and the README:
  - **"context provided"** — the full fused top-5, always logged, deterministic (debug/eval trace).
  - **"cited by the model"** — only the bracket-numbered chunks parsed from the answer text. This is the model's own signal, one step more trustworthy than a free-floating claim, but explicitly not independently verified ground truth.
- Fallback: if llama3 skips the bracket format entirely, citations fall back
  to showing all provided context, honestly labeled
  `"context provided (model did not cite specific passages)"` rather than
  silently presenting it as verified citation.

**System prompt and parsing logic** (final, ready to implement as written):

```python
SYSTEM_PROMPT = """You are a document assistant answering questions about FDA \
device recalls, adverse event reports, and device guidance/instructions-for-use \
documents.

Rules:
1. Answer ONLY using the numbered context passages below. Do not use outside \
knowledge, even if you believe it is correct.
2. Cite passages inline using their bracket number immediately after the \
sentence that relies on them, e.g. "...requires firmware 4.2.1 [1]." Use \
multiple brackets if a sentence draws on more than one passage, e.g. [1][3].
3. Only cite a passage number if you actually used its content. Do not cite \
all passages by default, and do not cite a passage you didn't rely on.
4. If the passages do not contain enough information to answer, respond \
exactly: "I don't know based on the available documents." Do not guess or \
fill gaps with general knowledge.
5. Keep answers concise — 2 to 4 sentences unless the question requires more.

Context passages:
{numbered_context_block}
"""

# numbered_context_block built per-query from the fused top-5:
# [1] Source: {source_type} | {doc_title/section} | Retrieved: {date} | URL: {url}
# {chunk_text}
# ... up to [5]

import re

def extract_citations(answer_text, chunk_metadata):
    cited_nums = {int(n) for n in re.findall(r'\[(\d)\]', answer_text)}
    citations = [chunk_metadata[n] for n in cited_nums if n in chunk_metadata]
    if not citations:
        # llama3 sometimes skips the bracket format — fall back to showing
        # all context provided, honestly labeled as unverified.
        citations = list(chunk_metadata.values())
        citation_label = "context provided (model did not cite specific passages)"
    else:
        citation_label = "cited by the model"
    return citations, citation_label
```

**Faithfulness check for v1:** the manual generation-faithfulness spot-check
described in §6 (15-20 question subset, pass/fail graded by hand) — no
automated hallucination-detection/LLM-judge layer for v1.

## 6. Evaluation / Benchmark Design

**Leakage-safe methodology (one-sentence version, repeatable verbatim in an
interview):** *questions and their gold source-document IDs are authored from
FDA's actual complaint/problem taxonomy fields before the retriever ever runs
against them, and the question file is frozen (committed) prior to any
retrieval scoring — so no question can be tuned to fit what retrieval happens
to return.*

**Taxonomy-grounded question sampling** (not ad-hoc):
- MAUDE `product_problems` — FDA's own controlled vocabulary (e.g. "Device
  Difficult to Program", "Break", "Material Deformation").
- Recall `root_cause_description` — categorical (e.g. "Design", "Software",
  "Nonconforming Material").
- Guidance docs / IFUs — questions map to real document sections
  (Indications for Use, Contraindications, Warnings), not invented topics.

**Fixed test set:** 50 questions total, ~12-13 per source type (recall,
MAUDE, guidance, IFU), each assigned a gold source-document ID (or small
acceptable set) at authoring time. Saved to a frozen JSON file, committed to
git, before any retrieval run scores against it. **Sequencing note:**
authoring this set requires the real ingested corpus (not the recon samples
pulled during design), so this is the first implementation task, immediately
after ingestion — not something done during design.

**Metrics:**
- **Primary: Hit Rate@5** (is the gold document in the fused top-5),
  reported as raw counts alongside percentages (e.g. "38/50", not just
  "76%"). At n=50, differences under ~10 percentage points between
  hybrid and dense-only/BM25-only can be noise (~±14pp margin at 95% CI) —
  the README must state this explicitly and treat retrieval-quality gaps
  between strategies as directional, not statistically definitive.
- **Ablation:** Hit Rate@5 reported separately for dense-only, BM25-only,
  and hybrid — so hybrid's value is demonstrated, not assumed.
- **Stratified by source type:** all retrieval metrics (Hit Rate@5, MRR)
  reported per source type (recall / MAUDE / guidance / IFU), not just
  pooled. Structured records (recall, MAUDE) and long-form chunked PDFs
  (guidance, IFU) have different retrieval difficulty, and hybrid's
  advantage is expected to concentrate in the exact-match-heavy structured
  sources — pooling would average that signal away.
- **Secondary: MRR** (mean reciprocal rank), to capture ranking quality
  beyond a binary hit/miss, relevant since RRF fusion is explicitly about
  rank position.
- **Corpus size context:** document count and chunk count per source type
  logged in the README next to the eval numbers, so readers can judge
  whether a given Hit Rate@5 reflects an easy (small pool) or hard (large
  pool) retrieval task.

**Generation-faithfulness spot-check** (on top of retrieval metrics, since
retrieval metrics only prove the right document was available, not that
generation used it correctly): a 15-20 question subset of the frozen 50,
manually graded pass/fail on "did the generated answer rely only on
retrieved content, no fabrication," with grading criteria documented. No
automation/LLM-judge for v1.

## 7. Serving & Deployment

**Streamlit**, deployed on Streamlit Community Cloud — chosen over a
FastAPI + frontend split because the actual phase-4 deliverable is a public
demo link, not a service architecture, and Streamlit reaches that fastest
within the 4-6 week budget. UI surfaces: question input, generated answer,
"cited by the model" passages (labeled distinctly from "context provided"
per §5), and source metadata/links per citation.

Demo-day operational note from §5 (Ollama keep_alive + warm-up request)
documented explicitly in deployment/demo instructions — a minor smoothing
step for a live walkthrough (avoids an ~11s idle-reload pause), not a
mitigation for a significant risk.

## 8. Testing & CI

Matching the forecasting project's bar: a dedicated `tests/` suite (pytest)
covering chunking logic, hybrid retrieval/RRF merge, and citation parsing
(`extract_citations`) as pure-function unit tests; GitHub Actions CI running
lint + tests on every push/PR.

## 9. Explicitly Out of Scope This Cycle

- Clinical SOPs as a data source (no legitimate access).
- A separate MCP project (shelved until this ships).
- Agentic fallback (vector search → web search on low-confidence queries) —
  stretch goal only, attempted only if core scope ships with time to spare.
  No architectural hooks are being pre-built for it in this design.
- Automated hallucination/grounding detection beyond the manual
  faithfulness spot-check — may be revisited in a later cycle.

## 10. Parameters Left Empirically Tunable

These have reasonable defaults above but are expected to be adjusted during
implementation based on the benchmark, not re-litigated at the design level:
chunk size/overlap for oversized PDF sections, BM25/dense top-N before
fusion (currently 20/20), RRF k constant (currently 60), fused top-k passed
to generation (currently 5).
