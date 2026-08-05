# Citation Grounding & Answer Generation — Design

Phase 3 of the architecture doc's scope
([`docs/superpowers/specs/2026-07-28-fda-device-rag-architecture-design.md`](2026-07-28-fda-device-rag-architecture-design.md)
§5-6). That doc already locks in the LLM choice, system prompt, and citation
strategy as "final, ready to implement as written" — this document covers
everything §5-6 left open: module layout, the Ollama call mechanism, error
handling, and how the manual faithfulness spot-check gets tooled.

## 1. What's Already Decided (from the architecture doc, not reopened here)

- **LLM: llama3, local via Ollama** — chosen on measured latency (11.4s cold,
  3.1s warm average), not benchmark leaderboards. Confirmed still installed
  locally with `llama3:latest` pulled.
- **System prompt** — exact text specified in §5, answer-only-from-context,
  explicit refusal string, concise 2-4 sentence answers.
- **Citation strategy** — inline bracket citations (`[1]`, `[1][3]`), parsed
  deterministically out of generated text via regex (not a self-reported
  citation list, to narrow the failure surface). Two distinct, always-separately-
  labeled concepts: **"context provided"** (full fused top-5, always shown) vs.
  **"cited by the model"** (only bracket-numbered chunks the model actually
  referenced). Fallback when the model skips brackets entirely: show all context,
  labeled `"context provided (model did not cite specific passages)"`.
- **`extract_citations(answer_text, chunk_metadata)`** — exact implementation
  given in §5, reused verbatim here.
- **Faithfulness check for v1**: manual pass/fail spot-check on a 15-20 question
  subset of the frozen 50. No automated hallucination-detection/LLM-judge.
- **Demo-day operational note**: `keep_alive` + a warm-up request avoid Ollama's
  5-minute idle-unload mid-demo. Documented as a deployment/demo step, not a
  code requirement.

## 2. Module Layout

**New package `src/fda_device_rag/generation/`:**

- `ollama_client.py`
  - `OllamaError` (base exception)
  - `OllamaNotReadyError(OllamaError)` — raised by the pre-flight check
  - `OllamaGenerationError(OllamaError)` — raised on a mid-request failure
  - `check_ollama_ready(base_url, model) -> None` — GETs `/api/tags`; raises
    `OllamaNotReadyError("Ollama not running. Start it, then retry.")` on
    connection failure, or `OllamaNotReadyError("Model 'llama3' not found. Run:
    ollama pull llama3")` if the server responds but the model isn't in the
    list. Returns normally if ready.
  - `generate(prompt, base_url, model, keep_alive="30m", timeout=60) -> str` —
    POSTs to `/api/generate` (non-streaming), sets `"keep_alive": "30m"` in the
    body per the architecture doc's demo-reliability note. Wraps the call in
    try/except; any failure (timeout, connection drop, non-200) raises
    `OllamaGenerationError` with a generic fallback message — kept distinct
    from `OllamaNotReadyError` so a mid-request failure (passed the pre-flight
    check, then Ollama crashed or timed out) stays distinguishable from a
    pre-flight failure. `ask.py` still only needs one catch site, via the
    shared `OllamaError` base (see §3).
- `prompt.py`
  - `build_prompt(question, chunks: list[ScoredChunk]) -> tuple[str, dict[int, ScoredChunk]]`
    — renders the architecture doc's `SYSTEM_PROMPT` with the chunks numbered
    1-indexed in retrieval order; returns the prompt text and a fresh
    `{number: ScoredChunk}` map (no state carried across calls) for citation
    lookup after generation.
- `citations.py`
  - `extract_citations(answer_text, chunk_map: dict[int, ScoredChunk]) -> tuple[list[ScoredChunk], str]`
    — exact logic from architecture doc §5, adapted to take the `chunk_map`
    produced by `build_prompt` directly (same regex-extraction and
    fallback-to-all-context behavior; returns the citation label as the second
    element).

**Extended `src/fda_device_rag/retrieval/`:**

- New `load_retrieval_stack(chroma_dir: Path, bm25_index_path: Path) -> RetrievalStack`
  (small dataclass: `bm25_index`, `dense_store`, `embedder`, `hybrid_retriever`).
  Replaces the identical ~6-line init block currently duplicated in `query.py`
  and `run_eval.py` (pickle-load BM25 → `ChromaStore` → `Embedder` →
  `HybridRetriever`). `query.py` and `run_eval.py` are both updated to call this
  instead of inlining it a second and third time; `ask.py` is the third
  consumer. `run_eval.py` continues to use `.bm25_index`/`.dense_store`
  individually for its ablation legs, not just the wrapped retriever.

**New `scripts/ask.py`** — end-to-end CLI:

```
check_ollama_ready(base_url, model)             [exits 1 with the specific message on OllamaError]
  → load_retrieval_stack()
  → hybrid_retriever.retrieve(question, top_k=5)
  → build_prompt(question, chunks)
  → generate(prompt, ...)                         [exits 1 with a fallback message on OllamaError]
  → extract_citations(answer_text, chunk_map)
  → print: question, generated answer,
           "cited by the model" (source_url/document_title/section_name per citation),
           "context provided" (always, full top-5, labeled distinctly)
```

`check_ollama_ready` runs first, before paying for index load/embedding/
retrieval — Ollama being down is flagged as the likely demo-day failure mode,
and retrieval has no dependency on it, so failing fast here is the entire
point of the pre-flight check.

Usage: `python scripts/ask.py "<question>"` — mirrors `query.py`'s CLI shape.
`query.py` itself is **not** modified into a dual-mode tool — its value
throughout this project has been fast, Ollama-independent retrieval debugging,
and that identity is worth keeping even behind an opt-in flag.

**New `scripts/spot_check_transcripts.py`** — selects a seeded subset of the
frozen 50 and writes a markdown transcript for hand-grading. See §4.

## 3. Error Handling

| Failure | Where caught | Behavior |
|---|---|---|
| `data/chroma/`/`data/bm25_index.pkl` missing | `ask.py` | Same pattern as `query.py`/`run_eval.py`: message naming the missing path + which script to run first, exit 1 |
| Ollama server unreachable | `check_ollama_ready` → `ask.py` catches `OllamaError` | `"Ollama not running. Start it, then retry."`, exit 1 |
| Model not pulled | `check_ollama_ready` → `ask.py` catches `OllamaError` | `"Model 'llama3' not found. Run: ollama pull llama3"`, exit 1 |
| Ollama crashes/times out mid-request (passed pre-flight) | `generate` raises `OllamaGenerationError` → `ask.py` catches `OllamaError` | Generic fallback message, exit 1 |
| Model skips bracket citations | `extract_citations` (not an error) | Falls back to showing all context, labeled `"context provided (model did not cite specific passages)"` |

`ask.py` has a single `except OllamaError` catch site; the two subclasses exist
so `ollama_client.py`'s raise sites stay accurate about *when* the failure
happened, without pushing that distinction onto the caller.

## 4. Faithfulness Spot-Check Tooling

Per architecture doc §6: 15-20 questions from the frozen 50, hand-graded
pass/fail on whether the generated answer relied only on retrieved content.
This design automates the mechanical transcription step only — grading itself
stays fully manual (no LLM-judge for v1, consistent with the architecture
doc's existing "no automation" decision).

`scripts/spot_check_transcripts.py`:

- Selects a proportionally-stratified subset (default 16 — 4 per source type ×
  4 source types, within the architecture doc's 15-20 range, comfortably
  inside each stratum's 12-13 available questions) via `random.Random(seed)`,
  mirroring `eval/sampling.py`'s seeded, no-manual-picking approach. Own
  `DEFAULT_SEED =
  "fda-device-rag-spotcheck-2026-08-04"`, distinct from the benchmark's own
  `DEFAULT_SEED` in `sample_eval_candidates.py` so the two sampling operations
  aren't correlated. `--seed` override lets a later re-check (e.g. after a
  generation-pipeline change) draw a different deterministic subset instead of
  either always regrading the same 16 questions or falling back to
  non-reproducible randomness.
- Calls `check_ollama_ready` and `load_retrieval_stack` once up front (same
  fail-fast rationale as `ask.py` — no point loading the index and retrieving
  for 16 questions only to find Ollama unreachable on the first `generate`
  call), then runs each selected question through the same `retrieve` →
  `build_prompt` → `generate` → `extract_citations` sequence `ask.py` uses per
  question — zero duplicated logic between the two scripts.
- Writes `data/eval/spot_check_transcripts.md` — gitignored and overwritten per
  run, same convention as `results.json` (commit a specific run's transcript
  only when it's actually being cited, e.g. as evidence of a completed grading
  pass). The file header records the seed used, so any committed copy is
  traceable to exactly which 16 questions were graded. One section per
  question: question text, context provided, generated answer, cited passages,
  and a blank verdict line for the human grader to fill in. Markdown, not
  JSON — the audience is a human grader, not a downstream metrics consumer.

## 5. Testing

| Component | Tested? | Rationale |
|---|---|---|
| `citations.py::extract_citations` | Yes — pure function | Matches architecture doc §8 exactly: happy path, no-citations fallback, out-of-range number ignored |
| `ollama_client.py::check_ollama_ready` | Yes — mocked `requests` | Three-state branching (unreachable / model missing / ready) is precision-critical logic embedded in an otherwise-thin wrapper — same risk category as `pull_corpus.py`'s dedup-logic exception, not "does the network call work" |
| `prompt.py::build_prompt` | Yes — pure function | 1-indexing and fresh-map-per-call behavior is real logic worth pinning |
| `ollama_client.py::generate` | No | Single linear POST-and-unwrap with no branching worth pinning — this justification is about the function's own shape, not borrowed from the `scripts/` "thin wrapper" exemption (this is library code in `generation/`, same tier as `check_ollama_ready`) |
| `scripts/ask.py`, `scripts/spot_check_transcripts.py` | No | Thin CLI/IO wrappers over already-tested `src/` logic, matching `query.py`/`pull_corpus.py`'s existing convention |
| `spot_check_transcripts.py`'s seeded selection | Yes | Determinism test (same seed → same question IDs), mirroring `tests/eval/test_sampling.py`'s existing determinism tests |

## 6. Out of Scope

- Automated hallucination/grounding detection beyond the manual spot-check —
  explicitly deferred per architecture doc §5.
- Any change to `run_eval.py`'s retrieval-only metrics — generation-faithfulness
  is a separate, manual signal, not folded into Hit Rate@5/MRR.
- Streaming responses, multi-turn conversation, or a served API — those belong
  to Phase 4 (deployed demo).
