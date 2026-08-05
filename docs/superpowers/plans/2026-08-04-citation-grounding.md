# Citation Grounding & Answer Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add end-to-end retrieve-then-generate answer generation with regex-parsed inline citations, grounded in llama3 via a local Ollama server.

**Architecture:** A new `generation/` package (Ollama HTTP client, prompt building, citation extraction) sits behind a new `scripts/ask.py` CLI that chains the existing `HybridRetriever` to generation. A shared `load_retrieval_stack()` helper (new, in `retrieval/`) replaces index-loading code currently duplicated in `query.py` and `run_eval.py`, and becomes the third consumer. A second script, `scripts/spot_check_transcripts.py`, automates transcript generation for the architecture doc's manual 15-20-question faithfulness spot-check (grading itself stays human).

**Tech Stack:** Python 3.11+, `requests` (already a dependency — no new deps), Ollama (local server, `llama3:latest` already pulled), pytest + `unittest.mock`.

## Global Constraints

- **No new dependencies.** Ollama is called via direct `requests` HTTP calls (`/api/tags`, `/api/generate`), not the `ollama` package — matches this project's stance on hand-rolling simple HTTP wrappers (e.g. Wilson interval vs. scipy).
- **`SYSTEM_PROMPT` and the citation-extraction regex are reused verbatim** from `docs/superpowers/specs/2026-07-28-fda-device-rag-architecture-design.md` §5 — do not rephrase or "improve" the prompt text or the `\[(\d)\]` regex.
- **Exception hierarchy:** `OllamaError` (base) → `OllamaNotReadyError` (pre-flight failure) and `OllamaGenerationError` (mid-request failure, after pre-flight passed). Callers catch `OllamaError` once; raise sites stay accurate about which subclass fired.
- **`keep_alive="30m"` must be set explicitly** in every `/api/generate` request body, per the architecture doc's demo-reliability note.
- **Pre-flight ordering:** `check_ollama_ready()` always runs before any index load / embedding / retrieval in both `ask.py` and `spot_check_transcripts.py` — fail fast on the likely demo-day failure mode before paying for anything else.
- **Two citation concepts stay separately labeled, always** — "context provided" (full top-5, always shown) and "cited by the model" (or its no-citation fallback label) are never collapsed into one list, in any script's output.
- **Verified against the live server** (not assumed): `/api/tags` returns `{"models": [{"name": "llama3:latest", ...}]}`; `/api/generate` (non-streaming) returns `{"response": "...", "done": true, ...}`. Confirmed via direct `curl` against the local Ollama instance before this plan was written.
- **Testing split** (from the design doc's §5 table): pure functions and precision-critical branching get unit tests (`extract_citations`, `build_prompt`, `check_ollama_ready`, `select_spot_check_questions`); thin single-shot IO (`generate`, both CLI scripts, `load_retrieval_stack`) does not — verified instead by a real run against the live corpus/Ollama server, matching `query.py`/`run_eval.py`'s existing convention.
- **Baseline regression values** (for Task 1's before/after check): `python scripts/query.py "Which infusion pumps were recalled for battery failure?"` currently returns, in order: `78369-10` (score≈0.03227), `recall-Z-0094-2024` (≈0.03102), `recall-Z-0091-2022` (≈0.02549), `event-2032227-2020-110249` (≈0.01639), `event-2032227-2020-110280` (≈0.01613). `python scripts/run_eval.py` currently reports pooled dense 38/50 (76.0%), bm25 39/50 (78.0%), hybrid 43/50 (86.0%).

---

## File Structure

```
src/fda_device_rag/
  retrieval/
    bootstrap.py          [NEW] RetrievalStack, load_retrieval_stack()
  generation/              [NEW package]
    __init__.py            [NEW]
    ollama_client.py       [NEW] OllamaError hierarchy, check_ollama_ready(), generate()
    prompt.py               [NEW] SYSTEM_PROMPT, build_prompt()
    citations.py            [NEW] extract_citations()
scripts/
  query.py                 [MODIFIED] use load_retrieval_stack()
  run_eval.py               [MODIFIED] use load_retrieval_stack()
  ask.py                    [NEW] end-to-end retrieve+generate CLI
  spot_check_transcripts.py [NEW] seeded faithfulness spot-check transcript generator
tests/
  retrieval/
    (no new test file — load_retrieval_stack is a thin IO wrapper, see Global Constraints)
  generation/               [NEW]
    __init__.py              [NEW]
    test_ollama_client.py    [NEW]
    test_prompt.py            [NEW]
    test_citations.py         [NEW]
  eval/
    test_spot_check_transcripts.py [NEW]
.gitignore                 [MODIFIED] add data/eval/spot_check_transcripts.md
README.md                  [MODIFIED] document ask.py and the spot-check script
```

---

### Task 1: Shared retrieval-stack loader + refactor `query.py`/`run_eval.py`

**Files:**
- Create: `src/fda_device_rag/retrieval/bootstrap.py`
- Modify: `scripts/query.py`
- Modify: `scripts/run_eval.py`

**Interfaces:**
- Produces: `RetrievalStack` dataclass (`bm25_index: BM25Index`, `dense_store: ChromaStore`, `embedder: Embedder`, `hybrid_retriever: HybridRetriever`), `load_retrieval_stack(chroma_dir: Path, bm25_index_path: Path) -> RetrievalStack`. Consumed by Task 5 (`ask.py`) and Task 6 (`spot_check_transcripts.py`).

This is a pure refactor (no new behavior), so it's verified by the existing test suite plus a real-corpus before/after comparison against the baseline values in Global Constraints, not new unit tests — matching how `query.py`/`run_eval.py`'s own index-loading was never separately unit tested.

- [ ] **Step 1: Create `src/fda_device_rag/retrieval/bootstrap.py`**

```python
from dataclasses import dataclass
from pathlib import Path
import pickle

from fda_device_rag.embedding.embedder import Embedder
from fda_device_rag.retrieval.bm25_index import BM25Index
from fda_device_rag.retrieval.hybrid_retriever import HybridRetriever
from fda_device_rag.store.chroma_store import ChromaStore


@dataclass
class RetrievalStack:
    bm25_index: BM25Index
    dense_store: ChromaStore
    embedder: Embedder
    hybrid_retriever: HybridRetriever


def load_retrieval_stack(chroma_dir: Path, bm25_index_path: Path) -> RetrievalStack:
    """Loads the BM25 pickle and Chroma collection built by build_index.py
    and wires them into a HybridRetriever. Safe to pickle.loads: this file
    is a local build artifact written by build_index.py on this machine,
    never fetched or accepted from an external source."""
    bm25_index = pickle.loads(bm25_index_path.read_bytes())
    dense_store = ChromaStore(persist_dir=str(chroma_dir))
    embedder = Embedder()
    hybrid_retriever = HybridRetriever(dense_store=dense_store, bm25_index=bm25_index, embedder=embedder)
    return RetrievalStack(
        bm25_index=bm25_index,
        dense_store=dense_store,
        embedder=embedder,
        hybrid_retriever=hybrid_retriever,
    )
```

- [ ] **Step 2: Rewrite `scripts/query.py` to use it**

Replace the file's contents entirely with:

```python
# scripts/query.py
"""Queries the index built by build_index.py using the hybrid retriever
(dense Chroma + BM25, fused with reciprocal rank fusion).

Usage: python scripts/query.py "<question>"
"""
import sys
from pathlib import Path

from fda_device_rag.retrieval.bootstrap import load_retrieval_stack

CHROMA_DIR = Path("data/chroma")
BM25_INDEX_PATH = Path("data/bm25_index.pkl")
TOP_K = 5
PREVIEW_CHARS = 300


def main() -> None:
    if len(sys.argv) < 2 or not sys.argv[1].strip():
        print('Usage: python scripts/query.py "<question>"')
        sys.exit(1)

    query_text = sys.argv[1]

    if not BM25_INDEX_PATH.exists() or not CHROMA_DIR.exists():
        print(f"Index not found ({CHROMA_DIR}, {BM25_INDEX_PATH}) -- run "
              f"scripts/pull_corpus.py then scripts/build_index.py first.")
        sys.exit(1)

    stack = load_retrieval_stack(CHROMA_DIR, BM25_INDEX_PATH)
    results = stack.hybrid_retriever.retrieve(query_text, top_k=TOP_K)

    print(f'Query: {query_text}')
    print(f"Top {len(results)} results (score = RRF fused rank score):\n")
    for rank, result in enumerate(results, start=1):
        preview = " ".join(result.text.split())[:PREVIEW_CHARS]
        if len(preview) == PREVIEW_CHARS:
            preview += "..."
        print(f"{rank}. {result.id}  (score={result.score:.5f})")
        print(f"   source_type: {result.metadata.source_type}")
        print(f"   title:       {result.metadata.document_title}")
        print(f"   text:        {preview}\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Update `scripts/run_eval.py`'s imports and index-loading block**

In the imports section, remove `import pickle`, remove the `from fda_device_rag.embedding.embedder import Embedder`, `from fda_device_rag.retrieval.hybrid_retriever import HybridRetriever`, and `from fda_device_rag.store.chroma_store import ChromaStore` lines, and add:

```python
from fda_device_rag.retrieval.bootstrap import load_retrieval_stack
```

Keep `from fda_device_rag.retrieval.bm25_index import BM25Index` (still used as a type hint in `_corpus_size`).

In `main()`, replace:

```python
    # Safe: this pickle is a local build artifact written by build_index.py on
    # this machine, never fetched or accepted from an external source. (It holds
    # a BM25Okapi object with fitted corpus statistics, not plain data, which is
    # why it is pickled rather than serialized to JSON.)
    bm25_index = pickle.loads(BM25_INDEX_PATH.read_bytes())
    dense_store = ChromaStore(persist_dir=str(CHROMA_DIR))
    embedder = Embedder()
    hybrid_retriever = HybridRetriever(dense_store=dense_store, bm25_index=bm25_index, embedder=embedder)
```

with:

```python
    stack = load_retrieval_stack(CHROMA_DIR, BM25_INDEX_PATH)
    bm25_index = stack.bm25_index
    dense_store = stack.dense_store
    embedder = stack.embedder
    hybrid_retriever = stack.hybrid_retriever
```

Leave the rest of `run_eval.py` (including the unused `DATA_DIR = Path("data/raw")` constant) untouched — out of scope for this refactor.

- [ ] **Step 4: Run the full test suite**

Run: `.venv/Scripts/python.exe -m pytest tests/ -q`
Expected: `133 passed` (same count as before this task — this refactor changes no tested behavior).

- [ ] **Step 5: Real-corpus regression check — `query.py`**

Run: `.venv/Scripts/python.exe scripts/query.py "Which infusion pumps were recalled for battery failure?"`
Expected: top-5 results in this exact order: `78369-10`, `recall-Z-0094-2024`, `recall-Z-0091-2022`, `event-2032227-2020-110249`, `event-2032227-2020-110280` (scores within rounding of 0.03227 / 0.03102 / 0.02549 / 0.01639 / 0.01613). If the order or ids differ, the refactor changed retrieval behavior — stop and investigate before continuing.

- [ ] **Step 6: Real-corpus regression check — `run_eval.py`**

Run: `.venv/Scripts/python.exe scripts/run_eval.py`
Expected: pooled results unchanged — dense 38/50 (76.0%), bm25 39/50 (78.0%), hybrid 43/50 (86.0%). If any number differs, stop and investigate before continuing.

- [ ] **Step 7: Commit**

```bash
git add src/fda_device_rag/retrieval/bootstrap.py scripts/query.py scripts/run_eval.py
git commit -m "refactor: extract load_retrieval_stack(), used by query.py and run_eval.py"
```

---

### Task 2: Ollama HTTP client

**Files:**
- Create: `src/fda_device_rag/generation/__init__.py` (empty)
- Create: `src/fda_device_rag/generation/ollama_client.py`
- Test: `tests/generation/__init__.py` (empty)
- Test: `tests/generation/test_ollama_client.py`

**Interfaces:**
- Consumes: `requests` (already a dependency).
- Produces: `OLLAMA_BASE_URL: str`, `MODEL: str`, `OllamaError`, `OllamaNotReadyError(OllamaError)`, `OllamaGenerationError(OllamaError)`, `check_ollama_ready(base_url: str, model: str) -> None`, `generate(prompt: str, base_url: str, model: str, keep_alive: str = "30m", timeout: int = 60) -> str`. Consumed by Task 5 (`ask.py`) and Task 6 (`spot_check_transcripts.py`).

- [ ] **Step 1: Write the failing tests**

Create `tests/generation/__init__.py` (empty file).

Create `tests/generation/test_ollama_client.py`:

```python
from unittest.mock import patch, MagicMock

import pytest
import requests

from fda_device_rag.generation.ollama_client import OllamaNotReadyError, check_ollama_ready


@patch("fda_device_rag.generation.ollama_client.requests.get")
def test_check_ollama_ready_raises_when_server_unreachable(mock_get):
    mock_get.side_effect = requests.ConnectionError()

    with pytest.raises(OllamaNotReadyError, match="not running"):
        check_ollama_ready("http://localhost:11434", "llama3")


@patch("fda_device_rag.generation.ollama_client.requests.get")
def test_check_ollama_ready_raises_when_model_not_pulled(mock_get):
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {"models": [{"name": "gemma:latest"}]}
    mock_get.return_value = mock_response

    with pytest.raises(OllamaNotReadyError, match="not found"):
        check_ollama_ready("http://localhost:11434", "llama3")


@patch("fda_device_rag.generation.ollama_client.requests.get")
def test_check_ollama_ready_passes_when_model_present(mock_get):
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {"models": [{"name": "llama3:latest"}]}
    mock_get.return_value = mock_response

    check_ollama_ready("http://localhost:11434", "llama3")  # must not raise
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/generation/test_ollama_client.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fda_device_rag.generation'`

- [ ] **Step 3: Create `src/fda_device_rag/generation/__init__.py`** (empty file)

- [ ] **Step 4: Write `src/fda_device_rag/generation/ollama_client.py`**

```python
"""Thin HTTP client for a local Ollama server -- direct requests calls, not
the ollama package, matching this project's stance on hand-rolling simple
HTTP wrappers over adding a dependency for marginal convenience.

Verified against a live Ollama instance before this module was written:
GET /api/tags returns {"models": [{"name": "llama3:latest", ...}]}; POST
/api/generate (non-streaming) returns {"response": "...", "done": true, ...}.
"""
import requests

OLLAMA_BASE_URL = "http://localhost:11434"
MODEL = "llama3"


class OllamaError(Exception):
    """Base class for generation-pipeline Ollama failures."""


class OllamaNotReadyError(OllamaError):
    """Raised by check_ollama_ready when the server is unreachable or the
    model isn't pulled -- a pre-flight failure, before any generation was
    attempted."""


class OllamaGenerationError(OllamaError):
    """Raised by generate() when the /api/generate call itself fails after
    the pre-flight check already passed -- kept distinct from
    OllamaNotReadyError so a mid-request failure (e.g. Ollama crashes
    between the check and the call) stays distinguishable from a pre-flight
    failure."""


def check_ollama_ready(base_url: str, model: str) -> None:
    """Raises OllamaNotReadyError with a specific message if the Ollama
    server isn't reachable (or returns an error status), or if it's
    reachable but `model` isn't pulled. Returns normally if ready."""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        response.raise_for_status()
    except requests.RequestException:
        raise OllamaNotReadyError("Ollama not running. Start it, then retry.")

    # Ollama's /api/tags returns tags with an explicit version, e.g.
    # "llama3:latest" -- match on the name before ':' so a bare "llama3"
    # check still matches whatever tag is actually pulled.
    pulled = {m["name"].split(":")[0] for m in response.json().get("models", [])}
    if model not in pulled:
        raise OllamaNotReadyError(f"Model {model!r} not found. Run: ollama pull {model}")


def generate(prompt: str, base_url: str, model: str, keep_alive: str = "30m", timeout: int = 60) -> str:
    """POSTs a non-streaming /api/generate request. Any failure (timeout,
    connection drop, non-200 response, unexpected response shape) raises
    OllamaGenerationError -- a single linear POST-and-unwrap with no
    branching worth unit-testing beyond this."""
    try:
        response = requests.post(
            f"{base_url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False, "keep_alive": keep_alive},
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()["response"]
    except (requests.RequestException, KeyError) as e:
        raise OllamaGenerationError(f"Generation failed: {e}")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/generation/test_ollama_client.py -v`
Expected: `3 passed`

- [ ] **Step 6: Commit**

```bash
git add src/fda_device_rag/generation/__init__.py src/fda_device_rag/generation/ollama_client.py tests/generation/__init__.py tests/generation/test_ollama_client.py
git commit -m "feat: add Ollama HTTP client (check_ollama_ready, generate)"
```

---

### Task 3: Prompt building

**Files:**
- Create: `src/fda_device_rag/generation/prompt.py`
- Test: `tests/generation/test_prompt.py`

**Interfaces:**
- Consumes: `ScoredChunk`, `ChunkMetadata` (`src/fda_device_rag/models.py` — existing).
- Produces: `SYSTEM_PROMPT: str`, `build_prompt(question: str, chunks: list[ScoredChunk]) -> tuple[str, dict[int, ScoredChunk]]`. Consumed by Task 5 (`ask.py`) and Task 6 (`spot_check_transcripts.py`); the returned `chunk_map` is consumed by Task 4's `extract_citations`.

- [ ] **Step 1: Write the failing test**

Create `tests/generation/test_prompt.py`:

```python
from fda_device_rag.generation.prompt import build_prompt
from fda_device_rag.models import ChunkMetadata, ScoredChunk


def _chunk(chunk_id, text, section_name=None):
    return ScoredChunk(
        id=chunk_id,
        score=1.0,
        text=text,
        metadata=ChunkMetadata(
            source_type="recall",
            source_url=f"http://example.com/{chunk_id}",
            document_title=f"Doc {chunk_id}",
            section_name=section_name,
            record_id=chunk_id,
            retrieved_date="2026-08-01",
        ),
    )


def test_build_prompt_numbers_chunks_1_indexed_in_order():
    chunk_a = _chunk("a", "Text A")
    chunk_b = _chunk("b", "Text B", section_name="Warnings")

    prompt, chunk_map = build_prompt("What is the warning?", [chunk_a, chunk_b])

    assert chunk_map == {1: chunk_a, 2: chunk_b}
    assert "[1]" in prompt
    assert "[2]" in prompt
    assert "Text A" in prompt
    assert "Text B" in prompt
    assert "What is the warning?" in prompt


def test_build_prompt_resets_chunk_map_per_call():
    chunk_b = _chunk("b", "Text B")

    _, first_map = build_prompt("Question one", [_chunk("a", "Text A"), chunk_b])
    _, second_map = build_prompt("Question two", [chunk_b])

    assert second_map == {1: chunk_b}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/generation/test_prompt.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fda_device_rag.generation.prompt'`

- [ ] **Step 3: Write `src/fda_device_rag/generation/prompt.py`**

```python
"""Builds the numbered-context generation prompt from retrieved chunks.
SYSTEM_PROMPT is reused verbatim from
docs/superpowers/specs/2026-07-28-fda-device-rag-architecture-design.md
section 5."""
from fda_device_rag.models import ScoredChunk

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


def build_prompt(question: str, chunks: list[ScoredChunk]) -> tuple[str, dict[int, ScoredChunk]]:
    """Numbers `chunks` 1-indexed in the order given (retrieval rank order),
    renders them into SYSTEM_PROMPT's context block, and appends the
    question. Returns the finished prompt and a fresh {number: chunk} map
    for citation lookup after generation -- no state is carried across
    calls."""
    chunk_map = {i: chunk for i, chunk in enumerate(chunks, start=1)}

    blocks = []
    for number, chunk in chunk_map.items():
        header = f"[{number}] Source: {chunk.metadata.source_type} | {chunk.metadata.document_title}"
        if chunk.metadata.section_name:
            header += f" | {chunk.metadata.section_name}"
        header += f" | Retrieved: {chunk.metadata.retrieved_date} | URL: {chunk.metadata.source_url}"
        blocks.append(f"{header}\n{chunk.text}")
    numbered_context_block = "\n\n".join(blocks)

    prompt = SYSTEM_PROMPT.format(numbered_context_block=numbered_context_block)
    prompt += f"\nQuestion: {question}\n\nAnswer:"
    return prompt, chunk_map
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/generation/test_prompt.py -v`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add src/fda_device_rag/generation/prompt.py tests/generation/test_prompt.py
git commit -m "feat: add build_prompt for numbered-context generation prompts"
```

---

### Task 4: Citation extraction

**Files:**
- Create: `src/fda_device_rag/generation/citations.py`
- Test: `tests/generation/test_citations.py`

**Interfaces:**
- Consumes: `ScoredChunk` (`src/fda_device_rag/models.py`); the `dict[int, ScoredChunk]` shape produced by Task 3's `build_prompt`.
- Produces: `extract_citations(answer_text: str, chunk_map: dict[int, ScoredChunk]) -> tuple[list[ScoredChunk], str]`. Consumed by Task 5 (`ask.py`) and Task 6 (`spot_check_transcripts.py`).

- [ ] **Step 1: Write the failing tests**

Create `tests/generation/test_citations.py`:

```python
from fda_device_rag.generation.citations import extract_citations
from fda_device_rag.models import ChunkMetadata, ScoredChunk


def _chunk(chunk_id):
    return ScoredChunk(
        id=chunk_id,
        score=1.0,
        text=f"Text {chunk_id}",
        metadata=ChunkMetadata(
            source_type="recall",
            source_url=f"http://example.com/{chunk_id}",
            document_title=f"Doc {chunk_id}",
            section_name=None,
            record_id=chunk_id,
            retrieved_date="2026-08-01",
        ),
    )


def test_extract_citations_happy_path_returns_cited_chunks_and_label():
    chunk_map = {1: _chunk("a"), 2: _chunk("b"), 3: _chunk("c")}

    citations, label = extract_citations("Uses passage one [1] and three [3].", chunk_map)

    assert citations == [chunk_map[1], chunk_map[3]]
    assert label == "cited by the model"


def test_extract_citations_falls_back_to_all_context_when_no_brackets():
    chunk_map = {1: _chunk("a"), 2: _chunk("b")}

    citations, label = extract_citations("I don't know based on the available documents.", chunk_map)

    assert citations == list(chunk_map.values())
    assert label == "context provided (model did not cite specific passages)"


def test_extract_citations_ignores_out_of_range_numbers():
    chunk_map = {1: _chunk("a")}

    citations, label = extract_citations("Cites passage [1] and a bogus [9].", chunk_map)

    assert citations == [chunk_map[1]]
    assert label == "cited by the model"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/generation/test_citations.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fda_device_rag.generation.citations'`

- [ ] **Step 3: Write `src/fda_device_rag/generation/citations.py`**

```python
"""Deterministic citation extraction from generated answer text -- reused
verbatim from
docs/superpowers/specs/2026-07-28-fda-device-rag-architecture-design.md
section 5, adapted to the {number: ScoredChunk} map build_prompt produces.
Regex-parses bracket-numbered citations the model wrote inline, rather than
trusting a self-reported citation list -- an 8B model can fabricate a
citations array as easily as it can fabricate an answer; parsing numbers it
already wrote inline is citation-by-what-it-did, not citation-by-what-it-
claims."""
import re

from fda_device_rag.models import ScoredChunk


def extract_citations(answer_text: str, chunk_map: dict[int, ScoredChunk]) -> tuple[list[ScoredChunk], str]:
    cited_nums = {int(n) for n in re.findall(r'\[(\d)\]', answer_text)}
    citations = [chunk_map[n] for n in cited_nums if n in chunk_map]
    if not citations:
        # The model sometimes skips the bracket format entirely -- fall back
        # to showing all context provided, honestly labeled as unverified.
        citations = list(chunk_map.values())
        citation_label = "context provided (model did not cite specific passages)"
    else:
        citation_label = "cited by the model"
    return citations, citation_label
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/generation/test_citations.py -v`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add src/fda_device_rag/generation/citations.py tests/generation/test_citations.py
git commit -m "feat: add extract_citations for regex-parsed bracket citations"
```

---

### Task 5: `scripts/ask.py` — end-to-end retrieve + generate CLI

**Files:**
- Create: `scripts/ask.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `load_retrieval_stack` (Task 1), `check_ollama_ready`/`generate`/`OllamaError`/`OLLAMA_BASE_URL`/`MODEL` (Task 2), `build_prompt` (Task 3), `extract_citations` (Task 4).

No new tests — thin CLI/IO wrapper over already-tested `src/` logic, matching `query.py`'s convention. Verified via a real run against the live corpus and Ollama server.

- [ ] **Step 1: Write `scripts/ask.py`**

```python
# scripts/ask.py
"""End-to-end retrieve-then-generate CLI: retrieves the top-5 chunks via the
hybrid retriever, generates a grounded answer with llama3 via Ollama, and
prints the answer alongside "context provided" (always, full top-5) and
"cited by the model" (only bracket-numbered chunks the model actually
referenced) -- the two always labeled distinctly, per
docs/superpowers/specs/2026-07-28-fda-device-rag-architecture-design.md
section 5.

Usage: python scripts/ask.py "<question>"
"""
import sys
from pathlib import Path

from fda_device_rag.generation.citations import extract_citations
from fda_device_rag.generation.ollama_client import (
    MODEL,
    OLLAMA_BASE_URL,
    OllamaError,
    check_ollama_ready,
    generate,
)
from fda_device_rag.generation.prompt import build_prompt
from fda_device_rag.retrieval.bootstrap import load_retrieval_stack

CHROMA_DIR = Path("data/chroma")
BM25_INDEX_PATH = Path("data/bm25_index.pkl")
TOP_K = 5


def _format_chunk_line(chunk) -> str:
    line = chunk.metadata.document_title
    if chunk.metadata.section_name:
        line += f" ({chunk.metadata.section_name})"
    return line


def main() -> None:
    if len(sys.argv) < 2 or not sys.argv[1].strip():
        print('Usage: python scripts/ask.py "<question>"')
        sys.exit(1)

    question = sys.argv[1]

    # Pre-flight check runs first, before paying for index load / embedding /
    # retrieval -- Ollama being down is the likely demo-day failure mode, and
    # retrieval has no dependency on it.
    try:
        check_ollama_ready(OLLAMA_BASE_URL, MODEL)
    except OllamaError as e:
        print(str(e))
        sys.exit(1)

    if not BM25_INDEX_PATH.exists() or not CHROMA_DIR.exists():
        print(f"Index not found ({CHROMA_DIR}, {BM25_INDEX_PATH}) -- run "
              f"scripts/pull_corpus.py then scripts/build_index.py first.")
        sys.exit(1)

    stack = load_retrieval_stack(CHROMA_DIR, BM25_INDEX_PATH)
    chunks = stack.hybrid_retriever.retrieve(question, top_k=TOP_K)

    prompt, chunk_map = build_prompt(question, chunks)

    try:
        answer = generate(prompt, OLLAMA_BASE_URL, MODEL)
    except OllamaError as e:
        print(str(e))
        sys.exit(1)

    citations, citation_label = extract_citations(answer, chunk_map)

    print(f"Question: {question}\n")
    print(f"Answer: {answer}\n")
    print(f"{citation_label}:")
    for chunk in citations:
        print(f"  - {_format_chunk_line(chunk)} -- {chunk.metadata.source_url}")
    print("\ncontext provided (full top-5):")
    for number, chunk in chunk_map.items():
        print(f"  [{number}] {_format_chunk_line(chunk)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Real-run check — happy path**

Ensure Ollama is running (`ollama list` should show `llama3:latest`; if not, run `ollama pull llama3` first).

Run: `.venv/Scripts/python.exe scripts/ask.py "Which infusion pumps were recalled for battery failure?"`
Expected: exits 0; prints a non-empty `Answer:` line; prints either a `cited by the model:` or `context provided (model did not cite specific passages):` section (not both, and not neither); prints a separate `context provided (full top-5):` section listing exactly 5 numbered entries.

- [ ] **Step 3: Real-run check — Ollama-down path**

Stop the Ollama server (Task Manager / `taskkill` on the Ollama process, or however it's running locally), then run the same command again.

Run: `.venv/Scripts/python.exe scripts/ask.py "Which infusion pumps were recalled for battery failure?"`
Expected: exits 1 immediately (no retrieval/index-loading delay) with `Ollama not running. Start it, then retry.` printed. Restart Ollama afterward before continuing.

- [ ] **Step 4: Document `ask.py` in README.md**

In the "Running the pipeline" section, after the existing `query.py` line, add:

```bash
python scripts/ask.py "Which infusion pumps were recalled for battery failure?"   # retrieve + generate a grounded, cited answer (requires Ollama running with llama3 pulled)
```

Add a short paragraph after the existing bullet list in that section:

```markdown
- `ask.py` retrieves the top-5 chunks via the same hybrid retriever, generates
  a grounded answer with llama3 via a local Ollama server, and prints the
  answer alongside two separately labeled citation lists: **"context
  provided"** (the full top-5, always shown) and **"cited by the model"**
  (only the bracket-numbered passages the model actually referenced in its
  answer — falls back to showing all context, honestly labeled, if the model
  skips the bracket format). Requires Ollama running locally with `llama3`
  pulled (`ollama pull llama3`); exits with a specific message if it isn't.
```

- [ ] **Step 5: Commit**

```bash
git add scripts/ask.py README.md
git commit -m "feat: add scripts/ask.py, end-to-end retrieve+generate CLI"
```

---

### Task 6: Faithfulness spot-check transcript generator

**Files:**
- Create: `scripts/spot_check_transcripts.py`
- Test: `tests/eval/test_spot_check_transcripts.py`
- Modify: `.gitignore`
- Modify: `README.md`

**Interfaces:**
- Consumes: `Question`/`load_questions` (`src/fda_device_rag/eval/questions.py` — existing), `load_retrieval_stack` (Task 1), `check_ollama_ready`/`generate`/`OllamaError`/`OLLAMA_BASE_URL`/`MODEL` (Task 2), `build_prompt` (Task 3), `extract_citations` (Task 4).
- Produces: `select_spot_check_questions(questions: list[Question], seed: str, per_stratum: int = 4) -> list[Question]` (importable from the script module, matching the existing `run_eval.py`/`_score_all` precedent of testing script-level pure functions directly).

- [ ] **Step 1: Write the failing tests**

Create `tests/eval/test_spot_check_transcripts.py`:

```python
from fda_device_rag.eval.questions import Question
from spot_check_transcripts import select_spot_check_questions


def _question(question_id, source_type):
    return Question(
        question_id=question_id, source_type=source_type, category=None,
        question=f"Question {question_id}?", gold={}, notes=None,
    )


def test_select_spot_check_questions_draws_per_stratum_count():
    questions = (
        [_question(f"recall-{i}", "recall") for i in range(13)]
        + [_question(f"maude-{i}", "maude") for i in range(13)]
        + [_question(f"guidance-{i}", "guidance") for i in range(12)]
        + [_question(f"ifu-{i}", "ifu") for i in range(12)]
    )

    selected = select_spot_check_questions(questions, seed="test-seed", per_stratum=4)

    assert len(selected) == 16
    by_type = {}
    for q in selected:
        by_type.setdefault(q.source_type, []).append(q)
    assert {st: len(qs) for st, qs in by_type.items()} == {
        "recall": 4, "maude": 4, "guidance": 4, "ifu": 4,
    }


def test_select_spot_check_questions_is_deterministic_for_a_given_seed():
    questions = [_question(f"recall-{i}", "recall") for i in range(13)]

    first = select_spot_check_questions(questions, seed="fixed-seed", per_stratum=4)
    second = select_spot_check_questions(questions, seed="fixed-seed", per_stratum=4)

    assert [q.question_id for q in first] == [q.question_id for q in second]


def test_select_spot_check_questions_strata_draw_independently():
    recall_questions = [_question(f"recall-{i}", "recall") for i in range(13)]
    maude_questions = [_question(f"maude-{i}", "maude") for i in range(13)]

    recall_only = select_spot_check_questions(recall_questions, seed="test-seed", per_stratum=4)
    combined = select_spot_check_questions(recall_questions + maude_questions, seed="test-seed", per_stratum=4)
    recall_from_combined = [q for q in combined if q.source_type == "recall"]

    assert [q.question_id for q in recall_only] == [q.question_id for q in recall_from_combined]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/eval/test_spot_check_transcripts.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'spot_check_transcripts'`

- [ ] **Step 3: Write `scripts/spot_check_transcripts.py`**

```python
# scripts/spot_check_transcripts.py
"""Generates a markdown transcript for the manual generation-faithfulness
spot-check described in
docs/superpowers/specs/2026-07-28-fda-device-rag-architecture-design.md
section 6: a seeded, stratified subset of the frozen 50-question benchmark
(data/eval/questions.json), run through the same retrieve-then-generate
pipeline as ask.py, written to data/eval/spot_check_transcripts.md for a
human to grade pass/fail by hand. Grading itself is not automated -- no
LLM-judge for v1.

Usage: python scripts/spot_check_transcripts.py [--seed SEED]
"""
import argparse
import random
import sys
from pathlib import Path

from fda_device_rag.eval.questions import Question, load_questions
from fda_device_rag.generation.citations import extract_citations
from fda_device_rag.generation.ollama_client import (
    MODEL,
    OLLAMA_BASE_URL,
    OllamaError,
    check_ollama_ready,
    generate,
)
from fda_device_rag.generation.prompt import build_prompt
from fda_device_rag.retrieval.bootstrap import load_retrieval_stack

CHROMA_DIR = Path("data/chroma")
BM25_INDEX_PATH = Path("data/bm25_index.pkl")
QUESTIONS_PATH = Path("data/eval/questions.json")
OUTPUT_PATH = Path("data/eval/spot_check_transcripts.md")
# Distinct from sample_eval_candidates.py's DEFAULT_SEED so the benchmark's
# own sampling and this spot-check's sampling aren't correlated draws.
DEFAULT_SEED = "fda-device-rag-spotcheck-2026-08-04"
SOURCE_TYPES = ("recall", "maude", "guidance", "ifu")
PER_STRATUM = 4
TOP_K = 5


def select_spot_check_questions(questions: list[Question], seed: str, per_stratum: int = PER_STRATUM) -> list[Question]:
    """Draws `per_stratum` questions from each of the 4 source types in
    `questions`, without replacement, via a seed derived deterministically
    from (seed, source_type) -- same pattern as eval/sampling.py's
    per-category draws -- so each stratum's selection is reproducible and
    independent of the others."""
    selected = []
    for source_type in SOURCE_TYPES:
        pool = [q for q in questions if q.source_type == source_type]
        rng = random.Random(f"{seed}-spotcheck-{source_type}")
        selected.extend(rng.sample(pool, per_stratum))
    return selected


def _format_chunk_line(chunk) -> str:
    line = chunk.metadata.document_title
    if chunk.metadata.section_name:
        line += f" ({chunk.metadata.section_name})"
    return line


def _render_transcript(seed: str, entries: list[dict]) -> str:
    lines = [f"# Faithfulness Spot-Check Transcript (seed={seed!r})", ""]
    for entry in entries:
        lines.append(f"## {entry['question_id']} ({entry['source_type']})")
        lines.append("")
        lines.append(f"**Question:** {entry['question']}")
        lines.append("")
        lines.append("**Context provided:**")
        for number, chunk in entry["chunk_map"].items():
            lines.append(f"- [{number}] {_format_chunk_line(chunk)}")
        lines.append("")
        lines.append(f"**Generated answer:** {entry['answer']}")
        lines.append("")
        lines.append(f"**{entry['citation_label']}:**")
        for chunk in entry["citations"]:
            lines.append(f"- {_format_chunk_line(chunk)}")
        lines.append("")
        lines.append("**Verdict (pass/fail):** ")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=DEFAULT_SEED)
    args = parser.parse_args()

    # Same fail-fast rationale as ask.py: no point loading the index and
    # retrieving for 16 questions only to find Ollama unreachable on the
    # first generate() call.
    try:
        check_ollama_ready(OLLAMA_BASE_URL, MODEL)
    except OllamaError as e:
        print(str(e))
        sys.exit(1)

    if not BM25_INDEX_PATH.exists() or not CHROMA_DIR.exists():
        print(f"Index not found ({CHROMA_DIR}, {BM25_INDEX_PATH}) -- run "
              f"scripts/pull_corpus.py then scripts/build_index.py first.")
        sys.exit(1)

    questions = load_questions(QUESTIONS_PATH)
    selected = select_spot_check_questions(questions, args.seed)

    stack = load_retrieval_stack(CHROMA_DIR, BM25_INDEX_PATH)

    entries = []
    for question in selected:
        chunks = stack.hybrid_retriever.retrieve(question.question, top_k=TOP_K)
        prompt, chunk_map = build_prompt(question.question, chunks)
        try:
            answer = generate(prompt, OLLAMA_BASE_URL, MODEL)
        except OllamaError as e:
            # A batch of 16 -- one flaky mid-run failure shouldn't lose the
            # rest of the transcript. ask.py (a single question) aborts
            # instead; this script logs and continues.
            print(f"Skipping {question.question_id}: {e}")
            continue
        citations, citation_label = extract_citations(answer, chunk_map)
        entries.append({
            "question_id": question.question_id,
            "source_type": question.source_type,
            "question": question.question,
            "chunk_map": chunk_map,
            "answer": answer,
            "citations": citations,
            "citation_label": citation_label,
        })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(_render_transcript(args.seed, entries))

    print(f"Wrote {len(entries)}/{len(selected)} transcripts (seed={args.seed!r}) -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/eval/test_spot_check_transcripts.py -v`
Expected: `3 passed`

- [ ] **Step 5: Add the output path to `.gitignore`**

Add this line to `.gitignore`, next to the existing `data/eval/results.json` line:

```
data/eval/spot_check_transcripts.md
```

- [ ] **Step 6: Real-run check**

Ensure Ollama is running with `llama3` pulled.

Run: `.venv/Scripts/python.exe scripts/spot_check_transcripts.py`
Expected: exits 0; prints `Wrote 16/16 transcripts (seed='fda-device-rag-spotcheck-2026-08-04') -> data\eval\spot_check_transcripts.md`; the written file has one `##` section per question (16 total), each with non-empty `**Question:**` and `**Generated answer:**` lines and a blank `**Verdict (pass/fail):**` line.

- [ ] **Step 7: Document the script in README.md**

In the "Retrieval-accuracy benchmark" section (or immediately after it), add:

```markdown
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
```

- [ ] **Step 8: Commit**

```bash
git add scripts/spot_check_transcripts.py tests/eval/test_spot_check_transcripts.py .gitignore README.md
git commit -m "feat: add spot_check_transcripts.py for manual faithfulness grading"
```

---

## Self-Review

**Spec coverage:** §1 (nothing to implement, reference only) ✓. §2 module layout — `ollama_client.py`/`prompt.py`/`citations.py` (Tasks 2-4), `load_retrieval_stack` (Task 1), `ask.py` (Task 5), `spot_check_transcripts.py` (Task 6) ✓. §3 error handling — pre-flight ordering (Task 5 Step 1, Task 6 Step 3), `OllamaError` hierarchy (Task 2), fallback-labeling (Task 4, already implemented) ✓. §4 faithfulness spot-check tooling ✓ (Task 6). §5 testing table — every row matches a task's testing decision exactly (checked against the table line by line above). §6 out of scope — nothing in this plan touches `run_eval.py`'s metrics, adds automation, or builds a served API ✓.

**Placeholder scan:** no TBD/TODO; every code step has complete code; every verification step has a real expected value (captured from the live corpus/Ollama server, not invented).

**Type consistency:** `build_prompt` (Task 3) returns `dict[int, ScoredChunk]`; `extract_citations` (Task 4) takes `chunk_map: dict[int, ScoredChunk]` — matches. `load_retrieval_stack` (Task 1) returns `RetrievalStack` with `.hybrid_retriever` used identically in `query.py`, `ask.py`, and `spot_check_transcripts.py`. `check_ollama_ready`/`generate` signatures (Task 2) match their call sites in Tasks 5 and 6 exactly (`(OLLAMA_BASE_URL, MODEL)` positional). `OllamaError` is the single catch type used in both `ask.py` and `spot_check_transcripts.py`, per the Global Constraints.
