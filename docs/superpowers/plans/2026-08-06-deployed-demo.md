# Deployed Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a public, live Q&A demo on Streamlit Community Cloud — retrieval over the real corpus plus Groq-hosted generation with citations — de-risked with a real-workload smoke test before generation is added.

**Architecture:** Two milestones, not six independent tasks in a row. **Milestone A** (Tasks 1-4) builds everything needed for a retrieval-only public demo and ends at **Checkpoint 1** — a controller-and-user-executed deployment where the real corpus/index goes to a Hugging Face Dataset, the retrieval-only app deploys to a real Streamlit Community Cloud app, and Streamlit's unpublished memory limits get tested against the real ~1,850-chunk workload. Only if that survives does **Milestone B** (Tasks 5-6) add Groq generation, citations, and the About/Benchmark section, ending at **Checkpoint 2** — pasting the Groq key into Streamlit's Secrets UI and verifying the full demo live. Checkpoints are explicitly **not** subagent-implementer tasks: they involve external accounts (Hugging Face, Groq, Streamlit Community Cloud's GitHub OAuth) that either require the user directly (account creation, OAuth, secret-pasting) or a live deploy of code that must already be merged to `master` first.

**Tech Stack:** Streamlit (new), Groq (new, hosted inference), Hugging Face Hub (`huggingface_hub`, new — dataset hosting), `tomllib` (stdlib, dependency-file generation).

## Global Constraints

- **`requirements.txt` is generated, never hand-edited.** Single source of truth is `pyproject.toml`; `scripts/generate_requirements.py` derives it. CI enforces this with `--check` — never edit `requirements.txt` directly in any task.
- **Demo-only dependency versions, verified at design time, not guessed:** `streamlit>=1.61`, `huggingface_hub>=1.26` (checked against `pip index versions` before this plan was written).
- **Groq model, verified live against Groq's production catalog, not memory:** `llama-3.1-8b-instant`. This is a **documented substitution** for the locally-benchmarked `llama3:latest` (Groq's catalog no longer hosts plain llama3) — every place the model is referenced in UI copy or docs must say so, not present it silently as the same model.
- **Groq endpoint, verified against Groq's own docs:** `https://api.groq.com/openai/v1/chat/completions`, `Authorization: Bearer <key>` header, OpenAI-compatible body (`model`, `messages`), response text at `choices[0].message.content`.
- **`groq_client.py` has no pre-flight "is it ready" check**, unlike `ollama_client.py`'s `check_ollama_ready`. Groq is a hosted, generally-always-up API — the "is the local server running" failure mode that justified Ollama's pre-flight check doesn't apply. `generate()` alone, wrapped in try/except, is the whole module — this is a deliberate scope decision, not an oversight.
- **`generate()`'s untested status matches `ollama_client.py`'s own precedent**: single linear POST-and-unwrap, no branching worth pinning. Do not add tests for it.
- **HF Dataset upload is scoped to what `load_retrieval_stack()` actually needs at query time — `data/chroma/` and `data/bm25_index.pkl` only, not `data/raw/`.** `data/raw/` is only needed to rebuild the index from scratch, which the deployed demo never does. This is a deliberate refinement from the design doc's literal list (which named all three), made because including `data/raw/` would upload/download real weight the demo never uses.
- **HF Dataset repo id:** `marshfalex/fda-device-rag-corpus`.
- **Session cap:** 10 questions per browser session (`st.session_state`), Milestone B only.
- **Checkpoints are controller + user, not delegated implementer tasks.** Do not dispatch a subagent for the "Checkpoint 1" or "Checkpoint 2" sections — they involve real external accounts and OAuth flows an isolated subagent cannot perform and should not be trusted with.
- **Never handle the Groq API key or an HF token as plain text in chat.** The user authenticates the `hf` CLI locally (`hf auth login`, or an `HF_TOKEN` environment variable they set themselves) and pastes the Groq key directly into Streamlit Cloud's Secrets UI — neither value is ever pasted into the conversation.

---

## File Structure

```
pyproject.toml                              [MODIFIED] add [project.optional-dependencies].demo group
scripts/generate_requirements.py            [NEW] pure build_requirements_content() + CLI wrapper
tests/test_generate_requirements.py         [NEW] unit tests for the pure function
requirements.txt                            [NEW] generated artifact, committed
.github/workflows/ci.yml                    [MODIFIED] add --check step
scripts/upload_corpus_to_hf.py              [NEW] uploads data/chroma/ + bm25_index.pkl to the HF Dataset
app.py                                      [NEW] the Streamlit app itself — Milestone A: retrieval only; Milestone B: + Groq generation, citations, About/Benchmark, session cap
src/fda_device_rag/generation/groq_client.py [NEW] GroqError, generate() — Milestone B
.gitignore                                  [MODIFIED] add .streamlit/secrets.toml — Milestone B
```

---

## Milestone A

### Task 1: `requirements.txt` generation — pure logic + tests

**Files:**
- Modify: `pyproject.toml`
- Create: `scripts/generate_requirements.py`
- Test: `tests/test_generate_requirements.py`

**Interfaces:**
- Produces: `build_requirements_content(pyproject_data: dict) -> str` (pure function, importable as `from generate_requirements import build_requirements_content`, matching this project's existing precedent of testing script-level pure functions directly via `pythonpath = ["src", "scripts"]`). `main()` (CLI wrapper, not directly tested). Consumed by Task 2 (regenerating the real `requirements.txt`) and by CI (Task 2's workflow step).

- [ ] **Step 1: Add the `demo` extras group to `pyproject.toml`**

Change:

```toml
[project.optional-dependencies]
dev = ["pytest>=8.0", "ruff>=0.16,<0.17"]
```

to:

```toml
[project.optional-dependencies]
dev = ["pytest>=8.0", "ruff>=0.16,<0.17"]
demo = ["streamlit>=1.61", "huggingface_hub>=1.26"]
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_generate_requirements.py`:

```python
from generate_requirements import build_requirements_content


def test_build_requirements_content_includes_core_deps_and_demo_extras():
    pyproject_data = {
        "project": {
            "dependencies": ["chromadb>=0.5", "requests>=2.31"],
            "optional-dependencies": {
                "dev": ["pytest>=8.0", "ruff>=0.16,<0.17"],
                "demo": ["streamlit>=1.61", "huggingface_hub>=1.26"],
            },
        }
    }

    content = build_requirements_content(pyproject_data)

    assert "chromadb>=0.5" in content
    assert "requests>=2.31" in content
    assert "streamlit>=1.61" in content
    assert "huggingface_hub>=1.26" in content
    # dev-only extras must not leak into the deployed demo's requirements
    assert "pytest>=8.0" not in content
    assert "ruff" not in content


def test_build_requirements_content_is_one_package_per_line():
    pyproject_data = {
        "project": {
            "dependencies": ["chromadb>=0.5"],
            "optional-dependencies": {"demo": ["streamlit>=1.61"]},
        }
    }

    content = build_requirements_content(pyproject_data)

    lines = content.strip().split("\n")
    assert lines == ["chromadb>=0.5", "streamlit>=1.61"]
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_generate_requirements.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'generate_requirements'`

- [ ] **Step 4: Write `scripts/generate_requirements.py`**

```python
# scripts/generate_requirements.py
"""Generates requirements.txt from pyproject.toml's core dependencies plus
the `demo` optional-dependencies group, so Streamlit Community Cloud (which
checks requirements.txt before pyproject.toml, and only supports Poetry-
style pyproject.toml dependency blocks -- this project uses plain PEP 621
[project.dependencies], not Poetry) has a dependency file it can use
without relying on untested parser compatibility.

pyproject.toml stays the single source of truth. Run this script after any
dependency change and commit the result; `--check` (used by CI) fails
instead of writing if the committed file doesn't match what this script
would currently generate.

Usage: python scripts/generate_requirements.py [--check]
"""
import argparse
import sys
import tomllib
from pathlib import Path

PYPROJECT_PATH = Path("pyproject.toml")
REQUIREMENTS_PATH = Path("requirements.txt")


def build_requirements_content(pyproject_data: dict) -> str:
    """Pure function: takes already-parsed pyproject.toml data (as
    tomllib.load/tomllib.loads would produce) and returns the exact
    requirements.txt text -- core project.dependencies plus
    project.optional-dependencies.demo, one package specifier per line.
    No file IO, no argv -- testable in isolation from the CLI wrapper."""
    core_deps = pyproject_data["project"]["dependencies"]
    demo_deps = pyproject_data["project"]["optional-dependencies"]["demo"]
    return "\n".join(core_deps + demo_deps) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    pyproject_data = tomllib.loads(PYPROJECT_PATH.read_text())
    content = build_requirements_content(pyproject_data)

    if args.check:
        existing = REQUIREMENTS_PATH.read_text() if REQUIREMENTS_PATH.exists() else ""
        if existing != content:
            print(f"{REQUIREMENTS_PATH} is out of date with {PYPROJECT_PATH} -- "
                  f"run 'python scripts/generate_requirements.py' and commit the result.")
            sys.exit(1)
        print(f"{REQUIREMENTS_PATH} matches {PYPROJECT_PATH}.")
    else:
        REQUIREMENTS_PATH.write_text(content)
        print(f"Wrote {REQUIREMENTS_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_generate_requirements.py -v`
Expected: `2 passed`

- [ ] **Step 6: Run the full suite to confirm no regressions**

Run: `.venv/Scripts/python.exe -m pytest tests/ -q`
Expected: `146 passed` (144 baseline + 2 new)

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml scripts/generate_requirements.py tests/test_generate_requirements.py
git commit -m "feat: add generate_requirements.py (pyproject.toml -> requirements.txt)"
```

---

### Task 2: Generate the real `requirements.txt`, wire into CI

**Files:**
- Create: `requirements.txt` (generated, not hand-written)
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: `scripts/generate_requirements.py`'s `main()` (Task 1).

- [ ] **Step 1: Generate the real `requirements.txt`**

Run: `.venv/Scripts/python.exe scripts/generate_requirements.py`
Expected: `Wrote requirements.txt`

- [ ] **Step 2: Verify its contents**

Run: `cat requirements.txt`
Expected: one line per package — `chromadb>=0.5`, `sentence-transformers>=3.0`, `rank_bm25>=0.2.2`, `requests>=2.31`, `pypdf>=4.0`, `langchain-text-splitters>=0.2`, `streamlit>=1.61`, `huggingface_hub>=1.26` (8 lines — the 6 core dependencies plus the 2 demo extras from Task 1; `pytest`/`ruff` must NOT appear).

- [ ] **Step 3: Verify `--check` passes against what was just written**

Run: `.venv/Scripts/python.exe scripts/generate_requirements.py --check`
Expected: `requirements.txt matches pyproject.toml.`

- [ ] **Step 4: Add the `--check` step to the CI workflow**

In `.github/workflows/ci.yml`, change:

```yaml
      - run: pip install -e ".[dev]"
      - run: ruff check .
      - run: pytest
```

to:

```yaml
      - run: pip install -e ".[dev]"
      - run: ruff check .
      - run: pytest
      - run: python scripts/generate_requirements.py --check
```

- [ ] **Step 5: Validate the YAML is still well-formed**

Run: `.venv/Scripts/python.exe -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml')); print('valid YAML')"`
Expected: `valid YAML`

- [ ] **Step 6: Commit**

```bash
git add requirements.txt .github/workflows/ci.yml
git commit -m "chore: generate requirements.txt, add drift-check to CI"
```

---

### Task 3: Corpus/index upload script

**Files:**
- Create: `scripts/upload_corpus_to_hf.py`

**Interfaces:** none — this is a thin IO wrapper (like `pull_corpus.py`/`build_index.py`), untested by this project's established convention, verified by a real run instead (which happens at Checkpoint 1, not in this task — this task only needs the code to exist and be syntactically correct, since actually running it requires real `hf` CLI authentication the implementer subagent will not have).

- [ ] **Step 1: Write `scripts/upload_corpus_to_hf.py`**

```python
# scripts/upload_corpus_to_hf.py
"""One-time (or re-run-when-the-corpus-changes) upload of the pre-built
retrieval index to a Hugging Face Hub Dataset repo, so the deployed demo
(app.py) can download it at startup instead of rebuilding against openFDA
live on every cold start.

Uploads only what load_retrieval_stack() actually needs at query time --
data/chroma/ and data/bm25_index.pkl -- not data/raw/, which is only
needed to rebuild the index from scratch, not to query an already-built
one.

Requires an authenticated `hf` CLI session (run `hf auth login`, or set
the HF_TOKEN environment variable, before running this script) -- this
script never accepts a token as a command-line argument or prompts for
one, so a credential is never visible in shell history or this script's
output.

Usage: python scripts/upload_corpus_to_hf.py [--repo-id REPO_ID]
"""
import argparse
from pathlib import Path

from huggingface_hub import HfApi

DEFAULT_REPO_ID = "marshfalex/fda-device-rag-corpus"
CHROMA_DIR = Path("data/chroma")
BM25_INDEX_PATH = Path("data/bm25_index.pkl")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    args = parser.parse_args()

    if not CHROMA_DIR.exists() or not BM25_INDEX_PATH.exists():
        print(f"Index not found ({CHROMA_DIR}, {BM25_INDEX_PATH}) -- run "
              f"scripts/pull_corpus.py then scripts/build_index.py first.")
        raise SystemExit(1)

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        repo_id=args.repo_id, repo_type="dataset",
        folder_path=str(CHROMA_DIR), path_in_repo="chroma",
    )
    api.upload_file(
        repo_id=args.repo_id, repo_type="dataset",
        path_or_fileobj=str(BM25_INDEX_PATH), path_in_repo="bm25_index.pkl",
    )

    print(f"Uploaded {CHROMA_DIR} and {BM25_INDEX_PATH} -> "
          f"https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it imports cleanly**

Run: `.venv/Scripts/python.exe -c "import ast; ast.parse(open('scripts/upload_corpus_to_hf.py').read()); print('syntax OK')"`
Expected: `syntax OK`

(`huggingface_hub` isn't installed in this worktree's venv yet — Task 4 installs it. This step only checks the script parses; Checkpoint 1 is where it actually runs, with real `hf` auth the implementer doesn't have.)

- [ ] **Step 3: Commit**

```bash
git add scripts/upload_corpus_to_hf.py
git commit -m "feat: add scripts/upload_corpus_to_hf.py"
```

---

### Task 4: `app.py` — retrieval-only Streamlit demo

**Files:**
- Create: `app.py`

**Interfaces:**
- Consumes: `load_retrieval_stack` (`src/fda_device_rag/retrieval/bootstrap.py`, from the citation-grounding phase — unchanged), `huggingface_hub.snapshot_download`.

No tests — thin UI wrapper over already-tested `load_retrieval_stack`, matching `query.py`/`ask.py`'s convention. Verified at Checkpoint 1 by a real deploy, not here.

- [ ] **Step 1: Install `huggingface_hub` and `streamlit` locally for iteration**

Run: `.venv/Scripts/python.exe -m pip install -e ".[demo]"`

- [ ] **Step 2: Write `app.py`**

```python
# app.py
"""Public demo: Q&A over the FDA device corpus (recalls, adverse events,
guidance, IFUs). This is the Milestone A version -- retrieval only, no
generation yet. Downloads the pre-built corpus/index from a Hugging Face
Dataset at startup (this deployed environment starts with none of the
gitignored corpus/index files -- see
docs/superpowers/specs/2026-08-06-deployed-demo-design.md section 5).

Generation (Groq-backed, with citations) is added in a later task, gated
on this retrieval-only version surviving a real Streamlit Community Cloud
deploy under the real corpus/index workload -- see that design doc's
section 6.
"""
from pathlib import Path

import streamlit as st
from huggingface_hub import snapshot_download

from fda_device_rag.retrieval.bootstrap import load_retrieval_stack

HF_DATASET_REPO = "marshfalex/fda-device-rag-corpus"
TOP_K = 5


@st.cache_resource
def get_retrieval_stack():
    local_dir = Path(snapshot_download(repo_id=HF_DATASET_REPO, repo_type="dataset"))
    return load_retrieval_stack(local_dir / "chroma", local_dir / "bm25_index.pkl")


st.title("FDA Device RAG")
st.caption(
    "Q&A over public FDA device recalls, adverse event reports, guidance "
    "documents, and manufacturer instructions for use."
)

question = st.text_input(
    "Ask a question about FDA device recalls, adverse events, guidance, or IFUs:"
)

if question:
    stack = get_retrieval_stack()
    results = stack.hybrid_retriever.retrieve(question, top_k=TOP_K)

    st.subheader(f"Top {len(results)} retrieved passages")
    for rank, result in enumerate(results, start=1):
        header = f"{rank}. {result.metadata.document_title} (score={result.score:.4f})"
        with st.expander(header):
            st.write(f"**Source type:** {result.metadata.source_type}")
            if result.metadata.section_name:
                st.write(f"**Section:** {result.metadata.section_name}")
            st.write(f"**URL:** {result.metadata.source_url}")
            preview = result.text[:500]
            if len(result.text) > 500:
                preview += "..."
            st.write(preview)
```

- [ ] **Step 3: Run it locally to confirm it boots without a Python error**

Run:

```bash
.venv/Scripts/python.exe -m streamlit run app.py --server.headless true &
STREAMLIT_PID=$!
sleep 5
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8501
kill $STREAMLIT_PID
```

Expected: `200`. This confirms the app boots and serves a page — a real Python import/syntax error in `app.py` would make Streamlit fail to start and this would show a connection error instead of `200`. It will still fail at the `get_retrieval_stack()` call if you actually submit a question through the browser, since `marshfalex/fda-device-rag-corpus` doesn't exist on Hugging Face yet (that happens at Checkpoint 1) — that's expected at this stage; this step only verifies the app itself boots and serves its initial page.

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat: add app.py, retrieval-only Streamlit demo"
```

---

## Checkpoint 1 (controller + user — not a delegated task)

**Do not dispatch a subagent for this section.** Everything here either requires the user directly (account/OAuth/secrets) or requires code already merged to `master` (Streamlit Community Cloud deploys from the real GitHub repo, not a worktree branch).

1. **Merge Milestone A to `master`** via `superpowers:finishing-a-development-branch`, same pattern as the citation-grounding and CI phases (verify tests, merge, clean up worktree). Push to `origin/master` (already configured from the CI phase).
2. **User authenticates the `hf` CLI**: run `hf auth login` in their own terminal, or set `HF_TOKEN` as a local environment variable — never paste the token into chat. Confirm with `hf auth whoami`.
3. **Controller runs the upload script** from the main checkout (which already has the real built `data/chroma/`/`data/bm25_index.pkl` from earlier phases): `python scripts/upload_corpus_to_hf.py`. Confirm it prints the expected `https://huggingface.co/datasets/marshfalex/fda-device-rag-corpus` URL and that the repo is visibly populated (chroma files + bm25_index.pkl) when checked.
4. **User connects Streamlit Community Cloud to GitHub** (OAuth, browser — `share.streamlit.io`, or the current Community Cloud entry point) and deploys `app.py` from `marshfalex/fda-device-rag`'s `master` branch.
5. **Verify the real deploy survives the real workload** — the entire point of sequencing it this way (see design doc §6): once the user has the deployed URL, the controller navigates to it (Browser tool) and submits a real question. Confirm: the app loads without a "This app has gone over its resource limits" error, the question returns real top-5 results (not an error), and the results look sane (real document titles/scores, not empty). If it fails here, stop — this is a real go/no-go gate on the entire Groq-integration milestone, not a soft check.

Only proceed to Milestone B once step 5 passes.

---

## Milestone B

### Task 5: `groq_client.py`

**Files:**
- Create: `src/fda_device_rag/generation/groq_client.py`

**Interfaces:**
- Produces: `GroqError`, `generate(prompt: str, api_key: str, base_url: str = GROQ_BASE_URL, model: str = GROQ_MODEL, timeout: int = 60) -> str`. Consumed by Task 6 (`app.py`'s generation extension).

No tests — single linear POST-and-unwrap, matching `ollama_client.py`'s own `generate()` precedent (Global Constraints).

- [ ] **Step 1: Write `src/fda_device_rag/generation/groq_client.py`**

```python
"""Thin HTTP client for Groq's OpenAI-compatible chat completions API --
direct requests calls, not the groq SDK, matching this project's stance on
hand-rolling simple HTTP wrappers (see ollama_client.py's identical
rationale) even though Groq's own docs recommend their SDK.

Verified against Groq's docs before this module was written: base URL
https://api.groq.com/openai/v1, chat completions endpoint appends
/chat/completions, OpenAI-compatible request/response shape (model,
messages -> choices[0].message.content).

Unlike ollama_client.py, there is no pre-flight "is it ready" check here --
Groq is a hosted, generally-always-up API, so the "is the local server
running" failure mode that justifies Ollama's check_ollama_ready doesn't
apply. This module is deliberately just generate() and an error type.
"""
import requests

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "llama-3.1-8b-instant"


class GroqError(Exception):
    """Raised when a Groq API call fails -- connection error, timeout,
    non-200 response, or an unexpected response shape."""


def generate(prompt: str, api_key: str, base_url: str = GROQ_BASE_URL, model: str = GROQ_MODEL, timeout: int = 60) -> str:
    """POSTs an OpenAI-compatible chat completion request. Any failure
    (timeout, connection drop, non-200 response, unexpected response
    shape) raises GroqError -- a single linear POST-and-unwrap with no
    branching worth unit-testing, same rationale as ollama_client.py's
    generate()."""
    try:
        response = requests.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": model, "messages": [{"role": "user", "content": prompt}]},
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except (requests.RequestException, KeyError, IndexError) as e:
        raise GroqError(f"Generation failed: {e}")
```

- [ ] **Step 2: Run the full test suite to confirm no regressions**

Run: `.venv/Scripts/python.exe -m pytest tests/ -q`
Expected: `146 passed` (unchanged — this task adds no tests, matching its untested-by-design status)

- [ ] **Step 3: Commit**

```bash
git add src/fda_device_rag/generation/groq_client.py
git commit -m "feat: add groq_client.py for hosted generation"
```

---

### Task 6: Extend `app.py` — generation, citations, About/Benchmark, session cap

**Files:**
- Modify: `app.py`

**Interfaces:**
- Consumes: `groq_client.generate`/`GroqError`/`GROQ_MODEL` (Task 5), `generation.prompt.build_prompt`, `generation.citations.extract_citations` (both from the citation-grounding phase, unchanged, provider-agnostic).

No tests — same rationale as Task 4 (thin UI wrapper over already-tested lower-level functions). Verified at Checkpoint 2 by a real deploy.

- [ ] **Step 1: Rewrite `app.py` with generation added**

Replace the file's contents entirely with:

```python
# app.py
"""Public demo: Q&A over the FDA device corpus (recalls, adverse events,
guidance, IFUs), with Groq-hosted generation and citation extraction.
Downloads the pre-built corpus/index from a Hugging Face Dataset at
startup -- see
docs/superpowers/specs/2026-08-06-deployed-demo-design.md section 5.

Generation uses llama-3.1-8b-instant via Groq -- a documented substitution
for the locally-benchmarked llama3:latest (Groq's production catalog no
longer hosts plain llama3; see that design doc's section 4). Local dev/eval
(scripts/ask.py, scripts/spot_check_transcripts.py) still uses Ollama with
llama3:latest, unchanged by this file.
"""
from pathlib import Path

import streamlit as st
from huggingface_hub import snapshot_download

from fda_device_rag.generation.citations import extract_citations
from fda_device_rag.generation.groq_client import GROQ_MODEL, GroqError, generate
from fda_device_rag.generation.prompt import build_prompt
from fda_device_rag.retrieval.bootstrap import load_retrieval_stack

HF_DATASET_REPO = "marshfalex/fda-device-rag-corpus"
TOP_K = 5
MAX_QUESTIONS_PER_SESSION = 10


@st.cache_resource
def get_retrieval_stack():
    local_dir = Path(snapshot_download(repo_id=HF_DATASET_REPO, repo_type="dataset"))
    return load_retrieval_stack(local_dir / "chroma", local_dir / "bm25_index.pkl")


def _format_chunk_line(chunk) -> str:
    line = chunk.metadata.document_title
    if chunk.metadata.section_name:
        line += f" ({chunk.metadata.section_name})"
    return line


st.title("FDA Device RAG")
st.caption(
    "Q&A over public FDA device recalls, adverse event reports, guidance "
    "documents, and manufacturer instructions for use, with citations "
    "traceable to the source record or document section."
)

with st.expander("About this demo / retrieval-accuracy benchmark"):
    st.markdown(
        f"""
This demo answers using **{GROQ_MODEL}** via Groq — a documented
substitution for the `llama3:latest` model actually used in the local
retrieval-accuracy benchmark below (Groq's hosted catalog no longer
includes plain llama3; `llama-3.1-8b-instant` is the closest available
match, same 8B size class).

**Pooled retrieval results, n=50 frozen benchmark questions** (full
methodology: [README](https://github.com/marshfalex/fda-device-rag)):

| Leg | Hit Rate@5 | 95% CI | MRR |
|---|---|---|---|
| Dense-only | 76.0% | [62.6, 85.7] | 0.638 |
| BM25-only | 78.0% | [64.8, 87.2] | 0.630 |
| Hybrid (RRF) | 86.0% | [73.8, 93.0] | 0.657 |

Hybrid had the highest pooled point estimate, but at n=50 the 95% Wilson
intervals for all three legs overlap substantially — this is a
**directional signal, not a statistically supported win**. See the
[full write-up](https://github.com/marshfalex/fda-device-rag#retrieval-accuracy-benchmark)
for the complete per-stratum breakdown and reasoning.
"""
    )

if "question_count" not in st.session_state:
    st.session_state.question_count = 0

question = st.text_input(
    "Ask a question about FDA device recalls, adverse events, guidance, or IFUs:"
)

if question:
    if st.session_state.question_count >= MAX_QUESTIONS_PER_SESSION:
        st.warning(
            f"You've reached the {MAX_QUESTIONS_PER_SESSION}-question limit "
            f"for this session. Refresh the page to reset."
        )
    else:
        st.session_state.question_count += 1

        stack = get_retrieval_stack()
        chunks = stack.hybrid_retriever.retrieve(question, top_k=TOP_K)
        prompt, chunk_map = build_prompt(question, chunks)

        groq_api_key = st.secrets.get("GROQ_API_KEY")
        if not groq_api_key:
            st.error("GROQ_API_KEY is not configured for this deployment.")
        else:
            try:
                answer = generate(prompt, groq_api_key)
            except GroqError as e:
                st.error(f"Generation failed: {e}")
            else:
                citations, citation_label = extract_citations(answer, chunk_map)

                st.subheader("Answer")
                st.write(answer)

                st.subheader(citation_label)
                for chunk in citations:
                    st.write(f"- {_format_chunk_line(chunk)} — {chunk.metadata.source_url}")

                st.subheader("Context provided (full top-5)")
                for number, chunk in chunk_map.items():
                    st.write(f"[{number}] {_format_chunk_line(chunk)}")
```

- [ ] **Step 2: Run the full test suite to confirm no regressions**

Run: `.venv/Scripts/python.exe -m pytest tests/ -q`
Expected: `146 passed`

- [ ] **Step 3: Verify the app still boots locally**

Run the same start/curl/kill sequence as Task 4 Step 3:

```bash
.venv/Scripts/python.exe -m streamlit run app.py --server.headless true &
STREAMLIT_PID=$!
sleep 5
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8501
kill $STREAMLIT_PID
```

Expected: `200`. This is a real boot, not just a syntax check — it exercises every new import this task added (`groq_client`, `citations`, `prompt`), so a typo'd import path would show up as a connection failure here rather than surfacing later at Checkpoint 2. Submitting a question through the browser will still fail without a real `GROQ_API_KEY` in local `.streamlit/secrets.toml` — that's expected; this step only verifies the app boots.

- [ ] **Step 4: Add local Streamlit secrets to `.gitignore`**

This task is the first point a local `GROQ_API_KEY` becomes relevant (for
manually testing generation locally, per Step 3's note) — add the line
before that file could ever be created, not after. Append to `.gitignore`:

```
.streamlit/secrets.toml
```

- [ ] **Step 5: Commit**

```bash
git add app.py .gitignore
git commit -m "feat: add Groq generation, citations, About/Benchmark section, session cap to app.py"
```

---

## Checkpoint 2 (controller + user — not a delegated task)

1. **Final whole-branch review + merge Milestone B to `master`** via the same subagent-driven-development + `finishing-a-development-branch` pattern as Milestone A and prior phases. Push to `origin/master` — Streamlit Community Cloud auto-redeploys on push to the connected branch.
2. **User creates a Groq account and API key** (`console.groq.com`) — the controller cannot create accounts.
3. **User pastes `GROQ_API_KEY` into the Streamlit Community Cloud app's Secrets UI** (Advanced settings → Secrets, TOML format: `GROQ_API_KEY = "..."`) — never through chat.
4. **Verify the full demo live**: controller navigates to the deployed URL (Browser tool), submits a real question, confirms a real Groq-generated answer appears with both "cited by the model"/"context provided (model did not cite specific passages)" and "context provided (full top-5)" sections, and that the About/Benchmark section renders the correct numbers. Submit 10 questions in one session and confirm the 11th shows the session-cap warning instead of another generation call.
5. **Update `README.md`**: add the live demo link, flip the Roadmap table's Phase 4 row to `Done`, update the status banner (drop "The deployed demo is the one remaining phase and is not implemented yet"). Commit and push directly (small doc-only change, same pattern as the CI phase's post-verification README commit).

---

## Self-Review

**Spec coverage:** Design doc §1 (Streamlit, UI surfaces) — Tasks 4/6. §2 (hosting platform) — Checkpoint 1 step 4. §3 (requirements.txt) — Tasks 1-2, including the test-coverage fix requested during spec review. §4 (Groq, model substitution) — Task 5, and Task 6's About section states the substitution explicitly. §5 (HF Dataset) — Task 3, Checkpoint 1 steps 2-3 (with the data/raw/ scoping refinement stated as a Global Constraint). §6 (real-workload smoke test) — Checkpoint 1 step 5, explicitly sequenced after the real upload (step 3) and before Milestone B begins. §7 (About/Benchmark UI) — Task 6. §8 (session cap) — Task 6. §9 (secrets) — Checkpoint 2 steps 2-3, plus `.streamlit/secrets.toml` added to `.gitignore` in Task 6 (a real gap caught in this self-review — the design doc named it but no task originally added it). §10 (out of scope) — nothing in this plan touches `ask.py`/`spot_check_transcripts.py`/Ollama, adds a paid tier, or rebuilds the benchmark.

**Placeholder scan:** no TBD/TODO; every code step has complete code; both design-review fixes (real-workload smoke test sequencing, generate_requirements.py test coverage) are reflected as actual tasks/steps, not notes.

**Type consistency:** `build_requirements_content(pyproject_data: dict) -> str` (Task 1) is called identically by `main()` in the same file and exercised identically by both test functions. `groq_client.generate(prompt, api_key, base_url=..., model=..., timeout=...) -> str` (Task 5) matches its call site in Task 6 (`generate(prompt, groq_api_key)`, using the two required positional args and both defaults). `load_retrieval_stack(chroma_dir, bm25_index_path)` (unchanged from the citation-grounding phase) is called identically in Tasks 4 and 6. `build_prompt`/`extract_citations` signatures (unchanged from the citation-grounding phase) match their Task 6 call sites exactly.
