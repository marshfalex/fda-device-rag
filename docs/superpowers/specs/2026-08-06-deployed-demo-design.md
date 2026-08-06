# Deployed Demo — Design

Phase 4 of the architecture doc's scope
([`docs/superpowers/specs/2026-07-28-fda-device-rag-architecture-design.md`](2026-07-28-fda-device-rag-architecture-design.md)
§7: Streamlit, deployed publicly, UI surfaces = question input, generated
answer, "cited by the model" vs. "context provided" citations, source
metadata/links). The architecture doc assumed Ollama would be reachable
wherever the demo runs — it wasn't written with a free-hosting-tier
constraint on running a local LLM server in mind. This document resolves
that gap and everything else §7 left open.

## 1. What's Already Decided (from the architecture doc, not reopened here)

- **Streamlit** as the UI framework — chosen because "the actual phase-4
  deliverable is a public demo link, not a service architecture."
- **UI surfaces**: question input, generated answer, "cited by the model"
  passages labeled distinctly from "context provided" (per the
  citation-grounding design, already implemented in `scripts/ask.py`),
  source metadata/links per citation.

## 2. Hosting Platform — Streamlit Community Cloud, Reconfirmed

Two platforms were evaluated live, not from documentation alone:

- **Hugging Face Spaces was explored and rejected.** HF's own docs
  contradicted each other (a dedicated Streamlit-SDK docs page implied it
  was still free; the Spaces Overview page said Gradio/Docker Spaces
  "require a paid plan"). Neither web research nor official docs resolved
  this — the actual live "New Space" creation UI was checked directly (a
  real screenshot, not a docs page): **only Static Spaces are free.**
  Gradio and Docker both prompt to subscribe to PRO. Streamlit isn't listed
  as its own SDK option anymore (folded into Docker, which is paid-gated).
  Static is client-side only — not viable for a Python backend serving
  retrieval + generation.
- **Streamlit Community Cloud is the fallback and the final choice** —
  the architecture doc's original pick, reconfirmed as correct once HF
  Spaces was ruled out empirically rather than assumed viable.

## 3. Dependency Management — `requirements.txt`, Generated From `pyproject.toml`, CI-Enforced

Streamlit Community Cloud's dependency detection (verified against its own
docs) checks `requirements.txt` before `pyproject.toml`, and explicitly
labels its `pyproject.toml` support "(poetry)". This project's
`pyproject.toml` uses standard PEP 621 `[project.dependencies]` with a
setuptools backend, not Poetry's `[tool.poetry.dependencies]` — relying on
Streamlit Cloud's `pyproject.toml` parser would be gambling on untested
compatibility. **A committed `requirements.txt` at the repo root removes
that risk entirely.**

To avoid `requirements.txt` silently drifting from `pyproject.toml` (a
version bump in one going unnoticed in the other, meaning the deployed demo
runs different dependency versions than what CI actually tests and
ruff-checks), `pyproject.toml` remains the single source of truth and
`requirements.txt` is a **generated, CI-verified artifact**, not a
second hand-maintained list:

- New `[project.optional-dependencies] demo = ["streamlit>=1.61",
  "huggingface_hub>=1.26"]` group in `pyproject.toml` (versions verified
  against `pip index versions` at design time, not guessed) — the two
  packages the demo needs beyond the core library's existing
  `dependencies`.
- New `scripts/generate_requirements.py` — reads `pyproject.toml` via
  stdlib `tomllib` (no extra dependency), writes `project.dependencies` +
  `project.optional-dependencies.demo` (core dependencies plus the two
  demo-only packages — the full core list, not a hand-pruned subset, so a
  future core-library import can't silently break the demo because
  `requirements.txt` was trimmed too aggressively) to `requirements.txt`,
  one package per line. Supports `--check`: instead of writing, compares
  generated content against the committed file and exits nonzero with a
  message if they differ.
- `.github/workflows/ci.yml` gets one more step:
  `python scripts/generate_requirements.py --check` — a forgotten
  regeneration after a `pyproject.toml` dependency change now fails CI,
  the same gate that already catches lint/test regressions, rather than
  silently shipping a stale file to the next deploy.

## 4. Generation — Groq, `llama-3.1-8b-instant`, Documented Model Substitution

**Provider: Groq**, chosen for its free tier and inference speed (keeps the
live demo responsive). Local dev/eval (`ask.py`, `spot_check_transcripts.py`,
the faithfulness spot-check) keeps using Ollama entirely unchanged — this is
additive, not a replacement of the existing generation pipeline.

**Model mismatch, verified and documented, not glossed over:** Groq's
production model catalog (checked live against
`https://console.groq.com/docs/models`, not assumed from memory) no longer
includes plain `llama3`. The two current options are `llama-3.1-8b-instant`
(8B — same size class as the locally-benchmarked `llama3:latest`) and
`llama-3.3-70b-versatile` (70B — a much larger behavioral gap from what was
actually measured). **Decision: `llama-3.1-8b-instant`**, with the UI and
README explicitly stating the deployed model differs from the
locally-benchmarked one and why (Groq doesn't host it), preserving rough
relevance of the local latency/quality reasoning rather than silently
presenting a different model under the same name.

**New `src/fda_device_rag/generation/groq_client.py`** — mirrors
`ollama_client.py`'s shape (an error hierarchy, a `generate()` function) but
is its own module rather than a branch on a provider flag inside
`ollama_client.py`: a hosted, API-key-authenticated REST call is different
enough from local unauthenticated HTTP that keeping them separate is
clearer than one function with two behaviors selected by a flag.
`build_prompt` and `extract_citations` (from Tasks 3-4 of the
citation-grounding phase) are already provider-agnostic pure functions and
are reused as-is — no changes needed there.

## 5. Corpus/Index Hosting — Hugging Face Hub Dataset

The deployed app starts with none of the gitignored corpus/index
(`data/raw/`, `data/chroma/`, `data/bm25_index.pkl`) — same situation the
CI hermeticity check already proved for a fresh checkout, now relevant to a
live deploy instead of a test run.

- **One-time upload**: a script pushes the already-built, already-verified
  corpus/index (1,848 chunks — the exact figures documented in the
  retrieval-accuracy benchmark section of the README) to a new HF Hub
  Dataset repo. HF Datasets have git-lfs built in, a natural fit for
  binary index files, without bloating the main code repo's git history.
- **`app.py` downloads this dataset at startup** (via `huggingface_hub`,
  cached on Streamlit's ephemeral disk) before building the retrieval
  stack — avoiding a live rebuild against openFDA on every cold start
  (slow, hits real rate limits, and risks silent drift from the corpus
  that was actually benchmarked, since FDA data can change over time).
- `load_retrieval_stack()` (from the citation-grounding phase's
  `retrieval/bootstrap.py`) is reused as-is once the downloaded files are
  in place — it already takes `chroma_dir`/`bm25_index_path` as
  parameters, so pointing it at the downloaded-and-cached paths requires
  no changes to that function.

## 6. Memory-Fit Risk — Smoke-Tested With the Real Workload, Not a Stand-In

Streamlit Community Cloud does not publish exact CPU/RAM numbers (its docs
state limits "may change at any time without notice"); `torch` +
`sentence-transformers` + `chromadb` together are a plausible tight fit on
a free tier. This can only be answered by an actual deploy, not more
research — so the implementation plan's **first task** is a retrieval-only
smoke test, sequenced as:

1. Upload the real, full corpus/index to the HF Dataset (§5) — this must
   happen first, since the smoke test is meaningless against anything
   smaller.
2. Deploy a minimal `app.py` (retrieval only, no Groq/generation code yet)
   to a real Streamlit Community Cloud app.
3. Confirm it downloads the **real** HF Dataset (all ~1,850 chunks, not a
   trimmed or synthetic subset), loads the full retrieval stack (the same
   `Embedder`, the same full Chroma collection, the same full BM25 index
   the finished app will use), and successfully answers a real query —
   before any generation code is added on top.

A smoke test against a smaller or synthetic corpus would report false
confidence: the actual risk (does the real ~1,850-chunk workload fit) would
only surface after the full app — including Groq integration — was already
built, which is exactly the failure mode this de-risking step exists to
avoid.

Only after this smoke test passes does the plan proceed to add Groq
generation, the About/Benchmark section, and the session request cap.

## 7. UI Scope

Beyond the architecture doc's locked-in surfaces (§1): a collapsible
**About / Benchmark** section showing the honest Hit Rate@5/MRR numbers
from the README's retrieval-accuracy benchmark section — reported with the
same "directional, not a statistically supported win" framing already
locked into the README (Wilson CIs overlap; this is not restated as a clean
hybrid win in the demo) — plus a link back to the GitHub repo and design
docs. Gives a visiting recruiter/interviewer context without cluttering the
main Q&A flow.

## 8. Abuse/Cost Guard

A simple `st.session_state` counter (e.g. 10 questions per browser
session), since this is the developer's own metered Groq API key behind a
public link. Beyond that, no rate-limiting infrastructure — graceful error
handling if Groq's own account-level quota is hit (a friendly "please try
again in a moment" message instead of a raw error), matching the actual
threat model of a low-traffic portfolio demo rather than engineering for
traffic this project has no reason to expect.

## 9. Secrets

`GROQ_API_KEY` via Streamlit Community Cloud's Secrets UI (`st.secrets`) —
verified directly against Streamlit's own docs: secrets are pasted into the
app's "Advanced settings" dialog, exposed to the app as `st.secrets`, and
explicitly never committed to the repo. Local dev uses
`.streamlit/secrets.toml`, added to `.gitignore`.

## 10. Out of Scope

- Any change to `ask.py`, `spot_check_transcripts.py`, or local
  Ollama-based dev/eval — untouched by this phase.
- A paid Streamlit Cloud tier, a paid HF Spaces tier, or self-hosted
  compute (option (c) from the original framing) — the free-tier hosted-API
  approach is sufficient once verified.
- Automated abuse detection beyond the simple session counter (§8).
- Rebuilding or re-verifying the retrieval-accuracy benchmark itself — this
  phase serves the existing benchmark results, it doesn't recompute them.
