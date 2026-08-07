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
import sys
from pathlib import Path

import streamlit as st
from huggingface_hub import snapshot_download

# Streamlit Community Cloud only reinstalls requirements.txt when its text
# changes, so a stale pip-installed fda_device_rag can lag behind app.py's
# own imports after a code-only push. The git checkout itself is always
# current, so importing straight from src/ avoids depending on the
# installed copy at all.
sys.path.insert(0, str(Path(__file__).parent / "src"))

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
        stack = get_retrieval_stack()
        chunks = stack.hybrid_retriever.retrieve(question, top_k=TOP_K)
        prompt, chunk_map = build_prompt(question, chunks)

        try:
            groq_api_key = st.secrets.get("GROQ_API_KEY")
        except FileNotFoundError:  # StreamlitSecretNotFoundError subclasses this
            groq_api_key = None

        if not groq_api_key:
            st.error("GROQ_API_KEY is not configured for this deployment.")
        else:
            try:
                answer = generate(prompt, groq_api_key)
            except GroqError:
                st.error("Generation failed — the Groq API call did not succeed. Try again.")
            else:
                st.session_state.question_count += 1
                citations, citation_label = extract_citations(answer, chunk_map)

                st.subheader("Answer")
                st.write(answer)

                st.subheader(citation_label)
                for chunk in citations:
                    st.write(f"- {_format_chunk_line(chunk)} — {chunk.metadata.source_url}")

                st.subheader("Context provided (full top-5)")
                for number, chunk in chunk_map.items():
                    st.write(f"[{number}] {_format_chunk_line(chunk)}")
