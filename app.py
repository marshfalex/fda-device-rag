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
