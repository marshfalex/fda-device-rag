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
