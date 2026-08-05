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
        # min() guards strata absent from `questions` (e.g. isolated-stratum
        # unit tests) -- the real 50-question benchmark has >= per_stratum
        # questions for all 4 source types, so this never clamps in
        # production use.
        selected.extend(rng.sample(pool, min(per_stratum, len(pool))))
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
    OUTPUT_PATH.write_text(_render_transcript(args.seed, entries), encoding="utf-8")

    print(f"Wrote {len(entries)}/{len(selected)} transcripts (seed={args.seed!r}) -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
