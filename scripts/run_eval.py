# scripts/run_eval.py
"""Scores the frozen benchmark (data/eval/questions.json) against the live
retrieval stack -- Hit Rate@5, MRR, and a dense-only/BM25-only/hybrid
ablation, stratified by source type. Refuses to run if the frozen file has
uncommitted changes (per docs/superpowers/specs/2026-08-03-eval-harness-design.md
section 7's freeze-enforcement guarantee).

Writes data/eval/results.json (gitignored -- overwritten on routine runs;
commit a specific run's results only when a number is actually being cited)
and prints a human-readable summary to stdout.

Usage: python scripts/run_eval.py
"""
import datetime
import json
import pickle
import sys
from pathlib import Path

from fda_device_rag.embedding.embedder import Embedder
from fda_device_rag.eval.freeze_check import FrozenFileError, assert_frozen_and_get_hash
from fda_device_rag.eval.gold_resolution import GoldResolutionError, resolve_gold
from fda_device_rag.eval.metrics import (
    hit_at_k,
    rank_of_first_gold_hit,
    summarize_hit_rate,
    summarize_mrr,
)
from fda_device_rag.eval.questions import load_questions
from fda_device_rag.retrieval.bm25_index import BM25Index
from fda_device_rag.retrieval.hybrid_retriever import HybridRetriever
from fda_device_rag.store.chroma_store import ChromaStore

DATA_DIR = Path("data/raw")
CHROMA_DIR = Path("data/chroma")
BM25_INDEX_PATH = Path("data/bm25_index.pkl")
QUESTIONS_PATH = Path("data/eval/questions.json")
RESULTS_PATH = Path("data/eval/results.json")

# Frozen file's benchmark-level source_type -> corpus's ChunkMetadata.source_type.
# Two different vocabularies for two different purposes -- see
# src/fda_device_rag/eval/gold_resolution.py's resolve_gold docstring.
CORPUS_SOURCE_TYPES = {
    "recall": "recall",
    "maude": "maude_event",
    "guidance": "guidance_pdf",
    "ifu": "ifu_pdf",
}
SOURCE_TYPES = ("recall", "maude", "guidance", "ifu")
LEGS = ("dense", "bm25", "hybrid")
GOLD_SET_SIZE_OUTLIER_THRESHOLD = 2


def _corpus_size(bm25_index: BM25Index) -> dict:
    """Per source type: document count and chunk count, computed at run
    time from the loaded index. recall/maude document count = chunk count
    directly (each record produces exactly one chunk, and their
    document_title field holds shared descriptive text, not a per-record
    key -- see this plan's Global Constraints). guidance/ifu document count
    = distinct document_title count (genuinely per-document there).
    """
    chunks_by_corpus_type: dict[str, list] = {}
    for chunk in bm25_index._chunks:
        chunks_by_corpus_type.setdefault(chunk.metadata.source_type, []).append(chunk)

    sizes = {}
    for benchmark_type, corpus_type in CORPUS_SOURCE_TYPES.items():
        type_chunks = chunks_by_corpus_type.get(corpus_type, [])
        chunk_count = len(type_chunks)
        if benchmark_type in ("recall", "maude"):
            document_count = chunk_count
        else:
            document_count = len({c.metadata.document_title for c in type_chunks})
        sizes[benchmark_type] = {"documents": document_count, "chunks": chunk_count}
    return sizes


def _score_question(question, bm25_index, dense_store, hybrid_retriever, embedder, corpus_total: int) -> dict:
    """Raises GoldResolutionError if the question's gold locator can't be
    resolved -- the caller must catch this and record it as a resolution
    error, never as a per_question entry with null ranks."""
    gold_ids = resolve_gold(question)

    query_embedding = embedder.embed_query(question.question)
    dense_results = dense_store.query(query_embedding, top_k=corpus_total)
    bm25_results = bm25_index.query(question.question, top_k=corpus_total)
    hybrid_results = hybrid_retriever.retrieve(question.question, top_k=corpus_total)

    dense_rank = rank_of_first_gold_hit([r.id for r in dense_results], gold_ids)
    bm25_rank = rank_of_first_gold_hit([r.id for r in bm25_results], gold_ids)
    hybrid_rank = rank_of_first_gold_hit([r.id for r in hybrid_results], gold_ids)

    return {
        "question_id": question.question_id,
        "source_type": question.source_type,
        "gold_set_size": len(gold_ids),
        "dense": {"hit_at_5": hit_at_k(dense_rank), "rank": dense_rank},
        "bm25": {"hit_at_5": hit_at_k(bm25_rank), "rank": bm25_rank},
        "hybrid": {"hit_at_5": hit_at_k(hybrid_rank), "rank": hybrid_rank},
    }


def _aggregate(per_question_subset: list[dict]) -> dict:
    result = {}
    for leg in LEGS:
        ranks = [q[leg]["rank"] for q in per_question_subset]
        result[leg] = {
            "n": len(ranks),
            "hit_rate_5": summarize_hit_rate(ranks),
            "mrr": summarize_mrr(ranks),
        }
    return result


MRR_PRECISION_NOTE = (
    "MRR has no confidence interval computed -- it is a mean of reciprocal ranks, "
    "not a binomial proportion, so the Wilson interval used for Hit Rate@5 does not "
    "apply. Treat MRR differences with the same caution as an unquantified "
    "statistic, not as precise as the adjacent Hit Rate@5 figures."
)


def _render_summary(results: dict) -> str:
    lines = []
    lines.append("=== Corpus Size ===")
    for source_type in SOURCE_TYPES:
        sizes = results["corpus_size"][source_type]
        lines.append(f"  {source_type}: {sizes['documents']} documents, {sizes['chunks']} chunks")

    if results["resolution_errors"]:
        lines.append("")
        lines.append(f"=== Resolution Errors ({len(results['resolution_errors'])}) ===")
        for err in results["resolution_errors"]:
            lines.append(f"  {err['question_id']}: {err['error']}")

    if results["gold_set_size_outliers"]:
        lines.append("")
        lines.append("=== Gold-Set-Size Outliers ===")
        for outlier in results["gold_set_size_outliers"]:
            lines.append(f"  {outlier['question_id']}: {outlier['gold_set_size']} gold chunks")

    total_questions = len(results["per_question"]) + len(results["resolution_errors"])
    valid_count = len(results["per_question"])
    lines.append("")
    lines.append(
        f"=== Results ({valid_count}/{total_questions} valid questions, "
        f"{len(results['resolution_errors'])} resolution errors) ==="
    )

    scopes = [("Pooled", results["metrics"]["pooled"])] + [
        (source_type.capitalize(), results["metrics"]["by_source_type"][source_type])
        for source_type in SOURCE_TYPES
    ]
    for scope_name, scope_data in scopes:
        lines.append("")
        lines.append(f"--- {scope_name} ---")
        for leg in LEGS:
            leg_data = scope_data[leg]
            hr = leg_data["hit_rate_5"]
            ci_lo, ci_hi = hr["wilson_ci_95"]
            lines.append(
                f"  {leg:8s}  Hit Rate@5: {hr['hits']}/{hr['n']} ({hr['pct']:.1f}%, "
                f"95% CI [{ci_lo:.1f}, {ci_hi:.1f}])  MRR: {leg_data['mrr']:.3f}"
            )

    lines.append("")
    lines.append(MRR_PRECISION_NOTE)
    lines.append("")
    lines.append(f"Frozen file hash: {results['frozen_file_hash']}")

    return "\n".join(lines)


def main() -> None:
    try:
        frozen_hash = assert_frozen_and_get_hash(QUESTIONS_PATH)
    except FrozenFileError as e:
        print(f"Cannot run eval: {e}")
        sys.exit(1)

    if not BM25_INDEX_PATH.exists() or not CHROMA_DIR.exists():
        print(f"Index not found ({CHROMA_DIR}, {BM25_INDEX_PATH}) -- run "
              f"scripts/pull_corpus.py then scripts/build_index.py first.")
        sys.exit(1)

    questions = load_questions(QUESTIONS_PATH)

    # Safe: this pickle is a local build artifact written by build_index.py on
    # this machine, never fetched or accepted from an external source. (It holds
    # a BM25Okapi object with fitted corpus statistics, not plain data, which is
    # why it is pickled rather than serialized to JSON.)
    bm25_index = pickle.loads(BM25_INDEX_PATH.read_bytes())
    dense_store = ChromaStore(persist_dir=str(CHROMA_DIR))
    embedder = Embedder()
    hybrid_retriever = HybridRetriever(dense_store=dense_store, bm25_index=bm25_index, embedder=embedder)

    corpus_size = _corpus_size(bm25_index)
    corpus_total = len(bm25_index._chunks)

    per_question = []
    resolution_errors = []
    for question in questions:
        try:
            per_question.append(
                _score_question(question, bm25_index, dense_store, hybrid_retriever, embedder, corpus_total)
            )
        except GoldResolutionError as e:
            resolution_errors.append({"question_id": question.question_id, "error": str(e)})

    gold_set_size_outliers = [
        {"question_id": pq["question_id"], "gold_set_size": pq["gold_set_size"]}
        for pq in per_question
        if pq["gold_set_size"] > GOLD_SET_SIZE_OUTLIER_THRESHOLD
    ]

    by_source_type = {
        source_type: _aggregate([pq for pq in per_question if pq["source_type"] == source_type])
        for source_type in SOURCE_TYPES
    }
    pooled = _aggregate(per_question)

    results = {
        "frozen_file_hash": frozen_hash,
        "run_date": datetime.date.today().isoformat(),
        "corpus_size": corpus_size,
        "gold_set_size_outliers": gold_set_size_outliers,
        "resolution_errors": resolution_errors,
        "per_question": per_question,
        "metrics": {"pooled": pooled, "by_source_type": by_source_type},
        "mrr_precision_note": MRR_PRECISION_NOTE,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))

    print(_render_summary(results))
    print(f"\nFull results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
