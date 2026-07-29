# scripts/query.py
"""Queries the index built by build_index.py using the hybrid retriever
(dense Chroma + BM25, fused with reciprocal rank fusion).

Usage: python scripts/query.py "<question>"
"""
import pickle
import sys
from pathlib import Path

from fda_device_rag.embedding.embedder import Embedder
from fda_device_rag.store.chroma_store import ChromaStore
from fda_device_rag.retrieval.hybrid_retriever import HybridRetriever

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

    # Safe: this pickle is a local build artifact written by build_index.py on
    # this machine, never fetched or accepted from an external source. (It holds
    # a BM25Okapi object with fitted corpus statistics, not plain data, which is
    # why it is pickled rather than serialized to JSON.)
    bm25 = pickle.loads(BM25_INDEX_PATH.read_bytes())
    store = ChromaStore(persist_dir=str(CHROMA_DIR))
    retriever = HybridRetriever(dense_store=store, bm25_index=bm25, embedder=Embedder())

    results = retriever.retrieve(query_text, top_k=TOP_K)

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
