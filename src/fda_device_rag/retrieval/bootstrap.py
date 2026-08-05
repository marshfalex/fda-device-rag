import pickle
from dataclasses import dataclass
from pathlib import Path

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
