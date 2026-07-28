from fda_device_rag.models import ScoredChunk, ChunkMetadata
from fda_device_rag.retrieval.hybrid_retriever import reciprocal_rank_fusion, HybridRetriever


def test_reciprocal_rank_fusion_ranks_items_in_both_lists_highest():
    dense_ids = ["a", "b", "c"]
    bm25_ids = ["b", "a", "d"]

    fused = reciprocal_rank_fusion(dense_ids, bm25_ids, k=60)
    fused_ids = [chunk_id for chunk_id, _score in fused]

    # "a" and "b" appear in both lists near the top, so they should outrank "c"/"d"
    assert fused_ids[0] in ("a", "b")
    assert fused_ids[1] in ("a", "b")
    assert set(fused_ids[2:]) == {"c", "d"}


def test_reciprocal_rank_fusion_handles_disjoint_lists():
    fused = reciprocal_rank_fusion(["a", "b"], ["c", "d"], k=60)
    fused_ids = {chunk_id for chunk_id, _score in fused}

    assert fused_ids == {"a", "b", "c", "d"}


def _scored(id_, score):
    return ScoredChunk(
        id=id_,
        score=score,
        text=f"text {id_}",
        metadata=ChunkMetadata(source_type="recall", source_url="https://example.com", document_title="t"),
    )


class _FakeDenseStore:
    def __init__(self, results):
        self._results = results

    def query(self, query_embedding, top_k):
        return self._results[:top_k]


class _FakeBM25Index:
    def __init__(self, results):
        self._results = results

    def query(self, query_text, top_k):
        return self._results[:top_k]


class _FakeEmbedder:
    def embed_query(self, text):
        return [0.1, 0.2]


def test_hybrid_retriever_fuses_dense_and_bm25_results():
    dense_results = [_scored("a", 0.9), _scored("b", 0.8), _scored("c", 0.7)]
    bm25_results = [_scored("b", 5.0), _scored("a", 3.0), _scored("d", 1.0)]

    retriever = HybridRetriever(
        dense_store=_FakeDenseStore(dense_results),
        bm25_index=_FakeBM25Index(bm25_results),
        embedder=_FakeEmbedder(),
    )

    top = retriever.retrieve("some query", top_k=2)

    assert len(top) == 2
    assert {c.id for c in top} == {"a", "b"}


def test_hybrid_retriever_returns_at_most_top_k():
    dense_results = [_scored("a", 0.9), _scored("b", 0.8)]
    bm25_results = [_scored("c", 5.0), _scored("d", 3.0)]

    retriever = HybridRetriever(
        dense_store=_FakeDenseStore(dense_results),
        bm25_index=_FakeBM25Index(bm25_results),
        embedder=_FakeEmbedder(),
    )

    top = retriever.retrieve("some query", top_k=3)

    assert len(top) == 3
