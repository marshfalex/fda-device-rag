from dataclasses import replace

from fda_device_rag.models import ScoredChunk

RRF_K = 60
DENSE_TOP_N = 20
BM25_TOP_N = 20
FUSED_TOP_K = 5


def reciprocal_rank_fusion(*ranked_id_lists: list[str], k: int = RRF_K) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranked_ids in ranked_id_lists:
        for rank, chunk_id in enumerate(ranked_ids):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


class HybridRetriever:
    def __init__(self, dense_store, bm25_index, embedder):
        self._dense_store = dense_store
        self._bm25_index = bm25_index
        self._embedder = embedder

    def retrieve(self, query_text: str, top_k: int = FUSED_TOP_K) -> list[ScoredChunk]:
        query_embedding = self._embedder.embed_query(query_text)
        dense_results = self._dense_store.query(query_embedding, top_k=DENSE_TOP_N)
        bm25_results = self._bm25_index.query(query_text, top_k=BM25_TOP_N)

        dense_ids = [r.id for r in dense_results]
        bm25_ids = [r.id for r in bm25_results]
        fused = reciprocal_rank_fusion(dense_ids, bm25_ids)

        by_id: dict[str, ScoredChunk] = {}
        for r in dense_results:
            by_id[r.id] = r
        for r in bm25_results:
            by_id.setdefault(r.id, r)

        # Return the RRF fused score, not the leg-native score. The legs use
        # different, incomparable scales (Chroma cosine similarity vs. raw BM25),
        # so returning whichever one `by_id` happened to hold would give a list
        # whose scores are neither mutually comparable nor monotonically
        # decreasing in the returned order.
        return [
            replace(by_id[chunk_id], score=rrf_score)
            for chunk_id, rrf_score in fused[:top_k]
            if chunk_id in by_id
        ]
