from dataclasses import asdict

import chromadb

from fda_device_rag.models import Chunk, ChunkMetadata, ScoredChunk


class ChromaStore:
    def __init__(self, persist_dir: str, collection_name: str = "fda_docs"):
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        # upsert, not add: Chroma's `add` silently no-ops on an already-present
        # id, so re-running scripts/build_index.py after a corpus change would
        # leave stale vectors here while the BM25 pickle is rewritten wholesale
        # -- the two retrieval legs would then describe different corpora.
        self._collection.upsert(
            ids=[c.id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[
                {k: v for k, v in asdict(c.metadata).items() if v is not None}
                for c in chunks
            ],
        )

    def query(self, query_embedding: list[float], top_k: int = 20) -> list[ScoredChunk]:
        result = self._collection.query(query_embeddings=[query_embedding], n_results=top_k)

        ids = result["ids"][0]
        distances = result["distances"][0]
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]

        scored: list[ScoredChunk] = []
        for id_, dist, doc, meta in zip(ids, distances, documents, metadatas):
            similarity = 1 - dist
            scored.append(ScoredChunk(id=id_, score=similarity, text=doc, metadata=ChunkMetadata(**meta)))
        return scored
