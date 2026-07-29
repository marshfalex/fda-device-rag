import pytest

from fda_device_rag.models import Chunk, ChunkMetadata
from fda_device_rag.store.chroma_store import ChromaStore


def _chunk(id_, text, source_type="recall"):
    return Chunk(
        id=id_,
        text=text,
        metadata=ChunkMetadata(
            source_type=source_type,
            source_url=f"https://example.com/{id_}",
            document_title=f"Title {id_}",
        ),
    )


def test_add_and_query_returns_closest_first(tmp_path):
    store = ChromaStore(persist_dir=str(tmp_path / "chroma"))

    chunks = [_chunk("a", "text a"), _chunk("b", "text b"), _chunk("c", "text c")]
    embeddings = [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]]
    store.add_chunks(chunks, embeddings)

    results = store.query(query_embedding=[1.0, 0.0], top_k=2)

    assert len(results) == 2
    assert results[0].id == "a"
    assert results[0].score > results[1].score
    assert results[0].metadata.source_type == "recall"


def test_add_chunks_upserts_existing_ids(tmp_path):
    """Re-indexing the same id must overwrite the stored text/embedding.
    Chroma's `add` silently no-ops on an existing id, which would leave the
    Chroma leg stale while build_index.py rewrites the BM25 pickle wholesale."""
    store = ChromaStore(persist_dir=str(tmp_path / "chroma"))

    store.add_chunks([_chunk("a", "original text")], [[1.0, 0.0]])
    store.add_chunks([_chunk("a", "updated text")], [[0.0, 1.0]])

    results = store.query(query_embedding=[0.0, 1.0], top_k=5)

    assert len(results) == 1
    assert results[0].id == "a"
    assert results[0].text == "updated text"
    assert results[0].score == pytest.approx(1.0, abs=1e-5)


def test_query_respects_top_k(tmp_path):
    store = ChromaStore(persist_dir=str(tmp_path / "chroma"))

    chunks = [_chunk("a", "text a"), _chunk("b", "text b"), _chunk("c", "text c")]
    embeddings = [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]]
    store.add_chunks(chunks, embeddings)

    results = store.query(query_embedding=[1.0, 0.0], top_k=1)

    assert len(results) == 1
