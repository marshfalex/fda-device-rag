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


def test_query_respects_top_k(tmp_path):
    store = ChromaStore(persist_dir=str(tmp_path / "chroma"))

    chunks = [_chunk("a", "text a"), _chunk("b", "text b"), _chunk("c", "text c")]
    embeddings = [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]]
    store.add_chunks(chunks, embeddings)

    results = store.query(query_embedding=[1.0, 0.0], top_k=1)

    assert len(results) == 1
