from fda_device_rag.models import Chunk, ChunkMetadata
from fda_device_rag.retrieval.bm25_index import BM25Index, tokenize


def _chunk(id_, text):
    return Chunk(
        id=id_,
        text=text,
        metadata=ChunkMetadata(source_type="recall", source_url="https://example.com", document_title="t"),
    )


def test_tokenize_lowercases_and_strips_light_punctuation_but_keeps_codes_verbatim():
    tokens = tokenize("Device code K123456 is a code!")

    assert tokens == ["device", "code", "k123456", "is", "a", "code"]


def test_bm25_index_ranks_exact_code_match_first():
    chunks = [
        _chunk("a", "This document discusses general safety information."),
        _chunk("b", "The affected device carries product code K850651 specifically."),
        _chunk("c", "Unrelated narrative about a different device entirely."),
    ]
    index = BM25Index(chunks)

    results = index.query("K850651", top_k=2)

    assert results[0].id == "b"
    assert len(results) == 2


def test_bm25_index_respects_top_k():
    chunks = [_chunk("a", "alpha beta"), _chunk("b", "alpha gamma"), _chunk("c", "alpha delta")]
    index = BM25Index(chunks)

    results = index.query("alpha", top_k=1)

    assert len(results) == 1
