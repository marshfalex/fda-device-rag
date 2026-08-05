from fda_device_rag.generation.citations import extract_citations
from fda_device_rag.models import ChunkMetadata, ScoredChunk


def _chunk(chunk_id):
    return ScoredChunk(
        id=chunk_id,
        score=1.0,
        text=f"Text {chunk_id}",
        metadata=ChunkMetadata(
            source_type="recall",
            source_url=f"http://example.com/{chunk_id}",
            document_title=f"Doc {chunk_id}",
            section_name=None,
            record_id=chunk_id,
            retrieved_date="2026-08-01",
        ),
    )


def test_extract_citations_happy_path_returns_cited_chunks_and_label():
    chunk_map = {1: _chunk("a"), 2: _chunk("b"), 3: _chunk("c")}

    citations, label = extract_citations("Uses passage one [1] and three [3].", chunk_map)

    assert citations == [chunk_map[1], chunk_map[3]]
    assert label == "cited by the model"


def test_extract_citations_falls_back_to_all_context_when_no_brackets():
    chunk_map = {1: _chunk("a"), 2: _chunk("b")}

    citations, label = extract_citations("I don't know based on the available documents.", chunk_map)

    assert citations == list(chunk_map.values())
    assert label == "context provided (model did not cite specific passages)"


def test_extract_citations_ignores_out_of_range_numbers():
    chunk_map = {1: _chunk("a")}

    citations, label = extract_citations("Cites passage [1] and a bogus [9].", chunk_map)

    assert citations == [chunk_map[1]]
    assert label == "cited by the model"
