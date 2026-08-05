from fda_device_rag.generation.prompt import build_prompt
from fda_device_rag.models import ChunkMetadata, ScoredChunk


def _chunk(chunk_id, text, section_name=None):
    return ScoredChunk(
        id=chunk_id,
        score=1.0,
        text=text,
        metadata=ChunkMetadata(
            source_type="recall",
            source_url=f"http://example.com/{chunk_id}",
            document_title=f"Doc {chunk_id}",
            section_name=section_name,
            record_id=chunk_id,
            retrieved_date="2026-08-01",
        ),
    )


def test_build_prompt_numbers_chunks_1_indexed_in_order():
    chunk_a = _chunk("a", "Text A")
    chunk_b = _chunk("b", "Text B", section_name="Warnings")

    prompt, chunk_map = build_prompt("What is the warning?", [chunk_a, chunk_b])

    assert chunk_map == {1: chunk_a, 2: chunk_b}
    assert "[1]" in prompt
    assert "[2]" in prompt
    assert "Text A" in prompt
    assert "Text B" in prompt
    assert "What is the warning?" in prompt


def test_build_prompt_resets_chunk_map_per_call():
    chunk_b = _chunk("b", "Text B")

    _, first_map = build_prompt("Question one", [_chunk("a", "Text A"), chunk_b])
    _, second_map = build_prompt("Question two", [chunk_b])

    assert second_map == {1: chunk_b}
