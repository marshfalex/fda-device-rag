from fda_device_rag.models import Chunk, ChunkMetadata, ScoredChunk


def test_chunk_metadata_defaults():
    meta = ChunkMetadata(
        source_type="recall",
        source_url="https://example.com/recall/1",
        document_title="Example Recall",
    )
    assert meta.section_name is None
    assert meta.record_id is None
    assert meta.retrieved_date == ""


def test_chunk_holds_text_and_metadata():
    meta = ChunkMetadata(
        source_type="recall",
        source_url="https://example.com/recall/1",
        document_title="Example Recall",
    )
    chunk = Chunk(id="recall-1", text="Some recall text", metadata=meta)
    assert chunk.id == "recall-1"
    assert chunk.text == "Some recall text"
    assert chunk.metadata is meta


def test_scored_chunk_holds_score():
    meta = ChunkMetadata(
        source_type="recall",
        source_url="https://example.com/recall/1",
        document_title="Example Recall",
    )
    scored = ScoredChunk(id="recall-1", score=0.87, text="Some recall text", metadata=meta)
    assert scored.score == 0.87
    assert scored.id == "recall-1"
