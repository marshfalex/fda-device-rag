from fda_device_rag.documents.structured import recall_to_chunk, event_to_chunk
from fda_device_rag.chunking.pdf_chunker import chunk_pdf_text
from fda_device_rag.store.chroma_store import ChromaStore
from fda_device_rag.retrieval.bm25_index import BM25Index
from fda_device_rag.retrieval.hybrid_retriever import HybridRetriever


_KEYWORDS = ("battery", "overheat", "firmware", "pacemaker")


class _StubEmbedder:
    """Deterministic, content-based (if simplistic) embedding: a 4-dim vector
    where each dimension is 1.0 if the corresponding keyword in _KEYWORDS
    appears (case-insensitive substring match) in the text, else 0.0. Both
    embed_documents and embed_query go through the same _vector method, so
    the query and recall_chunk's text land on the identical [1,1,1,1] point
    while the other two chunks (which share none of these keywords) land on
    [0,0,0,0] - genuinely and unambiguously closer to the query by cosine
    similarity, not a hash and not a constant tie-break.

    This means the dense leg now carries real signal: the test's top_k=1
    pass reflects actual hybrid retrieval fusion (dense + BM25 agreeing on
    recall-1), not an arbitrary insertion-order tie-break in Chroma/hnswlib.
    (Confirmed by simulation: reordering the `chunks` list below does not
    change the winner - see the corpus-reorder check in the task report.)
    Real embedding quality is validated separately by the Phase 2 benchmark.
    """

    def _vector(self, text: str) -> list[float]:
        lowered = text.lower()
        return [1.0 if kw in lowered else 0.0 for kw in _KEYWORDS]

    def embed_documents(self, texts):
        return [self._vector(t) for t in texts]

    def embed_query(self, text):
        return self._vector(text)


def test_pipeline_retrieves_recall_chunk_for_matching_query(tmp_path):
    recall_record = {
        "product_res_number": "Z-0001-04",
        "reason_for_recall": "Battery may overheat during normal use.",
        "action": "Firmware update issued to affected units.",
        "product_description": "Implantable Pacemaker Pulse-Generator Model X200",
    }
    event_record = {
        "report_number": "2",
        "mdr_text": [{"text": "Patient reported device beeping unexpectedly during sleep."}],
        "device": [{"generic_name": "MANUAL HOSPITAL BED"}],
    }
    guidance_chunks = chunk_pdf_text(
        "WARNINGS\nDo not expose the device to strong magnetic fields.",
        source_type="guidance_pdf",
        source_url="https://example.com/guidance.pdf",
        document_title="Example Guidance",
        retrieved_date="2026-07-28",
        id_prefix="guidance-1",
    )

    # Order is irrelevant here: recall_chunk is the only chunk whose text
    # contains all four _StubEmbedder keywords, so it is the unique nearest
    # neighbor by cosine similarity regardless of insertion order (verified
    # by re-running this test with event_chunk listed first).
    chunks = [
        recall_to_chunk(recall_record, retrieved_date="2026-07-28"),
        event_to_chunk(event_record, retrieved_date="2026-07-28"),
        *guidance_chunks,
    ]

    embedder = _StubEmbedder()
    store = ChromaStore(persist_dir=str(tmp_path / "chroma"))
    store.add_chunks(chunks, embedder.embed_documents([c.text for c in chunks]))
    bm25 = BM25Index(chunks)
    retriever = HybridRetriever(dense_store=store, bm25_index=bm25, embedder=embedder)

    results = retriever.retrieve("battery overheat pacemaker firmware update", top_k=1)

    assert results[0].id == "recall-Z-0001-04"
