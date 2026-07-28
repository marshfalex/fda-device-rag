from fda_device_rag.documents.structured import recall_to_chunk, event_to_chunk
from fda_device_rag.chunking.pdf_chunker import chunk_pdf_text
from fda_device_rag.store.chroma_store import ChromaStore
from fda_device_rag.retrieval.bm25_index import BM25Index
from fda_device_rag.retrieval.hybrid_retriever import HybridRetriever


class _StubEmbedder:
    """Deterministic pseudo-embedding: hashes text into a fixed-size vector.
    Only used to prove the pipeline wiring works end-to-end; real embedding
    quality is validated separately by the Phase 2 benchmark."""

    def _vector(self, text: str) -> list[float]:
        h = hash(text)
        return [((h >> (8 * i)) % 256) / 255.0 for i in range(4)]

    def embed_documents(self, texts):
        return [self._vector(t) for t in texts]

    def embed_query(self, text):
        return self._vector(text)


def test_pipeline_retrieves_recall_chunk_for_matching_query(tmp_path):
    recall_record = {
        "res_event_number": "1",
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

    results = retriever.retrieve("battery overheat pacemaker firmware update", top_k=3)

    assert any(r.id == "recall-1" for r in results)
