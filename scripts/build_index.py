# scripts/build_index.py
"""Chunks everything pulled by pull_corpus.py, embeds it locally, and
populates the Chroma store + BM25 index.

Usage: python scripts/build_index.py
"""
import json
import pickle
from pathlib import Path

from fda_device_rag.documents.structured import recall_to_chunk, event_to_chunk
from fda_device_rag.documents.pdf_document import extract_pdf_text
from fda_device_rag.chunking.pdf_chunker import chunk_pdf_text
from fda_device_rag.embedding.embedder import Embedder
from fda_device_rag.store.chroma_store import ChromaStore
from fda_device_rag.retrieval.bm25_index import BM25Index

DATA_DIR = Path("data/raw")
CHROMA_DIR = Path("data/chroma")
BM25_INDEX_PATH = Path("data/bm25_index.pkl")


def main() -> None:
    chunks = []

    recalls_path = DATA_DIR / "recalls.json"
    if recalls_path.exists():
        for record in json.loads(recalls_path.read_text()):
            chunks.append(recall_to_chunk(record, retrieved_date=""))

    events_path = DATA_DIR / "events.json"
    if events_path.exists():
        for record in json.loads(events_path.read_text()):
            chunks.append(event_to_chunk(record, retrieved_date=""))

    for pdf_dir, source_type in [(DATA_DIR / "guidance_pdfs", "guidance_pdf"), (DATA_DIR / "ifu_pdfs", "ifu_pdf")]:
        if not pdf_dir.exists():
            continue
        for pdf_path in sorted(pdf_dir.glob("*.pdf")):
            text = extract_pdf_text(pdf_path)
            chunks.extend(
                chunk_pdf_text(
                    text,
                    source_type=source_type,
                    source_url=str(pdf_path),
                    document_title=pdf_path.stem,
                    retrieved_date="",
                    id_prefix=pdf_path.stem,
                )
            )

    embedder = Embedder()
    embeddings = embedder.embed_documents([c.text for c in chunks])

    store = ChromaStore(persist_dir=str(CHROMA_DIR))
    store.add_chunks(chunks, embeddings)

    bm25 = BM25Index(chunks)
    BM25_INDEX_PATH.write_bytes(pickle.dumps(bm25))

    print(f"Indexed {len(chunks)} chunks into {CHROMA_DIR} and {BM25_INDEX_PATH}.")


if __name__ == "__main__":
    main()
