# scripts/build_index.py
"""Chunks everything pulled by pull_corpus.py, embeds it locally, and
populates the Chroma store + BM25 index.

Usage: python scripts/build_index.py
"""
import json
import pickle
from pathlib import Path

from fda_device_rag.chunking.pdf_chunker import chunk_pdf_text
from fda_device_rag.documents.pdf_document import extract_pdf_text
from fda_device_rag.documents.structured import event_to_chunk, recall_to_chunk
from fda_device_rag.embedding.embedder import Embedder
from fda_device_rag.retrieval.bm25_index import BM25Index
from fda_device_rag.store.chroma_store import ChromaStore

DATA_DIR = Path("data/raw")
CHROMA_DIR = Path("data/chroma")
BM25_INDEX_PATH = Path("data/bm25_index.pkl")
PULL_DATE_PATH = DATA_DIR / "pull_date.txt"


def _read_pull_date() -> str:
    """The pull date written by pull_corpus.py, carried into chunk metadata for
    citations. Degrades to "" (with a warning) rather than crashing, so the
    script still runs against manually-placed data in data/raw/."""
    if PULL_DATE_PATH.exists():
        return PULL_DATE_PATH.read_text().strip()
    print(f"WARNING: {PULL_DATE_PATH.as_posix()} not found -- indexing with an empty "
          f"retrieved_date. Run scripts/pull_corpus.py to record the pull date.")
    return ""


def main() -> None:
    retrieved_date = _read_pull_date()
    chunks = []

    recalls_path = DATA_DIR / "recalls.json"
    if recalls_path.exists():
        for record in json.loads(recalls_path.read_text()):
            chunk = recall_to_chunk(record, retrieved_date=retrieved_date)
            if chunk is not None:
                chunks.append(chunk)

    events_path = DATA_DIR / "events.json"
    if events_path.exists():
        for record in json.loads(events_path.read_text()):
            chunk = event_to_chunk(record, retrieved_date=retrieved_date)
            if chunk is not None:
                chunks.append(chunk)

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
                    retrieved_date=retrieved_date,
                    id_prefix=pdf_path.stem,
                )
            )

    # Guard before Embedder(), which loads a ~130MB model -- otherwise running
    # the scripts out of order costs a slow download just to fail on an empty
    # embed/BM25 call.
    if not chunks:
        raw_sources_exist = (
            recalls_path.exists()
            or events_path.exists()
            or (DATA_DIR / "guidance_pdfs").exists()
            or (DATA_DIR / "ifu_pdfs").exists()
        )
        if raw_sources_exist:
            # Source files exist but every record was filtered out (Fix 8's
            # content-free-record skipping, or empty curated PDF lists) --
            # distinct from "corpus never pulled", which would print below.
            print(f"Found source files in {DATA_DIR.as_posix()}/, but zero chunks were "
                  f"produced (all records may lack narrative content, or PDF lists are "
                  f"empty) -- check your data or curated PDF URL lists.")
        else:
            print(f"No corpus data found in {DATA_DIR.as_posix()}/ -- run scripts/pull_corpus.py first.")
        return

    embedder = Embedder()
    embeddings = embedder.embed_documents([c.text for c in chunks])

    store = ChromaStore(persist_dir=str(CHROMA_DIR))
    store.add_chunks(chunks, embeddings)

    bm25 = BM25Index(chunks)
    BM25_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    BM25_INDEX_PATH.write_bytes(pickle.dumps(bm25))

    print(f"Indexed {len(chunks)} chunks (retrieved_date={retrieved_date or 'unset'}) "
          f"into {CHROMA_DIR} and {BM25_INDEX_PATH}.")


if __name__ == "__main__":
    main()
