from langchain_text_splitters import RecursiveCharacterTextSplitter

from fda_device_rag.chunking.sections import detect_sections
from fda_device_rag.models import Chunk, ChunkMetadata

CHUNK_SIZE_CHARS = 1600
CHUNK_OVERLAP_CHARS = 240
MIN_CHUNK_CHARS = 40
MAX_HEADING_CHARS = 80  # sections.py caps detected heading length at 80
HEADING_BUDGET = MAX_HEADING_CHARS + 1  # +1 for the newline separator
EFFECTIVE_CHUNK_SIZE = CHUNK_SIZE_CHARS - HEADING_BUDGET


def chunk_pdf_text(
    text: str,
    source_type: str,
    source_url: str,
    document_title: str,
    retrieved_date: str,
    id_prefix: str,
) -> list[Chunk]:
    sections = detect_sections(text)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=EFFECTIVE_CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP_CHARS,
    )

    chunks: list[Chunk] = []
    for section in sections:
        if len(section.body) <= EFFECTIVE_CHUNK_SIZE:
            pieces = [section.body]
        else:
            pieces = splitter.split_text(section.body)

        for i, piece in enumerate(pieces):
            if len(piece) < MIN_CHUNK_CHARS:
                continue

            piece_text = f"{section.heading}\n{piece}" if i == 0 else piece
            index = len(chunks)
            metadata = ChunkMetadata(
                source_type=source_type,
                source_url=source_url,
                document_title=document_title,
                section_name=section.heading,
                record_id=None,
                retrieved_date=retrieved_date,
            )
            chunks.append(Chunk(id=f"{id_prefix}-{index}", text=piece_text, metadata=metadata))

    return chunks
