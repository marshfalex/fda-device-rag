from dataclasses import dataclass
from typing import Optional


@dataclass
class ChunkMetadata:
    source_type: str
    source_url: str
    document_title: str
    section_name: Optional[str] = None
    record_id: Optional[str] = None
    retrieved_date: str = ""


@dataclass
class Chunk:
    id: str
    text: str
    metadata: ChunkMetadata


@dataclass
class ScoredChunk:
    id: str
    score: float
    text: str
    metadata: ChunkMetadata
