"""Builds the numbered-context generation prompt from retrieved chunks.
SYSTEM_PROMPT is reused verbatim from
docs/superpowers/specs/2026-07-28-fda-device-rag-architecture-design.md
section 5."""
from fda_device_rag.models import ScoredChunk

SYSTEM_PROMPT = """You are a document assistant answering questions about FDA \
device recalls, adverse event reports, and device guidance/instructions-for-use \
documents.

Rules:
1. Answer ONLY using the numbered context passages below. Do not use outside \
knowledge, even if you believe it is correct.
2. Cite passages inline using their bracket number immediately after the \
sentence that relies on them, e.g. "...requires firmware 4.2.1 [1]." Use \
multiple brackets if a sentence draws on more than one passage, e.g. [1][3].
3. Only cite a passage number if you actually used its content. Do not cite \
all passages by default, and do not cite a passage you didn't rely on.
4. If the passages do not contain enough information to answer, respond \
exactly: "I don't know based on the available documents." Do not guess or \
fill gaps with general knowledge.
5. Keep answers concise — 2 to 4 sentences unless the question requires more.

Context passages:
{numbered_context_block}
"""


def build_prompt(question: str, chunks: list[ScoredChunk]) -> tuple[str, dict[int, ScoredChunk]]:
    """Numbers `chunks` 1-indexed in the order given (retrieval rank order),
    renders them into SYSTEM_PROMPT's context block, and appends the
    question. Returns the finished prompt and a fresh {number: chunk} map
    for citation lookup after generation -- no state is carried across
    calls."""
    chunk_map = {i: chunk for i, chunk in enumerate(chunks, start=1)}

    blocks = []
    for number, chunk in chunk_map.items():
        header = f"[{number}] Source: {chunk.metadata.source_type} | {chunk.metadata.document_title}"
        if chunk.metadata.section_name:
            header += f" | {chunk.metadata.section_name}"
        header += f" | Retrieved: {chunk.metadata.retrieved_date} | URL: {chunk.metadata.source_url}"
        blocks.append(f"{header}\n{chunk.text}")
    numbered_context_block = "\n\n".join(blocks)

    prompt = SYSTEM_PROMPT.format(numbered_context_block=numbered_context_block)
    prompt += f"\nQuestion: {question}\n\nAnswer:"
    return prompt, chunk_map
