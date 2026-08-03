from dataclasses import dataclass

from fda_device_rag.models import Chunk


@dataclass
class SectionInstance:
    document_title: str
    section_name: str
    instance_ordinal: int
    chunk_ids: list[str]
    text: str


def build_section_instances(chunks: list[Chunk]) -> list[SectionInstance]:
    """Groups a document's PDF chunks (in chunk_pdf_text's output order) into
    section instances: contiguous runs of chunks sharing the same
    section_name. A non-adjacent repeat of the same section_name -- e.g. the
    same mislabeled heading (`Contains Nonbinding Recommendations`)
    recurring later in the document -- is counted as a separate instance,
    numbered by instance_ordinal within (document_title, section_name).
    """
    instances: list[SectionInstance] = []
    ordinal_counts: dict[tuple[str, str], int] = {}

    current_doc = None
    current_name = None
    current_ids: list[str] = []
    current_texts: list[str] = []

    def flush():
        nonlocal current_ids, current_texts
        if current_ids:
            key = (current_doc, current_name)
            ordinal_counts[key] = ordinal_counts.get(key, 0) + 1
            instances.append(
                SectionInstance(
                    document_title=current_doc,
                    section_name=current_name,
                    instance_ordinal=ordinal_counts[key],
                    chunk_ids=list(current_ids),
                    text="\n".join(current_texts),
                )
            )
        current_ids = []
        current_texts = []

    for chunk in chunks:
        doc = chunk.metadata.document_title
        name = chunk.metadata.section_name
        if doc != current_doc or name != current_name:
            flush()
            current_doc = doc
            current_name = name
        current_ids.append(chunk.id)
        current_texts.append(chunk.text)

    flush()
    return instances
