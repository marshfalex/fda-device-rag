import json
from pathlib import Path

from fda_device_rag.chunking.pdf_chunker import chunk_pdf_text
from fda_device_rag.documents.pdf_document import extract_pdf_text
from fda_device_rag.documents.structured import event_to_chunk, recall_to_chunk
from fda_device_rag.eval.section_instances import build_section_instances

DEFAULT_DATA_DIR = Path("data/raw")

_STRUCTURED_SOURCES = {
    "recall": ("recalls.json", "product_res_number", recall_to_chunk),
    "maude": ("events.json", "report_number", event_to_chunk),
}


class GoldResolutionError(Exception):
    """Raised when a frozen question's gold locator can't be resolved
    against the current corpus -- a renamed heading, a removed document, an
    out-of-range instance_ordinal, or a record that no longer produces a
    chunk. Must never be silently swallowed into an empty gold set or a
    silent miss (design doc section 5)."""


def resolve_structured_gold(source_type: str, record_id: str, retrieved_date: str = "", data_dir: Path = DEFAULT_DATA_DIR) -> list[str]:
    if source_type not in _STRUCTURED_SOURCES:
        raise GoldResolutionError(f"unknown structured source_type {source_type!r}")

    filename, id_field, to_chunk = _STRUCTURED_SOURCES[source_type]
    path = Path(data_dir) / filename
    if not path.exists():
        raise GoldResolutionError(f"{path} not found -- run scripts/pull_corpus.py first")

    for record in json.loads(path.read_text()):
        if str(record.get(id_field, "")) == record_id:
            chunk = to_chunk(record, retrieved_date=retrieved_date)
            if chunk is None:
                raise GoldResolutionError(
                    f"{source_type} record {record_id!r} no longer produces a chunk (filtered as content-free)"
                )
            return [chunk.id]

    raise GoldResolutionError(f"{source_type} record {record_id!r} not found in {path}")


def resolve_pdf_gold_from_text(document_title: str, text: str, section_name: str, instance_ordinal: int, source_type: str = "guidance_pdf") -> list[str]:
    chunks = chunk_pdf_text(
        text,
        source_type=source_type,
        source_url="",
        document_title=document_title,
        retrieved_date="",
        id_prefix=document_title,
    )
    instances = build_section_instances(chunks)
    matches = [i for i in instances if i.section_name == section_name and i.instance_ordinal == instance_ordinal]

    if not matches:
        raise GoldResolutionError(
            f"no section instance found for document={document_title!r} section_name={section_name!r} "
            f"instance_ordinal={instance_ordinal} (heading may have been renamed, or ordinal is out of range)"
        )

    return matches[0].chunk_ids


def _find_pdf_path(document_title: str, data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    for subdir in ("guidance_pdfs", "ifu_pdfs"):
        candidate = Path(data_dir) / subdir / f"{document_title}.pdf"
        if candidate.exists():
            return candidate
    raise GoldResolutionError(f"no PDF found for document_title {document_title!r} under {data_dir}")


def resolve_pdf_gold(document_title: str, section_name: str, instance_ordinal: int, source_type: str = "guidance_pdf", data_dir: Path = DEFAULT_DATA_DIR) -> list[str]:
    pdf_path = _find_pdf_path(document_title, data_dir=data_dir)
    text = extract_pdf_text(pdf_path)
    return resolve_pdf_gold_from_text(document_title, text, section_name, instance_ordinal, source_type=source_type)


def resolve_gold(question, data_dir: Path = DEFAULT_DATA_DIR) -> list[str]:
    """Dispatches a frozen Question's gold locator to the right resolver.
    Translates the frozen file's benchmark-level source_type vocabulary
    ("recall"/"maude"/"guidance"/"ifu") to the corpus's ChunkMetadata
    vocabulary ("recall"/"maude_event"/"guidance_pdf"/"ifu_pdf") -- these are
    two different vocabularies for two different purposes, not a bug.

    A malformed gold dict (missing a field the question's source_type
    requires) is reported as a GoldResolutionError, not a bare KeyError, so
    that an eval runner's `except GoldResolutionError` handler records it as
    one question's resolution failure instead of aborting the whole run."""
    if question.source_type in ("recall", "maude"):
        record_id = question.gold.get("record_id")
        if record_id is None:
            raise GoldResolutionError(
                f"question {question.question_id!r} (source_type={question.source_type!r}) "
                f"is missing required gold field 'record_id'"
            )
        return resolve_structured_gold(question.source_type, record_id, data_dir=data_dir)
    if question.source_type in ("guidance", "ifu"):
        missing = [
            f for f in ("document_title", "section_name", "instance_ordinal")
            if question.gold.get(f) is None
        ]
        if missing:
            raise GoldResolutionError(
                f"question {question.question_id!r} (source_type={question.source_type!r}) "
                f"is missing required gold field(s): {missing}"
            )
        pdf_source_type = "guidance_pdf" if question.source_type == "guidance" else "ifu_pdf"
        return resolve_pdf_gold(
            question.gold["document_title"],
            question.gold["section_name"],
            question.gold["instance_ordinal"],
            source_type=pdf_source_type,
            data_dir=data_dir,
        )
    raise GoldResolutionError(f"unknown source_type {question.source_type!r}")
