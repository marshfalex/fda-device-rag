from fda_device_rag.models import Chunk, ChunkMetadata


def recall_to_chunk(record: dict, retrieved_date: str) -> Chunk | None:
    """Build one Chunk from an openFDA device/recall record.

    Returns None for content-free records (no reason, action, or product
    description), which would otherwise be indexed as scaffolding-only text.

    Keyed on ``product_res_number``, not ``res_event_number``: one recall event
    can cover many product records, so ``res_event_number`` is many-to-one with
    records (verified live: 500 records -> 169 distinct event numbers, but
    500 distinct product_res_numbers).
    """
    reason = record.get("reason_for_recall", "") or ""
    action = record.get("action", "") or ""
    product = record.get("product_description", "") or ""

    if not (reason or action or product):
        return None

    text = f"Recall reason: {reason}\nAction taken: {action}\nProduct: {product}"

    record_id = str(record.get("product_res_number", ""))
    metadata = ChunkMetadata(
        source_type="recall",
        # %22 wraps the id in quotes for an exact-phrase match. openFDA tokenizes
        # on hyphens, so an unquoted "Z-0001-2025" matches ~58k records instead
        # of the one being cited (verified live).
        source_url=f"https://api.fda.gov/device/recall.json?search=product_res_number:%22{record_id}%22",
        document_title=product[:120] if product else "FDA Device Recall",
        section_name=None,
        record_id=record_id,
        retrieved_date=retrieved_date,
    )
    return Chunk(id=f"recall-{record_id}", text=text, metadata=metadata)


def event_to_chunk(record: dict, retrieved_date: str) -> Chunk | None:
    """Build one Chunk from an openFDA device/event (MAUDE) record.

    Returns None when the record carries no narrative text at all, since the
    templated scaffolding alone conveys no retrievable information.
    """
    mdr_texts = record.get("mdr_text", []) or []
    narrative = "\n".join(t.get("text", "") for t in mdr_texts if t.get("text"))

    if not narrative.strip():
        return None

    report_number = str(record.get("report_number", ""))
    device_list = record.get("device", [])
    device_name = device_list[0].get("generic_name", "Unknown device") if device_list else "Unknown device"

    text = f"Adverse event narrative ({device_name}): {narrative}"
    metadata = ChunkMetadata(
        source_type="maude_event",
        # Quoted for the same reason as the recall url: MAUDE report numbers are
        # hyphenated, and unquoted they match millions of records.
        source_url=f"https://api.fda.gov/device/event.json?search=report_number:%22{report_number}%22",
        document_title=device_name,
        section_name=None,
        record_id=report_number,
        retrieved_date=retrieved_date,
    )
    return Chunk(id=f"event-{report_number}", text=text, metadata=metadata)
