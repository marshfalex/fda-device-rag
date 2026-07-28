from fda_device_rag.models import Chunk, ChunkMetadata


def recall_to_chunk(record: dict, retrieved_date: str) -> Chunk:
    reason = record.get("reason_for_recall", "")
    action = record.get("action", "")
    product = record.get("product_description", "")
    text = f"Recall reason: {reason}\nAction taken: {action}\nProduct: {product}"

    record_id = str(record.get("res_event_number", ""))
    metadata = ChunkMetadata(
        source_type="recall",
        source_url=f"https://api.fda.gov/device/recall.json?search=res_event_number:{record_id}",
        document_title=product[:120] if product else "FDA Device Recall",
        section_name=None,
        record_id=record_id,
        retrieved_date=retrieved_date,
    )
    return Chunk(id=f"recall-{record_id}", text=text, metadata=metadata)


def event_to_chunk(record: dict, retrieved_date: str) -> Chunk:
    mdr_texts = record.get("mdr_text", [])
    narrative = "\n".join(t.get("text", "") for t in mdr_texts if t.get("text"))

    report_number = str(record.get("report_number", ""))
    device_list = record.get("device", [])
    device_name = device_list[0].get("generic_name", "Unknown device") if device_list else "Unknown device"

    text = f"Adverse event narrative ({device_name}): {narrative}"
    metadata = ChunkMetadata(
        source_type="maude_event",
        source_url=f"https://api.fda.gov/device/event.json?search=report_number:{report_number}",
        document_title=device_name,
        section_name=None,
        record_id=report_number,
        retrieved_date=retrieved_date,
    )
    return Chunk(id=f"event-{report_number}", text=text, metadata=metadata)
