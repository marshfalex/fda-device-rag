from fda_device_rag.documents.structured import recall_to_chunk, event_to_chunk


def test_recall_to_chunk_builds_templated_text():
    record = {
        "res_event_number": "27549",
        "reason_for_recall": "Source-skin-distance non-compliance.",
        "action": "Safety notice sent to the direct account.",
        "product_description": "Sedecal SP-HF 4.0 Portable X-Ray System",
    }

    chunk = recall_to_chunk(record, retrieved_date="2026-07-28")

    assert chunk.id == "recall-27549"
    assert "Source-skin-distance non-compliance." in chunk.text
    assert "Safety notice sent to the direct account." in chunk.text
    assert "Sedecal SP-HF 4.0 Portable X-Ray System" in chunk.text
    assert chunk.metadata.source_type == "recall"
    assert chunk.metadata.record_id == "27549"
    assert chunk.metadata.retrieved_date == "2026-07-28"


def test_event_to_chunk_concatenates_mdr_text_entries():
    record = {
        "report_number": "10",
        "mdr_text": [
            {"text": "First narrative sentence."},
            {"text": "Second narrative sentence."},
        ],
        "device": [{"generic_name": "MANUAL HOSPITAL BED"}],
    }

    chunk = event_to_chunk(record, retrieved_date="2026-07-28")

    assert chunk.id == "event-10"
    assert "First narrative sentence." in chunk.text
    assert "Second narrative sentence." in chunk.text
    assert "MANUAL HOSPITAL BED" in chunk.text
    assert chunk.metadata.source_type == "maude_event"
    assert chunk.metadata.record_id == "10"


def test_event_to_chunk_handles_missing_device_list():
    record = {"report_number": "11", "mdr_text": [{"text": "Only narrative."}], "device": []}

    chunk = event_to_chunk(record, retrieved_date="2026-07-28")

    assert chunk.metadata.document_title == "Unknown device"
