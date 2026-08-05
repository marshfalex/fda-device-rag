from fda_device_rag.documents.structured import event_to_chunk, recall_to_chunk


def test_recall_to_chunk_builds_templated_text():
    record = {
        # res_event_number is deliberately kept here (real records carry both),
        # but it is many-to-one with records and must NOT be used as the key.
        "res_event_number": "27549",
        "product_res_number": "Z-0001-04",
        "reason_for_recall": "Source-skin-distance non-compliance.",
        "action": "Safety notice sent to the direct account.",
        "product_description": "Sedecal SP-HF 4.0 Portable X-Ray System",
    }

    chunk = recall_to_chunk(record, retrieved_date="2026-07-28")

    assert chunk.id == "recall-Z-0001-04"
    assert "Source-skin-distance non-compliance." in chunk.text
    assert "Safety notice sent to the direct account." in chunk.text
    assert "Sedecal SP-HF 4.0 Portable X-Ray System" in chunk.text
    assert chunk.metadata.source_type == "recall"
    assert chunk.metadata.record_id == "Z-0001-04"
    # Quoted (%22) for an exact-phrase match -- openFDA tokenizes on hyphens,
    # so an unquoted hyphenated id matches tens of thousands of records.
    assert chunk.metadata.source_url == (
        "https://api.fda.gov/device/recall.json?search=product_res_number:%22Z-0001-04%22"
    )
    assert chunk.metadata.retrieved_date == "2026-07-28"


def test_recall_to_chunk_ids_are_unique_across_one_shared_recall_event():
    """One recall event covers many product records; the chunk id must still
    be unique per record, or ChromaStore.add_chunks hits duplicate ids."""
    shared_event = "27549"
    records = [
        {
            "res_event_number": shared_event,
            "product_res_number": f"Z-000{n}-04",
            "reason_for_recall": "Same underlying defect.",
            "action": "Recall notice.",
            "product_description": f"Product variant {n}",
        }
        for n in (1, 2, 3)
    ]

    ids = [recall_to_chunk(r, retrieved_date="2026-07-28").id for r in records]

    assert ids == ["recall-Z-0001-04", "recall-Z-0002-04", "recall-Z-0003-04"]
    assert len(set(ids)) == 3


def test_recall_to_chunk_returns_none_when_all_narrative_fields_blank():
    record = {
        "product_res_number": "Z-0009-04",
        "reason_for_recall": "",
        "action": "",
        "product_description": "",
    }

    assert recall_to_chunk(record, retrieved_date="2026-07-28") is None


def test_recall_to_chunk_returns_none_when_narrative_fields_missing():
    assert recall_to_chunk({"product_res_number": "Z-0009-04"}, retrieved_date="2026-07-28") is None


def test_recall_to_chunk_returns_none_when_product_res_number_missing():
    """A chunk needs a valid id to be indexed at all -- a missing/empty
    product_res_number would otherwise silently produce record_id="" and
    Chunk.id == "recall-", colliding with any other such record (the same
    failure class as the original duplicate-id finding, via a different
    missing field)."""
    record = {
        "reason_for_recall": "Battery defect causing overheating.",
        "action": "Field safety notice issued.",
        "product_description": "Portable Infusion Pump Model Z",
    }

    assert recall_to_chunk(record, retrieved_date="2026-07-28") is None


def test_recall_to_chunk_returns_none_when_product_res_number_blank():
    record = {
        "product_res_number": "",
        "reason_for_recall": "Battery defect causing overheating.",
        "action": "Field safety notice issued.",
        "product_description": "Portable Infusion Pump Model Z",
    }

    assert recall_to_chunk(record, retrieved_date="2026-07-28") is None


def test_recall_to_chunk_kept_when_only_one_narrative_field_present():
    record = {"product_res_number": "Z-0010-04", "reason_for_recall": "Battery defect."}

    chunk = recall_to_chunk(record, retrieved_date="2026-07-28")

    assert chunk is not None
    assert "Battery defect." in chunk.text


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
    assert chunk.metadata.source_url == (
        "https://api.fda.gov/device/event.json?search=report_number:%2210%22"
    )


def test_event_to_chunk_handles_missing_device_list():
    record = {"report_number": "11", "mdr_text": [{"text": "Only narrative."}], "device": []}

    chunk = event_to_chunk(record, retrieved_date="2026-07-28")

    assert chunk.metadata.document_title == "Unknown device"


def test_event_to_chunk_returns_none_when_narrative_blank():
    record = {
        "report_number": "12",
        "mdr_text": [{"text": ""}, {"text": "   "}],
        "device": [{"generic_name": "MANUAL HOSPITAL BED"}],
    }

    assert event_to_chunk(record, retrieved_date="2026-07-28") is None


def test_event_to_chunk_returns_none_when_mdr_text_missing():
    record = {"report_number": "13", "device": [{"generic_name": "MANUAL HOSPITAL BED"}]}

    assert event_to_chunk(record, retrieved_date="2026-07-28") is None
