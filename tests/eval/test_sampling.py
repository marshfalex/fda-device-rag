import pytest

from fda_device_rag.eval.sampling import (
    EmptyCandidatePoolError,
    sample_recall_candidates,
    sample_event_candidates,
)


def _recall_record(product_res_number, category, reason="Some genuinely detailed recall reason text here.", action="Customers were notified by letter.", product_description="Example infusion pump device."):
    return {
        "product_res_number": product_res_number,
        "root_cause_description": category,
        "reason_for_recall": reason,
        "action": action,
        "product_description": product_description,
    }


def _event_record(report_number, product_problems, narrative="A detailed adverse event narrative describing what happened."):
    return {
        "report_number": report_number,
        "product_problems": product_problems,
        "mdr_text": [{"text": narrative}],
        "device": [{"generic_name": "Infusion Pump"}],
    }


def test_sample_recall_candidates_filters_by_category_and_returns_one_per_category():
    records = [
        _recall_record("Z-01", "Process control"),
        _recall_record("Z-02", "Device Design"),
        _recall_record("Z-03", "Process control"),
    ]

    results = sample_recall_candidates(
        records, retrieved_date="2026-08-03", base_seed="test-seed",
        categories=["Process control", "Device Design"],
    )

    by_category = {r["category"]: r for r in results}
    assert set(by_category) == {"Process control", "Device Design"}
    assert by_category["Process control"]["record_id"] in ("Z-01", "Z-03")
    assert by_category["Device Design"]["record_id"] == "Z-02"
    assert by_category["Process control"]["source_type"] == "recall"


def test_sample_recall_candidates_excludes_records_under_the_floor():
    records = [
        _recall_record("Z-01", "Process control", reason="short", action="", product_description=""),
        _recall_record("Z-02", "Process control", reason="A much longer reason for recall that clearly exceeds the length floor easily."),
    ]

    results = sample_recall_candidates(
        records, retrieved_date="2026-08-03", base_seed="test-seed",
        categories=["Process control"], floor=100,
    )

    assert results[0]["record_id"] == "Z-02"


def test_sample_recall_candidates_is_deterministic_for_a_given_seed():
    records = [_recall_record(f"Z-{i:02d}", "Process control") for i in range(10)]

    first = sample_recall_candidates(records, retrieved_date="2026-08-03", base_seed="fixed-seed", categories=["Process control"])
    second = sample_recall_candidates(records, retrieved_date="2026-08-03", base_seed="fixed-seed", categories=["Process control"])

    assert first[0]["record_id"] == second[0]["record_id"]


def test_sample_recall_candidates_raises_on_empty_category_pool():
    records = [_recall_record("Z-01", "Process control")]

    with pytest.raises(EmptyCandidatePoolError):
        sample_recall_candidates(records, retrieved_date="2026-08-03", base_seed="test-seed", categories=["Device Design"])


def test_sample_event_candidates_matches_by_list_membership_not_equality():
    records = [
        _event_record("E-01", ["Defective Device", "Pumping Stopped"]),
        _event_record("E-02", ["No Apparent Adverse Event"]),
    ]

    results = sample_event_candidates(
        records, retrieved_date="2026-08-03", base_seed="test-seed",
        categories=["Defective Device", "Pumping Stopped", "No Apparent Adverse Event"],
    )

    by_category = {r["category"]: r for r in results}
    assert by_category["Defective Device"]["record_id"] == "E-01"
    assert by_category["Pumping Stopped"]["record_id"] == "E-01"
    assert by_category["No Apparent Adverse Event"]["record_id"] == "E-02"
    assert by_category["Defective Device"]["source_type"] == "maude"


def test_sample_event_candidates_excludes_records_under_the_floor():
    records = [
        _event_record("E-01", ["Defective Device"], narrative="short"),
        _event_record("E-02", ["Defective Device"], narrative="A much longer adverse event narrative that clearly exceeds the length floor."),
    ]

    results = sample_event_candidates(
        records, retrieved_date="2026-08-03", base_seed="test-seed",
        categories=["Defective Device"], floor=100,
    )

    assert results[0]["record_id"] == "E-02"
