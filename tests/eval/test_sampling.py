import pytest

from fda_device_rag.eval.sampling import (
    EmptyCandidatePoolError,
    SectionDiversityError,
    check_section_diversity,
    sample_event_candidates,
    sample_pdf_section_candidates,
    sample_recall_candidates,
)
from fda_device_rag.models import Chunk, ChunkMetadata


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


def _pdf_chunk(id_, doc, section, text):
    return Chunk(
        id=id_,
        text=text,
        metadata=ChunkMetadata(
            source_type="ifu_pdf",
            source_url="https://example.com/doc.pdf",
            document_title=doc,
            section_name=section,
            retrieved_date="2026-08-03",
        ),
    )


def test_sample_pdf_section_candidates_excludes_instances_under_the_floor():
    chunks = [
        _pdf_chunk("doc-0", "doc", "TOO SHORT", "TOO SHORT\nTiny."),  # well under 72 chars
        _pdf_chunk("doc-1", "doc", "WARNINGS", "WARNINGS\n" + "A" * 80),
    ]

    selected, skip_log = sample_pdf_section_candidates("doc", chunks, quota=1, base_seed="test-seed", floor=72)

    assert len(selected) == 1
    assert selected[0].section_name == "WARNINGS"


def _furniture_chunks():
    """A document with one 2-member furniture cluster (the same footer
    template recurring under the same heading, differing only in page number)
    plus one genuinely distinct section."""
    long_body = "A" * 60
    return [
        _pdf_chunk("doc-0", "doc", "MAINTENANCE", f"MAINTENANCE\nZ-800F Instructions for Use.  15 \nP/N 800F-IFU-2602, Rev. O {long_body}"),
        _pdf_chunk("doc-1", "doc", "TROUBLESHOOTING", "TROUBLESHOOTING\nGenuinely distinct real content about troubleshooting alarms."),
        _pdf_chunk("doc-2", "doc", "MAINTENANCE", f"MAINTENANCE\nZ-800F Instructions for Use.  21 \nP/N 800F-IFU-2602, Rev. O {long_body}"),
    ]


def test_sample_pdf_section_candidates_excludes_furniture_duplicates():
    # Both members of the furniture cluster are excluded -- including the
    # first-seen one, which is only "first" by scan order over the pool.
    selected, skip_log = sample_pdf_section_candidates("doc", _furniture_chunks(), quota=1, base_seed="test-seed", floor=72)

    assert [i.section_name for i in selected] == ["TROUBLESHOOTING"]
    assert len(skip_log) == 2
    assert {entry[0] for entry in skip_log} == {"doc:MAINTENANCE#1", "doc:MAINTENANCE#2"}


def test_sample_pdf_section_candidates_furniture_cluster_does_not_backfill_the_quota():
    # Only one non-furniture instance survives, so a quota of 2 cannot be met
    # by falling back on a furniture cluster's first-seen representative.
    with pytest.raises(EmptyCandidatePoolError, match="only 1 non-furniture"):
        sample_pdf_section_candidates("doc", _furniture_chunks(), quota=2, base_seed="test-seed", floor=72)


def test_sample_pdf_section_candidates_excludes_every_member_of_a_three_member_cluster():
    # Three near-identical running footers (differing only by page number
    # 1/2/3, all under the same heading) plus one genuinely distinct section.
    # The whole cluster is undrawable; only the distinct instance is.
    long_body = "B" * 60
    chunks = [
        _pdf_chunk("doc-0", "doc", "FOOTER", f"FOOTER\nAcme 900X Instructions for Use. Page 1 \n{long_body}"),
        _pdf_chunk("doc-1", "doc", "CLEANING", "CLEANING\nWipe the exterior housing with a soft damp cloth after every use."),
        _pdf_chunk("doc-2", "doc", "FOOTER", f"FOOTER\nAcme 900X Instructions for Use. Page 2 \n{long_body}"),
        _pdf_chunk("doc-3", "doc", "CLEANING2", "CLEANING2\nA second distinct section so the pool is not trivially size one."),
        _pdf_chunk("doc-4", "doc", "FOOTER", f"FOOTER\nAcme 900X Instructions for Use. Page 3 \n{long_body}"),
    ]

    selected, skip_log = sample_pdf_section_candidates("doc", chunks, quota=2, base_seed="test-seed", floor=72)

    assert sorted(i.section_name for i in selected) == ["CLEANING", "CLEANING2"]
    # All three footer instances are logged as skipped, none survive the draw.
    assert len(skip_log) == 3
    assert {entry[0] for entry in skip_log} == {"doc:FOOTER#1", "doc:FOOTER#2", "doc:FOOTER#3"}
    assert not any(i.section_name == "FOOTER" for i in selected)


def test_sample_pdf_section_candidates_raises_when_only_a_furniture_cluster_is_eligible():
    long_body = "C" * 60
    chunks = [
        _pdf_chunk("doc-0", "doc", "FOOTER", f"FOOTER\nAcme 900X Instructions for Use. Page 1 \n{long_body}"),
        # Below the floor, so it never enters the eligible pool -- it only
        # breaks the contiguous run so the two footers are distinct instances.
        _pdf_chunk("doc-1", "doc", "TINY", "TINY\nShort."),
        _pdf_chunk("doc-2", "doc", "FOOTER", f"FOOTER\nAcme 900X Instructions for Use. Page 2 \n{long_body}"),
    ]

    with pytest.raises(EmptyCandidatePoolError, match="only 0 non-furniture"):
        sample_pdf_section_candidates("doc", chunks, quota=1, base_seed="test-seed", floor=72)


def test_sample_pdf_section_candidates_keeps_distinct_content_sharing_a_digit_template():
    chunks = [
        _pdf_chunk("doc-0", "doc", "Single-Needle Sets", "Single-Needle Sets\nLength Item # Residual Vol. p/ Box\n6 mm RMS12406 0.4 ml 20\n9 mm RMS12409 0.4 ml 20"),
        _pdf_chunk("doc-1", "doc", "Two-Needle Sets", "Two-Needle Sets\nLength Item # Residual Vol. p/ Box\n6 mm RMS22406 0.7 ml 10\n9 mm RMS22409 0.7 ml 10"),
    ]

    selected, skip_log = sample_pdf_section_candidates("doc", chunks, quota=2, base_seed="test-seed", floor=72)

    assert len(selected) == 2
    assert skip_log == []


def test_sample_pdf_section_candidates_is_deterministic_for_a_given_seed():
    chunks = [
        _pdf_chunk(f"doc-{i}", "doc", f"SECTION_{i}", f"SECTION_{i}\nModel ABC1000{i} specifications: voltage 10000V, current 200mA, power 2000W")
        for i in range(5)
    ]

    first, _ = sample_pdf_section_candidates("doc", chunks, quota=2, base_seed="fixed-seed", floor=72)
    second, _ = sample_pdf_section_candidates("doc", chunks, quota=2, base_seed="fixed-seed", floor=72)

    assert [i.section_name for i in first] == [i.section_name for i in second]


def test_sample_pdf_section_candidates_raises_when_pool_too_small_for_quota():
    chunks = [_pdf_chunk("doc-0", "doc", "ONLY SECTION", "ONLY SECTION\n" + "A" * 80)]

    with pytest.raises(EmptyCandidatePoolError):
        sample_pdf_section_candidates("doc", chunks, quota=2, base_seed="test-seed", floor=72)


def _pdf_candidate(document_title, section_name, instance_ordinal=1):
    return {
        "source_type": "ifu_pdf",
        "document_title": document_title,
        "section_name": section_name,
        "instance_ordinal": instance_ordinal,
        "text": "some candidate text",
    }


def test_check_section_diversity_raises_when_one_document_has_two_candidates_in_a_section():
    candidates = [
        _pdf_candidate("doc-a", "WARNINGS", instance_ordinal=1),
        _pdf_candidate("doc-a", "MAINTENANCE"),
        _pdf_candidate("doc-a", "WARNINGS", instance_ordinal=2),
    ]

    with pytest.raises(SectionDiversityError, match="WARNINGS"):
        check_section_diversity(candidates)


def test_check_section_diversity_accepts_distinct_sections_within_each_document():
    candidates = [
        _pdf_candidate("doc-a", "WARNINGS"),
        _pdf_candidate("doc-a", "MAINTENANCE"),
        _pdf_candidate("doc-b", "TROUBLESHOOTING"),
    ]

    assert check_section_diversity(candidates) is None


def test_check_section_diversity_allows_the_same_section_name_in_different_documents():
    # The invariant is per-document: two documents legitimately share
    # generic heading labels like "WARNINGS", and that is not a violation.
    candidates = [
        _pdf_candidate("doc-a", "WARNINGS"),
        _pdf_candidate("doc-b", "WARNINGS"),
    ]

    assert check_section_diversity(candidates) is None
