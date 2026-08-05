import json
from pathlib import Path
from unittest.mock import patch

import pytest

from fda_device_rag.eval.gold_resolution import (
    GoldResolutionError,
    _find_pdf_path,
    resolve_gold,
    resolve_pdf_gold_from_text,
    resolve_structured_gold,
)
from fda_device_rag.eval.questions import Question


def test_resolve_structured_gold_recall_returns_matching_chunk_id(tmp_path):
    (tmp_path / "recalls.json").write_text(json.dumps([
        {"product_res_number": "Z-01", "reason_for_recall": "A detailed reason.", "action": "Notified.", "product_description": "Pump."},
        {"product_res_number": "Z-02", "reason_for_recall": "Another reason.", "action": "Notified.", "product_description": "Valve."},
    ]))

    result = resolve_structured_gold("recall", "Z-01", data_dir=tmp_path)

    assert result == ["recall-Z-01"]


def test_resolve_structured_gold_maude_returns_matching_chunk_id(tmp_path):
    (tmp_path / "events.json").write_text(json.dumps([
        {"report_number": "E-01", "mdr_text": [{"text": "A narrative."}], "device": [{"generic_name": "Pump"}]},
    ]))

    result = resolve_structured_gold("maude", "E-01", data_dir=tmp_path)

    assert result == ["event-E-01"]


def test_resolve_structured_gold_raises_when_record_not_found(tmp_path):
    (tmp_path / "recalls.json").write_text(json.dumps([{"product_res_number": "Z-01", "reason_for_recall": "x", "action": "y", "product_description": "z"}]))

    with pytest.raises(GoldResolutionError, match="not found"):
        resolve_structured_gold("recall", "Z-99", data_dir=tmp_path)


def test_resolve_structured_gold_raises_when_matched_record_is_content_free(tmp_path):
    (tmp_path / "recalls.json").write_text(json.dumps([
        {"product_res_number": "Z-01", "reason_for_recall": "", "action": "", "product_description": ""},
    ]))

    with pytest.raises(GoldResolutionError, match="no longer produces a chunk"):
        resolve_structured_gold("recall", "Z-01", data_dir=tmp_path)


def test_resolve_pdf_gold_from_text_returns_chunk_ids_for_matching_instance():
    text = (
        "WARNINGS\n"
        "First warning content that is long enough to survive filtering easily.\n"
        "CONTRAINDICATIONS\n"
        "Do not use this device on patients with known allergies to latex.\n"
    )

    result = resolve_pdf_gold_from_text("doc", text, "CONTRAINDICATIONS", 1)

    assert result == ["doc-1"]


def test_resolve_pdf_gold_from_text_disambiguates_repeated_heading_by_ordinal():
    text = (
        "GETTING STARTED\n"
        "Page one content that is long enough to survive filtering easily here.\n"
        "MAINTENANCE\n"
        "Maintenance content that is also long enough to survive filtering here.\n"
        "GETTING STARTED\n"
        "Page two content that is long enough to survive filtering easily here too.\n"
    )

    first_instance = resolve_pdf_gold_from_text("doc", text, "GETTING STARTED", 1)
    second_instance = resolve_pdf_gold_from_text("doc", text, "GETTING STARTED", 2)

    assert first_instance != second_instance


def test_resolve_pdf_gold_from_text_raises_on_out_of_range_ordinal():
    text = "WARNINGS\nSome content that is long enough to survive the filter easily.\n"

    with pytest.raises(GoldResolutionError, match="no section instance found"):
        resolve_pdf_gold_from_text("doc", text, "WARNINGS", 2)


def test_find_pdf_path_locates_file_in_either_subdirectory(tmp_path):
    (tmp_path / "guidance_pdfs").mkdir()
    (tmp_path / "guidance_pdfs" / "188844.pdf").write_bytes(b"")
    (tmp_path / "ifu_pdfs").mkdir()
    (tmp_path / "ifu_pdfs" / "Z-800F.pdf").write_bytes(b"")

    assert _find_pdf_path("188844", data_dir=tmp_path) == tmp_path / "guidance_pdfs" / "188844.pdf"
    assert _find_pdf_path("Z-800F", data_dir=tmp_path) == tmp_path / "ifu_pdfs" / "Z-800F.pdf"


def test_find_pdf_path_raises_when_not_found(tmp_path):
    (tmp_path / "guidance_pdfs").mkdir()
    (tmp_path / "ifu_pdfs").mkdir()

    with pytest.raises(GoldResolutionError, match="no PDF found"):
        _find_pdf_path("missing", data_dir=tmp_path)


@patch("fda_device_rag.eval.gold_resolution.resolve_structured_gold")
def test_resolve_gold_dispatches_recall_and_maude_to_structured_resolver(mock_resolve):
    mock_resolve.return_value = ["recall-Z-01"]
    question = Question(question_id="q1", source_type="recall", category=None, question="x?", gold={"record_id": "Z-01"}, notes=None)

    result = resolve_gold(question)

    mock_resolve.assert_called_once_with("recall", "Z-01", data_dir=Path("data/raw"))
    assert result == ["recall-Z-01"]


@patch("fda_device_rag.eval.gold_resolution.resolve_pdf_gold")
def test_resolve_gold_dispatches_guidance_and_ifu_to_pdf_resolver(mock_resolve):
    mock_resolve.return_value = ["188844-3"]
    question = Question(
        question_id="q2", source_type="guidance", category=None, question="x?",
        gold={"document_title": "188844", "section_name": "Risk-Based Analysis", "instance_ordinal": 1}, notes=None,
    )

    result = resolve_gold(question)

    mock_resolve.assert_called_once_with(
        "188844", "Risk-Based Analysis", 1, source_type="guidance_pdf", data_dir=Path("data/raw")
    )
    assert result == ["188844-3"]


def test_resolve_gold_raises_gold_resolution_error_when_structured_gold_missing_record_id():
    # The frozen-file loader validates that a 'gold' key exists but not its
    # inner fields, so a malformed gold dict reaches resolve_gold. It must
    # surface as GoldResolutionError -- a bare KeyError would escape an eval
    # runner's `except GoldResolutionError` handler and abort the whole run
    # instead of being recorded as one question's resolution failure.
    question = Question(
        question_id="q1", source_type="recall", category=None, question="x?",
        gold={"reason": "oops, no record_id"}, notes=None,
    )

    with pytest.raises(GoldResolutionError, match="missing required gold field 'record_id'"):
        resolve_gold(question)


@pytest.mark.parametrize("missing_field", ["document_title", "section_name", "instance_ordinal"])
def test_resolve_gold_raises_gold_resolution_error_when_pdf_gold_missing_a_field(missing_field):
    gold = {"document_title": "188844", "section_name": "Risk-Based Analysis", "instance_ordinal": 1}
    del gold[missing_field]
    question = Question(
        question_id="q2", source_type="guidance", category=None, question="x?", gold=gold, notes=None,
    )

    with pytest.raises(GoldResolutionError, match=f"missing required gold field\\(s\\).*{missing_field}"):
        resolve_gold(question)
