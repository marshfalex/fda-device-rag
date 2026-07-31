from fda_device_rag.chunking.pdf_chunker import chunk_pdf_text


def test_chunk_pdf_text_keeps_short_section_as_one_chunk():
    text = "WARNINGS\nMay cause mild irritation in rare cases.\n"

    chunks = chunk_pdf_text(
        text,
        source_type="ifu_pdf",
        source_url="https://example.com/ifu-1.pdf",
        document_title="Example IFU",
        retrieved_date="2026-07-28",
        id_prefix="ifu-1",
    )

    assert len(chunks) == 1
    assert chunks[0].metadata.section_name == "WARNINGS"
    assert "mild irritation" in chunks[0].text
    assert chunks[0].metadata.source_type == "ifu_pdf"
    assert chunks[0].id == "ifu-1-0"


def test_chunk_pdf_text_splits_oversized_section_with_overlap():
    long_body = "This sentence repeats. " * 200  # well over 1600 chars
    text = f"CONTRAINDICATIONS\n{long_body}"

    chunks = chunk_pdf_text(
        text,
        source_type="guidance_pdf",
        source_url="https://example.com/guidance-1.pdf",
        document_title="Example Guidance",
        retrieved_date="2026-07-28",
        id_prefix="guidance-1",
    )

    assert len(chunks) > 1
    assert all(c.metadata.section_name == "CONTRAINDICATIONS" for c in chunks)
    assert all(len(c.text) <= 1600 for c in chunks)
    # ids are unique and ordered
    assert [c.id for c in chunks] == [f"guidance-1-{i}" for i in range(len(chunks))]


def test_chunk_pdf_text_handles_no_headings():
    text = "Just plain narrative text with no headings at all."

    chunks = chunk_pdf_text(
        text,
        source_type="guidance_pdf",
        source_url="https://example.com/guidance-2.pdf",
        document_title="Example Guidance 2",
        retrieved_date="2026-07-28",
        id_prefix="guidance-2",
    )

    assert len(chunks) == 1
    assert chunks[0].metadata.section_name == "Document"


def test_chunk_pdf_text_drops_chunks_under_min_length():
    text = "MAINTENANCE\nOK\nWARNINGS\nDo not operate this device while it is still charging today.\n"

    chunks = chunk_pdf_text(
        text,
        source_type="ifu_pdf",
        source_url="https://example.com/ifu-2.pdf",
        document_title="Example IFU 2",
        retrieved_date="2026-07-28",
        id_prefix="ifu-2",
    )

    assert len(chunks) == 1
    assert chunks[0].metadata.section_name == "WARNINGS"
    assert "Do not operate" in chunks[0].text
    assert chunks[0].id == "ifu-2-0"


def test_chunk_pdf_text_keeps_chunk_at_exactly_min_length_boundary():
    exactly_40_chars = "x" * 40
    assert len(exactly_40_chars) == 40
    text = f"WARNINGS\n{exactly_40_chars}\n"

    chunks = chunk_pdf_text(
        text,
        source_type="ifu_pdf",
        source_url="https://example.com/ifu-3.pdf",
        document_title="Example IFU 3",
        retrieved_date="2026-07-28",
        id_prefix="ifu-3",
    )

    assert len(chunks) == 1
    assert chunks[0].text == f"WARNINGS\n{exactly_40_chars}"


def test_chunk_pdf_text_prepends_heading_to_first_piece_of_each_section():
    text = "WARNINGS\nMay cause mild irritation in rare cases and requires monitoring.\n"

    chunks = chunk_pdf_text(
        text,
        source_type="ifu_pdf",
        source_url="https://example.com/ifu-4.pdf",
        document_title="Example IFU 4",
        retrieved_date="2026-07-28",
        id_prefix="ifu-4",
    )

    assert chunks[0].text.startswith("WARNINGS\n")


def test_chunk_pdf_text_drops_title_case_false_positive_heading_chunks():
    text = (
        "WARNINGS\n"
        "First warning content that is long enough to survive filtering easily.\n"
        "Short Label\n"
        "AB\n"
        "CONTRAINDICATIONS\n"
        "Do not use this device on patients with known allergies to latex.\n"
    )

    chunks = chunk_pdf_text(
        text,
        source_type="ifu_pdf",
        source_url="https://example.com/ifu-5.pdf",
        document_title="Example IFU 5",
        retrieved_date="2026-07-28",
        id_prefix="ifu-5",
    )

    section_names = [c.metadata.section_name for c in chunks]
    assert "Short Label" not in section_names
    assert "WARNINGS" in section_names
    assert "CONTRAINDICATIONS" in section_names
