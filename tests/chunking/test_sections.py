from fda_device_rag.chunking.sections import Section, detect_sections


def test_detect_sections_splits_on_all_caps_headings():
    text = (
        "INDICATIONS FOR USE\n"
        "This device is indicated for short-term use.\n"
        "It should be used under supervision.\n"
        "CONTRAINDICATIONS\n"
        "Do not use on patients with known allergies.\n"
    )

    sections = detect_sections(text)

    assert len(sections) == 2
    assert sections[0].heading == "INDICATIONS FOR USE"
    assert "short-term use" in sections[0].body
    assert sections[1].heading == "CONTRAINDICATIONS"
    assert "known allergies" in sections[1].body


def test_detect_sections_splits_on_numbered_headings():
    text = (
        "1. INDICATIONS FOR USE\n"
        "This device is indicated for short-term use.\n"
        "2. WARNINGS\n"
        "May cause irritation.\n"
    )

    sections = detect_sections(text)

    assert len(sections) == 2
    assert sections[0].heading == "1. INDICATIONS FOR USE"
    assert sections[1].heading == "2. WARNINGS"


def test_detect_sections_falls_back_to_single_document_section_when_no_headings():
    text = "Just plain narrative text with no headings at all, spanning a few sentences."

    sections = detect_sections(text)

    assert len(sections) == 1
    assert sections[0].heading == "Document"
    assert sections[0].body == text


def test_detect_sections_drops_empty_sections():
    text = "HEADING ONE\nHEADING TWO\nSome real body text here.\n"

    sections = detect_sections(text)

    assert len(sections) == 1
    assert sections[0].heading == "HEADING TWO"


def test_detect_sections_merges_repeated_running_header_across_pages():
    """extract_pdf_text joins pages with newlines, so a running page header
    appears once per page and would otherwise be detected as a fresh section
    boundary every time -- fragmenting one section into many identical stubs."""
    text = "\n".join(
        [
            "DEVICE INSTRUCTIONS FOR USE",
            "Page one body content about setup.",
            "DEVICE INSTRUCTIONS FOR USE",
            "Page two body content about operation.",
            "DEVICE INSTRUCTIONS FOR USE",
            "Page three body content about cleaning.",
        ]
    )

    sections = detect_sections(text)

    assert len(sections) == 1
    assert sections[0].heading == "DEVICE INSTRUCTIONS FOR USE"
    assert sections[0].body == (
        "Page one body content about setup.\n"
        "Page two body content about operation.\n"
        "Page three body content about cleaning."
    )


def test_detect_sections_does_not_merge_non_consecutive_repeats():
    text = (
        "WARNINGS\n"
        "First warning body.\n"
        "INDICATIONS\n"
        "Indication body.\n"
        "WARNINGS\n"
        "Second warning body.\n"
    )

    sections = detect_sections(text)

    assert [s.heading for s in sections] == ["WARNINGS", "INDICATIONS", "WARNINGS"]


def test_detect_sections_splits_on_multi_level_numbered_headings():
    text = (
        "2.1.4 RISKS OF USE\n"
        "Potential adverse reactions may occur.\n"
        "Always follow precautions.\n"
        "2.2 CONTRAINDICATIONS\n"
        "Do not use in patients with severe conditions.\n"
    )

    sections = detect_sections(text)

    assert len(sections) == 2
    assert sections[0].heading == "2.1.4 RISKS OF USE"
    assert "adverse reactions" in sections[0].body
    assert sections[1].heading == "2.2 CONTRAINDICATIONS"
    assert "severe conditions" in sections[1].body
