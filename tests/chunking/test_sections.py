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
