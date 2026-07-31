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


def test_detect_sections_detects_title_case_headings():
    text = (
        "Preface\n"
        "This section provides background context for readers\n"
        "Table of Contents\n"
        "See the following pages for a full listing of sections\n"
    )

    sections = detect_sections(text)

    assert len(sections) == 2
    assert sections[0].heading == "Preface"
    assert "background context" in sections[0].body
    assert sections[1].heading == "Table of Contents"
    assert "full listing" in sections[1].body


def test_detect_sections_does_not_treat_prose_as_title_case_heading():
    text = (
        "WARNINGS\n"
        "This document provides guidance for reviewers\n"
        "and covers additional safety considerations\n"
    )

    sections = detect_sections(text)

    assert len(sections) == 1
    assert sections[0].heading == "WARNINGS"
    assert "provides guidance for reviewers" in sections[0].body


def test_detect_sections_rejects_catalog_code_rows_as_headings():
    text = (
        "ACCESSORIES\n"
        "F120 F180 F275 F420 F500\n"
        "Compatible tubing sets are listed above\n"
    )

    sections = detect_sections(text)

    assert len(sections) == 1
    assert sections[0].heading == "ACCESSORIES"
    assert "F120 F180 F275 F420 F500" in sections[0].body


def test_detect_sections_rejects_lines_ending_in_period_as_headings():
    text = (
        "MAINTENANCE\n"
        "Clean the exterior\n"
        "Pump.\n"
        "Store in a cool location\n"
    )

    sections = detect_sections(text)

    assert len(sections) == 1
    assert sections[0].heading == "MAINTENANCE"
    assert "Pump." in sections[0].body


def test_detect_sections_rejects_numbered_list_items_as_headings():
    text = (
        "1. Introduction\n"
        "This is the intro paragraph\n"
        "5. Close the Roller Clamp and the Pinch Clamp.\n"
        "Continue with the next step\n"
    )

    sections = detect_sections(text)

    assert len(sections) == 1
    assert sections[0].heading == "1. Introduction"
    assert "5. Close the Roller Clamp and the Pinch Clamp." in sections[0].body


def test_detect_sections_does_not_treat_bare_page_number_as_numbered_heading():
    text = (
        "WARNINGS\n"
        "First warning content\n"
        "10 Device Manual Footer\n"
        "More warning content continues here\n"
    )

    sections = detect_sections(text)

    assert len(sections) == 1
    assert sections[0].heading == "WARNINGS"
    assert "10 Device Manual Footer" in sections[0].body


def test_detect_sections_suppresses_masthead_acronym_before_first_real_heading_but_not_after():
    text = (
        "CBER\n"
        "Cover page contact information here for reference\n"
        "PRODUCT OVERVIEW\n"
        "This section describes the product in detail here\n"
        "FLANGE\n"
        "Attach the flange to the connector body here\n"
    )

    sections = detect_sections(text)

    assert [s.heading for s in sections] == ["Document", "PRODUCT OVERVIEW", "FLANGE"]
    assert "CBER" in sections[0].body
    assert "Attach the flange" in sections[2].body


def test_detect_sections_rejects_lines_starting_with_lowercase_connector():
    text = (
        "WARNINGS\n"
        "First warning body content here for context\n"
        "and Open the access panel before proceeding\n"
        "CONTRAINDICATIONS\n"
        "Do not use in patients with known allergies\n"
    )

    sections = detect_sections(text)

    assert len(sections) == 2
    assert sections[0].heading == "WARNINGS"
    assert "and Open the access panel" in sections[0].body
    assert sections[1].heading == "CONTRAINDICATIONS"


def test_detect_sections_rejects_title_case_catalog_code_rows_as_headings():
    text = (
        "ACCESSORIES\n"
        "Widget1 Adapter2 Bracket3\n"
        "Compatible parts are listed above for reference\n"
    )

    sections = detect_sections(text)

    assert len(sections) == 1
    assert sections[0].heading == "ACCESSORIES"
    assert "Widget1 Adapter2 Bracket3" in sections[0].body
