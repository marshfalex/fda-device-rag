# Section Detection Title-Case Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix `detect_sections` so it recognizes Title-Case headings (not just ALL-CAPS), which currently causes 94-100% of three real FDA guidance PDFs to collapse into one catch-all section; and add a chunk-length filter so the table/list-content false positives this exposes don't produce near-empty, BM25-bait chunks.

**Architecture:** Two independent, self-contained changes to already-existing, already-tested functions: (1) replace the heading-detection logic in `src/fda_device_rag/chunking/sections.py` with a small classifier (Title-Case detection, a real-word content guard, position-scoped masthead suppression, trailing-period rejection, period-required numbered prefixes); (2) add a minimum-chunk-length filter to `src/fda_device_rag/chunking/pdf_chunker.py`. Neither task changes any public interface (`Section`, `Chunk`, `ChunkMetadata`, `detect_sections`, `chunk_pdf_text` signatures are unchanged) — nothing downstream (embedder, stores, retriever, ingestion scripts) needs to change.

**Tech Stack:** Python 3.11+, `re` (stdlib), `pytest`. No new dependencies.

## Global Constraints

- No public interface changes: `detect_sections(text: str) -> list[Section]` and `chunk_pdf_text(text, source_type, source_url, document_title, retrieved_date, id_prefix) -> list[Chunk]` keep their exact existing signatures.
- All 10 existing tests across `tests/chunking/test_sections.py` (7 tests) and `tests/chunking/test_pdf_chunker.py` (3 tests) must continue to pass unchanged — this is a behavior-preserving-except-for-the-fix change, not a rewrite.
- Minimum chunk length is a fixed constant: **40 characters**, empirically validated against all 7 real corpus documents (design doc §4) — not configurable, not a guess.
- Masthead suppression only applies to a bare single ALL-CAPS token, 2-6 characters, with no numbered prefix, before any real (non-suppressed) heading has been accepted anywhere earlier in the same document.

---

### Task 1: Title-Case Heading Classifier

**Files:**
- Modify: `src/fda_device_rag/chunking/sections.py` (full replacement of `HEADING_PATTERN`/`_is_heading`/the `detect_sections` loop body — the `Section` dataclass and the existing merge-consecutive-heading post-processing at the end of `detect_sections` are unchanged)
- Test: `tests/chunking/test_sections.py` (append new tests; do not modify the 7 existing tests)

**Interfaces:**
- Consumes: nothing new — still pure stdlib `re`.
- Produces: `Section(heading: str, body: str)` and `detect_sections(text: str) -> list[Section]` — same signatures Task 8's `chunk_pdf_text` already depends on. No changes needed in `pdf_chunker.py` for this task.

- [ ] **Step 1: Write the failing tests — append to `tests/chunking/test_sections.py`**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/chunking/test_sections.py -v`
Expected: the 7 pre-existing tests PASS (unchanged behavior so far); the 7 new tests FAIL — specifically:
- `test_detect_sections_detects_title_case_headings`: fails because `Preface`/`Table of Contents` aren't ALL-CAPS, so today's code puts everything under one `Document` section instead of 2.
- `test_detect_sections_does_not_treat_prose_as_title_case_heading`: currently passes accidentally (prose isn't ALL-CAPS either) — if it already passes, that's fine, it's a regression guard for the fix, not a red/green requirement in isolation. The important RED signal is the six other new tests.
- The rest fail because today's code has no trailing-period rejection, no period-required numbered-prefix check, and no masthead position-scoping (today's code has a fixed masthead-length-only rule — actually today's code has no masthead suppression logic at all yet; see current file content in Step 3 before changing it).

- [ ] **Step 3: Replace the implementation — full new content for `src/fda_device_rag/chunking/sections.py`**

```python
import re
from dataclasses import dataclass

NUMBERED_PREFIX = re.compile(r"^\d+(?:(?:\.\d+)+\.?|\.)\s+")
ALLCAPS_BODY = re.compile(r"^[A-Z][A-Z0-9 ,\-/:]{2,80}$")
MASTHEAD_TOKEN = re.compile(r"^[A-Z]{2,6}$")
MIN_ALPHA_TOKEN_LEN = 3
MAX_TITLE_WORDS = 8


@dataclass
class Section:
    heading: str
    body: str


def _has_real_word(body: str) -> bool:
    for token in body.split():
        cleaned = token.rstrip(":,.;")
        if len(cleaned) >= MIN_ALPHA_TOKEN_LEN and cleaned.isalpha():
            return True
    return False


def _is_title_case(body: str) -> bool:
    words = body.rstrip(":").split()
    if not words or len(words) > MAX_TITLE_WORDS:
        return False
    has_long_word = False
    for word in words:
        cleaned = word.strip(",")
        if not cleaned or not cleaned[0].isalpha():
            return False
        if len(cleaned) <= 3:
            continue
        if not cleaned[0].isupper():
            return False
        has_long_word = True
    return has_long_word


def _classify(line: str, seen_real_heading: bool) -> tuple[bool, bool]:
    """Returns (is_heading, updated_seen_real_heading)."""
    stripped = line.strip()
    if not stripped or len(stripped) > 80 or stripped.endswith("."):
        return False, seen_real_heading

    prefix_match = NUMBERED_PREFIX.match(stripped)
    body = stripped[prefix_match.end():] if prefix_match else stripped

    is_allcaps = bool(ALLCAPS_BODY.match(body))
    is_title = _is_title_case(body) if not is_allcaps else False

    if not (is_allcaps or is_title):
        return False, seen_real_heading
    if not _has_real_word(body):
        return False, seen_real_heading

    is_masthead_candidate = (
        not prefix_match
        and is_allcaps
        and " " not in body
        and bool(MASTHEAD_TOKEN.match(body))
    )
    if is_masthead_candidate and not seen_real_heading:
        return False, seen_real_heading

    return True, True


def detect_sections(text: str) -> list[Section]:
    lines = text.splitlines()
    sections: list[Section] = []
    current_heading = "Document"
    current_body_lines: list[str] = []
    seen_real_heading = False

    for line in lines:
        is_heading, seen_real_heading = _classify(line, seen_real_heading)
        if is_heading:
            if current_body_lines:
                sections.append(Section(current_heading, "\n".join(current_body_lines).strip()))
            current_heading = line.strip()
            current_body_lines = []
        else:
            current_body_lines.append(line)

    if current_body_lines:
        sections.append(Section(current_heading, "\n".join(current_body_lines).strip()))

    non_empty = [s for s in sections if s.body]

    # A running header/footer repeated on every page of a PDF looks like a new
    # heading on each page, fragmenting one logical section into many
    # identically-named tiny ones. Merge consecutive sections that share the
    # exact same heading text back into a single section.
    merged: list[Section] = []
    for section in non_empty:
        if merged and merged[-1].heading == section.heading:
            merged[-1] = Section(section.heading, f"{merged[-1].body}\n{section.body}")
        else:
            merged.append(section)

    return merged
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/chunking/test_sections.py -v`
Expected: 14 passed (7 pre-existing + 7 new).

- [ ] **Step 5: Run the full test suite to confirm no regressions elsewhere**

Run: `pytest`
Expected: all tests pass (this module has no other internal consumers besides `pdf_chunker.py`, which is unchanged in this task, so the only risk is `tests/chunking/test_pdf_chunker.py` — confirm it still passes with the new classifier).

- [ ] **Step 6: Commit**

```bash
git add src/fda_device_rag/chunking/sections.py tests/chunking/test_sections.py
git commit -m "fix: detect Title-Case section headings, not just ALL-CAPS"
```

---

### Task 2: Minimum Chunk-Length Filter

**Files:**
- Modify: `src/fda_device_rag/chunking/pdf_chunker.py`
- Test: `tests/chunking/test_pdf_chunker.py` (append new tests; do not modify the 3 existing tests)

**Interfaces:**
- Consumes: `Section` and `detect_sections` from `fda_device_rag.chunking.sections` (Task 1, signature unchanged) — no change needed to this import.
- Produces: `chunk_pdf_text(...) -> list[Chunk]` — same signature as before; only the internal filtering behavior changes. Nothing downstream needs to change.

- [ ] **Step 1: Write the failing tests — append to `tests/chunking/test_pdf_chunker.py`**

```python
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
    assert chunks[0].text == exactly_40_chars
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/chunking/test_pdf_chunker.py -v`
Expected: the 3 pre-existing tests PASS; `test_chunk_pdf_text_drops_chunks_under_min_length` FAILS (today's code produces 2 chunks — `MAINTENANCE`'s 2-character `"OK"` chunk and `WARNINGS`'s real chunk — with `len(chunks) == 1` asserted, so it fails on that count, and `chunks[0].id == "ifu-2-0"` would also fail against today's id numbering since `MAINTENANCE` would currently occupy index 0). `test_chunk_pdf_text_keeps_chunk_at_exactly_min_length_boundary` passes today (nothing filters yet) — that's fine, it's a regression guard for the boundary, not required to be RED in isolation.

- [ ] **Step 3: Add the minimum-length filter — modify `src/fda_device_rag/chunking/pdf_chunker.py`**

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

from fda_device_rag.chunking.sections import detect_sections
from fda_device_rag.models import Chunk, ChunkMetadata

CHUNK_SIZE_CHARS = 1600
CHUNK_OVERLAP_CHARS = 240
MIN_CHUNK_CHARS = 40


def chunk_pdf_text(
    text: str,
    source_type: str,
    source_url: str,
    document_title: str,
    retrieved_date: str,
    id_prefix: str,
) -> list[Chunk]:
    sections = detect_sections(text)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE_CHARS,
        chunk_overlap=CHUNK_OVERLAP_CHARS,
    )

    chunks: list[Chunk] = []
    for section in sections:
        if len(section.body) <= CHUNK_SIZE_CHARS:
            pieces = [section.body]
        else:
            pieces = splitter.split_text(section.body)

        for piece in pieces:
            if len(piece) < MIN_CHUNK_CHARS:
                continue

            index = len(chunks)
            metadata = ChunkMetadata(
                source_type=source_type,
                source_url=source_url,
                document_title=document_title,
                section_name=section.heading,
                record_id=None,
                retrieved_date=retrieved_date,
            )
            chunks.append(Chunk(id=f"{id_prefix}-{index}", text=piece, metadata=metadata))

    return chunks
```

(The only change from the existing file: the new `MIN_CHUNK_CHARS = 40` constant and the `if len(piece) < MIN_CHUNK_CHARS: continue` guard at the top of the inner loop. Everything else — the splitter setup, the id/metadata construction — is unchanged.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/chunking/test_pdf_chunker.py -v`
Expected: 5 passed (3 pre-existing + 2 new).

- [ ] **Step 5: Run the full test suite**

Run: `pytest`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/fda_device_rag/chunking/pdf_chunker.py tests/chunking/test_pdf_chunker.py
git commit -m "fix: drop near-empty chunks under 40 characters"
```

---

### Task 3: Re-Validate Against the Real Corpus

**Files:**
- None created or modified — this is a validation-only task confirming the fix works against real data, not synthetic fixtures, before this work is considered done.

**Interfaces:**
- Consumes: `scripts/build_index.py` (existing, unchanged), the real PDFs already downloaded to `data/raw/guidance_pdfs/` and `data/raw/ifu_pdfs/` in this repo.
- Produces: nothing new — this task's deliverable is a passing validation, reported back, not a code artifact.

- [ ] **Step 1: Rebuild the index against the real corpus with the fixed chunker**

Run: `python scripts/build_index.py`
Expected: completes without error, prints a chunk count.

- [ ] **Step 2: Re-measure the swallowing metric for the 3 previously-broken documents**

```python
import pickle
from collections import Counter

with open("data/bm25_index.pkl", "rb") as f:
    bm25 = pickle.load(f)

for doc_title in ["78369", "188844", "153781"]:
    chunks = [c for c in bm25._chunks if c.metadata.document_title == doc_title]
    sections = Counter(c.metadata.section_name for c in chunks)
    top_name, top_count = sections.most_common(1)[0]
    print(f"{doc_title}: {len(chunks)} chunks, largest section {top_name!r} = {top_count} ({100*top_count/len(chunks):.0f}%)")
```

Run this as a one-off Python script or inline `python -c`.
Expected: largest single section's share is well under the pre-fix 94-100% for all three documents (per design doc §3, ~1-10% is the validated result — some variance from the added chunk-length filter is expected and fine, the acceptance bar is "no single section anywhere near swallowing the whole document").

- [ ] **Step 3: Confirm the previously-correct documents (Z-800F, intera-3000) did not regress into swallowing**

Run the same snippet as Step 2 with `doc_title` values `"Z-800F_Instructions_for_Use_Rev_O"` and `"intera-3000-pump-ifu"`.
Expected: same acceptance bar — no single section swallows the document. (Some increase in total section/chunk count for these two is expected and accepted per the design doc §3 — that is not a regression on its own; only re-check for the swallowing failure mode.)

- [ ] **Step 4: Spot-check that the minimum-length filter actually reduced obvious noise**

```python
import pickle
with open("data/bm25_index.pkl", "rb") as f:
    bm25 = pickle.load(f)
lens = sorted(len(c.text) for c in bm25._chunks if c.metadata.source_type in ("guidance_pdf", "ifu_pdf"))
print("min chunk length among PDF chunks:", lens[0])
print("count under 40 chars (should be zero):", sum(1 for l in lens if l < 40))
```

Expected: `count under 40 chars` prints `0` — confirms the filter is active end-to-end against the real corpus, not just in unit tests.

- [ ] **Step 5: Report results**

No commit for this task (no files changed) — report the printed metrics from Steps 2-4 back to the user as confirmation the fix holds against real data, matching what the design doc's empirical validation (§3, §4) predicted.

---

## Self-Review

**Spec coverage:**
- Title-Case detection (design doc §2a) → Task 1. ✓
- Real-word content guard / catalog-code rejection (§2b) → Task 1. ✓
- Position-scoped masthead suppression (§2c) → Task 1. ✓
- Trailing-period rejection (§2d) → Task 1. ✓
- Period-required numbered prefix (§2e) → Task 1. ✓
- Minimum chunk-length filter at 40 characters (§4) → Task 2. ✓
- Re-validation against real corpus, not just synthetic fixtures (§3's methodology) → Task 3. ✓
- Existing tests must keep passing (Global Constraints) → explicitly checked in Task 1 Step 5 and Task 2 Step 5. ✓

**Placeholder scan:** No TBD/TODO markers. Task 3's validation script is inline/one-off by design (not a permanent test file) — this matches how the original Phase 1 plan's own "re-run against real data" moments were handled (e.g. the original ingestion scripts have no automated test coverage by design); it is not a placeholder, it's a deliberately non-committed verification step.

**Type consistency:** `Section(heading, body)` unchanged across Task 1 and its consumer in Task 2's `chunk_pdf_text`. `Chunk`/`ChunkMetadata` construction in Task 2 is byte-for-byte unchanged from the existing approved code except for the new filter guard — no signature drift.
