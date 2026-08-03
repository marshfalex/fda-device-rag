# Eval Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the deterministic candidate-sampling pipeline, gold-locator resolution
module, frozen-question-file loader, and freeze-enforcement utility specified in
`docs/superpowers/specs/2026-08-03-eval-harness-design.md`.

**Architecture:** A new `src/fda_device_rag/eval/` package holds all pure, testable
logic (section-instance aggregation, the page-furniture rule, candidate sampling, gold
resolution, question-file parsing, freeze enforcement). A thin CLI script
(`scripts/sample_eval_candidates.py`) wires the sampling functions to real files on
disk, matching the existing `scripts/pull_corpus.py` / `build_index.py` convention of
"pure logic in `src/`, file I/O glue in `scripts/`."

**Tech Stack:** Python 3.11, pytest, existing project dependencies only (no new
third-party packages — this plan uses `random`, `re`, `json`, `subprocess`, `pathlib`
from the standard library plus the project's existing `pypdf`/`langchain-text-splitters`
via already-existing functions).

## Global Constraints

- Recall/MAUDE candidate floor: **100 characters**, measured on the built chunk text
  (`recall_to_chunk`/`event_to_chunk` output), not the raw narrative field.
- PDF section-instance eligibility floor: **72 characters**.
- Page-furniture rule: two instances collapse only if they have the same total count
  of digit-runs AND every digit-run that differs between them is **≤3 digits long**.
- Recall categories (13, exact strings, top-frequency `root_cause_description`):
  `Process control`, `Device Design`, `Under Investigation by firm`, `Other`,
  `Nonconforming Material/Component`, `Employee error`, `Software design`,
  `Material/Component Contamination`, `Error in labeling`,
  `Packaging process control`, `Use error`,
  `Radiation Control for Health and Safety Act`, `Labeling design`.
- MAUDE categories (13, exact strings, top-frequency `product_problems`):
  `Patient Device Interaction Problem`,
  `Adverse Event Without Identified Device or Use Problem`,
  `Insufficient Device Problem Information`, `Defective Device`, `Pumping Stopped`,
  `Mechanical Problem`, `Defective Component`, `No Apparent Adverse Event`,
  `Material Integrity Problem`, `Failure to Power Up`, `Device Alarm System`,
  `Patient-Device Incompatibility`, `Device Dislodged or Dislocated`.
- Guidance PDF quota: 3 per document × `78369`, `188844`, `153781`, `73141`.
- IFU PDF quota: 4 per document × `Z-800F_Instructions_for_Use_Rev_O`,
  `intera-3000-pump-ifu`, `FreedomEdge_Domestic_IFU_347201_Rev_B`.
- Frozen file location: `data/eval/questions.json` (not gitignored). Candidates file:
  `data/eval/candidates.json` (must be gitignored — it's scratch material, never the
  frozen benchmark).
- Sampling must be reproducible: same seed → same selection, always.
- No changes to `Chunk`, `ChunkMetadata`, `Section`, or any existing module's public
  interface — this plan is purely additive.

---

## File Structure

New package:
- `src/fda_device_rag/eval/__init__.py` — empty package marker.
- `src/fda_device_rag/eval/section_instances.py` — groups a document's PDF chunks
  into section instances (shared by sampling and gold resolution).
- `src/fda_device_rag/eval/furniture.py` — the page-furniture rule.
- `src/fda_device_rag/eval/sampling.py` — recall/MAUDE candidate sampling + PDF
  section-instance candidate sampling.
- `src/fda_device_rag/eval/questions.py` — `Question` dataclass + frozen-file loader.
- `src/fda_device_rag/eval/gold_resolution.py` — resolves a `Question`'s gold locator
  to current chunk IDs.
- `src/fda_device_rag/eval/freeze_check.py` — git-backed freeze enforcement.

New script:
- `scripts/sample_eval_candidates.py` — CLI entry point producing
  `data/eval/candidates.json`.

Modified:
- `.gitignore` — add `data/eval/candidates.json`.

New tests (mirroring the package 1:1, matching the existing `tests/{module}/`
convention):
- `tests/eval/__init__.py`
- `tests/eval/test_section_instances.py`
- `tests/eval/test_furniture.py`
- `tests/eval/test_sampling.py`
- `tests/eval/test_questions.py`
- `tests/eval/test_gold_resolution.py`
- `tests/eval/test_freeze_check.py`

---

### Task 1: Section-Instance Aggregation

**Files:**
- Create: `src/fda_device_rag/eval/section_instances.py`
- Test: `tests/eval/test_section_instances.py`
- Create: `tests/eval/__init__.py` (empty)

**Interfaces:**
- Consumes: `fda_device_rag.models.Chunk`, `ChunkMetadata` (existing, unchanged).
- Produces: `SectionInstance` dataclass (`document_title: str, section_name: str,
  instance_ordinal: int, chunk_ids: list[str], text: str`) and
  `build_section_instances(chunks: list[Chunk]) -> list[SectionInstance]`. Both
  imported by `sampling.py` (Task 4) and `gold_resolution.py` (Task 6).

This groups a document's chunks (in the order `chunk_pdf_text` produced them) into
"instances": contiguous runs sharing the same `section_name`. A non-adjacent repeat of
the same `section_name` (a different real instance that happens to share a mislabeled
heading, e.g. `Contains Nonbinding Recommendations`) becomes a separate
`SectionInstance`, numbered by an ordinal scoped to `(document_title, section_name)` —
this is the exact instance-keying already validated in the design doc's §3b/§4.

- [ ] **Step 1: Write the failing test**

Create `tests/eval/__init__.py` (empty file).

```python
# tests/eval/test_section_instances.py
from fda_device_rag.eval.section_instances import build_section_instances
from fda_device_rag.models import Chunk, ChunkMetadata


def _chunk(id_, doc, section, text):
    return Chunk(
        id=id_,
        text=text,
        metadata=ChunkMetadata(
            source_type="guidance_pdf",
            source_url="https://example.com/doc.pdf",
            document_title=doc,
            section_name=section,
            retrieved_date="2026-08-03",
        ),
    )


def test_groups_consecutive_same_section_chunks_into_one_instance():
    chunks = [
        _chunk("doc-0", "doc", "WARNINGS", "WARNINGS\nFirst piece."),
        _chunk("doc-1", "doc", "WARNINGS", "Second piece."),
    ]

    instances = build_section_instances(chunks)

    assert len(instances) == 1
    assert instances[0].document_title == "doc"
    assert instances[0].section_name == "WARNINGS"
    assert instances[0].instance_ordinal == 1
    assert instances[0].chunk_ids == ["doc-0", "doc-1"]
    assert instances[0].text == "WARNINGS\nFirst piece.\nSecond piece."


def test_non_adjacent_repeat_of_same_section_name_is_a_separate_instance():
    chunks = [
        _chunk("doc-0", "doc", "GETTING STARTED", "GETTING STARTED\nPage 1 content."),
        _chunk("doc-1", "doc", "MAINTENANCE", "MAINTENANCE\nMaintenance content."),
        _chunk("doc-2", "doc", "GETTING STARTED", "GETTING STARTED\nPage 2 content."),
    ]

    instances = build_section_instances(chunks)

    assert len(instances) == 3
    getting_started = [i for i in instances if i.section_name == "GETTING STARTED"]
    assert [i.instance_ordinal for i in getting_started] == [1, 2]
    assert [i.chunk_ids for i in getting_started] == [["doc-0"], ["doc-2"]]


def test_instance_ordinal_counter_is_scoped_per_document():
    chunks = [
        _chunk("docA-0", "docA", "WARNINGS", "WARNINGS\nContent A."),
        _chunk("docB-0", "docB", "WARNINGS", "WARNINGS\nContent B."),
    ]

    instances = build_section_instances(chunks)

    assert [i.instance_ordinal for i in instances] == [1, 1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/eval/test_section_instances.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fda_device_rag.eval'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/fda_device_rag/eval/__init__.py
```
(empty file — package marker)

```python
# src/fda_device_rag/eval/section_instances.py
from dataclasses import dataclass

from fda_device_rag.models import Chunk


@dataclass
class SectionInstance:
    document_title: str
    section_name: str
    instance_ordinal: int
    chunk_ids: list[str]
    text: str


def build_section_instances(chunks: list[Chunk]) -> list[SectionInstance]:
    """Groups a document's PDF chunks (in chunk_pdf_text's output order) into
    section instances: contiguous runs of chunks sharing the same
    section_name. A non-adjacent repeat of the same section_name -- e.g. the
    same mislabeled heading (`Contains Nonbinding Recommendations`)
    recurring later in the document -- is counted as a separate instance,
    numbered by instance_ordinal within (document_title, section_name).
    """
    instances: list[SectionInstance] = []
    ordinal_counts: dict[tuple[str, str], int] = {}

    current_doc = None
    current_name = None
    current_ids: list[str] = []
    current_texts: list[str] = []

    def flush():
        nonlocal current_ids, current_texts
        if current_ids:
            key = (current_doc, current_name)
            ordinal_counts[key] = ordinal_counts.get(key, 0) + 1
            instances.append(
                SectionInstance(
                    document_title=current_doc,
                    section_name=current_name,
                    instance_ordinal=ordinal_counts[key],
                    chunk_ids=list(current_ids),
                    text="\n".join(current_texts),
                )
            )
        current_ids = []
        current_texts = []

    for chunk in chunks:
        doc = chunk.metadata.document_title
        name = chunk.metadata.section_name
        if doc != current_doc or name != current_name:
            flush()
            current_doc = doc
            current_name = name
        current_ids.append(chunk.id)
        current_texts.append(chunk.text)

    flush()
    return instances
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/eval/test_section_instances.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/fda_device_rag/eval/__init__.py src/fda_device_rag/eval/section_instances.py tests/eval/__init__.py tests/eval/test_section_instances.py
git commit -m "feat: add section-instance aggregation for eval harness"
```

---

### Task 2: The Page-Furniture Rule

**Files:**
- Create: `src/fda_device_rag/eval/furniture.py`
- Test: `tests/eval/test_furniture.py`

**Interfaces:**
- Produces: `is_page_furniture(text_a: str, text_b: str) -> bool`. Consumed by
  `sampling.py` (Task 4).

Implements the rule from design doc §4, using the real corpus strings that drove each
round of validation as test fixtures (not synthetic examples), so a future change that
breaks the rule breaks against the actual evidence that justified it.

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_furniture.py
from fda_device_rag.eval.furniture import is_page_furniture


def test_identical_footer_differing_only_by_page_number_is_furniture():
    a = "MAINTENANCE\nZ-800F Instructions for Use.  15 \nP/N 800F-IFU-2602, Rev. O"
    b = "MAINTENANCE\nZ-800F Instructions for Use.  21 \nP/N 800F-IFU-2602, Rev. O"

    assert is_page_furniture(a, b) is True


def test_short_leading_page_number_difference_is_furniture():
    a = "GETTING STARTED\n16  Z-800F Instructions for Use \nP/N 800F-IFU-2602, Rev. O"
    b = "GETTING STARTED\n18  Z-800F Instructions for Use \nP/N 800F-IFU-2602, Rev. O"

    assert is_page_furniture(a, b) is True


def test_distinct_citations_differing_by_a_long_digit_run_are_not_furniture():
    # Real guidance-doc citations that a naive digit-stripping rule wrongly
    # collapsed (design doc S4, round 2) -- different URLs, different topics,
    # coincidentally identical non-digit template text.
    a = ("Compliance on Off-the-Shelf Software Use in Medical Devices\n"
         "(http://www.fda.gov/MedicalDevices/DeviceRegulationandGuidance/GuidanceDocuments/ucm073778.htm).")
    b = ("Medical Device Design\n"
         "(http://www.fda.gov/MedicalDevices/DeviceRegulationandGuidance/GuidanceDocuments/ucm259748.htm).")

    assert is_page_furniture(a, b) is False


def test_distinct_spec_tables_with_long_differing_item_numbers_are_not_furniture():
    # Real needle-set spec tables that a naive digit-stripping rule wrongly
    # collapsed (design doc S4, round 2) -- different item numbers and
    # residual volumes, same table template.
    single_needle = ("Single-Needle Sets\nLength Item # Residual Vol. p/ Box\n"
                      "6 mm RMS12406 0.4 ml 20\n9 mm RMS12409 0.4 ml 20\n"
                      "12 mm RMS12412 0.4 ml 20\n14 mm RMS12414 0.4 ml 20")
    two_needle = ("Two-Needle Sets\nLength Item # Residual Vol. p/ Box\n"
                  "6 mm RMS22406 0.7 ml 10\n9 mm RMS22409 0.7 ml 10\n"
                  "12 mm RMS22412 0.7 ml 10\n14 mm RMS22414 0.7 ml 10")

    assert is_page_furniture(single_needle, two_needle) is False


def test_distinct_spec_tables_with_differing_digit_run_counts_are_not_furniture():
    # Real needle-set spec tables where one member has an extra wrapped
    # digit fragment, giving the two texts a different total digit-run count
    # (design doc S4, round 2's second false-positive class).
    single_needle = ("Single-Needle Sets\nLength Item # Residual Vol. p/ Box\n"
                      "4 mm RMS12604 0.1 ml 20\n6 mm RMS12606 0.1 ml 20\n"
                      "9 mm RMS12609 0.1 ml 20\n12 mm RMS12612 0.1 ml 20\n"
                      "14 mm RMS12614 0.1 ml 20")
    six_needle = ("Six-Needle Sets\nLength Item # Residual Vol. p/ Box\n"
                  "4 mm RMS62604 0.6 ml 10\n6 mm RMS62606 0.6 ml 10\n"
                  "9 mm RMS62609 0.6 ml 10\n12 mm RMS62612 0.6 ml 10\n"
                  "14 mm RMS62614 0.6 ml 10\n16")

    assert is_page_furniture(single_needle, six_needle) is False


def test_completely_unrelated_text_with_no_digits_is_not_furniture():
    assert is_page_furniture("WARNINGS\nDo not reuse this device.", "CAUTION\nKeep away from heat.") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/eval/test_furniture.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fda_device_rag.eval.furniture'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/fda_device_rag/eval/furniture.py
import re

DIGIT_RUN = re.compile(r"\d+")
MAX_FURNITURE_DIGIT_RUN_LEN = 3


def is_page_furniture(text_a: str, text_b: str) -> bool:
    """True if text_b is text_a's page-furniture duplicate: the same total
    count of digit-runs, and every digit-run that differs between the two is
    at most MAX_FURNITURE_DIGIT_RUN_LEN digits long (page-number-scale). A
    4+-digit differing run, or a differing digit-run count, means the two
    are distinct real content and must never be collapsed -- see
    docs/superpowers/specs/2026-08-03-eval-harness-design.md section 4 for
    the two false-positive classes (citation numbers, needle-set spec
    tables) this boundary was empirically set to exclude, and the
    monotonicity/isolation checks that confirmed the boundary itself.
    """
    runs_a = DIGIT_RUN.findall(text_a)
    runs_b = DIGIT_RUN.findall(text_b)
    if len(runs_a) != len(runs_b):
        return False
    for run_a, run_b in zip(runs_a, runs_b):
        if run_a != run_b and max(len(run_a), len(run_b)) > MAX_FURNITURE_DIGIT_RUN_LEN:
            return False
    return True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/eval/test_furniture.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/fda_device_rag/eval/furniture.py tests/eval/test_furniture.py
git commit -m "feat: add page-furniture detection rule for eval harness"
```

---

### Task 3: Recall / MAUDE Candidate Sampling

**Files:**
- Create: `src/fda_device_rag/eval/sampling.py`
- Test: `tests/eval/test_sampling.py`

**Interfaces:**
- Consumes: `fda_device_rag.documents.structured.recall_to_chunk`,
  `event_to_chunk` (existing, unchanged).
- Produces: `RECALL_CATEGORIES: list[str]`, `EVENT_CATEGORIES: list[str]`,
  `RECALL_FLOOR_CHARS = 100`, `EVENT_FLOOR_CHARS = 100`, `EmptyCandidatePoolError`,
  `sample_recall_candidates(records, retrieved_date, base_seed, categories=RECALL_CATEGORIES, floor=RECALL_FLOOR_CHARS) -> list[dict]`,
  `sample_event_candidates(records, retrieved_date, base_seed, categories=EVENT_CATEGORIES, floor=EVENT_FLOOR_CHARS) -> list[dict]`.
  Each returned dict: `{"source_type": "recall"|"maude", "category": str, "record_id": str, "record": dict}`.
  Consumed by `scripts/sample_eval_candidates.py` (Task 8).

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_sampling.py
import pytest

from fda_device_rag.eval.sampling import (
    EmptyCandidatePoolError,
    sample_recall_candidates,
    sample_event_candidates,
)


def _recall_record(product_res_number, category, reason="Some genuinely detailed recall reason text here."):
    return {
        "product_res_number": product_res_number,
        "root_cause_description": category,
        "reason_for_recall": reason,
        "action": "Customers were notified by letter.",
        "product_description": "Example infusion pump device.",
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
        _recall_record("Z-01", "Process control", reason="short"),
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/eval/test_sampling.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fda_device_rag.eval.sampling'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/fda_device_rag/eval/sampling.py
import random

from fda_device_rag.documents.structured import recall_to_chunk, event_to_chunk

RECALL_FLOOR_CHARS = 100
EVENT_FLOOR_CHARS = 100
PDF_SECTION_FLOOR_CHARS = 72

RECALL_CATEGORIES = [
    "Process control",
    "Device Design",
    "Under Investigation by firm",
    "Other",
    "Nonconforming Material/Component",
    "Employee error",
    "Software design",
    "Material/Component Contamination",
    "Error in labeling",
    "Packaging process control",
    "Use error",
    "Radiation Control for Health and Safety Act",
    "Labeling design",
]

EVENT_CATEGORIES = [
    "Patient Device Interaction Problem",
    "Adverse Event Without Identified Device or Use Problem",
    "Insufficient Device Problem Information",
    "Defective Device",
    "Pumping Stopped",
    "Mechanical Problem",
    "Defective Component",
    "No Apparent Adverse Event",
    "Material Integrity Problem",
    "Failure to Power Up",
    "Device Alarm System",
    "Patient-Device Incompatibility",
    "Device Dislodged or Dislocated",
]


class EmptyCandidatePoolError(Exception):
    """Raised when a category or document has zero eligible candidates after
    the floor filter -- a real data problem that must surface immediately,
    never silently skip a benchmark slot."""


def sample_recall_candidates(records, retrieved_date, base_seed, categories=RECALL_CATEGORIES, floor=RECALL_FLOOR_CHARS):
    """One candidate per category: filter records in `records` matching
    `category` (exact match on root_cause_description) to those whose built
    chunk text meets `floor`, sort by stable record id, then draw with a
    seed derived deterministically from (base_seed, category) so each
    category's draw is reproducible and independent of the others."""
    results = []
    for category in categories:
        pool = []
        for record in records:
            if record.get("root_cause_description") != category:
                continue
            chunk = recall_to_chunk(record, retrieved_date=retrieved_date)
            if chunk is None or len(chunk.text) < floor:
                continue
            pool.append(record)
        pool.sort(key=lambda r: str(r.get("product_res_number", "")))

        if not pool:
            raise EmptyCandidatePoolError(f"recall category {category!r} has zero eligible candidates")

        rng = random.Random(f"{base_seed}-recall-{category}")
        chosen = rng.choice(pool)
        results.append({
            "source_type": "recall",
            "category": category,
            "record_id": str(chosen.get("product_res_number", "")),
            "record": chosen,
        })
    return results


def sample_event_candidates(records, retrieved_date, base_seed, categories=EVENT_CATEGORIES, floor=EVENT_FLOOR_CHARS):
    """Same pipeline as sample_recall_candidates, but category membership is
    list-containment on product_problems (an event can carry several problem
    categories), not equality -- so the same record may legitimately be
    eligible for, and drawn into, more than one category's pool."""
    results = []
    for category in categories:
        pool = []
        for record in records:
            if category not in (record.get("product_problems") or []):
                continue
            chunk = event_to_chunk(record, retrieved_date=retrieved_date)
            if chunk is None or len(chunk.text) < floor:
                continue
            pool.append(record)
        pool.sort(key=lambda r: str(r.get("report_number", "")))

        if not pool:
            raise EmptyCandidatePoolError(f"MAUDE category {category!r} has zero eligible candidates")

        rng = random.Random(f"{base_seed}-maude-{category}")
        chosen = rng.choice(pool)
        results.append({
            "source_type": "maude",
            "category": category,
            "record_id": str(chosen.get("report_number", "")),
            "record": chosen,
        })
    return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/eval/test_sampling.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/fda_device_rag/eval/sampling.py tests/eval/test_sampling.py
git commit -m "feat: add recall/MAUDE candidate sampling for eval harness"
```

---

### Task 4: PDF Section-Instance Candidate Sampling

**Files:**
- Modify: `src/fda_device_rag/eval/sampling.py` (add PDF sampling alongside Task 3's
  recall/MAUDE sampling)
- Modify: `tests/eval/test_sampling.py` (add PDF sampling tests)

**Interfaces:**
- Consumes: `SectionInstance`, `build_section_instances` (Task 1);
  `is_page_furniture` (Task 2).
- Produces: `sample_pdf_section_candidates(document_title, chunks, quota, base_seed, floor=PDF_SECTION_FLOOR_CHARS) -> tuple[list[SectionInstance], list[tuple[str, str]]]`
  — returns `(selected_instances, skip_log)`, where `skip_log` entries are
  `(skipped_instance_key, matched_against_instance_key)` strings. Consumed by
  `scripts/sample_eval_candidates.py` (Task 8).

**Design note:** furniture status is computed once, pool-wide, across every eligible
instance in the document (an all-pairs check, exactly matching the design doc §4
round-3 full-pool validation) — not by checking a freshly-drawn candidate only against
instances already selected earlier in the same draw. A furniture instance whose
duplicate hasn't been drawn yet is still furniture; checking only against
already-selected instances would miss that.

- [ ] **Step 1: Write the failing test**

Append to `tests/eval/test_sampling.py`:

```python
from fda_device_rag.eval.sampling import sample_pdf_section_candidates
from fda_device_rag.models import Chunk, ChunkMetadata


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


def test_sample_pdf_section_candidates_excludes_furniture_duplicates():
    long_body = "A" * 60
    chunks = [
        _pdf_chunk("doc-0", "doc", "MAINTENANCE", f"MAINTENANCE\nZ-800F Instructions for Use.  15 \nP/N 800F-IFU-2602, Rev. O {long_body}"),
        _pdf_chunk("doc-1", "doc", "TROUBLESHOOTING", "TROUBLESHOOTING\nGenuinely distinct real content about troubleshooting alarms."),
        _pdf_chunk("doc-2", "doc", "MAINTENANCE", f"MAINTENANCE\nZ-800F Instructions for Use.  21 \nP/N 800F-IFU-2602, Rev. O {long_body}"),
    ]

    selected, skip_log = sample_pdf_section_candidates("doc", chunks, quota=2, base_seed="test-seed", floor=72)

    section_names = sorted(i.section_name for i in selected)
    assert section_names == ["MAINTENANCE", "TROUBLESHOOTING"]
    assert len(skip_log) == 1


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
        _pdf_chunk(f"doc-{i}", "doc", f"SECTION {i}", f"SECTION {i}\n" + "A" * 80)
        for i in range(5)
    ]

    first, _ = sample_pdf_section_candidates("doc", chunks, quota=2, base_seed="fixed-seed", floor=72)
    second, _ = sample_pdf_section_candidates("doc", chunks, quota=2, base_seed="fixed-seed", floor=72)

    assert [i.section_name for i in first] == [i.section_name for i in second]


def test_sample_pdf_section_candidates_raises_when_pool_too_small_for_quota():
    chunks = [_pdf_chunk("doc-0", "doc", "ONLY SECTION", "ONLY SECTION\n" + "A" * 80)]

    with pytest.raises(EmptyCandidatePoolError):
        sample_pdf_section_candidates("doc", chunks, quota=2, base_seed="test-seed", floor=72)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/eval/test_sampling.py -v`
Expected: FAIL with `ImportError: cannot import name 'sample_pdf_section_candidates'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/fda_device_rag/eval/sampling.py`:

```python
from fda_device_rag.eval.section_instances import build_section_instances
from fda_device_rag.eval.furniture import is_page_furniture


def _flag_furniture(eligible):
    """Returns {index_of_furniture_instance: index_of_first_instance_it_duplicates}
    for every instance in `eligible` that is a page-furniture duplicate of an
    earlier instance in the same list. Computed pool-wide (all pairs against
    instances kept so far), matching the design doc's full-pool validation
    methodology -- not scoped to what a live draw has selected so far."""
    furniture_of = {}
    kept = []
    for idx, instance in enumerate(eligible):
        match = next((k for k in kept if is_page_furniture(eligible[k].text, instance.text)), None)
        if match is not None:
            furniture_of[idx] = match
        else:
            kept.append(idx)
    return furniture_of


def sample_pdf_section_candidates(document_title, chunks, quota, base_seed, floor=PDF_SECTION_FLOOR_CHARS):
    """Selects `quota` distinct, non-furniture section instances from this
    document's chunks. Furniture exclusion happens before the random draw
    (pool-wide, per _flag_furniture), then `quota` instances are drawn
    without replacement from what remains."""
    instances = build_section_instances(chunks)
    eligible = [i for i in instances if len(i.text) >= floor]

    furniture_of = _flag_furniture(eligible)
    skip_log = [
        (
            f"{document_title}:{eligible[idx].section_name}#{eligible[idx].instance_ordinal}",
            f"{document_title}:{eligible[match].section_name}#{eligible[match].instance_ordinal}",
        )
        for idx, match in furniture_of.items()
    ]

    non_furniture = [inst for i, inst in enumerate(eligible) if i not in furniture_of]

    if len(non_furniture) < quota:
        raise EmptyCandidatePoolError(
            f"document {document_title!r} has only {len(non_furniture)} non-furniture "
            f"eligible section instances, needed {quota}"
        )

    rng = random.Random(f"{base_seed}-pdf-{document_title}")
    selected = rng.sample(non_furniture, quota)
    return selected, skip_log
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/eval/test_sampling.py -v`
Expected: PASS (11 tests total)

- [ ] **Step 5: Commit**

```bash
git add src/fda_device_rag/eval/sampling.py tests/eval/test_sampling.py
git commit -m "feat: add PDF section-instance candidate sampling with furniture exclusion"
```

---

### Task 5: Frozen Question-File Loader

**Files:**
- Create: `src/fda_device_rag/eval/questions.py`
- Test: `tests/eval/test_questions.py`

**Interfaces:**
- Produces: `Question` dataclass (`question_id: str, source_type: str,
  category: str | None, question: str, gold: dict, notes: str | None`),
  `QuestionsFileError`, `load_questions(path) -> list[Question]`. Consumed by
  `gold_resolution.py` (Task 6).

Implements the schema from design doc §7. This is the enforced contract for the
frozen file — the actual 50 questions are authored later, live, in the collaborative
session from design doc §8, not by this plan.

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_questions.py
import json

import pytest

from fda_device_rag.eval.questions import Question, QuestionsFileError, load_questions


def _write(tmp_path, data):
    path = tmp_path / "questions.json"
    path.write_text(json.dumps(data))
    return path


def test_load_questions_parses_valid_file(tmp_path):
    path = _write(tmp_path, {
        "version": 1,
        "frozen_date": "2026-08-03",
        "questions": [
            {
                "question_id": "recall-01",
                "source_type": "recall",
                "category": "Process control",
                "question": "Why was the XYZ pump recalled?",
                "gold": {"record_id": "Z-0001-2024"},
                "notes": None,
            },
            {
                "question_id": "guidance-01",
                "source_type": "guidance",
                "question": "What software validation activities does FDA recommend?",
                "gold": {"document_title": "188844", "section_name": "Risk-Based Analysis", "instance_ordinal": 1},
            },
        ],
    })

    questions = load_questions(path)

    assert len(questions) == 2
    assert questions[0] == Question(
        question_id="recall-01", source_type="recall", category="Process control",
        question="Why was the XYZ pump recalled?", gold={"record_id": "Z-0001-2024"}, notes=None,
    )
    # category/notes default to None when absent from the raw JSON
    assert questions[1].category is None
    assert questions[1].notes is None


def test_load_questions_raises_on_missing_questions_key(tmp_path):
    path = _write(tmp_path, {"version": 1})

    with pytest.raises(QuestionsFileError, match="missing top-level 'questions' key"):
        load_questions(path)


def test_load_questions_raises_on_missing_required_field(tmp_path):
    path = _write(tmp_path, {"questions": [{"question_id": "q1", "source_type": "recall", "gold": {}}]})

    with pytest.raises(QuestionsFileError, match="question"):
        load_questions(path)


def test_load_questions_raises_on_invalid_source_type(tmp_path):
    path = _write(tmp_path, {"questions": [{
        "question_id": "q1", "source_type": "bogus", "question": "x?", "gold": {},
    }]})

    with pytest.raises(QuestionsFileError, match="invalid source_type"):
        load_questions(path)


def test_load_questions_raises_on_duplicate_question_id(tmp_path):
    path = _write(tmp_path, {"questions": [
        {"question_id": "q1", "source_type": "recall", "question": "a?", "gold": {}},
        {"question_id": "q1", "source_type": "maude", "question": "b?", "gold": {}},
    ]})

    with pytest.raises(QuestionsFileError, match="duplicate question_id"):
        load_questions(path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/eval/test_questions.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fda_device_rag.eval.questions'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/fda_device_rag/eval/questions.py
from dataclasses import dataclass
from pathlib import Path
import json

VALID_SOURCE_TYPES = {"recall", "maude", "guidance", "ifu"}
REQUIRED_FIELDS = ("question_id", "source_type", "question", "gold")


@dataclass
class Question:
    question_id: str
    source_type: str
    category: str | None
    question: str
    gold: dict
    notes: str | None


class QuestionsFileError(Exception):
    """Raised when the frozen questions file is missing required fields or
    otherwise doesn't match the schema in
    docs/superpowers/specs/2026-08-03-eval-harness-design.md section 7."""


def load_questions(path) -> list[Question]:
    path = Path(path)
    data = json.loads(path.read_text())
    raw_questions = data.get("questions")
    if raw_questions is None:
        raise QuestionsFileError(f"{path}: missing top-level 'questions' key")

    questions = []
    seen_ids = set()
    for i, raw in enumerate(raw_questions):
        for field in REQUIRED_FIELDS:
            if field not in raw:
                raise QuestionsFileError(f"{path}: question at index {i} missing required field {field!r}")

        if raw["source_type"] not in VALID_SOURCE_TYPES:
            raise QuestionsFileError(
                f"{path}: question {raw['question_id']!r} has invalid source_type "
                f"{raw['source_type']!r} (must be one of {sorted(VALID_SOURCE_TYPES)})"
            )

        if raw["question_id"] in seen_ids:
            raise QuestionsFileError(f"{path}: duplicate question_id {raw['question_id']!r}")
        seen_ids.add(raw["question_id"])

        questions.append(Question(
            question_id=raw["question_id"],
            source_type=raw["source_type"],
            category=raw.get("category"),
            question=raw["question"],
            gold=raw["gold"],
            notes=raw.get("notes"),
        ))
    return questions
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/eval/test_questions.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/fda_device_rag/eval/questions.py tests/eval/test_questions.py
git commit -m "feat: add frozen question-file schema and loader"
```

---

### Task 6: Gold-Locator Resolution

**Files:**
- Create: `src/fda_device_rag/eval/gold_resolution.py`
- Test: `tests/eval/test_gold_resolution.py`

**Interfaces:**
- Consumes: `fda_device_rag.documents.structured.recall_to_chunk`, `event_to_chunk`;
  `fda_device_rag.documents.pdf_document.extract_pdf_text`;
  `fda_device_rag.chunking.pdf_chunker.chunk_pdf_text` (all existing, unchanged);
  `build_section_instances` (Task 1); `Question` (Task 5).
- Produces: `GoldResolutionError`,
  `resolve_structured_gold(source_type, record_id, retrieved_date="", data_dir=Path("data/raw")) -> list[str]`,
  `resolve_pdf_gold_from_text(document_title, text, section_name, instance_ordinal, source_type="guidance_pdf") -> list[str]`,
  `resolve_pdf_gold(document_title, section_name, instance_ordinal, source_type="guidance_pdf", data_dir=Path("data/raw")) -> list[str]`,
  `resolve_gold(question, data_dir=Path("data/raw")) -> list[str]`.

**Naming note:** the frozen file's `source_type` field uses the benchmark-level labels
`"recall"`/`"maude"`/`"guidance"`/`"ifu"` (design doc §7); the underlying corpus's
`ChunkMetadata.source_type` uses `"recall"`/`"maude_event"`/`"guidance_pdf"`/`"ifu_pdf"`
(existing code). `resolve_gold` translates between the two vocabularies — this is not
an inconsistency to fix, just two different vocabularies for two different purposes.

**Testability note:** `resolve_pdf_gold_from_text` takes already-extracted text
directly, so its instance-matching logic is unit-testable with plain string fixtures
— no real PDF file needed. `resolve_pdf_gold` (the thin wrapper that locates and reads
an actual PDF) is exercised separately via `tmp_path`-based fixtures for the
file-lookup logic only, since `extract_pdf_text` already has its own test coverage in
`tests/documents/test_pdf_document.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_gold_resolution.py
import json
from unittest.mock import patch

import pytest

from fda_device_rag.eval.gold_resolution import (
    GoldResolutionError,
    resolve_structured_gold,
    resolve_pdf_gold_from_text,
    resolve_gold,
    _find_pdf_path,
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

    mock_resolve.assert_called_once_with("recall", "Z-01")
    assert result == ["recall-Z-01"]


@patch("fda_device_rag.eval.gold_resolution.resolve_pdf_gold")
def test_resolve_gold_dispatches_guidance_and_ifu_to_pdf_resolver(mock_resolve):
    mock_resolve.return_value = ["188844-3"]
    question = Question(
        question_id="q2", source_type="guidance", category=None, question="x?",
        gold={"document_title": "188844", "section_name": "Risk-Based Analysis", "instance_ordinal": 1}, notes=None,
    )

    result = resolve_gold(question)

    mock_resolve.assert_called_once_with("188844", "Risk-Based Analysis", 1, source_type="guidance_pdf")
    assert result == ["188844-3"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/eval/test_gold_resolution.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fda_device_rag.eval.gold_resolution'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/fda_device_rag/eval/gold_resolution.py
import json
from pathlib import Path

from fda_device_rag.documents.structured import recall_to_chunk, event_to_chunk
from fda_device_rag.documents.pdf_document import extract_pdf_text
from fda_device_rag.chunking.pdf_chunker import chunk_pdf_text
from fda_device_rag.eval.section_instances import build_section_instances

DEFAULT_DATA_DIR = Path("data/raw")

_STRUCTURED_SOURCES = {
    "recall": ("recalls.json", "product_res_number", recall_to_chunk),
    "maude": ("events.json", "report_number", event_to_chunk),
}


class GoldResolutionError(Exception):
    """Raised when a frozen question's gold locator can't be resolved
    against the current corpus -- a renamed heading, a removed document, an
    out-of-range instance_ordinal, or a record that no longer produces a
    chunk. Must never be silently swallowed into an empty gold set or a
    silent miss (design doc section 5)."""


def resolve_structured_gold(source_type: str, record_id: str, retrieved_date: str = "", data_dir: Path = DEFAULT_DATA_DIR) -> list[str]:
    if source_type not in _STRUCTURED_SOURCES:
        raise GoldResolutionError(f"unknown structured source_type {source_type!r}")

    filename, id_field, to_chunk = _STRUCTURED_SOURCES[source_type]
    path = Path(data_dir) / filename
    if not path.exists():
        raise GoldResolutionError(f"{path} not found -- run scripts/pull_corpus.py first")

    for record in json.loads(path.read_text()):
        if str(record.get(id_field, "")) == record_id:
            chunk = to_chunk(record, retrieved_date=retrieved_date)
            if chunk is None:
                raise GoldResolutionError(
                    f"{source_type} record {record_id!r} no longer produces a chunk (filtered as content-free)"
                )
            return [chunk.id]

    raise GoldResolutionError(f"{source_type} record {record_id!r} not found in {path}")


def resolve_pdf_gold_from_text(document_title: str, text: str, section_name: str, instance_ordinal: int, source_type: str = "guidance_pdf") -> list[str]:
    chunks = chunk_pdf_text(
        text,
        source_type=source_type,
        source_url="",
        document_title=document_title,
        retrieved_date="",
        id_prefix=document_title,
    )
    instances = build_section_instances(chunks)
    matches = [i for i in instances if i.section_name == section_name and i.instance_ordinal == instance_ordinal]

    if not matches:
        raise GoldResolutionError(
            f"no section instance found for document={document_title!r} section_name={section_name!r} "
            f"instance_ordinal={instance_ordinal} (heading may have been renamed, or ordinal is out of range)"
        )

    return matches[0].chunk_ids


def _find_pdf_path(document_title: str, data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    for subdir in ("guidance_pdfs", "ifu_pdfs"):
        candidate = Path(data_dir) / subdir / f"{document_title}.pdf"
        if candidate.exists():
            return candidate
    raise GoldResolutionError(f"no PDF found for document_title {document_title!r} under {data_dir}")


def resolve_pdf_gold(document_title: str, section_name: str, instance_ordinal: int, source_type: str = "guidance_pdf", data_dir: Path = DEFAULT_DATA_DIR) -> list[str]:
    pdf_path = _find_pdf_path(document_title, data_dir=data_dir)
    text = extract_pdf_text(pdf_path)
    return resolve_pdf_gold_from_text(document_title, text, section_name, instance_ordinal, source_type=source_type)


def resolve_gold(question, data_dir: Path = DEFAULT_DATA_DIR) -> list[str]:
    """Dispatches a frozen Question's gold locator to the right resolver.
    Translates the frozen file's benchmark-level source_type vocabulary
    ("recall"/"maude"/"guidance"/"ifu") to the corpus's ChunkMetadata
    vocabulary ("recall"/"maude_event"/"guidance_pdf"/"ifu_pdf") -- these are
    two different vocabularies for two different purposes, not a bug."""
    if question.source_type in ("recall", "maude"):
        return resolve_structured_gold(question.source_type, question.gold["record_id"], data_dir=data_dir)
    if question.source_type in ("guidance", "ifu"):
        pdf_source_type = "guidance_pdf" if question.source_type == "guidance" else "ifu_pdf"
        return resolve_pdf_gold(
            question.gold["document_title"],
            question.gold["section_name"],
            question.gold["instance_ordinal"],
            source_type=pdf_source_type,
            data_dir=data_dir,
        )
    raise GoldResolutionError(f"unknown source_type {question.source_type!r}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/eval/test_gold_resolution.py -v`
Expected: PASS (11 tests)

- [ ] **Step 5: Commit**

```bash
git add src/fda_device_rag/eval/gold_resolution.py tests/eval/test_gold_resolution.py
git commit -m "feat: add gold-locator resolution for eval harness"
```

---

### Task 7: Freeze Enforcement

**Files:**
- Create: `src/fda_device_rag/eval/freeze_check.py`
- Test: `tests/eval/test_freeze_check.py`

**Interfaces:**
- Produces: `FrozenFileError`,
  `assert_frozen_and_get_hash(path, cwd=None) -> str`. Will be consumed by the
  eventual `scripts/run_eval.py` (out of scope for this plan, per design doc §9).

Implements design doc §7's mechanical freeze guarantee: refuse to proceed if the
frozen file has uncommitted changes, and return its committed blob hash for
per-eval-run logging.

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_freeze_check.py
import subprocess

import pytest

from fda_device_rag.eval.freeze_check import FrozenFileError, assert_frozen_and_get_hash


def _init_repo(tmp_path):
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, capture_output=True, check=True)


def test_returns_blob_hash_for_a_clean_committed_file(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "questions.json").write_text('{"questions": []}')
    subprocess.run(["git", "add", "questions.json"], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "freeze"], cwd=tmp_path, capture_output=True, check=True)

    result = assert_frozen_and_get_hash("questions.json", cwd=tmp_path)

    expected = subprocess.run(
        ["git", "rev-parse", "HEAD:questions.json"], cwd=tmp_path, capture_output=True, check=True,
    ).stdout.decode().strip()
    assert result == expected
    assert len(result) == 40


def test_raises_on_uncommitted_modification(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "questions.json").write_text('{"questions": []}')
    subprocess.run(["git", "add", "questions.json"], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "freeze"], cwd=tmp_path, capture_output=True, check=True)
    (tmp_path / "questions.json").write_text('{"questions": [{"edited": true}]}')

    with pytest.raises(FrozenFileError, match="uncommitted changes"):
        assert_frozen_and_get_hash("questions.json", cwd=tmp_path)


def test_raises_when_file_was_never_committed(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "questions.json").write_text('{"questions": []}')

    with pytest.raises(FrozenFileError, match="not committed"):
        assert_frozen_and_get_hash("questions.json", cwd=tmp_path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/eval/test_freeze_check.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fda_device_rag.eval.freeze_check'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/fda_device_rag/eval/freeze_check.py
import subprocess
from pathlib import Path


class FrozenFileError(Exception):
    """Raised when the frozen questions file isn't actually frozen: either
    the working tree has uncommitted changes to it, or it isn't tracked in
    HEAD at all. This makes 'frozen' a mechanically enforced guarantee
    (design doc section 7), not just a documented promise -- leakage can't
    happen by accident even if the procedural rule is forgotten."""


def assert_frozen_and_get_hash(path, cwd=None) -> str:
    """Raises FrozenFileError if `path` has uncommitted changes relative to
    HEAD (staged or not), or if it isn't committed in HEAD at all. Otherwise
    returns the git blob hash of the committed content, for permanent
    per-eval-run logging (a clean working tree alone can't distinguish a
    genuine one-time freeze from edit-rerun-recommit-rerun)."""
    path = Path(path)

    diff = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", str(path)],
        cwd=cwd, capture_output=True,
    )
    if diff.returncode not in (0, 1):
        raise FrozenFileError(
            f"git diff failed checking {path} (exit code {diff.returncode}): {diff.stderr.decode().strip()}"
        )
    if diff.returncode == 1:
        raise FrozenFileError(
            f"{path} has uncommitted changes -- commit it before running eval "
            f"(this file must be frozen before any retrieval scoring)"
        )

    hash_result = subprocess.run(
        ["git", "rev-parse", f"HEAD:{path.as_posix()}"],
        cwd=cwd, capture_output=True,
    )
    if hash_result.returncode != 0:
        raise FrozenFileError(
            f"{path} is not committed in HEAD -- commit it before running eval "
            f"({hash_result.stderr.decode().strip()})"
        )
    return hash_result.stdout.decode().strip()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/eval/test_freeze_check.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/fda_device_rag/eval/freeze_check.py tests/eval/test_freeze_check.py
git commit -m "feat: add git-backed freeze enforcement for the frozen questions file"
```

---

### Task 8: Candidate-Sampling CLI Script

**Files:**
- Create: `scripts/sample_eval_candidates.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: `sample_recall_candidates`, `sample_event_candidates`,
  `sample_pdf_section_candidates` (Task 3/4); `extract_pdf_text`; `chunk_pdf_text`
  (existing, unchanged).
- Produces: `data/eval/candidates.json` on disk (not a Python interface — this is the
  script that ends the "in scope" list from design doc §9).

This is a thin CLI wrapper with no independent business logic of its own (all logic
lives in already-tested `sampling.py` functions), matching the existing
`scripts/pull_corpus.py`/`build_index.py` convention where scripts only get direct
unit tests for extractable pure-logic helpers (see `tests/test_pull_corpus.py`'s
`_fetch_paginated` test) — there is no such helper here to extract, so this task has
no dedicated test file.

- [ ] **Step 1: Add the candidates file to `.gitignore`**

Read the current `.gitignore`, then add one line so the intermediate candidates file
can never be accidentally committed as if it were the frozen benchmark:

```
.worktrees/
__pycache__/
*.pyc
.venv/
venv/
data/raw/
data/chroma/
data/manifest.csv
data/bm25_index.pkl
data/eval/candidates.json
*.egg-info/
.pytest_cache/
```

- [ ] **Step 2: Write the script**

```python
# scripts/sample_eval_candidates.py
"""Runs the deterministic candidate-sampling pipeline (recall/MAUDE
categories + PDF section instances) and writes the intermediate,
uncommitted candidates file consumed by the collaborative question-phrasing
session (design doc docs/superpowers/specs/2026-08-03-eval-harness-design.md,
section 8, step 1). This is NOT the frozen benchmark -- see
data/eval/questions.json for that, and note data/eval/candidates.json is
gitignored specifically so it can never be committed in its place.

Usage: python scripts/sample_eval_candidates.py [--seed SEED]
"""
import argparse
import json
from pathlib import Path

from fda_device_rag.documents.pdf_document import extract_pdf_text
from fda_device_rag.chunking.pdf_chunker import chunk_pdf_text
from fda_device_rag.eval.sampling import (
    sample_recall_candidates,
    sample_event_candidates,
    sample_pdf_section_candidates,
)

DATA_DIR = Path("data/raw")
OUTPUT_PATH = Path("data/eval/candidates.json")
DEFAULT_SEED = "fda-device-rag-eval-2026-08-03"

GUIDANCE_DOCS = ["78369", "188844", "153781", "73141"]
GUIDANCE_QUOTA_PER_DOC = 3
IFU_DOCS = ["Z-800F_Instructions_for_Use_Rev_O", "intera-3000-pump-ifu", "FreedomEdge_Domestic_IFU_347201_Rev_B"]
IFU_QUOTA_PER_DOC = 4


def _read_pull_date() -> str:
    pull_date_path = DATA_DIR / "pull_date.txt"
    return pull_date_path.read_text().strip() if pull_date_path.exists() else ""


def _load_pdf_chunks(document_title: str, subdir: str, source_type: str, retrieved_date: str):
    pdf_path = DATA_DIR / subdir / f"{document_title}.pdf"
    text = extract_pdf_text(pdf_path)
    return chunk_pdf_text(
        text,
        source_type=source_type,
        source_url=str(pdf_path),
        document_title=document_title,
        retrieved_date=retrieved_date,
        id_prefix=document_title,
    )


def _sample_pdf_group(docs, subdir, source_type, benchmark_source_type, quota_per_doc, retrieved_date, seed):
    candidates = []
    skip_log = []
    for doc in docs:
        chunks = _load_pdf_chunks(doc, subdir, source_type, retrieved_date)
        selected, skips = sample_pdf_section_candidates(doc, chunks, quota_per_doc, base_seed=seed)
        skip_log.extend(skips)
        for inst in selected:
            candidates.append({
                "source_type": benchmark_source_type,
                "document_title": inst.document_title,
                "section_name": inst.section_name,
                "instance_ordinal": inst.instance_ordinal,
                "text": inst.text,
            })
    return candidates, skip_log


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=DEFAULT_SEED)
    args = parser.parse_args()

    retrieved_date = _read_pull_date()
    recalls = json.loads((DATA_DIR / "recalls.json").read_text())
    events = json.loads((DATA_DIR / "events.json").read_text())

    candidates = []
    candidates.extend(sample_recall_candidates(recalls, retrieved_date=retrieved_date, base_seed=args.seed))
    candidates.extend(sample_event_candidates(events, retrieved_date=retrieved_date, base_seed=args.seed))

    guidance_candidates, guidance_skips = _sample_pdf_group(
        GUIDANCE_DOCS, "guidance_pdfs", "guidance_pdf", "guidance", GUIDANCE_QUOTA_PER_DOC, retrieved_date, args.seed,
    )
    ifu_candidates, ifu_skips = _sample_pdf_group(
        IFU_DOCS, "ifu_pdfs", "ifu_pdf", "ifu", IFU_QUOTA_PER_DOC, retrieved_date, args.seed,
    )
    candidates.extend(guidance_candidates)
    candidates.extend(ifu_candidates)
    skip_log = guidance_skips + ifu_skips

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps({"seed": args.seed, "candidates": candidates, "furniture_skips": skip_log}, indent=2))

    print(f"Sampled {len(candidates)} candidates (seed={args.seed!r}) -> {OUTPUT_PATH}")
    print(f"Furniture skips: {len(skip_log)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the script against the real corpus to verify it produces output**

Run: `python scripts/sample_eval_candidates.py`
Expected: prints `Sampled 50 candidates (seed='fda-device-rag-eval-2026-08-03') -> data\eval\candidates.json`
and a `Furniture skips: N` line; `data/eval/candidates.json` exists afterward.

- [ ] **Step 4: Verify the candidates file is gitignored**

Run: `git status --porcelain data/eval/`
Expected: no output (an untracked-but-ignored file produces no `git status` line;
if `data/eval/candidates.json` appears as `??`, the `.gitignore` edit in Step 1 didn't
take effect — check for a typo in the added line).

- [ ] **Step 5: Commit**

```bash
git add scripts/sample_eval_candidates.py .gitignore
git commit -m "feat: add CLI script to sample eval-harness candidates"
```

---

## Self-Review

**Spec coverage:** design doc §3a (floor→sort→choice pipeline, recall+MAUDE) → Task 3.
§3b (instance-keyed PDF pool, 72-char floor) → Task 4. §4 (furniture rule, all three
validation rounds' evidence baked into test fixtures) → Task 2, exercised by Task 4.
§5 (gold schema: structured `record_id`, PDF locator, eval-time resolution, hard
resolution-failure errors) → Task 6. §6 (phrasing style) → not code, correctly no task
(it constrains the live authoring session in §8, out of scope here per the user's
explicit instruction). §7 (frozen file schema, freeze mechanics) → Task 5 (schema/
loader) + Task 7 (git enforcement). §8 (workflow) → Task 8 produces exactly the
candidates file step 1 describes; steps 2-4 are the live collaborative session,
correctly left undelegated. §9 scope → matches file list above exactly, including the
explicit exclusion of the eval-runner's scoring logic. §10 testing → every bullet maps
to a task's test file (sampling pipeline/PDF instance pool/furniture rule/gold
resolution/freeze enforcement all covered; no `Chunk`/`ChunkMetadata`/`Section`
changes anywhere in this plan).

**Placeholder scan:** no TBD/TODO/"add error handling"-style steps found — every step
has complete, runnable code.

**Type consistency:** `SectionInstance` (Task 1) fields (`document_title`,
`section_name`, `instance_ordinal`, `chunk_ids`, `text`) are used identically in Task 4
(`sample_pdf_section_candidates`) and Task 6 (`resolve_pdf_gold_from_text`). `Question`
(Task 5) fields (`source_type`, `gold`) match exactly how Task 6's `resolve_gold`
accesses them (`question.source_type`, `question.gold["record_id"]`,
`question.gold["document_title"]`, etc.). `EmptyCandidatePoolError` is defined once in
Task 3 and reused without redefinition in Task 4. `GoldResolutionError` is defined
once in Task 6 and used consistently across all of that task's functions.

---

Plan complete and saved to `docs/superpowers/plans/2026-08-03-eval-harness.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
