# Phase 1: Retrieval Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the working retrieval pipeline: ingest recall/MAUDE narrative text and curated guidance/IFU PDFs, chunk them (structured records untouched, PDFs structure-aware), embed locally, and retrieve via hybrid BM25+dense with Reciprocal Rank Fusion.

**Architecture:** A small, layered Python package (`src/fda_device_rag/`) with one module per responsibility — openFDA ingestion, PDF ingestion, structured/PDF document building, chunking, embedding, vector store, BM25 index, and the hybrid retriever that fuses the two. Each layer is unit-tested in isolation with fakes/mocks for external services (HTTP, the embedding model, Chroma against a tmp directory); a final integration test wires all real modules together (with a stub embedder for speed) against a tiny fixture corpus to prove the pipeline works end-to-end.

**Tech Stack:** Python 3.11+, ChromaDB (persistent, cosine space), `sentence-transformers` (`BAAI/bge-small-en-v1.5`), `rank_bm25`, `langchain-text-splitters` (`RecursiveCharacterTextSplitter`), `pypdf`, `requests`, `pytest`.

## Global Constraints

- Corpus sources: openFDA `recall` (`reason_for_recall`+`action`+`product_description`), openFDA `event`/MAUDE (`mdr_text[].text`), curated FDA guidance PDFs, curated manufacturer IFU PDFs. 510(k) is metadata-only, not retrievable text (per design doc §2).
- Structured records (recall, MAUDE): no chunking, one record = one document via field-templating.
- PDF chunking: structure-aware — detect section headings first, then `RecursiveCharacterTextSplitter` within an oversized section only. Target chunk size 1600 characters (~400 tokens), overlap 240 characters (~60 tokens) (design doc §3).
- Embedding model: local `sentence-transformers` `BAAI/bge-small-en-v1.5`, cosine similarity, normalized embeddings. BGE query prefix `"Represent this sentence for searching relevant passages: "` applied to queries only, never to documents (design doc §3).
- Retrieval: hybrid — dense top-20 (Chroma cosine) + BM25 top-20 (lowercase + light punctuation strip, no stemming, so codes like `K123456` stay verbatim), merged via Reciprocal Rank Fusion with `k=60`, fused top-5 returned (design doc §4).
- No live network calls in unit tests — HTTP and PDF downloads are mocked; only the final integration test may use real (but tiny, local) components.

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `src/fda_device_rag/__init__.py`
- Create: `src/fda_device_rag/ingestion/__init__.py`
- Create: `src/fda_device_rag/documents/__init__.py`
- Create: `src/fda_device_rag/chunking/__init__.py`
- Create: `src/fda_device_rag/embedding/__init__.py`
- Create: `src/fda_device_rag/store/__init__.py`
- Create: `src/fda_device_rag/retrieval/__init__.py`
- Create: `tests/__init__.py`

**Interfaces:**
- Produces: an installable `fda_device_rag` package importable from `tests/` via pytest's `pythonpath` config, and a working `pytest` command.

- [ ] **Step 1: Create the package directories and empty `__init__.py` files**

```bash
mkdir -p src/fda_device_rag/ingestion src/fda_device_rag/documents src/fda_device_rag/chunking src/fda_device_rag/embedding src/fda_device_rag/store src/fda_device_rag/retrieval tests
touch src/fda_device_rag/__init__.py src/fda_device_rag/ingestion/__init__.py src/fda_device_rag/documents/__init__.py src/fda_device_rag/chunking/__init__.py src/fda_device_rag/embedding/__init__.py src/fda_device_rag/store/__init__.py src/fda_device_rag/retrieval/__init__.py tests/__init__.py
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "fda-device-rag"
version = "0.1.0"
description = "Agentic RAG document assistant over public FDA medical-device documentation"
requires-python = ">=3.11"
dependencies = [
    "chromadb>=0.5",
    "sentence-transformers>=3.0",
    "rank_bm25>=0.2.2",
    "requests>=2.31",
    "pypdf>=4.0",
    "langchain-text-splitters>=0.2",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

- [ ] **Step 3: Write `.gitignore`**

```
__pycache__/
*.pyc
.venv/
venv/
data/raw/
data/chroma/
data/manifest.csv
*.egg-info/
.pytest_cache/
```

- [ ] **Step 4: Install the package in editable mode with dev dependencies**

Run: `pip install -e ".[dev]"`
Expected: install completes without errors.

- [ ] **Step 5: Verify pytest runs (no tests yet)**

Run: `pytest`
Expected: `no tests ran` (exit code 5) — confirms pytest can find the config and package.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore src tests
git commit -m "chore: scaffold fda_device_rag package structure"
```

---

### Task 2: Core Data Models

**Files:**
- Create: `src/fda_device_rag/models.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Produces: `ChunkMetadata(source_type, source_url, document_title, section_name=None, record_id=None, retrieved_date="")` dataclass; `Chunk(id, text, metadata)` dataclass; `ScoredChunk(id, score, text, metadata)` dataclass — used by every later task.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_models.py
from fda_device_rag.models import Chunk, ChunkMetadata, ScoredChunk


def test_chunk_metadata_defaults():
    meta = ChunkMetadata(
        source_type="recall",
        source_url="https://example.com/recall/1",
        document_title="Example Recall",
    )
    assert meta.section_name is None
    assert meta.record_id is None
    assert meta.retrieved_date == ""


def test_chunk_holds_text_and_metadata():
    meta = ChunkMetadata(
        source_type="recall",
        source_url="https://example.com/recall/1",
        document_title="Example Recall",
    )
    chunk = Chunk(id="recall-1", text="Some recall text", metadata=meta)
    assert chunk.id == "recall-1"
    assert chunk.text == "Some recall text"
    assert chunk.metadata is meta


def test_scored_chunk_holds_score():
    meta = ChunkMetadata(
        source_type="recall",
        source_url="https://example.com/recall/1",
        document_title="Example Recall",
    )
    scored = ScoredChunk(id="recall-1", score=0.87, text="Some recall text", metadata=meta)
    assert scored.score == 0.87
    assert scored.id == "recall-1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fda_device_rag.models'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/fda_device_rag/models.py
from dataclasses import dataclass
from typing import Optional


@dataclass
class ChunkMetadata:
    source_type: str
    source_url: str
    document_title: str
    section_name: Optional[str] = None
    record_id: Optional[str] = None
    retrieved_date: str = ""


@dataclass
class Chunk:
    id: str
    text: str
    metadata: ChunkMetadata


@dataclass
class ScoredChunk:
    id: str
    score: float
    text: str
    metadata: ChunkMetadata
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_models.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/fda_device_rag/models.py tests/test_models.py
git commit -m "feat: add Chunk/ChunkMetadata/ScoredChunk data models"
```

---

### Task 3: openFDA Client

**Files:**
- Create: `src/fda_device_rag/ingestion/openfda_client.py`
- Test: `tests/ingestion/test_openfda_client.py`
- Create: `tests/ingestion/__init__.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `fetch_recalls(search=None, limit=100, skip=0) -> list[dict]`, `fetch_events(search=None, limit=100, skip=0) -> list[dict]` — raw openFDA JSON records, used by Task 5.

- [ ] **Step 1: Create the ingestion test package**

```bash
mkdir -p tests/ingestion
touch tests/ingestion/__init__.py
```

- [ ] **Step 2: Write the failing test**

```python
# tests/ingestion/test_openfda_client.py
from unittest.mock import patch, MagicMock

from fda_device_rag.ingestion.openfda_client import fetch_recalls, fetch_events


@patch("fda_device_rag.ingestion.openfda_client.requests.get")
def test_fetch_recalls_returns_results_list(mock_get):
    mock_response = MagicMock()
    mock_response.json.return_value = {"results": [{"reason_for_recall": "test reason"}]}
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    results = fetch_recalls(limit=5, skip=0)

    assert results == [{"reason_for_recall": "test reason"}]
    called_url = mock_get.call_args.args[0]
    called_params = mock_get.call_args.kwargs["params"]
    assert called_url == "https://api.fda.gov/device/recall.json"
    assert called_params == {"limit": 5, "skip": 0}


@patch("fda_device_rag.ingestion.openfda_client.requests.get")
def test_fetch_events_passes_search_param(mock_get):
    mock_response = MagicMock()
    mock_response.json.return_value = {"results": [{"report_number": "10"}]}
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    results = fetch_events(search="device.generic_name:pacemaker", limit=10, skip=0)

    assert results == [{"report_number": "10"}]
    called_params = mock_get.call_args.kwargs["params"]
    assert called_params == {"limit": 10, "skip": 0, "search": "device.generic_name:pacemaker"}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/ingestion/test_openfda_client.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fda_device_rag.ingestion.openfda_client'`

- [ ] **Step 4: Write minimal implementation**

```python
# src/fda_device_rag/ingestion/openfda_client.py
import requests

OPENFDA_RECALL_URL = "https://api.fda.gov/device/recall.json"
OPENFDA_EVENT_URL = "https://api.fda.gov/device/event.json"


def _fetch(url: str, search: str | None, limit: int, skip: int) -> list[dict]:
    params = {"limit": limit, "skip": skip}
    if search:
        params["search"] = search
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()["results"]


def fetch_recalls(search: str | None = None, limit: int = 100, skip: int = 0) -> list[dict]:
    return _fetch(OPENFDA_RECALL_URL, search, limit, skip)


def fetch_events(search: str | None = None, limit: int = 100, skip: int = 0) -> list[dict]:
    return _fetch(OPENFDA_EVENT_URL, search, limit, skip)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/ingestion/test_openfda_client.py -v`
Expected: 2 passed

- [ ] **Step 6: Commit**

```bash
git add src/fda_device_rag/ingestion/openfda_client.py tests/ingestion/test_openfda_client.py tests/ingestion/__init__.py
git commit -m "feat: add openFDA recall/event client"
```

---

### Task 4: Ingestion Manifest Writer

**Files:**
- Create: `src/fda_device_rag/ingestion/manifest.py`
- Test: `tests/ingestion/test_manifest.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `ManifestEntry(source_type, identifier, url, retrieved_date)` dataclass; `ManifestWriter(path).write(entry)` / `.close()` — used by the Task 13 ingestion script.

- [ ] **Step 1: Write the failing test**

```python
# tests/ingestion/test_manifest.py
import csv

from fda_device_rag.ingestion.manifest import ManifestEntry, ManifestWriter


def test_manifest_writer_writes_header_and_rows(tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    writer = ManifestWriter(manifest_path)
    writer.write(ManifestEntry("recall", "Z-0001-04", "https://example.com/1", "2026-07-28"))
    writer.write(ManifestEntry("guidance_pdf", "guidance-1", "https://example.com/2.pdf", "2026-07-28"))
    writer.close()

    with open(manifest_path, newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2
    assert rows[0]["source_type"] == "recall"
    assert rows[0]["identifier"] == "Z-0001-04"
    assert rows[1]["source_type"] == "guidance_pdf"


def test_manifest_writer_appends_without_duplicate_header(tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    writer = ManifestWriter(manifest_path)
    writer.write(ManifestEntry("recall", "Z-0001-04", "https://example.com/1", "2026-07-28"))
    writer.close()

    writer2 = ManifestWriter(manifest_path)
    writer2.write(ManifestEntry("recall", "Z-0002-04", "https://example.com/2", "2026-07-28"))
    writer2.close()

    with open(manifest_path, newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/ingestion/test_manifest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fda_device_rag.ingestion.manifest'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/fda_device_rag/ingestion/manifest.py
import csv
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class ManifestEntry:
    source_type: str
    identifier: str
    url: str
    retrieved_date: str


class ManifestWriter:
    FIELDNAMES = ["source_type", "identifier", "url", "retrieved_date"]

    def __init__(self, path: Path):
        self._path = Path(path)
        is_new = not self._path.exists()
        self._file = open(self._path, "a", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDNAMES)
        if is_new:
            self._writer.writeheader()

    def write(self, entry: ManifestEntry) -> None:
        self._writer.writerow(asdict(entry))
        self._file.flush()

    def close(self) -> None:
        self._file.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/ingestion/test_manifest.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/fda_device_rag/ingestion/manifest.py tests/ingestion/test_manifest.py
git commit -m "feat: add ingestion manifest writer"
```

---

### Task 5: Structured Document Builders (recall / MAUDE)

**Files:**
- Create: `src/fda_device_rag/documents/structured.py`
- Test: `tests/documents/test_structured.py`
- Create: `tests/documents/__init__.py`

**Interfaces:**
- Consumes: `Chunk`, `ChunkMetadata` from `fda_device_rag.models` (Task 2).
- Produces: `recall_to_chunk(record: dict, retrieved_date: str) -> Chunk`, `event_to_chunk(record: dict, retrieved_date: str) -> Chunk` — used by the Task 13 ingestion script.

- [ ] **Step 1: Create the documents test package**

```bash
mkdir -p tests/documents
touch tests/documents/__init__.py
```

- [ ] **Step 2: Write the failing test**

```python
# tests/documents/test_structured.py
from fda_device_rag.documents.structured import recall_to_chunk, event_to_chunk


def test_recall_to_chunk_builds_templated_text():
    record = {
        "res_event_number": "27549",
        "reason_for_recall": "Source-skin-distance non-compliance.",
        "action": "Safety notice sent to the direct account.",
        "product_description": "Sedecal SP-HF 4.0 Portable X-Ray System",
    }

    chunk = recall_to_chunk(record, retrieved_date="2026-07-28")

    assert chunk.id == "recall-27549"
    assert "Source-skin-distance non-compliance." in chunk.text
    assert "Safety notice sent to the direct account." in chunk.text
    assert "Sedecal SP-HF 4.0 Portable X-Ray System" in chunk.text
    assert chunk.metadata.source_type == "recall"
    assert chunk.metadata.record_id == "27549"
    assert chunk.metadata.retrieved_date == "2026-07-28"


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


def test_event_to_chunk_handles_missing_device_list():
    record = {"report_number": "11", "mdr_text": [{"text": "Only narrative."}], "device": []}

    chunk = event_to_chunk(record, retrieved_date="2026-07-28")

    assert chunk.metadata.document_title == "Unknown device"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/documents/test_structured.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fda_device_rag.documents.structured'`

- [ ] **Step 4: Write minimal implementation**

```python
# src/fda_device_rag/documents/structured.py
from fda_device_rag.models import Chunk, ChunkMetadata


def recall_to_chunk(record: dict, retrieved_date: str) -> Chunk:
    reason = record.get("reason_for_recall", "")
    action = record.get("action", "")
    product = record.get("product_description", "")
    text = f"Recall reason: {reason}\nAction taken: {action}\nProduct: {product}"

    record_id = str(record.get("res_event_number", ""))
    metadata = ChunkMetadata(
        source_type="recall",
        source_url=f"https://api.fda.gov/device/recall.json?search=res_event_number:{record_id}",
        document_title=product[:120] if product else "FDA Device Recall",
        section_name=None,
        record_id=record_id,
        retrieved_date=retrieved_date,
    )
    return Chunk(id=f"recall-{record_id}", text=text, metadata=metadata)


def event_to_chunk(record: dict, retrieved_date: str) -> Chunk:
    mdr_texts = record.get("mdr_text", [])
    narrative = "\n".join(t.get("text", "") for t in mdr_texts if t.get("text"))

    report_number = str(record.get("report_number", ""))
    device_list = record.get("device", [])
    device_name = device_list[0].get("generic_name", "Unknown device") if device_list else "Unknown device"

    text = f"Adverse event narrative ({device_name}): {narrative}"
    metadata = ChunkMetadata(
        source_type="maude_event",
        source_url=f"https://api.fda.gov/device/event.json?search=report_number:{report_number}",
        document_title=device_name,
        section_name=None,
        record_id=report_number,
        retrieved_date=retrieved_date,
    )
    return Chunk(id=f"event-{report_number}", text=text, metadata=metadata)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/documents/test_structured.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add src/fda_device_rag/documents/structured.py tests/documents/test_structured.py tests/documents/__init__.py
git commit -m "feat: build templated chunks from recall/MAUDE records"
```

---

### Task 6: PDF Download and Text Extraction

**Files:**
- Create: `src/fda_device_rag/ingestion/pdf_fetch.py`
- Create: `src/fda_device_rag/documents/pdf_document.py`
- Test: `tests/ingestion/test_pdf_fetch.py`
- Test: `tests/documents/test_pdf_document.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `download_pdfs(url_list: list[str], dest_dir: Path) -> list[Path]`; `extract_pdf_text(path: Path) -> str` — used by the Task 13 ingestion script and by Task 7 (section detection).

- [ ] **Step 1: Write the failing test for PDF download**

```python
# tests/ingestion/test_pdf_fetch.py
from unittest.mock import patch, MagicMock

from fda_device_rag.ingestion.pdf_fetch import download_pdfs


@patch("fda_device_rag.ingestion.pdf_fetch.requests.get")
def test_download_pdfs_writes_files_to_dest_dir(mock_get, tmp_path):
    mock_response = MagicMock()
    mock_response.content = b"%PDF-1.4 fake content"
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    dest_dir = tmp_path / "pdfs"
    paths = download_pdfs(["https://example.com/docs/guidance-1.pdf"], dest_dir)

    assert len(paths) == 1
    assert paths[0].name == "guidance-1.pdf"
    assert paths[0].read_bytes() == b"%PDF-1.4 fake content"


@patch("fda_device_rag.ingestion.pdf_fetch.requests.get")
def test_download_pdfs_appends_pdf_extension_if_missing(mock_get, tmp_path):
    mock_response = MagicMock()
    mock_response.content = b"%PDF-1.4 fake content"
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    dest_dir = tmp_path / "pdfs"
    paths = download_pdfs(["https://example.com/docs/guidance-2"], dest_dir)

    assert paths[0].name == "guidance-2.pdf"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/ingestion/test_pdf_fetch.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fda_device_rag.ingestion.pdf_fetch'`

- [ ] **Step 3: Write minimal implementation for PDF download**

```python
# src/fda_device_rag/ingestion/pdf_fetch.py
from pathlib import Path

import requests


def download_pdfs(url_list: list[str], dest_dir: Path) -> list[Path]:
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    downloaded = []
    for url in url_list:
        filename = url.rsplit("/", 1)[-1]
        if not filename.lower().endswith(".pdf"):
            filename += ".pdf"
        dest_path = dest_dir / filename

        response = requests.get(url, timeout=60)
        response.raise_for_status()
        dest_path.write_bytes(response.content)
        downloaded.append(dest_path)

    return downloaded
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/ingestion/test_pdf_fetch.py -v`
Expected: 2 passed

- [ ] **Step 5: Write the failing test for PDF text extraction**

```python
# tests/documents/test_pdf_document.py
from unittest.mock import patch, MagicMock

from fda_device_rag.documents.pdf_document import extract_pdf_text


@patch("fda_device_rag.documents.pdf_document.PdfReader")
def test_extract_pdf_text_joins_pages_with_newline(mock_reader_cls):
    page1 = MagicMock()
    page1.extract_text.return_value = "Page one text."
    page2 = MagicMock()
    page2.extract_text.return_value = "Page two text."
    mock_reader_cls.return_value.pages = [page1, page2]

    text = extract_pdf_text("fake/path.pdf")

    assert text == "Page one text.\nPage two text."


@patch("fda_device_rag.documents.pdf_document.PdfReader")
def test_extract_pdf_text_handles_none_from_extract_text(mock_reader_cls):
    page1 = MagicMock()
    page1.extract_text.return_value = None
    mock_reader_cls.return_value.pages = [page1]

    text = extract_pdf_text("fake/path.pdf")

    assert text == ""
```

- [ ] **Step 6: Run test to verify it fails**

Run: `pytest tests/documents/test_pdf_document.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fda_device_rag.documents.pdf_document'`

- [ ] **Step 7: Write minimal implementation for PDF text extraction**

```python
# src/fda_device_rag/documents/pdf_document.py
from pypdf import PdfReader


def extract_pdf_text(path) -> str:
    reader = PdfReader(str(path))
    pages_text = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages_text)
```

- [ ] **Step 8: Run test to verify it passes**

Run: `pytest tests/documents/test_pdf_document.py -v`
Expected: 2 passed

- [ ] **Step 9: Commit**

```bash
git add src/fda_device_rag/ingestion/pdf_fetch.py src/fda_device_rag/documents/pdf_document.py tests/ingestion/test_pdf_fetch.py tests/documents/test_pdf_document.py
git commit -m "feat: add PDF download and text extraction"
```

---

### Task 7: PDF Section Detection

**Files:**
- Create: `src/fda_device_rag/chunking/sections.py`
- Test: `tests/chunking/test_sections.py`
- Create: `tests/chunking/__init__.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (operates on plain extracted text from Task 6's `extract_pdf_text`).
- Produces: `Section(heading: str, body: str)` dataclass; `detect_sections(text: str) -> list[Section]` — used by Task 8 (chunking).

- [ ] **Step 1: Create the chunking test package**

```bash
mkdir -p tests/chunking
touch tests/chunking/__init__.py
```

- [ ] **Step 2: Write the failing test**

```python
# tests/chunking/test_sections.py
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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/chunking/test_sections.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fda_device_rag.chunking.sections'`

- [ ] **Step 4: Write minimal implementation**

```python
# src/fda_device_rag/chunking/sections.py
import re
from dataclasses import dataclass

HEADING_PATTERN = re.compile(r"^(?:\d+(?:\.\d+)*\s+)?([A-Z][A-Z0-9 ,\-/:]{2,80})$")


@dataclass
class Section:
    heading: str
    body: str


def _is_heading(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and len(stripped) <= 80 and bool(HEADING_PATTERN.match(stripped))


def detect_sections(text: str) -> list[Section]:
    lines = text.splitlines()
    sections: list[Section] = []
    current_heading = "Document"
    current_body_lines: list[str] = []

    for line in lines:
        if _is_heading(line):
            if current_body_lines:
                sections.append(Section(current_heading, "\n".join(current_body_lines).strip()))
            current_heading = line.strip()
            current_body_lines = []
        else:
            current_body_lines.append(line)

    if current_body_lines:
        sections.append(Section(current_heading, "\n".join(current_body_lines).strip()))

    return [s for s in sections if s.body]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/chunking/test_sections.py -v`
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add src/fda_device_rag/chunking/sections.py tests/chunking/test_sections.py tests/chunking/__init__.py
git commit -m "feat: detect section headings in extracted PDF text"
```

---

### Task 8: Structure-Aware PDF Chunking

**Files:**
- Create: `src/fda_device_rag/chunking/pdf_chunker.py`
- Test: `tests/chunking/test_pdf_chunker.py`

**Interfaces:**
- Consumes: `Section` from `fda_device_rag.chunking.sections` (Task 7); `Chunk`, `ChunkMetadata` from `fda_device_rag.models` (Task 2).
- Produces: `chunk_pdf_text(text: str, source_type: str, source_url: str, document_title: str, retrieved_date: str, id_prefix: str) -> list[Chunk]` — used by the Task 13 ingestion script.

- [ ] **Step 1: Write the failing test**

```python
# tests/chunking/test_pdf_chunker.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/chunking/test_pdf_chunker.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fda_device_rag.chunking.pdf_chunker'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/fda_device_rag/chunking/pdf_chunker.py
from langchain_text_splitters import RecursiveCharacterTextSplitter

from fda_device_rag.chunking.sections import detect_sections
from fda_device_rag.models import Chunk, ChunkMetadata

CHUNK_SIZE_CHARS = 1600
CHUNK_OVERLAP_CHARS = 240


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

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/chunking/test_pdf_chunker.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/fda_device_rag/chunking/pdf_chunker.py tests/chunking/test_pdf_chunker.py
git commit -m "feat: structure-aware chunking for guidance/IFU PDFs"
```

---

### Task 9: Local Embedder

**Files:**
- Create: `src/fda_device_rag/embedding/embedder.py`
- Test: `tests/embedding/test_embedder.py`
- Create: `tests/embedding/__init__.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `Embedder(model_name="BAAI/bge-small-en-v1.5")` with `.embed_documents(texts: list[str]) -> list[list[float]]` and `.embed_query(text: str) -> list[float]` — used by Task 10 (vector store) and Task 12 (hybrid retriever).

- [ ] **Step 1: Create the embedding test package**

```bash
mkdir -p tests/embedding
touch tests/embedding/__init__.py
```

- [ ] **Step 2: Write the failing test (mocks `SentenceTransformer` — no real model download in this unit test)**

```python
# tests/embedding/test_embedder.py
from unittest.mock import patch, MagicMock

import numpy as np

from fda_device_rag.embedding.embedder import Embedder, QUERY_PREFIX


@patch("fda_device_rag.embedding.embedder.SentenceTransformer")
def test_embed_documents_returns_list_of_lists_no_prefix(mock_st_cls):
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
    mock_st_cls.return_value = mock_model

    embedder = Embedder()
    result = embedder.embed_documents(["doc one", "doc two"])

    assert result == [[0.1, 0.2], [0.3, 0.4]]
    mock_model.encode.assert_called_once_with(["doc one", "doc two"], normalize_embeddings=True)


@patch("fda_device_rag.embedding.embedder.SentenceTransformer")
def test_embed_query_applies_bge_query_prefix(mock_st_cls):
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([0.5, 0.6])
    mock_st_cls.return_value = mock_model

    embedder = Embedder()
    result = embedder.embed_query("what is the recall reason")

    assert result == [0.5, 0.6]
    mock_model.encode.assert_called_once_with(
        QUERY_PREFIX + "what is the recall reason", normalize_embeddings=True
    )
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/embedding/test_embedder.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fda_device_rag.embedding.embedder'`

- [ ] **Step 4: Write minimal implementation**

```python
# src/fda_device_rag/embedding/embedder.py
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    def __init__(self, model_name: str = MODEL_NAME):
        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self._model.encode(QUERY_PREFIX + text, normalize_embeddings=True)
        return vector.tolist()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/embedding/test_embedder.py -v`
Expected: 2 passed

- [ ] **Step 6: Commit**

```bash
git add src/fda_device_rag/embedding/embedder.py tests/embedding/test_embedder.py tests/embedding/__init__.py
git commit -m "feat: add local bge-small embedder with query/document asymmetry"
```

---

### Task 10: Chroma Vector Store

**Files:**
- Create: `src/fda_device_rag/store/chroma_store.py`
- Test: `tests/store/test_chroma_store.py`
- Create: `tests/store/__init__.py`

**Interfaces:**
- Consumes: `Chunk`, `ChunkMetadata`, `ScoredChunk` from `fda_device_rag.models` (Task 2).
- Produces: `ChromaStore(persist_dir, collection_name="fda_docs")` with `.add_chunks(chunks: list[Chunk], embeddings: list[list[float]])` and `.query(query_embedding: list[float], top_k=20) -> list[ScoredChunk]` — used by Task 12 (hybrid retriever).

- [ ] **Step 1: Create the store test package**

```bash
mkdir -p tests/store
touch tests/store/__init__.py
```

- [ ] **Step 2: Write the failing test (real Chroma against a tmp_path directory — no network involved)**

```python
# tests/store/test_chroma_store.py
from fda_device_rag.models import Chunk, ChunkMetadata
from fda_device_rag.store.chroma_store import ChromaStore


def _chunk(id_, text, source_type="recall"):
    return Chunk(
        id=id_,
        text=text,
        metadata=ChunkMetadata(
            source_type=source_type,
            source_url=f"https://example.com/{id_}",
            document_title=f"Title {id_}",
        ),
    )


def test_add_and_query_returns_closest_first(tmp_path):
    store = ChromaStore(persist_dir=str(tmp_path / "chroma"))

    chunks = [_chunk("a", "text a"), _chunk("b", "text b"), _chunk("c", "text c")]
    embeddings = [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]]
    store.add_chunks(chunks, embeddings)

    results = store.query(query_embedding=[1.0, 0.0], top_k=2)

    assert len(results) == 2
    assert results[0].id == "a"
    assert results[0].score > results[1].score
    assert results[0].metadata.source_type == "recall"


def test_query_respects_top_k(tmp_path):
    store = ChromaStore(persist_dir=str(tmp_path / "chroma"))

    chunks = [_chunk("a", "text a"), _chunk("b", "text b"), _chunk("c", "text c")]
    embeddings = [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]]
    store.add_chunks(chunks, embeddings)

    results = store.query(query_embedding=[1.0, 0.0], top_k=1)

    assert len(results) == 1
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/store/test_chroma_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fda_device_rag.store.chroma_store'`

- [ ] **Step 4: Write minimal implementation**

```python
# src/fda_device_rag/store/chroma_store.py
from dataclasses import asdict

import chromadb

from fda_device_rag.models import Chunk, ChunkMetadata, ScoredChunk


class ChromaStore:
    def __init__(self, persist_dir: str, collection_name: str = "fda_docs"):
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        self._collection.add(
            ids=[c.id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[asdict(c.metadata) for c in chunks],
        )

    def query(self, query_embedding: list[float], top_k: int = 20) -> list[ScoredChunk]:
        result = self._collection.query(query_embeddings=[query_embedding], n_results=top_k)

        ids = result["ids"][0]
        distances = result["distances"][0]
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]

        scored: list[ScoredChunk] = []
        for id_, dist, doc, meta in zip(ids, distances, documents, metadatas):
            similarity = 1 - dist
            scored.append(ScoredChunk(id=id_, score=similarity, text=doc, metadata=ChunkMetadata(**meta)))
        return scored
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/store/test_chroma_store.py -v`
Expected: 2 passed

- [ ] **Step 6: Commit**

```bash
git add src/fda_device_rag/store/chroma_store.py tests/store/test_chroma_store.py tests/store/__init__.py
git commit -m "feat: add Chroma persistent vector store wrapper"
```

---

### Task 11: BM25 Index

**Files:**
- Create: `src/fda_device_rag/retrieval/bm25_index.py`
- Test: `tests/retrieval/test_bm25_index.py`
- Create: `tests/retrieval/__init__.py`

**Interfaces:**
- Consumes: `Chunk`, `ScoredChunk`, `ChunkMetadata` from `fda_device_rag.models` (Task 2).
- Produces: `tokenize(text: str) -> list[str]`; `BM25Index(chunks: list[Chunk])` with `.query(query_text: str, top_k=20) -> list[ScoredChunk]` — used by Task 12 (hybrid retriever).

- [ ] **Step 1: Create the retrieval test package**

```bash
mkdir -p tests/retrieval
touch tests/retrieval/__init__.py
```

- [ ] **Step 2: Write the failing test**

```python
# tests/retrieval/test_bm25_index.py
from fda_device_rag.models import Chunk, ChunkMetadata
from fda_device_rag.retrieval.bm25_index import BM25Index, tokenize


def _chunk(id_, text):
    return Chunk(
        id=id_,
        text=text,
        metadata=ChunkMetadata(source_type="recall", source_url="https://example.com", document_title="t"),
    )


def test_tokenize_lowercases_and_strips_light_punctuation_but_keeps_codes_verbatim():
    tokens = tokenize("Device code K123456 is a code!")

    assert tokens == ["device", "code", "k123456", "is", "a", "code"]


def test_bm25_index_ranks_exact_code_match_first():
    chunks = [
        _chunk("a", "This document discusses general safety information."),
        _chunk("b", "The affected device carries product code K850651 specifically."),
        _chunk("c", "Unrelated narrative about a different device entirely."),
    ]
    index = BM25Index(chunks)

    results = index.query("K850651", top_k=2)

    assert results[0].id == "b"
    assert len(results) == 2


def test_bm25_index_respects_top_k():
    chunks = [_chunk("a", "alpha beta"), _chunk("b", "alpha gamma"), _chunk("c", "alpha delta")]
    index = BM25Index(chunks)

    results = index.query("alpha", top_k=1)

    assert len(results) == 1
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/retrieval/test_bm25_index.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fda_device_rag.retrieval.bm25_index'`

- [ ] **Step 4: Write minimal implementation**

```python
# src/fda_device_rag/retrieval/bm25_index.py
import re

from rank_bm25 import BM25Okapi

from fda_device_rag.models import Chunk, ScoredChunk

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


class BM25Index:
    def __init__(self, chunks: list[Chunk]):
        self._chunks = chunks
        tokenized_corpus = [tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(tokenized_corpus)

    def query(self, query_text: str, top_k: int = 20) -> list[ScoredChunk]:
        tokenized_query = tokenize(query_text)
        scores = self._bm25.get_scores(tokenized_query)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        return [
            ScoredChunk(
                id=self._chunks[i].id,
                score=float(scores[i]),
                text=self._chunks[i].text,
                metadata=self._chunks[i].metadata,
            )
            for i in ranked_indices
        ]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/retrieval/test_bm25_index.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add src/fda_device_rag/retrieval/bm25_index.py tests/retrieval/test_bm25_index.py tests/retrieval/__init__.py
git commit -m "feat: add BM25 sparse index with code-preserving tokenizer"
```

---

### Task 12: Hybrid Retriever (RRF Fusion)

**Files:**
- Create: `src/fda_device_rag/retrieval/hybrid_retriever.py`
- Test: `tests/retrieval/test_hybrid_retriever.py`

**Interfaces:**
- Consumes: `ScoredChunk` from `fda_device_rag.models` (Task 2); duck-typed `dense_store` with `.query(query_embedding, top_k) -> list[ScoredChunk]` (Task 10's `ChromaStore`); duck-typed `bm25_index` with `.query(query_text, top_k) -> list[ScoredChunk]` (Task 11's `BM25Index`); duck-typed `embedder` with `.embed_query(text) -> list[float]` (Task 9's `Embedder`).
- Produces: `reciprocal_rank_fusion(*ranked_id_lists, k=60) -> list[tuple[str, float]]`; `HybridRetriever(dense_store, bm25_index, embedder)` with `.retrieve(query_text: str, top_k=5) -> list[ScoredChunk]` — used by Phase 3's generation layer (out of scope for this plan).

- [ ] **Step 1: Write the failing test for the pure RRF merge function**

```python
# tests/retrieval/test_hybrid_retriever.py
from fda_device_rag.models import ScoredChunk, ChunkMetadata
from fda_device_rag.retrieval.hybrid_retriever import reciprocal_rank_fusion, HybridRetriever


def test_reciprocal_rank_fusion_ranks_items_in_both_lists_highest():
    dense_ids = ["a", "b", "c"]
    bm25_ids = ["b", "a", "d"]

    fused = reciprocal_rank_fusion(dense_ids, bm25_ids, k=60)
    fused_ids = [chunk_id for chunk_id, _score in fused]

    # "a" and "b" appear in both lists near the top, so they should outrank "c"/"d"
    assert fused_ids[0] in ("a", "b")
    assert fused_ids[1] in ("a", "b")
    assert set(fused_ids[2:]) == {"c", "d"}


def test_reciprocal_rank_fusion_handles_disjoint_lists():
    fused = reciprocal_rank_fusion(["a", "b"], ["c", "d"], k=60)
    fused_ids = {chunk_id for chunk_id, _score in fused}

    assert fused_ids == {"a", "b", "c", "d"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/retrieval/test_hybrid_retriever.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fda_device_rag.retrieval.hybrid_retriever'`

- [ ] **Step 3: Write the RRF implementation**

```python
# src/fda_device_rag/retrieval/hybrid_retriever.py
from fda_device_rag.models import ScoredChunk

RRF_K = 60
DENSE_TOP_N = 20
BM25_TOP_N = 20
FUSED_TOP_K = 5


def reciprocal_rank_fusion(*ranked_id_lists: list[str], k: int = RRF_K) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranked_ids in ranked_id_lists:
        for rank, chunk_id in enumerate(ranked_ids):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/retrieval/test_hybrid_retriever.py -v`
Expected: 2 passed

- [ ] **Step 5: Write the failing test for `HybridRetriever` wiring (using test doubles, not real Chroma/BM25/model)**

```python
# append to tests/retrieval/test_hybrid_retriever.py
def _scored(id_, score):
    return ScoredChunk(
        id=id_,
        score=score,
        text=f"text {id_}",
        metadata=ChunkMetadata(source_type="recall", source_url="https://example.com", document_title="t"),
    )


class _FakeDenseStore:
    def __init__(self, results):
        self._results = results

    def query(self, query_embedding, top_k):
        return self._results[:top_k]


class _FakeBM25Index:
    def __init__(self, results):
        self._results = results

    def query(self, query_text, top_k):
        return self._results[:top_k]


class _FakeEmbedder:
    def embed_query(self, text):
        return [0.1, 0.2]


def test_hybrid_retriever_fuses_dense_and_bm25_results():
    dense_results = [_scored("a", 0.9), _scored("b", 0.8), _scored("c", 0.7)]
    bm25_results = [_scored("b", 5.0), _scored("a", 3.0), _scored("d", 1.0)]

    retriever = HybridRetriever(
        dense_store=_FakeDenseStore(dense_results),
        bm25_index=_FakeBM25Index(bm25_results),
        embedder=_FakeEmbedder(),
    )

    top = retriever.retrieve("some query", top_k=2)

    assert len(top) == 2
    assert {c.id for c in top} == {"a", "b"}


def test_hybrid_retriever_returns_at_most_top_k():
    dense_results = [_scored("a", 0.9), _scored("b", 0.8)]
    bm25_results = [_scored("c", 5.0), _scored("d", 3.0)]

    retriever = HybridRetriever(
        dense_store=_FakeDenseStore(dense_results),
        bm25_index=_FakeBM25Index(bm25_results),
        embedder=_FakeEmbedder(),
    )

    top = retriever.retrieve("some query", top_k=3)

    assert len(top) == 3
```

- [ ] **Step 6: Run test to verify it fails**

Run: `pytest tests/retrieval/test_hybrid_retriever.py -v`
Expected: FAIL with `ImportError: cannot import name 'HybridRetriever'`

- [ ] **Step 7: Write the `HybridRetriever` implementation**

```python
# append to src/fda_device_rag/retrieval/hybrid_retriever.py

class HybridRetriever:
    def __init__(self, dense_store, bm25_index, embedder):
        self._dense_store = dense_store
        self._bm25_index = bm25_index
        self._embedder = embedder

    def retrieve(self, query_text: str, top_k: int = FUSED_TOP_K) -> list[ScoredChunk]:
        query_embedding = self._embedder.embed_query(query_text)
        dense_results = self._dense_store.query(query_embedding, top_k=DENSE_TOP_N)
        bm25_results = self._bm25_index.query(query_text, top_k=BM25_TOP_N)

        dense_ids = [r.id for r in dense_results]
        bm25_ids = [r.id for r in bm25_results]
        fused = reciprocal_rank_fusion(dense_ids, bm25_ids)

        by_id: dict[str, ScoredChunk] = {}
        for r in dense_results:
            by_id[r.id] = r
        for r in bm25_results:
            by_id.setdefault(r.id, r)

        return [by_id[chunk_id] for chunk_id, _score in fused[:top_k] if chunk_id in by_id]
```

- [ ] **Step 8: Run test to verify it passes**

Run: `pytest tests/retrieval/test_hybrid_retriever.py -v`
Expected: 4 passed

- [ ] **Step 9: Commit**

```bash
git add src/fda_device_rag/retrieval/hybrid_retriever.py tests/retrieval/test_hybrid_retriever.py
git commit -m "feat: add hybrid retriever with RRF fusion of dense + BM25"
```

---

### Task 13: End-to-End Ingestion Script and Integration Test

**Files:**
- Create: `scripts/pull_corpus.py`
- Create: `scripts/build_index.py`
- Test: `tests/test_integration_pipeline.py`

**Interfaces:**
- Consumes: every module from Tasks 2-12 (`openfda_client`, `manifest`, `structured`, `pdf_fetch`, `pdf_document`, `pdf_chunker`, `Embedder`, `ChromaStore`, `BM25Index`, `HybridRetriever`).
- Produces: `scripts/pull_corpus.py` (CLI: fetches recall/event records + downloads curated PDFs, writes `data/raw/` and `data/manifest.csv`); `scripts/build_index.py` (CLI: chunks everything in `data/raw/`, embeds, and populates the Chroma store + BM25 index). Nothing downstream in this plan consumes these — they are the operator-facing entry points.

- [ ] **Step 1: Write the failing integration test (stub embedder for speed — no real model download; verifies wiring, not embedding quality)**

```python
# tests/test_integration_pipeline.py
from fda_device_rag.documents.structured import recall_to_chunk, event_to_chunk
from fda_device_rag.chunking.pdf_chunker import chunk_pdf_text
from fda_device_rag.store.chroma_store import ChromaStore
from fda_device_rag.retrieval.bm25_index import BM25Index
from fda_device_rag.retrieval.hybrid_retriever import HybridRetriever


class _StubEmbedder:
    """Deterministic pseudo-embedding: hashes text into a fixed-size vector.
    Only used to prove the pipeline wiring works end-to-end; real embedding
    quality is validated separately by the Phase 2 benchmark."""

    def _vector(self, text: str) -> list[float]:
        h = hash(text)
        return [((h >> (8 * i)) % 256) / 255.0 for i in range(4)]

    def embed_documents(self, texts):
        return [self._vector(t) for t in texts]

    def embed_query(self, text):
        return self._vector(text)


def test_pipeline_retrieves_recall_chunk_for_matching_query(tmp_path):
    recall_record = {
        "res_event_number": "1",
        "reason_for_recall": "Battery may overheat during normal use.",
        "action": "Firmware update issued to affected units.",
        "product_description": "Implantable Pacemaker Pulse-Generator Model X200",
    }
    event_record = {
        "report_number": "2",
        "mdr_text": [{"text": "Patient reported device beeping unexpectedly during sleep."}],
        "device": [{"generic_name": "MANUAL HOSPITAL BED"}],
    }
    guidance_chunks = chunk_pdf_text(
        "WARNINGS\nDo not expose the device to strong magnetic fields.",
        source_type="guidance_pdf",
        source_url="https://example.com/guidance.pdf",
        document_title="Example Guidance",
        retrieved_date="2026-07-28",
        id_prefix="guidance-1",
    )

    chunks = [
        recall_to_chunk(recall_record, retrieved_date="2026-07-28"),
        event_to_chunk(event_record, retrieved_date="2026-07-28"),
        *guidance_chunks,
    ]

    embedder = _StubEmbedder()
    store = ChromaStore(persist_dir=str(tmp_path / "chroma"))
    store.add_chunks(chunks, embedder.embed_documents([c.text for c in chunks]))
    bm25 = BM25Index(chunks)
    retriever = HybridRetriever(dense_store=store, bm25_index=bm25, embedder=embedder)

    results = retriever.retrieve("battery overheat pacemaker firmware update", top_k=3)

    assert any(r.id == "recall-1" for r in results)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_integration_pipeline.py -v`
Expected: FAIL — `ChromaStore`/`BM25Index` exist, but this specific assertion may fail if the stub embedder's hash-based vectors don't place `recall-1` in the dense top-20; run it to confirm the *actual* failure mode before proceeding (it should fail on the final `assert`, not an import error, since all modules already exist from Tasks 2-12).

- [ ] **Step 3: Fix the test to rely on BM25 (which is lexical, not hash-based) rather than the stub dense embeddings for relevance**

The stub embedder's hash-based vectors are only meant to exercise the pipeline's wiring, not carry real semantic meaning — dense similarity from a hash is arbitrary. The query text shares real vocabulary with the recall record ("battery", "overheat", "firmware"), so BM25 alone should surface it; adjust the assertion to check it appears in the *fused* top-3, which is guaranteed since BM25 will rank it highly regardless of the arbitrary dense scores:

```python
    results = retriever.retrieve("battery overheat pacemaker firmware update", top_k=3)

    result_ids = [r.id for r in results]
    assert "recall-1" in result_ids
```

(This is the same assertion — if it still fails, the fix is in the fixture text, not the test: verify `reason_for_recall` and `action` in `recall_record` share enough vocabulary with the query. The fixture above already shares "battery", "overheat", "firmware", "pacemaker" — keep these terms if editing.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_integration_pipeline.py -v`
Expected: 1 passed

- [ ] **Step 5: Write `scripts/pull_corpus.py`**

```python
# scripts/pull_corpus.py
"""Ingests recall/MAUDE narrative records and curated guidance/IFU PDFs into
data/raw/, logging every source to data/manifest.csv.

Usage: python scripts/pull_corpus.py
"""
import datetime
import json
from pathlib import Path

from fda_device_rag.ingestion.openfda_client import fetch_recalls, fetch_events
from fda_device_rag.ingestion.manifest import ManifestEntry, ManifestWriter
from fda_device_rag.ingestion.pdf_fetch import download_pdfs

# Hand-curated, defensible list -- each URL was chosen and verified by hand,
# not scraped. Fill in with real guidance/IFU PDF URLs before running.
GUIDANCE_PDF_URLS: list[str] = []
IFU_PDF_URLS: list[str] = []

DATA_DIR = Path("data/raw")
MANIFEST_PATH = Path("data/manifest.csv")


def main() -> None:
    retrieved_date = datetime.date.today().isoformat()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest = ManifestWriter(MANIFEST_PATH)

    recalls = fetch_recalls(limit=100)
    (DATA_DIR / "recalls.json").write_text(json.dumps(recalls))
    for r in recalls:
        manifest.write(ManifestEntry("recall", str(r.get("res_event_number", "")), "openfda:device/recall", retrieved_date))

    events = fetch_events(limit=100)
    (DATA_DIR / "events.json").write_text(json.dumps(events))
    for e in events:
        manifest.write(ManifestEntry("maude_event", str(e.get("report_number", "")), "openfda:device/event", retrieved_date))

    if GUIDANCE_PDF_URLS:
        paths = download_pdfs(GUIDANCE_PDF_URLS, DATA_DIR / "guidance_pdfs")
        for url, path in zip(GUIDANCE_PDF_URLS, paths):
            manifest.write(ManifestEntry("guidance_pdf", path.name, url, retrieved_date))

    if IFU_PDF_URLS:
        paths = download_pdfs(IFU_PDF_URLS, DATA_DIR / "ifu_pdfs")
        for url, path in zip(IFU_PDF_URLS, paths):
            manifest.write(ManifestEntry("ifu_pdf", path.name, url, retrieved_date))

    manifest.close()
    print(f"Pulled {len(recalls)} recalls, {len(events)} events, "
          f"{len(GUIDANCE_PDF_URLS)} guidance PDFs, {len(IFU_PDF_URLS)} IFU PDFs "
          f"on {retrieved_date}. Update the README's first paragraph with these counts.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Write `scripts/build_index.py`**

```python
# scripts/build_index.py
"""Chunks everything pulled by pull_corpus.py, embeds it locally, and
populates the Chroma store + BM25 index.

Usage: python scripts/build_index.py
"""
import json
import pickle
from pathlib import Path

from fda_device_rag.documents.structured import recall_to_chunk, event_to_chunk
from fda_device_rag.documents.pdf_document import extract_pdf_text
from fda_device_rag.chunking.pdf_chunker import chunk_pdf_text
from fda_device_rag.embedding.embedder import Embedder
from fda_device_rag.store.chroma_store import ChromaStore
from fda_device_rag.retrieval.bm25_index import BM25Index

DATA_DIR = Path("data/raw")
CHROMA_DIR = Path("data/chroma")
BM25_INDEX_PATH = Path("data/bm25_index.pkl")


def main() -> None:
    chunks = []

    recalls_path = DATA_DIR / "recalls.json"
    if recalls_path.exists():
        for record in json.loads(recalls_path.read_text()):
            chunks.append(recall_to_chunk(record, retrieved_date=""))

    events_path = DATA_DIR / "events.json"
    if events_path.exists():
        for record in json.loads(events_path.read_text()):
            chunks.append(event_to_chunk(record, retrieved_date=""))

    for pdf_dir, source_type in [(DATA_DIR / "guidance_pdfs", "guidance_pdf"), (DATA_DIR / "ifu_pdfs", "ifu_pdf")]:
        if not pdf_dir.exists():
            continue
        for pdf_path in sorted(pdf_dir.glob("*.pdf")):
            text = extract_pdf_text(pdf_path)
            chunks.extend(
                chunk_pdf_text(
                    text,
                    source_type=source_type,
                    source_url=str(pdf_path),
                    document_title=pdf_path.stem,
                    retrieved_date="",
                    id_prefix=pdf_path.stem,
                )
            )

    embedder = Embedder()
    embeddings = embedder.embed_documents([c.text for c in chunks])

    store = ChromaStore(persist_dir=str(CHROMA_DIR))
    store.add_chunks(chunks, embeddings)

    bm25 = BM25Index(chunks)
    BM25_INDEX_PATH.write_bytes(pickle.dumps(bm25))

    print(f"Indexed {len(chunks)} chunks into {CHROMA_DIR} and {BM25_INDEX_PATH}.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Run the full test suite to confirm nothing is broken**

Run: `pytest -v`
Expected: all tests from Tasks 2-13 pass.

- [ ] **Step 8: Commit**

```bash
git add scripts/pull_corpus.py scripts/build_index.py tests/test_integration_pipeline.py
git commit -m "feat: add end-to-end ingestion/index-build scripts and integration test"
```

---

## Self-Review

**Spec coverage:**
- Corpus sources (recall, MAUDE, guidance PDFs, IFU PDFs, 510(k)-as-metadata-only) → Tasks 3, 5, 6, 13. ✓
- Structured record templating (no chunking) → Task 5. ✓
- Structure-aware PDF chunking (section-first, recursive-split-within-section) → Tasks 7, 8. ✓
- Local embedding (bge-small, cosine, query/document asymmetry) → Task 9. ✓
- Hybrid retrieval (BM25 + dense, RRF k=60, top-20/20 → top-5) → Tasks 10, 11, 12. ✓
- Ingestion manifest (backs README corpus line) → Task 4, wired in Task 13. ✓
- End-to-end wiring proof → Task 13. ✓
- Generation/citation grounding, eval harness, deployment, tests/CI: intentionally **not** in this plan — they are Phases 2-5 per the design doc's shipping order, each to get its own plan once this one ships.

**Placeholder scan:** No TBD/TODO markers. `GUIDANCE_PDF_URLS`/`IFU_PDF_URLS` are empty lists in `scripts/pull_corpus.py` by design — they're operator input (the hand-curated URL list), not a plan placeholder; the script runs correctly (produces zero PDF chunks) either way, and populating them is a data-curation task for whoever runs the script, not an implementation step.

**Type consistency:** `Chunk`/`ChunkMetadata`/`ScoredChunk` (Task 2) used identically across Tasks 5, 8, 10, 11, 12, 13. `Embedder.embed_documents`/`embed_query` (Task 9) signatures match their use in Tasks 10 and 12. `ChromaStore.query` and `BM25Index.query` both return `list[ScoredChunk]`, matching what `HybridRetriever.retrieve` expects. `reciprocal_rank_fusion` signature (`*ranked_id_lists, k=60`) matches its two call-sites (test and `HybridRetriever.retrieve`).
