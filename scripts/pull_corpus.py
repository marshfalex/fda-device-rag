# scripts/pull_corpus.py
"""Ingests recall/MAUDE narrative records and curated guidance/IFU PDFs into
data/raw/, logging every source to data/manifest.csv.

The openFDA pull uses a fixed, explicit date-range query and paginates to a
hard record cap, so the same command produces the same result set against the
same underlying FDA data. (The pull date recorded in the manifest is what
"as of" refers to -- FDA can and does add records to historical ranges.)

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
PULL_DATE_PATH = DATA_DIR / "pull_date.txt"

# Fixed queries, verified live against the API. Note the date formats differ
# between endpoints: device/recall's event_date_posted is ISO (YYYY-MM-DD),
# device/event's date_received is YYYYMMDD. Ranges are written with spaces --
# requests url-encodes the space to the "+" that openFDA's Lucene range syntax
# expects; a literal "+" in the string would be encoded to %2B and 500.
RECALL_SEARCH = "event_date_posted:[2015-01-01 TO 2024-12-31]"
EVENT_SEARCH = "date_received:[20150101 TO 20241231]"

PAGE_SIZE = 100
MAX_RECORDS = 500


def _fetch_paginated(fetch_fn, search: str, max_records: int = MAX_RECORDS) -> list[dict]:
    """Page through an openFDA endpoint in PAGE_SIZE increments up to a hard cap."""
    records: list[dict] = []
    for skip in range(0, max_records, PAGE_SIZE):
        page = fetch_fn(search=search, limit=PAGE_SIZE, skip=skip)
        if not page:
            break
        records.extend(page)
        if len(page) < PAGE_SIZE:
            break
    return records[:max_records]


def main() -> None:
    retrieved_date = datetime.date.today().isoformat()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest = ManifestWriter(MANIFEST_PATH)

    recalls = _fetch_paginated(fetch_recalls, RECALL_SEARCH)
    (DATA_DIR / "recalls.json").write_text(json.dumps(recalls))
    for r in recalls:
        # product_res_number, not res_event_number: one recall event covers many
        # product records, so res_event_number is not unique per record.
        manifest.write(ManifestEntry("recall", str(r.get("product_res_number", "")), "openfda:device/recall", retrieved_date))

    events = _fetch_paginated(fetch_events, EVENT_SEARCH)
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

    # Single source of truth for the pull date, read back by build_index.py so
    # retrieved_date actually reaches the indexed chunk metadata (citations).
    PULL_DATE_PATH.write_text(retrieved_date)

    print(f"Pulled {len(recalls)} recalls, {len(events)} events, "
          f"{len(GUIDANCE_PDF_URLS)} guidance PDFs, {len(IFU_PDF_URLS)} IFU PDFs "
          f"on {retrieved_date}.")
    print("Reproducibility -- fixed queries used for this pull:")
    print(f"  device/recall  search={RECALL_SEARCH!r}  limit={PAGE_SIZE} skip=0..{MAX_RECORDS - PAGE_SIZE} -> {len(recalls)} records")
    print(f"  device/event   search={EVENT_SEARCH!r}  limit={PAGE_SIZE} skip=0..{MAX_RECORDS - PAGE_SIZE} -> {len(events)} records")
    print(f"Manifest: {MANIFEST_PATH} | pull date written to {PULL_DATE_PATH}")


if __name__ == "__main__":
    main()
