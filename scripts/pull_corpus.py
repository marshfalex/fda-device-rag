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
