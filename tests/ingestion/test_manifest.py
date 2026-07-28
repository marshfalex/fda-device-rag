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
