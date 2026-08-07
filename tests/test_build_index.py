from unittest.mock import patch

from build_index import _chunk_pdf_dir


@patch("build_index.extract_pdf_text")
def test_chunk_pdf_dir_resolves_real_url_not_local_path(mock_extract, tmp_path):
    mock_extract.return_value = "WARNINGS\nDo not use this device on patients with known allergies.\n"
    pdf_dir = tmp_path / "guidance_pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "78369.pdf").write_bytes(b"%PDF-1.4 fake")

    chunks = _chunk_pdf_dir(
        pdf_dir,
        source_type="guidance_pdf",
        url_list=["https://www.fda.gov/media/78369/download"],
        retrieved_date="2026-08-07",
    )

    assert len(chunks) == 1
    assert chunks[0].metadata.source_url == "https://www.fda.gov/media/78369/download"
    assert chunks[0].metadata.source_url.startswith("https://")
    assert str(tmp_path) not in chunks[0].metadata.source_url


def test_chunk_pdf_dir_falls_back_to_local_path_with_warning_for_unknown_file(tmp_path, capsys):
    pdf_dir = tmp_path / "guidance_pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "unknown.pdf").write_bytes(b"%PDF-1.4 fake")

    with patch(
        "build_index.extract_pdf_text",
        return_value="WARNINGS\nSome warning text here that is long enough to survive filtering.\n",
    ):
        chunks = _chunk_pdf_dir(pdf_dir, source_type="guidance_pdf", url_list=[], retrieved_date="2026-08-07")

    # No known URL for this file -- degrades to the local path (still usable
    # for local dev/debugging) rather than crashing, but warns loudly since a
    # local path isn't a usable citation link once deployed.
    assert chunks[0].metadata.source_url == str(pdf_dir / "unknown.pdf")
    assert "WARNING" in capsys.readouterr().out
