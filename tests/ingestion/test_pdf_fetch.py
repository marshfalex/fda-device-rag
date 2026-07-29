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


@patch("fda_device_rag.ingestion.pdf_fetch.requests.get")
def test_download_pdfs_disambiguates_generic_download_endpoints(mock_get, tmp_path):
    # Real-world case: FDA's media host serves every document at a URL ending
    # literally in "/download" (e.g. fda.gov/media/78369/download), so naively
    # taking the last path segment collides across every such URL. Fall back
    # to the preceding segment (the media id) when the last segment is the
    # generic word "download".
    mock_response = MagicMock()
    mock_response.content = b"%PDF-1.4 fake content"
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    dest_dir = tmp_path / "pdfs"
    paths = download_pdfs(
        [
            "https://www.fda.gov/media/78369/download",
            "https://www.fda.gov/media/73141/download",
        ],
        dest_dir,
    )

    names = [p.name for p in paths]
    assert names == ["78369.pdf", "73141.pdf"]
    assert len(set(names)) == 2
