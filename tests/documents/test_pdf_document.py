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
