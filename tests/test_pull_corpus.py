from unittest.mock import MagicMock, patch

import pull_corpus
from pull_corpus import _fetch_paginated


def _mock_response(results):
    response = MagicMock()
    response.json.return_value = {"results": results}
    response.raise_for_status.return_value = None
    return response


@patch("fda_device_rag.ingestion.openfda_client.requests.get")
def test_fetch_paginated_dedups_records_repeated_across_pages(mock_get, monkeypatch):
    """openFDA's result ordering isn't pinned by an explicit `sort` param, so
    the same record could land on two pages if ordering shifts between
    requests. _fetch_paginated must not return duplicate ids in that case,
    since a duplicate id would raise ChromaStore's DuplicateIDError when the
    corpus is later indexed."""
    from fda_device_rag.ingestion.openfda_client import fetch_recalls

    # Shrink PAGE_SIZE to 2 so a two-record mocked page reads as a "full"
    # page (triggering the next skip) rather than a short/final page.
    monkeypatch.setattr(pull_corpus, "PAGE_SIZE", 2)

    page_one = [
        {"product_res_number": "Z-0001-04"},
        {"product_res_number": "Z-0002-04"},
    ]
    # Second page's ordering shifted and re-returned Z-0002-04 from page one,
    # followed by one genuinely new record.
    page_two = [
        {"product_res_number": "Z-0002-04"},
        {"product_res_number": "Z-0003-04"},
    ]
    mock_get.side_effect = [_mock_response(page_one), _mock_response(page_two)]

    records = _fetch_paginated(
        fetch_recalls, "some search", id_field="product_res_number", max_records=4
    )

    ids = [r["product_res_number"] for r in records]
    assert ids == ["Z-0001-04", "Z-0002-04", "Z-0003-04"]
    assert len(ids) == len(set(ids))
