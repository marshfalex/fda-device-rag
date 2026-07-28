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
