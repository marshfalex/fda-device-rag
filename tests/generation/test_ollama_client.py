from unittest.mock import patch, MagicMock

import pytest
import requests

from fda_device_rag.generation.ollama_client import OllamaNotReadyError, check_ollama_ready


@patch("fda_device_rag.generation.ollama_client.requests.get")
def test_check_ollama_ready_raises_when_server_unreachable(mock_get):
    mock_get.side_effect = requests.ConnectionError()

    with pytest.raises(OllamaNotReadyError, match="not running"):
        check_ollama_ready("http://localhost:11434", "llama3")


@patch("fda_device_rag.generation.ollama_client.requests.get")
def test_check_ollama_ready_raises_when_model_not_pulled(mock_get):
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {"models": [{"name": "gemma:latest"}]}
    mock_get.return_value = mock_response

    with pytest.raises(OllamaNotReadyError, match="not found"):
        check_ollama_ready("http://localhost:11434", "llama3")


@patch("fda_device_rag.generation.ollama_client.requests.get")
def test_check_ollama_ready_passes_when_model_present(mock_get):
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {"models": [{"name": "llama3:latest"}]}
    mock_get.return_value = mock_response

    check_ollama_ready("http://localhost:11434", "llama3")  # must not raise
