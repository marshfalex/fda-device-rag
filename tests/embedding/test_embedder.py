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
