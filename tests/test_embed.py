"""Tests for glm_embed tool."""
from unittest.mock import MagicMock, patch


def test_glm_embed_returns_float_list():
    """glm_embed returns list of floats."""
    expected_vector = [0.1, 0.2, 0.3, 0.4]

    mock_response = MagicMock()
    mock_response.data[0].embedding = expected_vector

    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = mock_response

    with patch("glm_mcp.tools.embed.get_client", return_value=mock_client):
        from glm_mcp.tools.embed import glm_embed
        result = glm_embed("Hello")

    assert result == expected_vector


def test_glm_embed_wraps_text_in_list():
    """glm_embed passes text as a single-element list to API."""
    mock_response = MagicMock()
    mock_response.data[0].embedding = [0.1]

    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = mock_response

    with patch("glm_mcp.tools.embed.get_client", return_value=mock_client):
        from glm_mcp.tools.embed import glm_embed
        glm_embed("test text")

    call_args = mock_client.embeddings.create.call_args
    assert call_args.kwargs["input"] == ["test text"]


def test_glm_embed_uses_specified_model():
    """glm_embed passes model parameter to API."""
    mock_response = MagicMock()
    mock_response.data[0].embedding = [0.1]

    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = mock_response

    with patch("glm_mcp.tools.embed.get_client", return_value=mock_client):
        from glm_mcp.tools.embed import glm_embed
        glm_embed("Hello", model="embedding-2")

    call_args = mock_client.embeddings.create.call_args
    assert call_args.kwargs["model"] == "embedding-2"


def test_glm_embed_uses_default_model():
    """glm_embed defaults to embedding-3 model."""
    mock_response = MagicMock()
    mock_response.data[0].embedding = [0.1]

    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = mock_response

    with patch("glm_mcp.tools.embed.get_client", return_value=mock_client):
        from glm_mcp.tools.embed import glm_embed
        glm_embed("Hello")

    call_args = mock_client.embeddings.create.call_args
    assert call_args.kwargs["model"] == "embedding-3"


def test_glm_embed_logs_token_usage():
    """glm_embed calls log_usage with prompt_tokens and output_tokens=0."""
    mock_response = MagicMock()
    mock_response.data[0].embedding = [0.1]
    mock_response.usage.prompt_tokens = 42

    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = mock_response

    with patch("glm_mcp.tools.embed.get_client", return_value=mock_client), \
         patch("glm_mcp.tools.embed.log_usage") as mock_log:
        from glm_mcp.tools.embed import glm_embed
        glm_embed("Hello", model="embedding-3")

    mock_log.assert_called_once_with("glm_embed", "embedding-3", 42, 0)
