"""Tests for glm_embed tool."""
from unittest.mock import MagicMock, patch

import httpx
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError


def _make_request() -> httpx.Request:
    return httpx.Request("POST", "https://open.bigmodel.cn/api/paas/v4/embeddings")


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


def test_glm_embed_raises_runtime_error_on_timeout():
    """glm_embed raises RuntimeError when API times out."""
    mock_client = MagicMock()
    mock_client.embeddings.create.side_effect = APITimeoutError(request=_make_request())

    with patch("glm_mcp.tools.embed.get_client", return_value=mock_client):
        from glm_mcp.tools.embed import glm_embed
        with pytest.raises(RuntimeError, match="timed out"):
            glm_embed("Hello")


def test_glm_embed_raises_runtime_error_on_connection_error():
    """glm_embed raises RuntimeError when network is unavailable."""
    mock_client = MagicMock()
    mock_client.embeddings.create.side_effect = APIConnectionError(
        request=_make_request()
    )

    with patch("glm_mcp.tools.embed.get_client", return_value=mock_client):
        from glm_mcp.tools.embed import glm_embed
        with pytest.raises(RuntimeError, match="Could not reach"):
            glm_embed("Hello")


def test_glm_embed_raises_runtime_error_on_api_status_error():
    """glm_embed raises RuntimeError when API returns error status."""
    mock_client = MagicMock()
    response = httpx.Response(
        401, request=_make_request(), content=b'{"error":"unauthorized"}'
    )
    mock_client.embeddings.create.side_effect = APIStatusError(
        "Unauthorized", response=response, body=None
    )

    with patch("glm_mcp.tools.embed.get_client", return_value=mock_client):
        from glm_mcp.tools.embed import glm_embed
        with pytest.raises(RuntimeError, match="401"):
            glm_embed("Hello")
