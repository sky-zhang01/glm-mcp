"""Tests for glm_chat tool."""
import httpx
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError
from unittest.mock import MagicMock, patch


def _make_request() -> httpx.Request:
    return httpx.Request("POST", "https://open.bigmodel.cn/api/paas/v4/chat/completions")


def test_glm_chat_returns_text():
    """glm_chat returns text response from API."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Hello, I'm GLM!"

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("glm_mcp.tools.chat.get_client", return_value=mock_client):
        from glm_mcp.tools.chat import glm_chat
        result = glm_chat("Hello")

    assert result == "Hello, I'm GLM!"


def test_glm_chat_with_system_prompt_includes_system_message():
    """glm_chat includes system message when system is provided."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Response"

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("glm_mcp.tools.chat.get_client", return_value=mock_client):
        from glm_mcp.tools.chat import glm_chat
        glm_chat("Hello", system="Be helpful")

    call_args = mock_client.chat.completions.create.call_args
    messages = call_args.kwargs["messages"]
    assert messages[0] == {"role": "system", "content": "Be helpful"}
    assert messages[1] == {"role": "user", "content": "Hello"}


def test_glm_chat_without_system_excludes_system_message():
    """glm_chat sends only user message when system is empty."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Response"

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("glm_mcp.tools.chat.get_client", return_value=mock_client):
        from glm_mcp.tools.chat import glm_chat
        glm_chat("Hello")

    call_args = mock_client.chat.completions.create.call_args
    messages = call_args.kwargs["messages"]
    assert len(messages) == 1
    assert messages[0] == {"role": "user", "content": "Hello"}


def test_glm_chat_uses_specified_model():
    """glm_chat passes model parameter to API."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Response"

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("glm_mcp.tools.chat.get_client", return_value=mock_client):
        from glm_mcp.tools.chat import glm_chat
        glm_chat("Hello", model="glm-4")

    call_args = mock_client.chat.completions.create.call_args
    assert call_args.kwargs["model"] == "glm-4"


def test_glm_chat_uses_default_model():
    """glm_chat defaults to glm-4-flash model."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Response"

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("glm_mcp.tools.chat.get_client", return_value=mock_client):
        from glm_mcp.tools.chat import glm_chat
        glm_chat("Hello")

    call_args = mock_client.chat.completions.create.call_args
    assert call_args.kwargs["model"] == "glm-4-flash"


def test_glm_chat_logs_token_usage():
    """glm_chat calls log_usage with prompt and completion token counts."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Hello!"
    mock_response.usage.prompt_tokens = 150
    mock_response.usage.completion_tokens = 320

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("glm_mcp.tools.chat.get_client", return_value=mock_client), \
         patch("glm_mcp.tools.chat.log_usage") as mock_log:
        from glm_mcp.tools.chat import glm_chat
        glm_chat("Hello", model="glm-4-flash")

    mock_log.assert_called_once_with("glm_chat", "glm-4-flash", 150, 320)


def test_glm_chat_raises_runtime_error_on_timeout():
    """glm_chat raises RuntimeError when API times out."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = APITimeoutError(request=_make_request())

    with patch("glm_mcp.tools.chat.get_client", return_value=mock_client):
        from glm_mcp.tools.chat import glm_chat
        with pytest.raises(RuntimeError, match="timed out"):
            glm_chat("Hello")


def test_glm_chat_raises_runtime_error_on_connection_error():
    """glm_chat raises RuntimeError when network is unavailable."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = APIConnectionError(request=_make_request())

    with patch("glm_mcp.tools.chat.get_client", return_value=mock_client):
        from glm_mcp.tools.chat import glm_chat
        with pytest.raises(RuntimeError, match="Could not reach"):
            glm_chat("Hello")


def test_glm_chat_raises_runtime_error_on_api_status_error():
    """glm_chat raises RuntimeError when API returns error status."""
    mock_client = MagicMock()
    response = httpx.Response(429, request=_make_request(), content=b'{"error":"rate limited"}')
    mock_client.chat.completions.create.side_effect = APIStatusError(
        "Rate limited", response=response, body=None
    )

    with patch("glm_mcp.tools.chat.get_client", return_value=mock_client):
        from glm_mcp.tools.chat import glm_chat
        with pytest.raises(RuntimeError, match="429"):
            glm_chat("Hello")


def test_glm_chat_raises_runtime_error_when_content_is_none():
    """glm_chat raises RuntimeError when API returns None content."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = None

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("glm_mcp.tools.chat.get_client", return_value=mock_client):
        from glm_mcp.tools.chat import glm_chat
        with pytest.raises(RuntimeError, match="no text content"):
            glm_chat("Hello")
