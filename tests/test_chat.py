"""Tests for glm_chat tool."""
import datetime
from unittest.mock import MagicMock, patch

import httpx
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError

import glm_mcp.tools._core as core_module


def _make_request() -> httpx.Request:
    return httpx.Request("POST", "https://open.bigmodel.cn/api/paas/v4/chat/completions")


def test_glm_chat_returns_text():
    """glm_chat returns text response from API."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Hello, I'm GLM!"

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("glm_mcp.tools._core.get_client", return_value=mock_client):
        from glm_mcp.tools.chat import glm_chat
        result = glm_chat("Hello")

    assert result == "Hello, I'm GLM!"


def test_glm_chat_with_system_prompt_includes_system_message():
    """glm_chat includes system message when system is provided."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Response"

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("glm_mcp.tools._core.get_client", return_value=mock_client):
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

    with patch("glm_mcp.tools._core.get_client", return_value=mock_client):
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

    with patch("glm_mcp.tools._core.get_client", return_value=mock_client):
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

    with patch("glm_mcp.tools._core.get_client", return_value=mock_client):
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

    with patch("glm_mcp.tools._core.get_client", return_value=mock_client), \
         patch("glm_mcp.tools._core.log_usage") as mock_log:
        from glm_mcp.tools.chat import glm_chat
        glm_chat("Hello", model="glm-4-flash")

    mock_log.assert_called_once_with(
        "glm_chat", "glm-4-flash", 150, 320,
        fallback_used=False, original_model=None, fallback_reason=None,
    )


def test_glm_chat_raises_runtime_error_on_timeout():
    """glm_chat raises RuntimeError when API times out (auto_fallback=False)."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = APITimeoutError(
        request=_make_request()
    )

    with patch("glm_mcp.tools._core.get_client", return_value=mock_client):
        from glm_mcp.tools.chat import glm_chat
        with pytest.raises(RuntimeError, match="timed out"):
            glm_chat("Hello", auto_fallback=False)


def test_glm_chat_raises_runtime_error_on_connection_error():
    """glm_chat raises RuntimeError when network is unavailable (auto_fallback=False)."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = APIConnectionError(
        request=_make_request()
    )

    with patch("glm_mcp.tools._core.get_client", return_value=mock_client):
        from glm_mcp.tools.chat import glm_chat
        with pytest.raises(RuntimeError, match="Could not reach"):
            glm_chat("Hello", auto_fallback=False)


def test_glm_chat_raises_runtime_error_on_api_status_error():
    """glm_chat raises RuntimeError when API returns error status (auto_fallback=False)."""
    mock_client = MagicMock()
    response = httpx.Response(
        429, request=_make_request(), content=b'{"error":"rate limited"}'
    )
    mock_client.chat.completions.create.side_effect = APIStatusError(
        "Rate limited", response=response, body=None
    )

    with patch("glm_mcp.tools._core.get_client", return_value=mock_client):
        from glm_mcp.tools.chat import glm_chat
        with pytest.raises(RuntimeError, match="429"):
            glm_chat("Hello", auto_fallback=False)


def test_glm_chat_raises_runtime_error_when_content_is_none():
    """glm_chat raises RuntimeError when API returns None content."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = None

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("glm_mcp.tools._core.get_client", return_value=mock_client):
        from glm_mcp.tools.chat import glm_chat
        with pytest.raises(RuntimeError, match="no text content"):
            glm_chat("Hello")


# --- v0.2.0: Multi-turn conversation (UT-CHT-11 ~ UT-CHT-18) ---


def test_glm_chat_multi_turn_passes_messages_to_api():
    """UT-CHT-11: glm_chat passes messages list directly to API in multi-turn mode."""
    history = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
        {"role": "user", "content": "Bye"},
    ]
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Goodbye!"

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("glm_mcp.tools._core.get_client", return_value=mock_client):
        from glm_mcp.tools.chat import glm_chat
        result = glm_chat(messages=history)

    assert result == "Goodbye!"
    call_args = mock_client.chat.completions.create.call_args
    assert call_args.kwargs["messages"] == history


def test_glm_chat_multi_turn_does_not_append_extra_user_message():
    """UT-CHT-12: glm_chat does not append extra message when messages= provided."""
    history = [{"role": "user", "content": "Hi"}]
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Hello!"

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("glm_mcp.tools._core.get_client", return_value=mock_client):
        from glm_mcp.tools.chat import glm_chat
        glm_chat(messages=history)

    call_args = mock_client.chat.completions.create.call_args
    assert call_args.kwargs["messages"] == history
    assert len(call_args.kwargs["messages"]) == 1


def test_glm_chat_raises_value_error_when_both_message_and_messages_provided():
    """UT-CHT-13: glm_chat raises ValueError when both message and messages are given."""
    with patch("glm_mcp.tools._core.get_client"):
        from glm_mcp.tools.chat import glm_chat
        with pytest.raises(ValueError, match="message"):
            glm_chat(message="Hello", messages=[{"role": "user", "content": "Hi"}])


def test_glm_chat_raises_value_error_for_empty_messages_list():
    """UT-CHT-14: glm_chat raises ValueError when messages is an empty list."""
    with patch("glm_mcp.tools._core.get_client"):
        from glm_mcp.tools.chat import glm_chat
        with pytest.raises(ValueError, match="empty"):
            glm_chat(messages=[])


def test_glm_chat_multi_turn_logs_token_usage():
    """UT-CHT-15: glm_chat calls log_usage in multi-turn mode."""
    history = [{"role": "user", "content": "Hi"}]
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Hello!"
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("glm_mcp.tools._core.get_client", return_value=mock_client), \
         patch("glm_mcp.tools._core.log_usage") as mock_log:
        from glm_mcp.tools.chat import glm_chat
        glm_chat(messages=history, model="glm-4")

    mock_log.assert_called_once_with(
        "glm_chat", "glm-4", 10, 5,
        fallback_used=False, original_model=None, fallback_reason=None,
    )


def test_glm_chat_multi_turn_ignores_system_param():
    """UT-CHT-16: glm_chat ignores system param when messages= is provided."""
    history = [{"role": "user", "content": "Hi"}]
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Hello!"

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("glm_mcp.tools._core.get_client", return_value=mock_client):
        from glm_mcp.tools.chat import glm_chat
        glm_chat(messages=history, system="Ignored system prompt")

    call_args = mock_client.chat.completions.create.call_args
    # messages list passed as-is; no extra system message prepended
    assert call_args.kwargs["messages"] == history


def test_glm_chat_single_turn_still_works_as_positional():
    """UT-CHT-17: glm_chat single-turn still works with positional arg after signature change."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Hi!"

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("glm_mcp.tools._core.get_client", return_value=mock_client):
        from glm_mcp.tools.chat import glm_chat
        result = glm_chat("Hello")  # positional arg

    assert result == "Hi!"
    call_args = mock_client.chat.completions.create.call_args
    assert call_args.kwargs["messages"] == [{"role": "user", "content": "Hello"}]


def test_glm_chat_raises_descriptive_error_on_context_window_exceeded():
    """UT-CHT-18: glm_chat raises descriptive RuntimeError when context window exceeded."""
    mock_client = MagicMock()
    response = httpx.Response(
        400,
        request=_make_request(),
        content=b'{"error":{"message":"This model maximum context length exceeded","type":"invalid_request_error"}}',
    )
    mock_client.chat.completions.create.side_effect = APIStatusError(
        "context length exceeded", response=response, body=None
    )

    with patch("glm_mcp.tools._core.get_client", return_value=mock_client):
        from glm_mcp.tools.chat import glm_chat
        with pytest.raises(RuntimeError, match="context window"):
            glm_chat("Hello")


# --- v0.4.0: Fallback resilience (UT-CHT-19 ~ UT-CHT-30) ---


def test_glm_chat_auto_fallback_on_429_uses_default_fallback_model():
    """UT-CHT-19: auto_fallback=True + 429 → auto-switch to default glm-4.7."""
    mock_client = MagicMock()
    response_429 = httpx.Response(429, request=_make_request(), content=b'{"error":"rate limited"}')
    fallback_response = MagicMock()
    fallback_response.choices[0].message.content = "Fallback response"
    fallback_response.usage.prompt_tokens = 10
    fallback_response.usage.completion_tokens = 5
    mock_client.chat.completions.create.side_effect = [
        APIStatusError("Rate limited", response=response_429, body=None),
        fallback_response,
    ]
    with patch("glm_mcp.tools._core.get_client", return_value=mock_client), \
         patch("glm_mcp.tools._core.log_usage") as mock_log:
        from glm_mcp.tools.chat import glm_chat
        result = glm_chat("Hello", model="GLM-5", auto_fallback=True)
    assert result == "Fallback response"
    # Second call must use glm-4.7
    second_call = mock_client.chat.completions.create.call_args_list[1]
    assert second_call.kwargs["model"] == "glm-4.7"
    mock_log.assert_called_once_with(
        "glm_chat", "glm-4.7", 10, 5,
        fallback_used=True, original_model="GLM-5", fallback_reason="429",
    )


def test_glm_chat_auto_fallback_on_429_uses_specified_fallback_model():
    """UT-CHT-20: auto_fallback=True + 429 + fallback_model='glm-4-flash' → switch to glm-4-flash."""
    mock_client = MagicMock()
    response_429 = httpx.Response(429, request=_make_request(), content=b'{"error":"rate limited"}')
    fallback_response = MagicMock()
    fallback_response.choices[0].message.content = "Flash fallback"
    fallback_response.usage.prompt_tokens = 5
    fallback_response.usage.completion_tokens = 3
    mock_client.chat.completions.create.side_effect = [
        APIStatusError("Rate limited", response=response_429, body=None),
        fallback_response,
    ]
    with patch("glm_mcp.tools._core.get_client", return_value=mock_client), \
         patch("glm_mcp.tools._core.log_usage") as mock_log:
        from glm_mcp.tools.chat import glm_chat
        result = glm_chat("Hello", model="GLM-5", fallback_model="glm-4-flash", auto_fallback=True)
    assert result == "Flash fallback"
    second_call = mock_client.chat.completions.create.call_args_list[1]
    assert second_call.kwargs["model"] == "glm-4-flash"
    mock_log.assert_called_once_with(
        "glm_chat", "glm-4-flash", 5, 3,
        fallback_used=True, original_model="GLM-5", fallback_reason="429",
    )


def test_glm_chat_no_fallback_on_429_when_auto_fallback_disabled():
    """UT-CHT-21: auto_fallback=False + 429 → raises RuntimeError immediately."""
    mock_client = MagicMock()
    response_429 = httpx.Response(429, request=_make_request(), content=b'{"error":"rate limited"}')
    mock_client.chat.completions.create.side_effect = APIStatusError(
        "Rate limited", response=response_429, body=None
    )
    with patch("glm_mcp.tools._core.get_client", return_value=mock_client):
        from glm_mcp.tools.chat import glm_chat
        with pytest.raises(RuntimeError, match="429"):
            glm_chat("Hello", auto_fallback=False)


def test_glm_chat_auto_fallback_on_503():
    """UT-CHT-22: auto_fallback=True + 503 → auto-switch to glm-4.7."""
    mock_client = MagicMock()
    response_503 = httpx.Response(503, request=_make_request(), content=b'{"error":"service unavailable"}')
    fallback_response = MagicMock()
    fallback_response.choices[0].message.content = "503 fallback"
    fallback_response.usage.prompt_tokens = 8
    fallback_response.usage.completion_tokens = 4
    mock_client.chat.completions.create.side_effect = [
        APIStatusError("Service unavailable", response=response_503, body=None),
        fallback_response,
    ]
    with patch("glm_mcp.tools._core.get_client", return_value=mock_client), \
         patch("glm_mcp.tools._core.log_usage") as mock_log:
        from glm_mcp.tools.chat import glm_chat
        result = glm_chat("Hello", model="GLM-5", auto_fallback=True)
    assert result == "503 fallback"
    mock_log.assert_called_once_with(
        "glm_chat", "glm-4.7", 8, 4,
        fallback_used=True, original_model="GLM-5", fallback_reason="503",
    )


def test_glm_chat_no_fallback_on_non_retriable_status():
    """UT-CHT-23: 400/401/422 → raises RuntimeError regardless of auto_fallback."""
    for status_code in [401, 422]:
        mock_client = MagicMock()
        response = httpx.Response(status_code, request=_make_request(), content=b'{"error":"bad request"}')
        mock_client.chat.completions.create.side_effect = APIStatusError(
            f"Error {status_code}", response=response, body=None
        )
        with patch("glm_mcp.tools._core.get_client", return_value=mock_client):
            from glm_mcp.tools.chat import glm_chat
            with pytest.raises(RuntimeError):
                glm_chat("Hello", auto_fallback=True)
        # Verify only one API call was made (no fallback)
        assert mock_client.chat.completions.create.call_count == 1


def test_glm_chat_avoid_peak_hours_during_peak_skips_primary():
    """UT-CHT-24: auto_fallback=True + avoid_peak_hours=True + peak → skips primary model."""
    fallback_response = MagicMock()
    fallback_response.choices[0].message.content = "Peak skip response"
    fallback_response.usage.prompt_tokens = 5
    fallback_response.usage.completion_tokens = 3
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = fallback_response
    with patch("glm_mcp.tools._core.get_client", return_value=mock_client), \
         patch("glm_mcp.tools._core._is_peak_hours", return_value=True), \
         patch("glm_mcp.tools._core.log_usage") as mock_log:
        from glm_mcp.tools.chat import glm_chat
        result = glm_chat("Hello", model="GLM-5", auto_fallback=True, avoid_peak_hours=True)
    assert result == "Peak skip response"
    # Only one API call (fallback), primary was skipped
    assert mock_client.chat.completions.create.call_count == 1
    call_args = mock_client.chat.completions.create.call_args
    assert call_args.kwargs["model"] == "glm-4.7"
    mock_log.assert_called_once_with(
        "glm_chat", "glm-4.7", 5, 3,
        fallback_used=True, original_model="GLM-5", fallback_reason="peak_hours",
    )


def test_glm_chat_avoid_peak_hours_outside_peak_uses_primary():
    """UT-CHT-25: auto_fallback=True + avoid_peak_hours=True + non-peak → uses primary model."""
    primary_response = MagicMock()
    primary_response.choices[0].message.content = "Primary response"
    primary_response.usage.prompt_tokens = 10
    primary_response.usage.completion_tokens = 5
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = primary_response
    with patch("glm_mcp.tools._core.get_client", return_value=mock_client), \
         patch("glm_mcp.tools._core._is_peak_hours", return_value=False), \
         patch("glm_mcp.tools._core.log_usage") as mock_log:
        from glm_mcp.tools.chat import glm_chat
        result = glm_chat("Hello", model="GLM-5", auto_fallback=True, avoid_peak_hours=True)
    assert result == "Primary response"
    call_args = mock_client.chat.completions.create.call_args
    assert call_args.kwargs["model"] == "GLM-5"
    mock_log.assert_called_once_with(
        "glm_chat", "GLM-5", 10, 5,
        fallback_used=False, original_model=None, fallback_reason=None,
    )


def test_glm_chat_auto_fallback_false_overrides_avoid_peak_hours():
    """UT-CHT-26: auto_fallback=False + avoid_peak_hours=True + peak → uses primary model (switch wins)."""
    primary_response = MagicMock()
    primary_response.choices[0].message.content = "Primary despite peak"
    primary_response.usage.prompt_tokens = 10
    primary_response.usage.completion_tokens = 5
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = primary_response
    with patch("glm_mcp.tools._core.get_client", return_value=mock_client), \
         patch("glm_mcp.tools._core._is_peak_hours", return_value=True), \
         patch("glm_mcp.tools._core.log_usage"):
        from glm_mcp.tools.chat import glm_chat
        result = glm_chat("Hello", model="GLM-5", auto_fallback=False, avoid_peak_hours=True)
    assert result == "Primary despite peak"
    call_args = mock_client.chat.completions.create.call_args
    assert call_args.kwargs["model"] == "GLM-5"


def test_glm_chat_raises_runtime_error_when_fallback_also_fails():
    """UT-CHT-27: fallback itself fails → raises RuntimeError (no second fallback)."""
    mock_client = MagicMock()
    response_429 = httpx.Response(429, request=_make_request(), content=b'{"error":"rate limited"}')
    response_500 = httpx.Response(500, request=_make_request(), content=b'{"error":"server error"}')
    mock_client.chat.completions.create.side_effect = [
        APIStatusError("Rate limited", response=response_429, body=None),
        APIStatusError("Server error", response=response_500, body=None),
    ]
    with patch("glm_mcp.tools._core.get_client", return_value=mock_client):
        from glm_mcp.tools.chat import glm_chat
        with pytest.raises(RuntimeError):
            glm_chat("Hello", auto_fallback=True)
    # Exactly 2 calls: primary (429) + fallback (500)
    assert mock_client.chat.completions.create.call_count == 2


def test_glm_chat_fallback_reason_distinguishes_trigger():
    """UT-CHT-28: fallback_reason field distinguishes trigger: '429'/'503'/'peak_hours'."""
    # Test 503 specifically to confirm fallback_reason="503"
    mock_client = MagicMock()
    response_503 = httpx.Response(503, request=_make_request(), content=b'{"error":"unavailable"}')
    fallback_response = MagicMock()
    fallback_response.choices[0].message.content = "503 result"
    fallback_response.usage.prompt_tokens = 3
    fallback_response.usage.completion_tokens = 2
    mock_client.chat.completions.create.side_effect = [
        APIStatusError("Unavailable", response=response_503, body=None),
        fallback_response,
    ]
    with patch("glm_mcp.tools._core.get_client", return_value=mock_client), \
         patch("glm_mcp.tools._core.log_usage") as mock_log:
        from glm_mcp.tools.chat import glm_chat
        glm_chat("Hello", model="GLM-5", auto_fallback=True)
    call_kwargs = mock_log.call_args.kwargs
    assert call_kwargs["fallback_reason"] == "503"


def test_glm_chat_success_logs_fallback_used_false():
    """UT-CHT-29: successful call (no fallback) → logs fallback_used=False, original_model=None."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Success"
    mock_response.usage.prompt_tokens = 20
    mock_response.usage.completion_tokens = 10
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    with patch("glm_mcp.tools._core.get_client", return_value=mock_client), \
         patch("glm_mcp.tools._core.log_usage") as mock_log:
        from glm_mcp.tools.chat import glm_chat
        glm_chat("Hello", model="GLM-5", auto_fallback=True)
    mock_log.assert_called_once_with(
        "glm_chat", "GLM-5", 20, 10,
        fallback_used=False, original_model=None, fallback_reason=None,
    )


def test_is_peak_hours_returns_bool():
    """UT-CHT-30: _is_peak_hours() returns bool based on UTC+8 14:00-18:00."""
    # Test that _is_peak_hours exists and returns bool.
    # Implementation uses `from datetime import datetime` then `datetime.now(_TZ_UTC8)`.
    # We patch `glm_mcp.tools.chat.datetime` (the datetime class) to control .now().

    # Peak: UTC+8 15:00 → hour=15 ∈ [14,18) → True
    peak_time = datetime.datetime(2026, 2, 22, 15, 0, 0, tzinfo=datetime.timezone(datetime.timedelta(hours=8)))
    with patch("glm_mcp.tools._core.datetime") as mock_dt:
        mock_dt.now.return_value = peak_time
        result = core_module._is_peak_hours()
        mock_dt.now.assert_called_once_with(core_module._TZ_UTC8)
        assert isinstance(result, bool)
        assert result is True

    # Non-peak: UTC+8 10:00 → hour=10 ∉ [14,18) → False
    non_peak_time = datetime.datetime(2026, 2, 22, 10, 0, 0, tzinfo=datetime.timezone(datetime.timedelta(hours=8)))
    with patch("glm_mcp.tools._core.datetime") as mock_dt:
        mock_dt.now.return_value = non_peak_time
        result = core_module._is_peak_hours()
        mock_dt.now.assert_called_once_with(core_module._TZ_UTC8)
        assert isinstance(result, bool)
        assert result is False


# --- Coverage: _do_fallback edge cases ---


def test_glm_chat_raises_runtime_error_when_fallback_returns_none_content():
    """UT-CHT-31: fallback response with content=None → raises RuntimeError."""
    mock_client = MagicMock()
    response_429 = httpx.Response(429, request=_make_request(), content=b'{"error":"rate limited"}')
    fallback_response = MagicMock()
    fallback_response.choices[0].message.content = None
    mock_client.chat.completions.create.side_effect = [
        APIStatusError("Rate limited", response=response_429, body=None),
        fallback_response,
    ]
    with patch("glm_mcp.tools._core.get_client", return_value=mock_client):
        from glm_mcp.tools.chat import glm_chat
        with pytest.raises(RuntimeError, match="no text content"):
            glm_chat("Hello", auto_fallback=True)


def test_glm_chat_raises_runtime_error_when_fallback_raises_runtime_error():
    """UT-CHT-32: _do_fallback wraps non-APIStatusError as RuntimeError."""
    mock_client = MagicMock()
    response_429 = httpx.Response(429, request=_make_request(), content=b'{"error":"rate limited"}')
    # Second call raises APITimeoutError, which _do_fallback catches and wraps as RuntimeError
    mock_client.chat.completions.create.side_effect = [
        APIStatusError("Rate limited", response=response_429, body=None),
        APITimeoutError(request=_make_request()),
    ]
    with patch("glm_mcp.tools._core.get_client", return_value=mock_client):
        from glm_mcp.tools.chat import glm_chat
        with pytest.raises(RuntimeError, match="also failed"):
            glm_chat("Hello", auto_fallback=True)
    assert mock_client.chat.completions.create.call_count == 2


# --- Decision B: APITimeoutError / APIConnectionError → fallback ---


def test_glm_chat_auto_fallback_on_timeout():
    """UT-CHT-33: auto_fallback=True + APITimeoutError → auto-switch to fallback model."""
    mock_client = MagicMock()
    fallback_response = MagicMock()
    fallback_response.choices[0].message.content = "Timeout fallback"
    fallback_response.usage.prompt_tokens = 10
    fallback_response.usage.completion_tokens = 5
    mock_client.chat.completions.create.side_effect = [
        APITimeoutError(request=_make_request()),
        fallback_response,
    ]
    with patch("glm_mcp.tools._core.get_client", return_value=mock_client), \
         patch("glm_mcp.tools._core.log_usage") as mock_log:
        from glm_mcp.tools.chat import glm_chat
        result = glm_chat("Hello", model="GLM-5", auto_fallback=True)
    assert result == "Timeout fallback"
    assert mock_client.chat.completions.create.call_count == 2
    second_call = mock_client.chat.completions.create.call_args_list[1]
    assert second_call.kwargs["model"] == "glm-4.7"
    mock_log.assert_called_once_with(
        "glm_chat", "glm-4.7", 10, 5,
        fallback_used=True, original_model="GLM-5", fallback_reason="timeout",
    )


def test_glm_chat_auto_fallback_on_connection_error():
    """UT-CHT-34: auto_fallback=True + APIConnectionError → auto-switch to fallback model."""
    mock_client = MagicMock()
    fallback_response = MagicMock()
    fallback_response.choices[0].message.content = "Connection fallback"
    fallback_response.usage.prompt_tokens = 8
    fallback_response.usage.completion_tokens = 4
    mock_client.chat.completions.create.side_effect = [
        APIConnectionError(request=_make_request()),
        fallback_response,
    ]
    with patch("glm_mcp.tools._core.get_client", return_value=mock_client), \
         patch("glm_mcp.tools._core.log_usage") as mock_log:
        from glm_mcp.tools.chat import glm_chat
        result = glm_chat("Hello", model="GLM-5", auto_fallback=True)
    assert result == "Connection fallback"
    assert mock_client.chat.completions.create.call_count == 2
    second_call = mock_client.chat.completions.create.call_args_list[1]
    assert second_call.kwargs["model"] == "glm-4.7"
    mock_log.assert_called_once_with(
        "glm_chat", "glm-4.7", 8, 4,
        fallback_used=True, original_model="GLM-5", fallback_reason="connection",
    )


def test_glm_chat_default_top_p_is_0_95():
    """UT-CHT-35: glm_chat passes top_p=0.95 by default (GLM-5 demo recommendation)."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Hello"
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("glm_mcp.tools._core.get_client", return_value=mock_client):
        from glm_mcp.tools.chat import glm_chat
        glm_chat("Hello")

    call_args = mock_client.chat.completions.create.call_args
    assert call_args.kwargs.get("top_p") == 0.95


def test_glm_chat_custom_top_p_passed_to_api():
    """UT-CHT-36: glm_chat forwards caller-supplied top_p to the API."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Hello"
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("glm_mcp.tools._core.get_client", return_value=mock_client):
        from glm_mcp.tools.chat import glm_chat
        glm_chat("Hello", top_p=0.7)

    call_args = mock_client.chat.completions.create.call_args
    assert call_args.kwargs.get("top_p") == 0.7


def test_glm_chat_top_p_none_not_sent_to_api():
    """UT-CHT-37: When top_p=None, the 'top_p' key must not appear in the API call kwargs."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Hello"
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("glm_mcp.tools._core.get_client", return_value=mock_client):
        from glm_mcp.tools.chat import glm_chat
        glm_chat("Hello", top_p=None)

    call_args = mock_client.chat.completions.create.call_args
    assert "top_p" not in call_args.kwargs
