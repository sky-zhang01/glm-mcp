"""Tests for glm_vision tool — UT-VIS-01 ~ UT-VIS-26.

Design spec target values (all tests assert these):
  _DEFAULT_VISION_MODEL          = "glm-4.6v"
  _DEFAULT_VISION_FALLBACK_MODEL = "glm-4.6v-flash"
  exposed temperature default    = 0.2
  exposed top_p default          = 0.9

Every test imports glm_mcp.tools.vision and checks at least one design-spec
constant so that EVERY test fails until the implementation is updated.
"""
import asyncio
from unittest.mock import MagicMock, patch

import httpx
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError

# Design-spec constants (authoritative values for this test suite)
_SPEC_DEFAULT_MODEL = "glm-4.6v"
_SPEC_DEFAULT_FALLBACK = "glm-4.6v-flash"
_SPEC_TEMPERATURE = 0.2
_SPEC_TOP_P = 0.9


def _make_request() -> httpx.Request:
    return httpx.Request("POST", "https://open.bigmodel.cn/api/paas/v4/chat/completions")


def _assert_spec_constants():
    """Assert that the module-level constants match the design spec.

    This helper is called at the top of every test so every test is anchored
    to the design-spec values. Any test will FAIL until vision.py defines
    _DEFAULT_VISION_MODEL = "glm-4.6v" and
    _DEFAULT_VISION_FALLBACK_MODEL = "glm-4.6v-flash".
    """
    import glm_mcp.tools.vision as vision_module
    assert vision_module._DEFAULT_VISION_MODEL == _SPEC_DEFAULT_MODEL, (
        f"Expected _DEFAULT_VISION_MODEL='{_SPEC_DEFAULT_MODEL}', "
        f"got '{vision_module._DEFAULT_VISION_MODEL}'"
    )
    assert vision_module._DEFAULT_VISION_FALLBACK_MODEL == _SPEC_DEFAULT_FALLBACK, (
        f"Expected _DEFAULT_VISION_FALLBACK_MODEL='{_SPEC_DEFAULT_FALLBACK}', "
        f"got '{vision_module._DEFAULT_VISION_FALLBACK_MODEL}'"
    )


# ---------------------------------------------------------------------------
# AC-1  HTTPS URL → non-empty str
# ---------------------------------------------------------------------------

def test_vis_01_https_url_returns_str():
    """UT-VIS-01: glm_vision(image_url='https://...', prompt='Describe') returns non-empty str."""
    _assert_spec_constants()

    mock_response = MagicMock()
    mock_response.choices[0].message.content = "A scenic mountain view."

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("glm_mcp.tools._core.get_client", return_value=mock_client):
        from glm_mcp.tools.vision import glm_vision
        result = glm_vision("https://example.com/mountain.jpg", "Describe this image")

    assert isinstance(result, str)
    assert result != ""
    assert result == "A scenic mountain view."


# ---------------------------------------------------------------------------
# AC-2  data:image/png;base64,… URI → non-empty str
# ---------------------------------------------------------------------------

def test_vis_02_data_uri_returns_str():
    """UT-VIS-02: glm_vision(image_url='data:image/png;base64,...', prompt='...') returns non-empty str."""
    _assert_spec_constants()

    mock_response = MagicMock()
    mock_response.choices[0].message.content = "A simple diagram."

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    data_uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
    with patch("glm_mcp.tools._core.get_client", return_value=mock_client):
        from glm_mcp.tools.vision import glm_vision
        result = glm_vision(data_uri, "What does this show?")

    assert isinstance(result, str)
    assert result != ""


# ---------------------------------------------------------------------------
# AC-3  Bare base64 (no data: prefix) → auto-prefixed as data:image/png;base64,
# ---------------------------------------------------------------------------

def test_vis_03_bare_base64_auto_prefixed():
    """UT-VIS-03: Bare base64 string without data: prefix is auto-prefixed to data:image/png;base64,..."""
    _assert_spec_constants()

    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Result"

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    bare_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
    with patch("glm_mcp.tools._core.get_client", return_value=mock_client):
        from glm_mcp.tools.vision import glm_vision
        glm_vision(bare_b64, "Describe")

    call_args = mock_client.chat.completions.create.call_args
    messages = call_args.kwargs["messages"]
    actual_url = messages[0]["content"][0]["image_url"]["url"]
    assert actual_url == f"data:image/png;base64,{bare_b64}"


# ---------------------------------------------------------------------------
# AC-4  Default model is "glm-4.6v"
# ---------------------------------------------------------------------------

def test_vis_04_default_model_is_glm_4v_plus():
    """UT-VIS-04: Default model is 'glm-4.6v'."""
    _assert_spec_constants()

    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Result"

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("glm_mcp.tools._core.get_client", return_value=mock_client):
        from glm_mcp.tools.vision import glm_vision
        glm_vision("https://example.com/img.png", "Describe")

    call_args = mock_client.chat.completions.create.call_args
    assert call_args.kwargs["model"] == _SPEC_DEFAULT_MODEL


# ---------------------------------------------------------------------------
# AC-5  Custom model param is passed through
# ---------------------------------------------------------------------------

def test_vis_05_custom_model_passed_to_api():
    """UT-VIS-05: Custom model parameter is forwarded to the API call."""
    _assert_spec_constants()

    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Result"

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("glm_mcp.tools._core.get_client", return_value=mock_client):
        from glm_mcp.tools.vision import glm_vision
        glm_vision("https://example.com/img.png", "Describe", model="glm-4v")

    call_args = mock_client.chat.completions.create.call_args
    assert call_args.kwargs["model"] == "glm-4v"


# ---------------------------------------------------------------------------
# AC-6  detail param appears in image_url object
# ---------------------------------------------------------------------------

def test_vis_06_detail_param_in_image_url_object():
    """UT-VIS-06: detail parameter is included inside the image_url object in the message."""
    _assert_spec_constants()

    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Result"

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("glm_mcp.tools._core.get_client", return_value=mock_client):
        from glm_mcp.tools.vision import glm_vision
        glm_vision("https://example.com/img.png", "Describe", detail="high")

    call_args = mock_client.chat.completions.create.call_args
    messages = call_args.kwargs["messages"]
    image_part = messages[0]["content"][0]
    assert image_part["image_url"]["detail"] == "high"


# ---------------------------------------------------------------------------
# AC-7  Usage logged with tool="glm_vision" and model="glm-4v-plus"
# ---------------------------------------------------------------------------

def test_vis_07_usage_logged_with_tool_name_glm_vision():
    """UT-VIS-07: Usage is logged with tool='glm_vision' and model='glm-4.6v' (default)."""
    _assert_spec_constants()

    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Result"
    mock_response.usage.prompt_tokens = 120
    mock_response.usage.completion_tokens = 60

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("glm_mcp.tools._core.get_client", return_value=mock_client), \
         patch("glm_mcp.tools._core.log_usage") as mock_log:
        from glm_mcp.tools.vision import glm_vision
        glm_vision("https://example.com/img.png", "Describe")

    mock_log.assert_called_once_with(
        "glm_vision", _SPEC_DEFAULT_MODEL, 120, 60,
        fallback_used=False, original_model=None, fallback_reason=None,
    )


# ---------------------------------------------------------------------------
# AC-8  Timeout + auto_fallback=False → RuntimeError
# ---------------------------------------------------------------------------

def test_vis_08_timeout_no_fallback_raises_runtime_error():
    """UT-VIS-08: Timeout with auto_fallback=False raises RuntimeError."""
    _assert_spec_constants()

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = APITimeoutError(
        request=_make_request()
    )

    with patch("glm_mcp.tools._core.get_client", return_value=mock_client):
        from glm_mcp.tools.vision import glm_vision
        with pytest.raises(RuntimeError, match="timed out"):
            glm_vision("https://example.com/img.png", "Describe", auto_fallback=False)


# ---------------------------------------------------------------------------
# AC-9  Connection error + auto_fallback=False → RuntimeError
# ---------------------------------------------------------------------------

def test_vis_09_connection_error_no_fallback_raises_runtime_error():
    """UT-VIS-09: Connection error with auto_fallback=False raises RuntimeError."""
    _assert_spec_constants()

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = APIConnectionError(
        request=_make_request()
    )

    with patch("glm_mcp.tools._core.get_client", return_value=mock_client):
        from glm_mcp.tools.vision import glm_vision
        with pytest.raises(RuntimeError, match="Could not reach"):
            glm_vision("https://example.com/img.png", "Describe", auto_fallback=False)


# ---------------------------------------------------------------------------
# AC-10  API status error + auto_fallback=False → RuntimeError
# ---------------------------------------------------------------------------

def test_vis_10_api_status_error_no_fallback_raises_runtime_error():
    """UT-VIS-10: APIStatusError with auto_fallback=False raises RuntimeError."""
    _assert_spec_constants()

    mock_client = MagicMock()
    response = httpx.Response(
        429, request=_make_request(), content=b'{"error":"rate limited"}'
    )
    mock_client.chat.completions.create.side_effect = APIStatusError(
        "Rate limited", response=response, body=None
    )

    with patch("glm_mcp.tools._core.get_client", return_value=mock_client):
        from glm_mcp.tools.vision import glm_vision
        with pytest.raises(RuntimeError, match="429"):
            glm_vision("https://example.com/img.png", "Describe", auto_fallback=False)


# ---------------------------------------------------------------------------
# AC-11  None content in response → RuntimeError
# ---------------------------------------------------------------------------

def test_vis_11_none_content_raises_runtime_error():
    """UT-VIS-11: Response with content=None raises RuntimeError with 'no text content'."""
    _assert_spec_constants()

    mock_response = MagicMock()
    mock_response.choices[0].message.content = None

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("glm_mcp.tools._core.get_client", return_value=mock_client):
        from glm_mcp.tools.vision import glm_vision
        with pytest.raises(RuntimeError, match="no text content"):
            glm_vision("https://example.com/img.png", "Describe")


# ---------------------------------------------------------------------------
# AC-12  429 + auto_fallback=True → uses fallback_model ("glm-4.6v-flash" default)
# ---------------------------------------------------------------------------

def test_vis_12_429_auto_fallback_uses_default_fallback_model():
    """UT-VIS-12: 429 + auto_fallback=True triggers switch to default fallback model 'glm-4.6v-flash'."""
    _assert_spec_constants()

    mock_client = MagicMock()
    response_429 = httpx.Response(
        429, request=_make_request(), content=b'{"error":"rate limited"}'
    )
    fallback_response = MagicMock()
    fallback_response.choices[0].message.content = "Fallback result"
    fallback_response.usage.prompt_tokens = 80
    fallback_response.usage.completion_tokens = 40
    mock_client.chat.completions.create.side_effect = [
        APIStatusError("Rate limited", response=response_429, body=None),
        fallback_response,
    ]

    with patch("glm_mcp.tools._core.get_client", return_value=mock_client), \
         patch("glm_mcp.tools._core.log_usage"):
        from glm_mcp.tools.vision import glm_vision
        result = glm_vision(
            "https://example.com/img.png", "Describe",
            model=_SPEC_DEFAULT_MODEL, auto_fallback=True,
        )

    assert result == "Fallback result"
    second_call = mock_client.chat.completions.create.call_args_list[1]
    assert second_call.kwargs["model"] == _SPEC_DEFAULT_FALLBACK


# ---------------------------------------------------------------------------
# AC-13  503 + auto_fallback=True → uses fallback_model ("glm-4.6v-flash" default)
# ---------------------------------------------------------------------------

def test_vis_13_503_auto_fallback_uses_default_fallback_model():
    """UT-VIS-13: 503 + auto_fallback=True triggers switch to default fallback model 'glm-4.6v-flash'."""
    _assert_spec_constants()

    mock_client = MagicMock()
    response_503 = httpx.Response(
        503, request=_make_request(), content=b'{"error":"service unavailable"}'
    )
    fallback_response = MagicMock()
    fallback_response.choices[0].message.content = "503 fallback result"
    fallback_response.usage.prompt_tokens = 60
    fallback_response.usage.completion_tokens = 30
    mock_client.chat.completions.create.side_effect = [
        APIStatusError("Service Unavailable", response=response_503, body=None),
        fallback_response,
    ]

    with patch("glm_mcp.tools._core.get_client", return_value=mock_client), \
         patch("glm_mcp.tools._core.log_usage"):
        from glm_mcp.tools.vision import glm_vision
        result = glm_vision(
            "https://example.com/img.png", "Describe",
            model=_SPEC_DEFAULT_MODEL, auto_fallback=True,
        )

    assert result == "503 fallback result"
    second_call = mock_client.chat.completions.create.call_args_list[1]
    assert second_call.kwargs["model"] == _SPEC_DEFAULT_FALLBACK


# ---------------------------------------------------------------------------
# AC-14  avoid_peak_hours=True during peak → pre-emptively uses fallback_model
# ---------------------------------------------------------------------------

def test_vis_14_avoid_peak_hours_during_peak_uses_fallback():
    """UT-VIS-14: avoid_peak_hours=True + peak hours → skips primary, uses fallback 'glm-4.6v-flash'."""
    _assert_spec_constants()

    fallback_response = MagicMock()
    fallback_response.choices[0].message.content = "Peak skip result"
    fallback_response.usage.prompt_tokens = 50
    fallback_response.usage.completion_tokens = 25

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = fallback_response

    with patch("glm_mcp.tools._core.get_client", return_value=mock_client), \
         patch("glm_mcp.tools._core._is_peak_hours", return_value=True), \
         patch("glm_mcp.tools._core.log_usage") as mock_log:
        from glm_mcp.tools.vision import glm_vision
        result = glm_vision(
            "https://example.com/img.png", "Describe",
            model=_SPEC_DEFAULT_MODEL,
            auto_fallback=True, avoid_peak_hours=True,
        )

    assert result == "Peak skip result"
    # Only one API call (fallback), primary was skipped
    assert mock_client.chat.completions.create.call_count == 1
    call_args = mock_client.chat.completions.create.call_args
    assert call_args.kwargs["model"] == _SPEC_DEFAULT_FALLBACK
    mock_log.assert_called_once_with(
        "glm_vision", _SPEC_DEFAULT_FALLBACK, 50, 25,
        fallback_used=True, original_model=_SPEC_DEFAULT_MODEL,
        fallback_reason="peak_hours",
    )


# ---------------------------------------------------------------------------
# AC-15  Custom fallback_model param is respected
# ---------------------------------------------------------------------------

def test_vis_15_custom_fallback_model_is_respected():
    """UT-VIS-15: Custom fallback_model parameter overrides the default 'glm-4.6v-flash'."""
    _assert_spec_constants()

    mock_client = MagicMock()
    response_429 = httpx.Response(
        429, request=_make_request(), content=b'{"error":"rate limited"}'
    )
    fallback_response = MagicMock()
    fallback_response.choices[0].message.content = "Custom fallback result"
    fallback_response.usage.prompt_tokens = 20
    fallback_response.usage.completion_tokens = 10
    mock_client.chat.completions.create.side_effect = [
        APIStatusError("Rate limited", response=response_429, body=None),
        fallback_response,
    ]

    with patch("glm_mcp.tools._core.get_client", return_value=mock_client), \
         patch("glm_mcp.tools._core.log_usage"):
        from glm_mcp.tools.vision import glm_vision
        result = glm_vision(
            "https://example.com/img.png", "Describe",
            model=_SPEC_DEFAULT_MODEL,
            fallback_model="glm-4.6v", auto_fallback=True,
        )

    assert result == "Custom fallback result"
    second_call = mock_client.chat.completions.create.call_args_list[1]
    assert second_call.kwargs["model"] == "glm-4.6v"


# ---------------------------------------------------------------------------
# AC-16  Empty prompt → ValueError
# ---------------------------------------------------------------------------

def test_vis_16_empty_prompt_raises_value_error():
    """UT-VIS-16: Empty prompt raises ValueError."""
    _assert_spec_constants()

    with patch("glm_mcp.tools._core.get_client"):
        from glm_mcp.tools.vision import glm_vision
        with pytest.raises(ValueError, match="prompt"):
            glm_vision("https://example.com/img.png", "")


# ---------------------------------------------------------------------------
# AC-17  Empty image_url → ValueError
# ---------------------------------------------------------------------------

def test_vis_17_empty_image_url_raises_value_error():
    """UT-VIS-17: Empty image_url raises ValueError."""
    _assert_spec_constants()

    with patch("glm_mcp.tools._core.get_client"):
        from glm_mcp.tools.vision import glm_vision
        with pytest.raises(ValueError, match="image_url"):
            glm_vision("", "Describe this image")


# ---------------------------------------------------------------------------
# AC-18  Multimodal message format contract
# ---------------------------------------------------------------------------

def test_vis_18_multimodal_message_format_contract():
    """UT-VIS-18: content is a list — image_url object first, text object second."""
    _assert_spec_constants()

    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Result"

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    url = "https://example.com/photo.jpg"
    prompt_text = "Describe this photo in detail"

    with patch("glm_mcp.tools._core.get_client", return_value=mock_client):
        from glm_mcp.tools.vision import glm_vision
        glm_vision(url, prompt_text, detail="low")

    call_args = mock_client.chat.completions.create.call_args
    messages = call_args.kwargs["messages"]

    # Single user message
    assert len(messages) == 1
    msg = messages[0]
    assert msg["role"] == "user"

    # content must be a list with exactly 2 parts
    content = msg["content"]
    assert isinstance(content, list)
    assert len(content) == 2

    # Part 0: image_url — must come FIRST
    img_part = content[0]
    assert img_part["type"] == "image_url"
    assert img_part["image_url"]["url"] == url
    assert img_part["image_url"]["detail"] == "low"

    # Part 1: text — must come SECOND
    text_part = content[1]
    assert text_part["type"] == "text"
    assert text_part["text"] == prompt_text


# ---------------------------------------------------------------------------
# AC-19  Tool registered in server.py via mcp.add_tool(glm_vision)
# ---------------------------------------------------------------------------

def test_vis_19_tool_registered_in_server():
    """UT-VIS-19: glm_vision is registered as an MCP tool in server.py."""
    _assert_spec_constants()

    from glm_mcp.server import mcp
    tool = asyncio.run(mcp.get_tool("glm_vision"))
    assert tool is not None
    assert tool.name == "glm_vision"


# ---------------------------------------------------------------------------
# AC-20  Does NOT instantiate OpenAI client directly (uses get_client)
#         And verifies default temperature=0.2 per design spec
# ---------------------------------------------------------------------------

def test_vis_20_uses_get_client_not_direct_openai_and_default_temperature():
    """UT-VIS-20: glm_vision uses get_client (not OpenAI directly) and passes temperature=0.2 by default."""
    _assert_spec_constants()

    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Result"

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("glm_mcp.tools._core.get_client", return_value=mock_client) as mock_get_client, \
         patch("openai.OpenAI") as mock_openai_cls:
        from glm_mcp.tools.vision import glm_vision
        glm_vision("https://example.com/img.png", "Describe")

    # Must use the factory, never instantiate OpenAI directly
    mock_get_client.assert_called()
    mock_openai_cls.assert_not_called()

    # Design spec: default temperature is 0.2 (exposed parameter, not internal constant)
    call_args = mock_client.chat.completions.create.call_args
    assert call_args.kwargs["temperature"] == _SPEC_TEMPERATURE


# ---------------------------------------------------------------------------
# AC-21  max_tokens <= 0 → ValueError
# ---------------------------------------------------------------------------

def test_vis_21_non_positive_max_tokens_raises_value_error():
    """UT-VIS-21: max_tokens <= 0 raises ValueError with 'max_tokens'."""
    _assert_spec_constants()

    with patch("glm_mcp.tools._core.get_client"):
        from glm_mcp.tools.vision import glm_vision
        with pytest.raises(ValueError, match="max_tokens"):
            glm_vision("https://example.com/img.png", "Describe", max_tokens=0)
        with pytest.raises(ValueError, match="max_tokens"):
            glm_vision("https://example.com/img.png", "Describe", max_tokens=-1)


# ---------------------------------------------------------------------------
# AC-22  Empty string content (reasoning model token exhaustion) → RuntimeError
# ---------------------------------------------------------------------------

def test_vis_22_empty_string_content_raises_runtime_error():
    """UT-VIS-22: Response with content='' (empty string) raises RuntimeError.

    Reasoning models like glm-4.6v return empty string '' instead of None
    when max_tokens is too low to produce output after reasoning tokens are
    consumed. The safety check must treat '' as falsy, not rely on `is None`.
    """
    _assert_spec_constants()

    mock_response = MagicMock()
    mock_response.choices[0].message.content = ""

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("glm_mcp.tools._core.get_client", return_value=mock_client):
        from glm_mcp.tools.vision import glm_vision
        with pytest.raises(RuntimeError, match="no text content"):
            glm_vision("https://example.com/img.png", "Describe")


# ---------------------------------------------------------------------------
# AC-23  Custom temperature is passed through to the API call
# ---------------------------------------------------------------------------

def test_vis_23_custom_temperature_passed_to_api():
    """UT-VIS-23: glm_vision forwards caller-supplied temperature to the API."""
    _assert_spec_constants()

    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Result"

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("glm_mcp.tools._core.get_client", return_value=mock_client):
        from glm_mcp.tools.vision import glm_vision
        glm_vision("https://example.com/img.png", "Describe", temperature=0.5)

    call_args = mock_client.chat.completions.create.call_args
    assert call_args.kwargs["temperature"] == 0.5


# ---------------------------------------------------------------------------
# AC-24  Default top_p=0.9 is passed to the API call
# ---------------------------------------------------------------------------

def test_vis_24_default_top_p_passed_to_api():
    """UT-VIS-24: glm_vision passes top_p=0.9 by default."""
    _assert_spec_constants()

    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Result"

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("glm_mcp.tools._core.get_client", return_value=mock_client):
        from glm_mcp.tools.vision import glm_vision
        glm_vision("https://example.com/img.png", "Describe")

    call_args = mock_client.chat.completions.create.call_args
    assert call_args.kwargs.get("top_p") == _SPEC_TOP_P


# ---------------------------------------------------------------------------
# AC-25  Custom top_p is passed through to the API call
# ---------------------------------------------------------------------------

def test_vis_25_custom_top_p_passed_to_api():
    """UT-VIS-25: glm_vision forwards caller-supplied top_p to the API."""
    _assert_spec_constants()

    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Result"

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("glm_mcp.tools._core.get_client", return_value=mock_client):
        from glm_mcp.tools.vision import glm_vision
        glm_vision("https://example.com/img.png", "Describe", top_p=0.5)

    call_args = mock_client.chat.completions.create.call_args
    assert call_args.kwargs.get("top_p") == 0.5


# ---------------------------------------------------------------------------
# AC-26  top_p=None → top_p kwarg must NOT be present in the API call
# ---------------------------------------------------------------------------

def test_vis_26_top_p_none_not_sent_to_api():
    """UT-VIS-26: When top_p=None, the 'top_p' key must not appear in the API call kwargs."""
    _assert_spec_constants()

    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Result"

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("glm_mcp.tools._core.get_client", return_value=mock_client):
        from glm_mcp.tools.vision import glm_vision
        glm_vision("https://example.com/img.png", "Describe", top_p=None)

    call_args = mock_client.chat.completions.create.call_args
    assert "top_p" not in call_args.kwargs
