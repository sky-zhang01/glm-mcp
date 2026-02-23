"""Tests for glm_translate tool (UT-TRN-01 ~ UT-TRN-12)."""
import asyncio
import inspect
from unittest.mock import MagicMock, patch

import httpx
from openai import APIStatusError


def _make_request() -> httpx.Request:
    return httpx.Request("POST", "https://open.bigmodel.cn/api/paas/v4/chat/completions")


# UT-TRN-01: basic success → returns non-empty string
def test_glm_translate_returns_text():
    """UT-TRN-01: glm_translate returns a non-empty string."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "こんにちは"
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("glm_mcp.tools._core.get_client", return_value=mock_client):
        from glm_mcp.tools.translate import glm_translate
        result = glm_translate("Hello", "ja")

    assert isinstance(result, str)
    assert len(result) > 0


# UT-TRN-02: style="formal" → system prompt contains "formal"
def test_glm_translate_formal_style_includes_formal_in_system_prompt():
    """UT-TRN-02: style='formal' produces system prompt with formal register description."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "翻訳結果"
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("glm_mcp.tools._core.get_client", return_value=mock_client):
        from glm_mcp.tools.translate import glm_translate
        glm_translate("Hello", "ja", style="formal")

    call_args = mock_client.chat.completions.create.call_args
    system_msg = call_args.kwargs["messages"][0]
    assert system_msg["role"] == "system"
    assert "formal" in system_msg["content"].lower()


# UT-TRN-03: style="casual" → system prompt contains "casual"
def test_glm_translate_casual_style_includes_casual_in_system_prompt():
    """UT-TRN-03: style='casual' produces system prompt with casual register description."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "やあ"
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("glm_mcp.tools._core.get_client", return_value=mock_client):
        from glm_mcp.tools.translate import glm_translate
        glm_translate("Hello", "ja", style="casual")

    call_args = mock_client.chat.completions.create.call_args
    system_msg = call_args.kwargs["messages"][0]
    assert system_msg["role"] == "system"
    assert "casual" in system_msg["content"].lower()


# UT-TRN-04: system prompt contains "ONLY" and "Do NOT mix"
def test_glm_translate_system_prompt_enforces_language_constraint():
    """UT-TRN-04: system prompt contains both 'ONLY' and 'Do NOT mix' constraints."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "翻訳"
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("glm_mcp.tools._core.get_client", return_value=mock_client):
        from glm_mcp.tools.translate import glm_translate
        glm_translate("Hello", "ja")

    call_args = mock_client.chat.completions.create.call_args
    system_content = call_args.kwargs["messages"][0]["content"]
    assert "ONLY" in system_content
    assert "Do NOT mix" in system_content


# UT-TRN-05: target_lang="zh" → system prompt references Chinese
def test_glm_translate_target_lang_zh_references_chinese_in_system_prompt():
    """UT-TRN-05: target_lang='zh' produces system prompt referencing Chinese/中文."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "你好"
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("glm_mcp.tools._core.get_client", return_value=mock_client):
        from glm_mcp.tools.translate import glm_translate
        glm_translate("Hello", "zh")

    call_args = mock_client.chat.completions.create.call_args
    system_content = call_args.kwargs["messages"][0]["content"]
    assert "Chinese" in system_content or "中文" in system_content


# UT-TRN-06: target_lang="en" → system prompt references English
def test_glm_translate_target_lang_en_references_english_in_system_prompt():
    """UT-TRN-06: target_lang='en' produces system prompt referencing English."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Hello"
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("glm_mcp.tools._core.get_client", return_value=mock_client):
        from glm_mcp.tools.translate import glm_translate
        glm_translate("こんにちは", "en")

    call_args = mock_client.chat.completions.create.call_args
    system_content = call_args.kwargs["messages"][0]["content"]
    assert "English" in system_content


# UT-TRN-07: fallback_model passed through to underlying call
def test_glm_translate_uses_specified_fallback_model():
    """UT-TRN-07: fallback_model='glm-4-flash' is used on 429 error."""
    mock_client = MagicMock()
    response_429 = httpx.Response(429, request=_make_request(), content=b'{"error":"rate limited"}')
    fallback_response = MagicMock()
    fallback_response.choices[0].message.content = "Fallback translation"
    fallback_response.usage.prompt_tokens = 5
    fallback_response.usage.completion_tokens = 3
    mock_client.chat.completions.create.side_effect = [
        APIStatusError("Rate limited", response=response_429, body=None),
        fallback_response,
    ]

    with patch("glm_mcp.tools._core.get_client", return_value=mock_client), \
         patch("glm_mcp.tools._core.log_usage"):
        from glm_mcp.tools.translate import glm_translate
        result = glm_translate("Hello", "ja", fallback_model="glm-4-flash")

    assert result == "Fallback translation"
    second_call = mock_client.chat.completions.create.call_args_list[1]
    assert second_call.kwargs["model"] == "glm-4-flash"


# UT-TRN-08: log_usage called with tool="glm_translate"
def test_glm_translate_logs_usage_with_translate_tool_name():
    """UT-TRN-08: log_usage is called with tool_name='glm_translate'."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "翻訳"
    mock_response.usage.prompt_tokens = 20
    mock_response.usage.completion_tokens = 10
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("glm_mcp.tools._core.get_client", return_value=mock_client), \
         patch("glm_mcp.tools._core.log_usage") as mock_log:
        from glm_mcp.tools.translate import glm_translate
        glm_translate("Hello", "ja")

    mock_log.assert_called_once_with(
        "glm_translate", "glm-4.7", 20, 10,
        fallback_used=False, original_model=None, fallback_reason=None,
    )


# UT-TRN-09: auto_fallback=True (underlying default) → 429 → fallback triggered
def test_glm_translate_auto_fallback_triggers_on_429():
    """UT-TRN-09: glm_translate uses auto_fallback=True by default; 429 triggers fallback."""
    mock_client = MagicMock()
    response_429 = httpx.Response(429, request=_make_request(), content=b'{"error":"rate limited"}')
    fallback_response = MagicMock()
    fallback_response.choices[0].message.content = "Fallback"
    fallback_response.usage.prompt_tokens = 5
    fallback_response.usage.completion_tokens = 3
    mock_client.chat.completions.create.side_effect = [
        APIStatusError("Rate limited", response=response_429, body=None),
        fallback_response,
    ]

    with patch("glm_mcp.tools._core.get_client", return_value=mock_client), \
         patch("glm_mcp.tools._core.log_usage"):
        from glm_mcp.tools.translate import glm_translate
        result = glm_translate("Hello", "ja")

    assert result == "Fallback"
    assert mock_client.chat.completions.create.call_count == 2


# UT-TRN-10: translate.py does NOT instantiate OpenAI() directly
def test_glm_translate_does_not_instantiate_openai_directly():
    """UT-TRN-10: translate.py source must not contain 'OpenAI(' (no direct client creation)."""
    from glm_mcp.tools import translate
    source = inspect.getsource(translate)
    assert "OpenAI(" not in source


# UT-TRN-11: glm_translate registered as MCP tool in server
def test_server_registers_glm_translate_tool():
    """UT-TRN-11: glm_translate is registered as an MCP tool in server.py."""
    from glm_mcp.server import mcp
    assert asyncio.run(mcp.get_tool("glm_translate")) is not None


# UT-TRN-12: default model is "glm-4.7"
def test_glm_translate_default_model_is_glm_4_7():
    """UT-TRN-12: glm_translate uses model='glm-4.7' by default."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "翻訳"
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("glm_mcp.tools._core.get_client", return_value=mock_client):
        from glm_mcp.tools.translate import glm_translate
        glm_translate("Hello", "ja")

    call_args = mock_client.chat.completions.create.call_args
    assert call_args.kwargs["model"] == "glm-4.7"


# UT-TRN-13: default temperature=1.0 passed to API
def test_glm_translate_default_temperature_is_1_0():
    """UT-TRN-13: glm_translate passes temperature=1.0 by default (GLM-4.7 Plan B neutral)."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "翻訳"
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("glm_mcp.tools._core.get_client", return_value=mock_client):
        from glm_mcp.tools.translate import glm_translate
        glm_translate("Hello", "ja")

    call_args = mock_client.chat.completions.create.call_args
    assert call_args.kwargs["temperature"] == 1.0


# UT-TRN-14: custom temperature is passed through to API
def test_glm_translate_custom_temperature_passed_to_api():
    """UT-TRN-14: glm_translate forwards caller-supplied temperature to the API."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "翻訳"
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("glm_mcp.tools._core.get_client", return_value=mock_client):
        from glm_mcp.tools.translate import glm_translate
        glm_translate("Hello", "ja", temperature=0.3)

    call_args = mock_client.chat.completions.create.call_args
    assert call_args.kwargs["temperature"] == 0.3


# UT-TRN-15: default top_p=0.8 passed to API
def test_glm_translate_default_top_p_is_0_8():
    """UT-TRN-15: glm_translate passes top_p=0.8 by default (GLM-4.7 Plan B stable)."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "翻訳"
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("glm_mcp.tools._core.get_client", return_value=mock_client):
        from glm_mcp.tools.translate import glm_translate
        glm_translate("Hello", "ja")

    call_args = mock_client.chat.completions.create.call_args
    assert call_args.kwargs.get("top_p") == 0.8


# UT-TRN-16: custom top_p is passed through to API
def test_glm_translate_custom_top_p_passed_to_api():
    """UT-TRN-16: glm_translate forwards caller-supplied top_p to the API."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "翻訳"
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("glm_mcp.tools._core.get_client", return_value=mock_client):
        from glm_mcp.tools.translate import glm_translate
        glm_translate("Hello", "ja", top_p=0.5)

    call_args = mock_client.chat.completions.create.call_args
    assert call_args.kwargs.get("top_p") == 0.5


# UT-TRN-17: top_p=None → "top_p" must NOT be present in the API call kwargs
def test_glm_translate_top_p_none_not_sent_to_api():
    """UT-TRN-17: When top_p=None, the 'top_p' key must not appear in the API call kwargs."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "翻訳"
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("glm_mcp.tools._core.get_client", return_value=mock_client):
        from glm_mcp.tools.translate import glm_translate
        glm_translate("Hello", "ja", top_p=None)

    call_args = mock_client.chat.completions.create.call_args
    assert "top_p" not in call_args.kwargs
