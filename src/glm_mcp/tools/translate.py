"""GLM translation tool."""
from openai.types.chat import ChatCompletionMessageParam

from glm_mcp.tools import _core

_LANG_NAMES = {
    "ja": "Japanese (日本語)",
    "zh": "Chinese (中文)",
    "en": "English",
}

_STYLE_DESCS = {
    "formal": "in a formal, professional register suitable for official documents",
    "casual": "in a casual, conversational register",
}

_TRANSLATE_MAX_TOKENS = 4096


def _build_system_prompt(target_lang: str, style: str) -> str:
    lang_name = _LANG_NAMES.get(target_lang, target_lang)
    style_desc = _STYLE_DESCS.get(style, "")
    return (
        f"You are a professional translator. "
        f"Translate the following text into {lang_name} {style_desc}. "
        f"Output ONLY the translated text in pure {lang_name}. "
        f"Do NOT mix any other language."
    )


def glm_translate(
    text: str,
    target_lang: str,
    source_lang: str = "auto",
    style: str = "formal",
    model: str = "glm-4.7",
    fallback_model: str | None = None,
    temperature: float = 1.0,
    top_p: float | None = 0.8,
) -> str:
    """Translate text into the target language using GLM.

    Produces pure single-language output without mixing other languages,
    solving the Chinese–Japanese mixed-output problem common with general LLMs.

    Args:
        text: The text to translate.
        target_lang: Target language code ('ja', 'zh', 'en').
        source_lang: Source language code (default: 'auto' for auto-detection).
        style: Translation style — 'formal' (official documents) or 'casual'
            (conversational). Defaults to 'formal'.
        model: GLM model to use (default: 'glm-4.7').
        fallback_model: Model to switch to on retriable errors.
            Defaults to 'glm-4.7' when None.
        temperature: Sampling temperature (0.0–2.0). Defaults to 1.0 (neutral
            value for GLM-4.7 Plan B where top_p is the primary control).
        top_p: Nucleus sampling probability (0.0–1.0). Defaults to 0.8
            (GLM-4.7 Plan B stable technical output). When None, omitted
            from the API call.

    Returns:
        The translated text in the target language.

    Raises:
        RuntimeError: If the API call fails or returns empty content.
    """
    system_prompt = _build_system_prompt(target_lang, style)
    api_messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]
    actual_fallback_model = (
        fallback_model if fallback_model is not None else _core._DEFAULT_FALLBACK_MODEL
    )
    return _core._execute_chat_call(
        "glm_translate", model, api_messages,
        temperature, _TRANSLATE_MAX_TOKENS,
        actual_fallback_model, False, True, top_p,
    )
