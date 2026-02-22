"""GLM vision analysis tool."""
from typing import Literal, cast

from openai.types.chat import ChatCompletionMessageParam

from glm_mcp.tools import _core

_DEFAULT_VISION_MODEL = "glm-4v-plus"
_DEFAULT_VISION_FALLBACK_MODEL = "glm-4v"
_VISION_TEMPERATURE = 0.0


def glm_vision(
    image_url: str,
    prompt: str,
    model: str = _DEFAULT_VISION_MODEL,
    max_tokens: int = 2048,
    detail: Literal["auto", "low", "high"] = "auto",
    fallback_model: str | None = None,
    avoid_peak_hours: bool = False,
    auto_fallback: bool = True,
) -> str:
    """Analyze an image using the GLM multimodal vision API.

    Accepts HTTP/HTTPS image URLs or Base64-encoded image data. Bare Base64
    strings (without a ``data:`` prefix) are automatically prefixed with
    ``data:image/png;base64,``.

    Args:
        image_url: HTTP/HTTPS URL or Base64-encoded image string.
        prompt: Instruction or question about the image.
        model: Vision model to use. Defaults to ``glm-4v-plus``.
        max_tokens: Maximum tokens in the response. Must be > 0.
        detail: Image detail level for the API: "auto", "low", or "high".
        fallback_model: Model used when fallback is triggered. Defaults to
            ``glm-4v`` (preserves image context).
        avoid_peak_hours: Pre-emptively switch to fallback during peak hours
            (UTC+8 14:00–18:00) when auto_fallback is also True.
        auto_fallback: Enable automatic fallback on retriable errors
            (429, 503, timeout, connection errors).

    Returns:
        The model's text response describing or analyzing the image.

    Raises:
        ValueError: If ``image_url`` or ``prompt`` is empty, or if
            ``max_tokens`` is not positive.
        RuntimeError: If the API call fails and fallback is disabled or
            both primary and fallback calls fail.
    """
    if not image_url:
        raise ValueError("'image_url' must not be empty.")
    if not prompt:
        raise ValueError("'prompt' must not be empty.")
    if max_tokens <= 0:
        raise ValueError("'max_tokens' must be > 0.")

    if not image_url.startswith(("http://", "https://", "data:")):
        image_url = f"data:image/png;base64,{image_url}"

    actual_fallback_model = (
        fallback_model if fallback_model is not None else _DEFAULT_VISION_FALLBACK_MODEL
    )

    messages = cast(
        list[ChatCompletionMessageParam],
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url, "detail": detail},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )

    return _core._execute_chat_call(
        "glm_vision",
        model,
        messages,
        _VISION_TEMPERATURE,
        max_tokens,
        actual_fallback_model,
        avoid_peak_hours,
        auto_fallback,
    )
