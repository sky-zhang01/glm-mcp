"""GLM chat completion tool."""
from typing import cast

from openai.types.chat import ChatCompletionMessageParam

from glm_mcp.tools import _core


def glm_chat(
    message: str = "",
    model: str = "glm-4-flash",
    system: str = "",
    temperature: float = 0.7,
    max_tokens: int = 2048,
    messages: list[dict[str, str]] | None = None,
    fallback_model: str | None = None,
    avoid_peak_hours: bool = False,
    auto_fallback: bool = True,
    top_p: float | None = 0.95,
) -> str:
    """Send a message to the GLM chat API and return the response text.

    Single-turn usage (backward compatible):
        glm_chat(message="Hello")
        glm_chat(message="Hello", system="Be helpful")

    Multi-turn usage:
        glm_chat(messages=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"},
        ])
        Note: include system prompt as {"role": "system", ...} inside messages.

    Args:
        message: The user message (single-turn). Mutually exclusive with messages.
        model: The GLM model to use (default: glm-4-flash).
        system: Optional system prompt (single-turn only; ignored when
            messages= is used).
        temperature: Sampling temperature between 0.0 and 2.0 (default: 0.7).
        max_tokens: Maximum tokens in the response (default: 2048).
        messages: Full conversation history (multi-turn). Mutually
            exclusive with message.
        fallback_model: Model to switch to when fallback is triggered.
            Defaults to 'glm-4.7' when None.
        avoid_peak_hours: If True and auto_fallback=True, pre-emptively switch
            to fallback model during peak hours (UTC+8 14:00-18:00).
        auto_fallback: Global fallback switch. When False, disables all fallback
            logic and raises RuntimeError on 429/503/timeout/connection errors
            (original behavior).
        top_p: Nucleus sampling probability (0.0–1.0). Defaults to 0.95
            (GLM-5 recommended range 0.9–0.95 for general chat). When None,
            omitted from the API call.

    Returns:
        The text content of the model's response.

    Raises:
        ValueError: If both message and messages are provided, or messages is empty.
        RuntimeError: If the API call fails or returns empty content.
    """
    if messages is not None and message:
        raise ValueError(
            "Provide either 'message' (single-turn) or 'messages' "
            "(multi-turn), not both."
        )
    if messages is not None and len(messages) == 0:
        raise ValueError("'messages' list must not be empty.")

    if messages is not None:
        api_messages = cast(list[ChatCompletionMessageParam], messages)
    else:
        api_messages = []
        if system:
            api_messages.append({"role": "system", "content": system})
        api_messages.append({"role": "user", "content": message})

    actual_fallback_model = (
        fallback_model if fallback_model is not None else _core._DEFAULT_FALLBACK_MODEL
    )

    return _core._execute_chat_call(
        "glm_chat", model, api_messages, temperature, max_tokens,
        actual_fallback_model, avoid_peak_hours, auto_fallback, top_p,
    )
