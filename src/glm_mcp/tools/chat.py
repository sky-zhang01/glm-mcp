"""GLM chat completion tool."""
import logging
from typing import cast

from openai import APIConnectionError, APIStatusError, APITimeoutError
from openai.types.chat import ChatCompletionMessageParam

from glm_mcp.client import get_client
from glm_mcp.usage_log import log_usage

logger = logging.getLogger(__name__)

_CONTEXT_KEYWORDS = ("context", "length", "token", "exceed")


def glm_chat(
    message: str = "",
    model: str = "glm-4-flash",
    system: str = "",
    temperature: float = 0.7,
    max_tokens: int = 2048,
    messages: list[dict[str, str]] | None = None,
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

    client = get_client()

    if messages is not None:
        api_messages = cast(list[ChatCompletionMessageParam], messages)
    else:
        api_messages = []
        if system:
            api_messages.append({"role": "system", "content": system})
        api_messages.append({"role": "user", "content": message})

    try:
        response = client.chat.completions.create(
            model=model,
            messages=api_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except APITimeoutError as exc:
        logger.error("GLM chat timed out: %s", exc)
        raise RuntimeError("GLM API request timed out. Please retry.") from exc
    except APIConnectionError as exc:
        logger.error("GLM chat connection error: %s", exc)
        raise RuntimeError(
            "Could not reach GLM API. Check network connectivity."
        ) from exc
    except APIStatusError as exc:
        error_body = str(exc.body) if exc.body else str(exc.message)
        if exc.status_code in (400, 413) and any(
            kw in error_body.lower() for kw in _CONTEXT_KEYWORDS
        ):
            raise RuntimeError(
                "Messages exceed model context window. "
                "Reduce conversation history length."
            ) from exc
        logger.error("GLM chat API error %s: %s", exc.status_code, exc.message)
        raise RuntimeError(f"GLM API returned error {exc.status_code}.") from exc

    if response.usage is not None:
        log_usage(
            "glm_chat", model,
            response.usage.prompt_tokens, response.usage.completion_tokens,
        )
    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError("GLM API returned no text content.")
    return content
