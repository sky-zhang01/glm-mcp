"""GLM chat completion tool."""
import logging
from datetime import datetime, timedelta, timezone
from typing import cast

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
from openai.types.chat import ChatCompletionMessageParam

from glm_mcp.client import get_client
from glm_mcp.usage_log import FallbackReason, log_usage

logger = logging.getLogger(__name__)

_CONTEXT_KEYWORDS = ("context", "length", "token", "exceed")
_DEFAULT_FALLBACK_MODEL = "glm-4.7"
_RETRIABLE_STATUSES = (429, 503)
_TZ_UTC8 = timezone(timedelta(hours=8))
_PEAK_HOUR_START = 14
_PEAK_HOUR_END = 18


def _is_peak_hours() -> bool:
    """Check if current time is in peak hours (UTC+8 14:00-18:00).

    Returns:
        True if current UTC+8 hour is in [14, 18), False otherwise.
    """
    now_utc8 = datetime.now(_TZ_UTC8)
    return _PEAK_HOUR_START <= now_utc8.hour < _PEAK_HOUR_END


def _do_fallback(
    client: OpenAI,
    fallback_model: str,
    messages: list[ChatCompletionMessageParam],
    temperature: float,
    max_tokens: int,
    original_model: str,
    fallback_reason: FallbackReason,
) -> str:
    """Execute one fallback API call. Raises RuntimeError if fallback also fails.

    Args:
        client: The OpenAI-compatible client instance.
        fallback_model: Model name to use for the fallback call.
        messages: Message list to send to the API.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in response.
        original_model: The originally requested model name (for logging).
        fallback_reason: Why fallback was triggered ('429', '503', 'peak_hours',
            'timeout', 'connection').

    Returns:
        The text content of the fallback model's response.

    Raises:
        RuntimeError: If the fallback call fails for any reason.
    """
    try:
        response = client.chat.completions.create(
            model=fallback_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if response.usage is not None:
            log_usage(
                "glm_chat",
                fallback_model,
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
                fallback_used=True,
                original_model=original_model,
                fallback_reason=fallback_reason,
            )
        content: str | None = response.choices[0].message.content
        if content is None:
            raise RuntimeError("GLM API returned no text content in fallback response.")
        return content
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(
            f"GLM fallback model {fallback_model} also failed"
            f" ({type(e).__name__}): {e}"
        ) from e


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

    actual_fallback_model = (
        fallback_model if fallback_model is not None else _DEFAULT_FALLBACK_MODEL
    )

    # Pre-emptive peak-hours fallback: skip primary model entirely
    if auto_fallback and avoid_peak_hours and _is_peak_hours():
        return _do_fallback(
            client, actual_fallback_model, api_messages, temperature, max_tokens,
            model, "peak_hours",
        )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=api_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except APITimeoutError as exc:
        if auto_fallback:
            return _do_fallback(
                client, actual_fallback_model, api_messages, temperature, max_tokens,
                model, "timeout",
            )
        logger.error("GLM chat timed out: %s", exc)
        raise RuntimeError("GLM API request timed out. Please retry.") from exc
    except APIConnectionError as exc:
        if auto_fallback:
            return _do_fallback(
                client, actual_fallback_model, api_messages, temperature, max_tokens,
                model, "connection",
            )
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
        if auto_fallback and exc.status_code in _RETRIABLE_STATUSES:
            return _do_fallback(
                client, actual_fallback_model, api_messages, temperature, max_tokens,
                model, str(exc.status_code),  # type: ignore[arg-type]
            )
        logger.error("GLM chat API error %s: %s", exc.status_code, exc.message)
        raise RuntimeError(f"GLM API returned error {exc.status_code}.") from exc

    if response.usage is not None:
        log_usage(
            "glm_chat", model,
            response.usage.prompt_tokens, response.usage.completion_tokens,
            fallback_used=False,
            original_model=None,
            fallback_reason=None,
        )
    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError("GLM API returned no text content.")
    return content
