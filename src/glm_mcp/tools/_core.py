"""Shared core logic for GLM MCP tools (chat execution, fallback, peak-hours)."""
import logging
from datetime import datetime, timedelta, timezone

from openai import APIConnectionError, APIStatusError, APITimeoutError
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
_ERR_TIMEOUT = "GLM API request timed out. Please retry."
_ERR_CONNECTION = "Could not reach GLM API. Check network connectivity."


def _build_llm_kwargs(top_p: float | None) -> dict[str, object]:
    return {"top_p": top_p} if top_p is not None else {}


def _is_peak_hours() -> bool:
    """Check if current time is in peak hours (UTC+8 14:00-18:00).

    Returns:
        True if current UTC+8 hour is in [14, 18), False otherwise.
    """
    now_utc8 = datetime.now(_TZ_UTC8)
    return _PEAK_HOUR_START <= now_utc8.hour < _PEAK_HOUR_END


def _do_fallback(
    tool_name: str,
    fallback_model: str,
    messages: list[ChatCompletionMessageParam],
    temperature: float,
    max_tokens: int,
    original_model: str,
    fallback_reason: FallbackReason,
    top_p: float | None = None,
) -> str:
    """Execute one fallback API call. Raises RuntimeError if fallback also fails.

    Args:
        tool_name: Name of the calling tool (for usage logging).
        fallback_model: Model name to use for the fallback call.
        messages: Message list to send to the API.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in response.
        original_model: The originally requested model name (for logging).
        fallback_reason: Why fallback was triggered.
        top_p: Nucleus sampling probability. When None, not passed to the API.

    Returns:
        The text content of the fallback model's response.

    Raises:
        RuntimeError: If the fallback call fails for any reason.
    """
    try:
        client = get_client()
        response = client.chat.completions.create(  # type: ignore[call-overload]
            model=fallback_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **_build_llm_kwargs(top_p),
        )
        if response.usage is not None:
            log_usage(
                tool_name,
                fallback_model,
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
                fallback_used=True,
                original_model=original_model,
                fallback_reason=fallback_reason,
            )
        content: str | None = response.choices[0].message.content
        if not content:
            raise RuntimeError("GLM API returned no text content in fallback response.")
        return content
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(
            f"GLM fallback model {fallback_model} also failed"
            f" ({type(e).__name__}): {e}"
        ) from e


def _execute_chat_call(
    tool_name: str,
    model: str,
    messages: list[ChatCompletionMessageParam],
    temperature: float,
    max_tokens: int,
    fallback_model: str,
    avoid_peak_hours: bool,
    auto_fallback: bool,
    top_p: float | None = None,
) -> str:
    """Execute a chat completion call with fallback support.

    Args:
        tool_name: Name of the calling tool (for usage logging).
        model: Primary model to use.
        messages: Message list to send to the API.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in response.
        fallback_model: Model to switch to on retriable errors or peak hours.
        avoid_peak_hours: If True and auto_fallback=True, pre-emptively switch
            to fallback model during peak hours (UTC+8 14:00-18:00).
        auto_fallback: When False, disables all fallback logic and raises
            RuntimeError on retriable errors.
        top_p: Nucleus sampling probability. When None, not passed to the API.

    Returns:
        The text content of the model's response.

    Raises:
        RuntimeError: If the API call fails or returns empty content.
    """
    client = get_client()

    if auto_fallback and avoid_peak_hours and _is_peak_hours():
        return _do_fallback(
            tool_name, fallback_model, messages,
            temperature, max_tokens, model, "peak_hours", top_p,
        )

    try:
        response = client.chat.completions.create(  # type: ignore[call-overload]
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **_build_llm_kwargs(top_p),
        )
    except APITimeoutError as exc:
        if auto_fallback:
            return _do_fallback(
                tool_name, fallback_model, messages,
                temperature, max_tokens, model, "timeout", top_p,
            )
        raise RuntimeError(_ERR_TIMEOUT) from exc
    except APIConnectionError as exc:
        if auto_fallback:
            return _do_fallback(
                tool_name, fallback_model, messages,
                temperature, max_tokens, model, "connection", top_p,
            )
        raise RuntimeError(_ERR_CONNECTION) from exc
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
                tool_name, fallback_model, messages,
                temperature, max_tokens, model,
                str(exc.status_code),  # type: ignore[arg-type]
                top_p,
            )
        raise RuntimeError(f"GLM API returned error {exc.status_code}.") from exc

    if response.usage is not None:
        log_usage(
            tool_name, model,
            response.usage.prompt_tokens, response.usage.completion_tokens,
            fallback_used=False, original_model=None, fallback_reason=None,
        )
    content: str | None = response.choices[0].message.content
    if not content:
        raise RuntimeError("GLM API returned no text content.")
    return content
