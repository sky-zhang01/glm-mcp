"""GLM chat completion tool."""
import logging

from openai import APIConnectionError, APIStatusError, APITimeoutError
from openai.types.chat import ChatCompletionMessageParam

from glm_mcp.client import get_client
from glm_mcp.usage_log import log_usage

logger = logging.getLogger(__name__)


def glm_chat(
    message: str,
    model: str = "glm-4-flash",
    system: str = "",
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> str:
    """Send a message to the GLM chat API and return the response text.

    Args:
        message: The user message to send.
        model: The GLM model to use (default: glm-4-flash).
        system: Optional system prompt.
        temperature: Sampling temperature between 0.0 and 2.0 (default: 0.7).
        max_tokens: Maximum tokens in the response (default: 2048).

    Returns:
        The text content of the model's response.

    Raises:
        RuntimeError: If the API call fails or returns empty content.
    """
    client = get_client()
    messages: list[ChatCompletionMessageParam] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": message})
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
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
