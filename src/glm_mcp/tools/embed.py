"""GLM text embedding tool."""
import logging

from openai import APIConnectionError, APIStatusError, APITimeoutError

from glm_mcp.client import get_client
from glm_mcp.tools._core import _ERR_CONNECTION, _ERR_TIMEOUT
from glm_mcp.usage_log import log_usage

logger = logging.getLogger(__name__)


def glm_embed(text: str, model: str = "embedding-3") -> list[float]:
    """Generate a text embedding vector using the GLM API.

    Args:
        text: The text to embed.
        model: The embedding model to use (default: embedding-3).

    Returns:
        A list of floats representing the embedding vector.

    Raises:
        RuntimeError: If the API call fails.
    """
    client = get_client()
    try:
        response = client.embeddings.create(model=model, input=[text])
    except APITimeoutError as exc:
        logger.error("GLM embed timed out: %s", exc)
        raise RuntimeError(_ERR_TIMEOUT) from exc
    except APIConnectionError as exc:
        logger.error("GLM embed connection error: %s", exc)
        raise RuntimeError(_ERR_CONNECTION) from exc
    except APIStatusError as exc:
        logger.error("GLM embed API error %s: %s", exc.status_code, exc.message)
        raise RuntimeError(f"GLM API returned error {exc.status_code}.") from exc
    log_usage("glm_embed", model, response.usage.prompt_tokens, 0)
    return response.data[0].embedding
