"""GLM text embedding tool."""
from glm_mcp.client import get_client
from glm_mcp.usage_log import log_usage


def glm_embed(text: str, model: str = "embedding-3") -> list[float]:
    """Generate a text embedding vector using the GLM API.

    Args:
        text: The text to embed.
        model: The embedding model to use (default: embedding-3).

    Returns:
        A list of floats representing the embedding vector.
    """
    client = get_client()
    response = client.embeddings.create(model=model, input=[text])
    log_usage("glm_embed", model, response.usage.prompt_tokens, 0)
    return response.data[0].embedding
