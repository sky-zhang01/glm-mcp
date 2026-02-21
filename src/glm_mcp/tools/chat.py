"""GLM chat completion tool."""
from glm_mcp.client import get_client
from glm_mcp.usage_log import log_usage


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
        temperature: Sampling temperature between 0 and 1 (default: 0.7).
        max_tokens: Maximum tokens in the response (default: 2048).

    Returns:
        The text content of the model's response.
    """
    client = get_client()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": message})
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    log_usage("glm_chat", model, response.usage.prompt_tokens, response.usage.completion_tokens)
    return response.choices[0].message.content
