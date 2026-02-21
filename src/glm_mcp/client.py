"""GLM API client factory using OpenAI-compatible interface."""
import os

from openai import OpenAI

_DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"


def get_client() -> OpenAI:
    """Return an OpenAI client configured for the GLM API.

    Raises:
        EnvironmentError: If GLM_API_KEY environment variable is not set.
    """
    api_key = os.getenv("GLM_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GLM_API_KEY environment variable is not set. "
            "Get your API key at https://open.bigmodel.cn/"
        )
    base_url = os.getenv("GLM_BASE_URL", _DEFAULT_BASE_URL)
    return OpenAI(api_key=api_key, base_url=base_url)
