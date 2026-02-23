"""GLM API client factory using OpenAI-compatible interface."""
import functools
import os

from openai import OpenAI

_DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"


@functools.cache
def _get_cached_client(api_key: str, base_url: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url)


def get_client() -> OpenAI:
    """Return an OpenAI client configured for the GLM API.

    Raises:
        EnvironmentError: If GLM_API_KEY environment variable is not set.
    """
    api_key = os.getenv("GLM_API_KEY")
    if not api_key:
        raise OSError(
            "GLM_API_KEY environment variable is not set. "
            "Get your API key at https://open.bigmodel.cn/"
        )
    base_url = os.getenv("GLM_BASE_URL", _DEFAULT_BASE_URL)
    return _get_cached_client(api_key, base_url)


def get_api_config() -> tuple[str, str]:
    """Return (api_key, base_url) for direct HTTP calls (non-OpenAI endpoints).

    Raises:
        OSError: If GLM_API_KEY environment variable is not set.
    """
    api_key = os.getenv("GLM_API_KEY")
    if not api_key:
        raise OSError(
            "GLM_API_KEY environment variable is not set. "
            "Get your API key at https://open.bigmodel.cn/"
        )
    base_url = os.getenv("GLM_BASE_URL", _DEFAULT_BASE_URL)
    return api_key, base_url
