"""Tests for GLM client factory."""
from unittest.mock import patch

import pytest
from openai import OpenAI


@pytest.fixture(autouse=True)
def _clear_client_cache():
    """Clear lru_cache before and after each test to prevent cross-test pollution."""
    try:
        from glm_mcp.client import _get_cached_client
        _get_cached_client.cache_clear()
    except (ImportError, AttributeError):
        pass
    yield
    try:
        from glm_mcp.client import _get_cached_client
        _get_cached_client.cache_clear()
    except (ImportError, AttributeError):
        pass


def test_get_client_raises_when_key_missing(monkeypatch):
    """get_client raises EnvironmentError when GLM_API_KEY is not set."""
    monkeypatch.delenv("GLM_API_KEY", raising=False)

    from glm_mcp.client import get_client

    with pytest.raises(EnvironmentError, match="GLM_API_KEY"):
        get_client()


def test_get_client_returns_openai_instance(monkeypatch):
    """get_client returns OpenAI instance when API key is set."""
    monkeypatch.setenv("GLM_API_KEY", "test-key-12345")

    from glm_mcp.client import get_client

    client = get_client()
    assert isinstance(client, OpenAI)


def test_get_client_uses_default_base_url(monkeypatch):
    """get_client uses ZhipuAI endpoint when GLM_BASE_URL is not set."""
    monkeypatch.setenv("GLM_API_KEY", "test-key")
    monkeypatch.delenv("GLM_BASE_URL", raising=False)

    with patch("glm_mcp.client.OpenAI") as mock_openai:
        from glm_mcp.client import get_client
        get_client()

    _, kwargs = mock_openai.call_args
    assert kwargs["base_url"] == "https://open.bigmodel.cn/api/paas/v4/"


def test_get_client_uses_custom_base_url(monkeypatch):
    """get_client uses GLM_BASE_URL env var when set."""
    monkeypatch.setenv("GLM_API_KEY", "test-key")
    monkeypatch.setenv("GLM_BASE_URL", "https://custom.example.com/v1/")

    with patch("glm_mcp.client.OpenAI") as mock_openai:
        from glm_mcp.client import get_client
        get_client()

    _, kwargs = mock_openai.call_args
    assert kwargs["base_url"] == "https://custom.example.com/v1/"


def test_get_client_caches_instance_for_same_credentials(monkeypatch):
    """get_client returns the same OpenAI instance for identical credentials."""
    monkeypatch.setenv("GLM_API_KEY", "test-key")
    monkeypatch.delenv("GLM_BASE_URL", raising=False)

    from glm_mcp.client import get_client

    client1 = get_client()
    client2 = get_client()
    assert client1 is client2
