"""Tests for GLM client factory."""
import pytest
from unittest.mock import patch
from openai import OpenAI


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
