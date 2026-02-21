"""Tests for token usage logging."""
import json
import logging
from pathlib import Path
from unittest.mock import patch

from glm_mcp.usage_log import log_usage


def test_log_usage_creates_dir_if_not_exists(tmp_path):
    """log_usage creates the log directory when it does not exist."""
    log_dir = tmp_path / ".glm-mcp"
    assert not log_dir.exists()

    with patch("glm_mcp.usage_log._get_log_dir", return_value=log_dir):
        log_usage("glm_chat", "glm-4-flash", 100, 200)

    assert log_dir.is_dir()


def test_log_usage_writes_jsonl_entry(tmp_path):
    """log_usage writes one JSON line with all required fields."""
    log_dir = tmp_path / ".glm-mcp"

    with patch("glm_mcp.usage_log._get_log_dir", return_value=log_dir):
        log_usage("glm_chat", "glm-4-flash", 150, 320)

    entry = json.loads((log_dir / "usage.jsonl").read_text().strip())
    assert entry["tool"] == "glm_chat"
    assert entry["model"] == "glm-4-flash"
    assert entry["input_tokens"] == 150
    assert entry["output_tokens"] == 320
    assert "timestamp" in entry


def test_log_usage_appends_to_existing_file(tmp_path):
    """log_usage appends entries without overwriting previous ones."""
    log_dir = tmp_path / ".glm-mcp"

    with patch("glm_mcp.usage_log._get_log_dir", return_value=log_dir):
        log_usage("glm_chat", "glm-4-flash", 100, 200)
        log_usage("glm_embed", "embedding-3", 50, 0)

    lines = (log_dir / "usage.jsonl").read_text().strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["tool"] == "glm_chat"
    assert json.loads(lines[1])["tool"] == "glm_embed"


def test_log_usage_does_not_raise_on_error(caplog):
    """log_usage does not raise if logging fails (best-effort), but logs a warning."""
    with patch("glm_mcp.usage_log._get_log_dir") as mock_get_log_dir:
        mock_log_dir = mock_get_log_dir.return_value
        mock_log_dir.mkdir.side_effect = OSError("permission denied")
        with caplog.at_level(logging.WARNING, logger="glm_mcp.usage_log"):
            log_usage("glm_chat", "glm-4-flash", 100, 200)  # must not raise

    assert "failed to write usage log" in caplog.text


def test_log_usage_uses_glm_mcp_log_dir_when_set(monkeypatch, tmp_path):
    """log_usage writes to GLM_MCP_LOG_DIR when the env var is set."""
    custom_dir = tmp_path / "custom-logs"
    monkeypatch.setenv("GLM_MCP_LOG_DIR", str(custom_dir))

    log_usage("glm_chat", "glm-4-flash", 10, 20)

    assert (custom_dir / "usage.jsonl").exists()


def test_log_usage_uses_home_dir_when_glm_mcp_log_dir_not_set(monkeypatch):
    """log_usage defaults to ~/.glm-mcp when GLM_MCP_LOG_DIR is not set."""
    monkeypatch.delenv("GLM_MCP_LOG_DIR", raising=False)

    from glm_mcp.usage_log import _get_log_dir

    assert _get_log_dir() == Path.home() / ".glm-mcp"
