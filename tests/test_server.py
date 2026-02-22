"""Tests for FastMCP server and main entry point."""
import asyncio
from unittest.mock import MagicMock, patch


def test_server_registers_five_tools():
    """FastMCP server exposes glm_chat, glm_embed, glm_usage_summary, glm_translate, and glm_vision tools."""
    from glm_mcp.server import mcp

    assert asyncio.run(mcp.get_tool("glm_chat")) is not None
    assert asyncio.run(mcp.get_tool("glm_embed")) is not None
    assert asyncio.run(mcp.get_tool("glm_usage_summary")) is not None
    assert asyncio.run(mcp.get_tool("glm_translate")) is not None
    assert asyncio.run(mcp.get_tool("glm_vision")) is not None


def test_mcp_is_accessible_at_module_level():
    """mcp instance is importable from glm_mcp package directly."""
    import glm_mcp

    assert hasattr(glm_mcp, "mcp")
    assert glm_mcp.mcp is not None


def test_main_calls_mcp_run():
    """main() calls mcp.run() to start the server."""
    mock_mcp = MagicMock()
    with patch("glm_mcp.mcp", mock_mcp):
        from glm_mcp import main
        main()

    mock_mcp.run.assert_called_once()
