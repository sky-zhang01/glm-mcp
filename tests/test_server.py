"""Tests for FastMCP server and main entry point."""
import asyncio
from unittest.mock import patch, MagicMock


def test_server_registers_two_tools():
    """FastMCP server exposes glm_chat and glm_embed tools."""
    from glm_mcp.server import mcp

    assert asyncio.run(mcp.get_tool("glm_chat")) is not None
    assert asyncio.run(mcp.get_tool("glm_embed")) is not None


def test_main_calls_mcp_run():
    """main() calls mcp.run() to start the server."""
    mock_mcp = MagicMock()
    with patch("glm_mcp.server.mcp", mock_mcp):
        from glm_mcp import main
        main()

    mock_mcp.run.assert_called_once()
