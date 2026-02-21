"""FastMCP server exposing GLM tools."""
from fastmcp import FastMCP

from glm_mcp.tools.chat import glm_chat
from glm_mcp.tools.embed import glm_embed

mcp = FastMCP("GLM MCP Server")
mcp.add_tool(glm_chat)
mcp.add_tool(glm_embed)
