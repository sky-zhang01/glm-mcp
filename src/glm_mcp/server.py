"""FastMCP server exposing GLM tools."""
from fastmcp import FastMCP

from glm_mcp.tools.chat import glm_chat
from glm_mcp.tools.embed import glm_embed
from glm_mcp.tools.usage_summary import glm_usage_summary

mcp = FastMCP("GLM MCP Server")
mcp.add_tool(glm_chat)
mcp.add_tool(glm_embed)
mcp.add_tool(glm_usage_summary)
