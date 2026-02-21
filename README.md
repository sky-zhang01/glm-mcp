# glm-mcp

MCP server for [ZhipuAI GLM](https://open.bigmodel.cn/) — exposes chat and text embeddings to Claude Code (and any MCP-compatible client) via the OpenAI-compatible API.

## Tools

| Tool | Description |
|------|-------------|
| `glm_chat` | Text completion — default model `glm-4-flash`, pass `model=` to use any GLM chat model (e.g. `glm-5`). Supports single-turn and multi-turn (`messages=` parameter). |
| `glm_embed` | Text embeddings — default model `embedding-3`, pass `model=` to override |
| `glm_usage_summary` | Query token usage from `~/.glm-mcp/usage.jsonl`. Parameters: `days` (default 7), `model` (optional filter). Returns period, total tokens, by_tool, by_model. |

## Quick Start

### Install via uvx (recommended)

```bash
uvx glm-mcp
```

### Add to Claude Code

Add to `~/.claude.json`:

```json
{
  "mcpServers": {
    "glm-mcp": {
      "type": "stdio",
      "command": "uvx",
      "args": ["glm-mcp"],
      "env": {
        "GLM_API_KEY": "your_api_key_here"
      }
    }
  }
}
```

Get your API key at <https://open.bigmodel.cn/>.

### Run from source

```bash
git clone https://github.com/sky-zhang01/glm-mcp
cd glm-mcp
uv sync
GLM_API_KEY=your_key uv run glm-mcp
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GLM_API_KEY` | Yes | — | ZhipuAI API key |
| `GLM_BASE_URL` | No | `https://open.bigmodel.cn/api/paas/v4/` | API endpoint override |
| `GLM_MCP_LOG_DIR` | No | `~/.glm-mcp/` | Directory for `usage.jsonl` token log |

## Token Usage Logging

Each tool call appends a JSON line to `~/.glm-mcp/usage.jsonl`:

```json
{"timestamp": "...", "tool": "glm_chat", "model": "glm-4-flash", "input_tokens": 13, "output_tokens": 15}
```

## Development

```bash
uv sync --dev
uv run pytest --cov=glm_mcp --cov-report=term-missing
```

## License

MIT
