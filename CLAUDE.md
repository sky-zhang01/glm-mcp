# glm-mcp — Developer Reference

## Architecture

MCP server exposing ZhipuAI GLM capabilities via OpenAI-compatible API.

```
src/glm_mcp/
├── __init__.py      # Entry point: calls mcp.run()
├── client.py        # OpenAI client factory (reads GLM_API_KEY env var)
├── server.py        # FastMCP server, registers 3 tools
└── tools/
    ├── chat.py      # glm_chat — text completion
    ├── embed.py     # glm_embed — text embeddings
    └── image.py     # glm_image — image generation (CogView)
```

## Tech Stack

- Python ≥ 3.10
- [fastmcp](https://github.com/jlowin/fastmcp) ≥ 2.14 — MCP server framework
- [openai](https://github.com/openai/openai-python) ≥ 1.0 — HTTP client (GLM is OpenAI-compatible)
- GLM base URL: `https://open.bigmodel.cn/api/paas/v4/`

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GLM_API_KEY` | Yes | ZhipuAI API key |

## Commands

```bash
# Install dependencies
uv sync --dev

# Run tests with coverage
uv run pytest --cov=glm_mcp --cov-report=term-missing

# Run the MCP server
uv run glm-mcp

# Install as uvx tool
uvx glm-mcp
```

## Git Conventions

Branch naming: `feat/<name>`, `fix/<name>`, `chore/<name>`
Commit types: feat, fix, refactor, docs, test, chore
