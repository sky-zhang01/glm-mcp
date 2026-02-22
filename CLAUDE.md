# glm-mcp — Developer Reference

## Architecture

MCP server exposing ZhipuAI GLM capabilities via OpenAI-compatible API.

```
src/glm_mcp/
├── __init__.py      # Entry point: module-level mcp import, calls mcp.run()
├── client.py        # OpenAI client factory (cached via functools.cache)
├── server.py        # FastMCP server, registers 5 tools
├── usage_log.py     # Append-only token usage log (~/.glm-mcp/usage.jsonl)
└── tools/
    ├── _core.py          # Shared core: _execute_chat_call, _do_fallback, _is_peak_hours
    ├── chat.py           # glm_chat — text completion (single-turn + multi-turn + auto-fallback)
    ├── embed.py          # glm_embed — text embeddings
    ├── translate.py      # glm_translate — pure single-language translation
    ├── usage_summary.py  # glm_usage_summary — query ~/.glm-mcp/usage.jsonl
    └── vision.py         # glm_vision — multimodal image analysis (CoT vision, auto-fallback)
```

## Tech Stack

- Python ≥ 3.10
- [fastmcp](https://github.com/jlowin/fastmcp) ≥ 2.14 — MCP server framework
- [openai](https://github.com/openai/openai-python) ≥ 1.0 — HTTP client (GLM is OpenAI-compatible)
- GLM base URL: `https://open.bigmodel.cn/api/paas/v4/`

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GLM_API_KEY` | Yes | — | ZhipuAI API key |
| `GLM_BASE_URL` | No | `https://open.bigmodel.cn/api/paas/v4/` | Override GLM API base URL |
| `GLM_MCP_LOG_DIR` | No | `~/.glm-mcp/` | Directory for `usage.jsonl` token log |

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

## Documentation Update Triggers

When code changes, **all affected documents must be updated before PR/commit**. Cross-check this table at Phase 11 (PR Pre-Submission):

| Code Change | Must Update |
|-------------|-------------|
| `glm_chat` parameter added/removed | `README.md` → Tools table + `glm_chat` description |
| New/removed environment variable | `README.md` → Environment Variables table; `CLAUDE.md` → Environment Variables section |
| New field in `usage.jsonl` | `README.md` → Token Usage Logging JSON examples |
| New/removed tool | `README.md` → Tools table; `CLAUDE.md` → Architecture tree |
| Source file added/renamed/deleted | `CLAUDE.md` → Architecture tree |
| Fallback behavior changed | `README.md` → `glm_chat` description + Token Usage Logging |
| New/changed default model | `README.md` → Tools table |

## Git Conventions

Branch naming: `feat/<name>`, `fix/<name>`, `chore/<name>`
Commit types: feat, fix, refactor, docs, test, chore
