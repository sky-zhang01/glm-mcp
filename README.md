# glm-mcp

MCP server for [ZhipuAI GLM](https://open.bigmodel.cn/) — exposes chat, text embeddings, translation, vision, and OCR to Claude Code (and any MCP-compatible client) via the OpenAI-compatible API.

## Tools

| Tool | Description |
|------|-------------|
| `glm_chat` | Text completion — default model `glm-4-flash`, pass `model=` to use any GLM chat model (e.g. `glm-5`). Supports single-turn and multi-turn (`messages=` parameter). Sampling: `temperature=0.7`, `top_p=0.95` (pass `top_p=None` to omit). Auto-fallback on transient errors (429/503/timeout/connection) via `auto_fallback=True` (default). Use `avoid_peak_hours=True` to pre-emptively switch during peak hours (UTC+8 14:00–18:00). |
| `glm_embed` | Text embeddings — default model `embedding-3`, pass `model=` to override |
| `glm_usage_summary` | Query token usage from `~/.glm-mcp/usage.jsonl`. Parameters: `days` (default 7), `model` (optional filter). Returns period, total tokens, by_tool, by_model. |
| `glm_translate` | Pure single-language translation — default model `glm-4.7`. Parameters: `text`, `target_lang` (`"ja"`, `"zh"`, `"en"`), `source_lang` (default `"auto"`), `style` (`"formal"` or `"casual"`, default `"formal"`). Sampling: `temperature=1.0`, `top_p=0.8` (GLM-4.7 Plan B: top_p as primary control; pass `top_p=None` to omit). Outputs ONLY the target language, solving the mixed Chinese–Japanese output problem common with general LLMs. |
| `glm_vision` | Multimodal image analysis — default model `glm-4.6v`. Parameters: `image_url` (HTTP/HTTPS URL or Base64 string), `prompt`, `detail` (`"auto"`, `"low"`, `"high"`), `max_tokens` (default 2048). Sampling: `temperature=0.2`, `top_p=0.9` (focused stable analysis; pass `top_p=None` to omit). Auto-fallback to `glm-4.6v-flash` on 429/503/timeout. Bare Base64 strings are automatically prefixed with `data:image/png;base64,`. |
| `glm_ocr` | Document and image OCR — default model `glm-ocr`. Parameters: `file` (HTTP/HTTPS URL, Base64 string, `data:` URI, or local file path), `model`, `start_page_id`, `end_page_id`. Returns extracted text as Markdown. Local files are auto-encoded as Base64; bare Base64 strings are prefixed with `data:application/pdf;base64,`. |

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
{"timestamp": "...", "tool": "glm_chat", "model": "glm-4-flash", "input_tokens": 13, "output_tokens": 15, "fallback_used": false, "original_model": null, "fallback_reason": null}
```

When fallback is triggered:

```json
{"timestamp": "...", "tool": "glm_chat", "model": "glm-4.7", "input_tokens": 13, "output_tokens": 15, "fallback_used": true, "original_model": "GLM-5", "fallback_reason": "429"}
```

## Development

```bash
uv sync --dev
uv run pytest --cov=glm_mcp --cov-report=term-missing
```

## License

MIT
