"""Token usage logging for GLM API calls."""
import json
from datetime import datetime, timezone
from pathlib import Path

_LOG_DIR = Path.home() / ".glm-mcp"
_LOG_FILE = "usage.jsonl"


def log_usage(tool: str, model: str, input_tokens: int, output_tokens: int) -> None:
    """Append one token usage record to ~/.glm-mcp/usage.jsonl.

    Args:
        tool: The MCP tool name (e.g. 'glm_chat', 'glm_embed').
        model: The GLM model that was called.
        input_tokens: Number of prompt/input tokens consumed.
        output_tokens: Number of completion/output tokens consumed.

    Note:
        Best-effort: errors are silently ignored so callers are never broken.
    """
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tool": tool,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
        with (_LOG_DIR / _LOG_FILE).open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass
