"""Token usage logging for GLM API calls."""
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

_LOG_FILE = "usage.jsonl"

logger = logging.getLogger(__name__)


def _get_log_dir() -> Path:
    """Return the directory for writing usage logs.

    Uses GLM_MCP_LOG_DIR environment variable when set,
    otherwise defaults to ~/.glm-mcp.
    """
    env = os.getenv("GLM_MCP_LOG_DIR")
    return Path(env) if env else Path.home() / ".glm-mcp"


def log_usage(tool: str, model: str, input_tokens: int, output_tokens: int) -> None:
    """Append one token usage record to the usage log file.

    Args:
        tool: The MCP tool name (e.g. 'glm_chat', 'glm_embed').
        model: The GLM model that was called.
        input_tokens: Number of prompt/input tokens consumed.
        output_tokens: Number of completion/output tokens consumed.

    Note:
        Best-effort: failures are logged as warnings so callers are never broken.
    """
    try:
        log_dir = _get_log_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tool": tool,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
        with (log_dir / _LOG_FILE).open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as exc:  # noqa: BLE001 — intentional best-effort
        logger.warning("glm-mcp: failed to write usage log: %s", exc)
