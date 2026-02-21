"""glm_usage_summary MCP tool — query token usage from usage.jsonl."""
import json
from datetime import datetime, timedelta, timezone
from typing import Any

from glm_mcp.usage_log import _get_log_dir

_LOG_FILE = "usage.jsonl"


def glm_usage_summary(
    days: int = 7,
    model: str | None = None,
) -> dict[str, Any]:
    """Return token usage summary from ~/.glm-mcp/usage.jsonl.

    Args:
        days: Number of past days to summarize (default: 7).
        model: Filter by model name (optional).

    Returns:
        Dict with keys: period, total_input_tokens, total_output_tokens,
        by_tool, by_model, record_count.
    """
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)

    period_start = cutoff.date().isoformat()
    period_end = now.date().isoformat()

    zero_summary = {
        "period": f"{period_start} ~ {period_end}",
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "record_count": 0,
        "by_tool": {},
        "by_model": {},
    }

    log_path = _get_log_dir() / _LOG_FILE
    if not log_path.exists():
        return zero_summary

    total_input = 0
    total_output = 0
    record_count = 0
    by_tool: dict[str, int] = {}
    by_model: dict[str, int] = {}

    with log_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                ts_str = entry["timestamp"]
                tool_name = entry["tool"]
                model_name = entry["model"]
                input_tokens = entry["input_tokens"]
                output_tokens = entry["output_tokens"]
            except (json.JSONDecodeError, KeyError):
                continue

            try:
                ts = datetime.fromisoformat(ts_str)
            except ValueError:
                continue

            if ts < cutoff:
                continue

            if model is not None and model_name != model:
                continue

            total_input += input_tokens
            total_output += output_tokens
            record_count += 1
            by_tool[tool_name] = by_tool.get(tool_name, 0) + 1
            by_model[model_name] = by_model.get(model_name, 0) + 1

    return {
        "period": f"{period_start} ~ {period_end}",
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "record_count": record_count,
        "by_tool": by_tool,
        "by_model": by_model,
    }
