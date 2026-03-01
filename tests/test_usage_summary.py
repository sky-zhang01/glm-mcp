"""Tests for glm_usage_summary tool.

Test IDs: UT-SUM-01 ~ UT-SUM-10
"""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from glm_mcp.tools.usage_summary import glm_usage_summary

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts(days_ago: int = 0) -> str:
    """Return an ISO 8601 UTC timestamp N days before today."""
    dt = datetime.now(timezone.utc) - timedelta(days=days_ago)
    return dt.isoformat()


def _write_log(log_dir: Path, entries: list[dict]) -> None:
    """Write entries to usage.jsonl in log_dir."""
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / "usage.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# UT-SUM-01: file does not exist → zero summary, no exception
# ---------------------------------------------------------------------------

def test_missing_log_file_returns_zero_summary(tmp_path):
    """UT-SUM-01: Missing usage.jsonl returns a zero summary without raising."""
    log_dir = tmp_path / ".glm-mcp"
    # do NOT create the file

    with patch("glm_mcp.tools.usage_summary.get_log_dir", return_value=log_dir):
        result = glm_usage_summary()

    assert result["record_count"] == 0
    assert result["total_input_tokens"] == 0
    assert result["total_output_tokens"] == 0
    assert result["by_tool"] == {}
    assert result["by_model"] == {}


# ---------------------------------------------------------------------------
# UT-SUM-02: records within range are aggregated correctly
# ---------------------------------------------------------------------------

def test_aggregates_records_within_default_window(tmp_path):
    """UT-SUM-02: Records within 7 days are summed; outside are excluded."""
    log_dir = tmp_path / ".glm-mcp"
    entries = [
        {"timestamp": _ts(0), "tool": "glm_chat",  "model": "glm-4-flash",  "input_tokens": 100, "output_tokens": 200},
        {"timestamp": _ts(3), "tool": "glm_embed", "model": "embedding-3",   "input_tokens":  50, "output_tokens":   0},
        {"timestamp": _ts(8), "tool": "glm_chat",  "model": "glm-4-flash",  "input_tokens": 999, "output_tokens": 999},  # outside window
    ]
    _write_log(log_dir, entries)

    with patch("glm_mcp.tools.usage_summary.get_log_dir", return_value=log_dir):
        result = glm_usage_summary(days=7)

    assert result["record_count"] == 2
    assert result["total_input_tokens"] == 150
    assert result["total_output_tokens"] == 200
    assert result["by_tool"] == {"glm_chat": 1, "glm_embed": 1}
    assert result["by_model"] == {"glm-4-flash": 1, "embedding-3": 1}


# ---------------------------------------------------------------------------
# UT-SUM-03: model filter
# ---------------------------------------------------------------------------

def test_model_filter_returns_only_matching_records(tmp_path):
    """UT-SUM-03: model= parameter filters to that model only."""
    log_dir = tmp_path / ".glm-mcp"
    entries = [
        {"timestamp": _ts(0), "tool": "glm_chat",  "model": "glm-4-flash", "input_tokens": 100, "output_tokens": 200},
        {"timestamp": _ts(0), "tool": "glm_chat",  "model": "glm-4",       "input_tokens":  80, "output_tokens": 150},
        {"timestamp": _ts(0), "tool": "glm_embed", "model": "embedding-3",  "input_tokens":  50, "output_tokens":   0},
    ]
    _write_log(log_dir, entries)

    with patch("glm_mcp.tools.usage_summary.get_log_dir", return_value=log_dir):
        result = glm_usage_summary(days=7, model="glm-4-flash")

    assert result["record_count"] == 1
    assert result["total_input_tokens"] == 100
    assert result["total_output_tokens"] == 200
    assert result["by_model"] == {"glm-4-flash": 1}


# ---------------------------------------------------------------------------
# UT-SUM-04: no records in range → zero summary
# ---------------------------------------------------------------------------

def test_no_records_in_range_returns_zero_summary(tmp_path):
    """UT-SUM-04: File exists but all records are outside the window."""
    log_dir = tmp_path / ".glm-mcp"
    entries = [
        {"timestamp": _ts(30), "tool": "glm_chat", "model": "glm-4-flash", "input_tokens": 100, "output_tokens": 200},
    ]
    _write_log(log_dir, entries)

    with patch("glm_mcp.tools.usage_summary.get_log_dir", return_value=log_dir):
        result = glm_usage_summary(days=7)

    assert result["record_count"] == 0
    assert result["total_input_tokens"] == 0
    assert result["by_tool"] == {}
    assert result["by_model"] == {}


# ---------------------------------------------------------------------------
# UT-SUM-05: period field format
# ---------------------------------------------------------------------------

def test_period_field_format(tmp_path):
    """UT-SUM-05: period is 'YYYY-MM-DD ~ YYYY-MM-DD' covering the requested window."""
    log_dir = tmp_path / ".glm-mcp"
    _write_log(log_dir, [])  # empty but file exists

    with patch("glm_mcp.tools.usage_summary.get_log_dir", return_value=log_dir):
        result = glm_usage_summary(days=7)

    period = result["period"]
    assert "~" in period
    parts = [p.strip() for p in period.split("~")]
    assert len(parts) == 2
    # each part should be a valid date string YYYY-MM-DD
    from datetime import date
    date.fromisoformat(parts[0])
    date.fromisoformat(parts[1])


# ---------------------------------------------------------------------------
# UT-SUM-06: by_tool / by_model only contain keys with count > 0
# ---------------------------------------------------------------------------

def test_by_tool_and_by_model_exclude_zero_count_keys(tmp_path):
    """UT-SUM-06: Only tools/models that appear in filtered records are in the dicts."""
    log_dir = tmp_path / ".glm-mcp"
    entries = [
        {"timestamp": _ts(0),  "tool": "glm_chat",  "model": "glm-4-flash", "input_tokens": 10, "output_tokens": 20},
        {"timestamp": _ts(30), "tool": "glm_embed", "model": "embedding-3",  "input_tokens": 50, "output_tokens":  0},
    ]
    _write_log(log_dir, entries)

    with patch("glm_mcp.tools.usage_summary.get_log_dir", return_value=log_dir):
        result = glm_usage_summary(days=7)

    assert "glm_embed" not in result["by_tool"]
    assert "embedding-3" not in result["by_model"]


# ---------------------------------------------------------------------------
# UT-SUM-07: malformed lines are skipped silently
# ---------------------------------------------------------------------------

def test_malformed_lines_are_skipped(tmp_path):
    """UT-SUM-07: Lines that cannot be parsed or are missing fields are skipped."""
    log_dir = tmp_path / ".glm-mcp"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / "usage.jsonl"
    with path.open("w", encoding="utf-8") as f:
        f.write("not-json\n")
        f.write(json.dumps({"timestamp": _ts(0), "tool": "glm_chat", "model": "glm-4-flash",
                             "input_tokens": 10, "output_tokens": 20}) + "\n")
        f.write(json.dumps({"timestamp": _ts(0)}) + "\n")  # missing fields

    with patch("glm_mcp.tools.usage_summary.get_log_dir", return_value=log_dir):
        result = glm_usage_summary(days=7)

    # only the valid line counts
    assert result["record_count"] == 1
    assert result["total_input_tokens"] == 10


# ---------------------------------------------------------------------------
# UT-SUM-08: days=1 boundary — today only
# ---------------------------------------------------------------------------

def test_days_1_includes_only_today(tmp_path):
    """UT-SUM-08: days=1 includes records from today, excludes yesterday."""
    log_dir = tmp_path / ".glm-mcp"
    entries = [
        {"timestamp": _ts(0), "tool": "glm_chat", "model": "glm-4-flash", "input_tokens": 10, "output_tokens": 20},
        {"timestamp": _ts(1), "tool": "glm_chat", "model": "glm-4-flash", "input_tokens": 99, "output_tokens": 99},
    ]
    _write_log(log_dir, entries)

    with patch("glm_mcp.tools.usage_summary.get_log_dir", return_value=log_dir):
        result = glm_usage_summary(days=1)

    assert result["record_count"] == 1
    assert result["total_input_tokens"] == 10


# ---------------------------------------------------------------------------
# UT-SUM-09: multiple records same tool/model — counts accumulate
# ---------------------------------------------------------------------------

def test_multiple_records_same_tool_accumulate(tmp_path):
    """UT-SUM-09: Multiple records with same tool/model accumulate token sums."""
    log_dir = tmp_path / ".glm-mcp"
    entries = [
        {"timestamp": _ts(0), "tool": "glm_chat", "model": "glm-4-flash", "input_tokens": 100, "output_tokens": 200},
        {"timestamp": _ts(0), "tool": "glm_chat", "model": "glm-4-flash", "input_tokens":  50, "output_tokens": 100},
    ]
    _write_log(log_dir, entries)

    with patch("glm_mcp.tools.usage_summary.get_log_dir", return_value=log_dir):
        result = glm_usage_summary(days=7)

    assert result["record_count"] == 2
    assert result["by_tool"]["glm_chat"] == 2
    assert result["by_model"]["glm-4-flash"] == 2
    assert result["total_input_tokens"] == 150
    assert result["total_output_tokens"] == 300


# ---------------------------------------------------------------------------
# UT-SUM-10: tool is registered in server
# ---------------------------------------------------------------------------

def test_glm_usage_summary_registered_in_server():
    """UT-SUM-10: glm_usage_summary is registered as an MCP tool in server.py."""
    import asyncio

    from glm_mcp.server import mcp
    tool = asyncio.run(mcp.get_tool("glm_usage_summary"))
    assert tool is not None
    assert tool.name == "glm_usage_summary"


# ---------------------------------------------------------------------------
# UT-SUM-11: empty lines in file are skipped silently
# ---------------------------------------------------------------------------

def test_empty_lines_in_file_are_skipped(tmp_path):
    """UT-SUM-11: Empty lines (blank lines) in usage.jsonl are silently skipped."""
    log_dir = tmp_path / ".glm-mcp"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / "usage.jsonl"
    with path.open("w", encoding="utf-8") as f:
        f.write("\n")  # blank line
        f.write(json.dumps({"timestamp": _ts(0), "tool": "glm_chat", "model": "glm-4-flash",
                             "input_tokens": 10, "output_tokens": 20}) + "\n")
        f.write("   \n")  # whitespace-only line

    with patch("glm_mcp.tools.usage_summary.get_log_dir", return_value=log_dir):
        result = glm_usage_summary(days=7)

    assert result["record_count"] == 1
    assert result["total_input_tokens"] == 10


# ---------------------------------------------------------------------------
# UT-SUM-12: valid JSON with invalid timestamp is skipped
# ---------------------------------------------------------------------------

def test_invalid_timestamp_format_is_skipped(tmp_path):
    """UT-SUM-12: Valid JSON but non-ISO timestamp is silently skipped."""
    log_dir = tmp_path / ".glm-mcp"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / "usage.jsonl"
    with path.open("w", encoding="utf-8") as f:
        # Valid JSON but timestamp cannot be parsed by datetime.fromisoformat
        f.write(json.dumps({"timestamp": "not-a-date", "tool": "glm_chat",
                             "model": "glm-4-flash", "input_tokens": 99,
                             "output_tokens": 99}) + "\n")
        f.write(json.dumps({"timestamp": _ts(0), "tool": "glm_chat", "model": "glm-4-flash",
                             "input_tokens": 10, "output_tokens": 20}) + "\n")

    with patch("glm_mcp.tools.usage_summary.get_log_dir", return_value=log_dir):
        result = glm_usage_summary(days=7)

    assert result["record_count"] == 1
    assert result["total_input_tokens"] == 10
