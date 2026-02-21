# Changelog

All notable changes to glm-mcp will be documented in this file.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
Versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html)

---

## [0.4.0] - 2026-02-22

### Added

- **Auto-fallback resilience** (`glm_chat`): automatic model switch when the primary
  model encounters transient errors.
  - New parameters: `auto_fallback` (bool, default `True`), `avoid_peak_hours` (bool,
    default `False`), `fallback_model` (str, default `"glm-4.7"`).
  - Triggers: HTTP 429 / 503, `APITimeoutError`, `APIConnectionError`.
  - Pre-emptive mode: `avoid_peak_hours=True` skips primary model during UTC+8 14:00–18:00.
  - `auto_fallback=False` disables all fallback and restores pre-v0.4.0 behavior.
- **`FallbackReason` type alias** (`usage_log`): `Literal["429", "503", "peak_hours",
  "timeout", "connection"]` exported from `glm_mcp.usage_log`.

### Fixed

- `_do_fallback`: added `if response.usage is not None:` guard before `log_usage` call.
- `_do_fallback`: `log_usage` now called before content check (correct ordering).
- `_do_fallback`: exception message now includes `type(e).__name__` for easier diagnosis.
- `_is_peak_hours`: magic numbers `14`/`18` replaced with named constants
  `_PEAK_HOUR_START` / `_PEAK_HOUR_END`.
- `_do_fallback`: `client` parameter type changed from `Any` to `OpenAI` (mypy-safe).

### Tests

- 2 new unit tests (UT-CHT-33 ~ UT-CHT-34): timeout/connection auto-fallback.
- Updated UT-CHT-7/8: explicit `auto_fallback=False` to preserve error-raise semantics.
- Updated UT-CHT-30: added timezone argument assertion for `_is_peak_hours`.
- Total: **70 UT, 100% coverage**.

---

## [0.3.0] - 2026-02-21

### Added

- **`glm_usage_summary` MCP tool**: query token usage from `~/.glm-mcp/usage.jsonl`.
  - Parameters: `days` (int, default 7), `model` (str, optional filter).
  - Returns: `period`, `total_input_tokens`, `total_output_tokens`, `record_count`,
    `by_tool` (call count per tool), `by_model` (call count per model).
  - Malformed lines and invalid timestamps are silently skipped.

### Tests

- 12 new unit tests (UT-SUM-01 ~ UT-SUM-12).
- Total: 52 UT, 100% coverage.

### Chore

- ruff `per-file-ignores`: exclude E501 from `tests/**` (test data lines are intentionally wide).

---

## [0.2.0] - 2026-02-21

### Added

- **Multi-turn conversation support** (`glm_chat`): new `messages` parameter accepts
  a full conversation history list, enabling multi-turn dialogue via MCP.
- **Context window error handling**: `glm_chat` now raises a descriptive
  `RuntimeError("Messages exceed model context window. ...")` instead of a generic
  HTTP 400 error when the API reports a context length exceeded condition.

### Changed

- `glm_chat` signature: `message` parameter changed from required positional to
  optional keyword arg (default `""`). Fully backward-compatible.

### Tests

- 8 new unit tests (UT-CHT-11 ~ UT-CHT-18) covering multi-turn mode, edge cases,
  and context window error handling.
- Total: 40 UT, 100% coverage.

---

## [0.1.2] - 2026-02-21

### Added

- `GLM_BASE_URL` env var for custom API endpoint override (default: ZhipuAI).
- `@functools.cache` on internal client factory for connection reuse.
- `GLM_MCP_LOG_DIR` env var for custom usage log directory.

### Tests

- 4 new unit tests (UT-CLT-05, UT-SRV-03, UT-LOG-05, UT-LOG-06).
- Total: 32 UT, 100% coverage.

---

## [0.1.1] - 2026-02-21

### Added

- Error handling for API timeout, connection failure, and HTTP status errors.
- `glm_chat` raises descriptive `RuntimeError` for each error class.
- `glm_embed` raises descriptive `RuntimeError` for each error class.

### Fixed

- Corrected env var names from `ZHIPUAI_*` to `GLM_*`.
- `get_client` now raises `EnvironmentError` (not `ValueError`) when key is missing.

### Tests

- 7 new error-path tests (UT-CHT-07~10, UT-EMB-06~08).
- Total: 28 UT, 100% coverage.

---

## [0.1.0] - 2026-02-21

### Initial Release

- MCP server exposing `glm_chat` and `glm_embed` tools via fastmcp 3.0.
- OpenAI-compatible SDK targeting ZhipuAI GLM API.
- Usage logging to `~/.glm-mcp/usage.jsonl`.
- 21 unit tests across 5 modules, 100% coverage.
