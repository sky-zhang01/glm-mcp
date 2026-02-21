# Changelog

All notable changes to glm-mcp will be documented in this file.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
Versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html)

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
