# Changelog

All notable changes to glm-mcp will be documented in this file.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
Versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html)

---

## [0.8.0] - 2026-02-23

### Added

- **`temperature` and `top_p` parameters** exposed on `glm_chat`, `glm_translate`, and `glm_vision`
  with research-backed defaults from GLM official documentation.
  - `glm_chat`: `temperature=0.7` (existing, unchanged), `top_p=0.95` (new; GLM-5 recommended 0.9–0.95).
  - `glm_translate`: `temperature=1.0` (neutral, top_p is the primary control), `top_p=0.8` (new;
    GLM-4.7 Plan B stable technical output).
  - `glm_vision`: `temperature=0.2` (focused stable analysis, previously a private constant `_VISION_TEMPERATURE`),
    `top_p=0.9` (new).
  - All tools: pass `top_p=None` to omit the parameter from the API call entirely.

### Changed

- `glm_translate`: `temperature` promoted from private constant `_TRANSLATE_TEMPERATURE=0.1` to
  a public parameter defaulting to `1.0` (GLM-4.7 Plan B neutral value).
- `glm_vision`: `temperature` promoted from private constant `_VISION_TEMPERATURE=0.0` to a public
  parameter defaulting to `0.2`.
- `_core._execute_chat_call` and `_core._do_fallback`: added `top_p: float | None = None` parameter;
  uses conditional kwargs (`**{"top_p": top_p} if top_p is not None else {}`) to cleanly omit the
  parameter when not needed.

### Tests

- 12 new unit tests covering default and custom values for `temperature` and `top_p`, plus `top_p=None`
  omission: UT-VIS-23~26 (vision), UT-TRN-13~17 (translate), UT-CHT-35~37 (chat).
- UT-VIS-20 updated: function renamed from `...temperature_is_zero` to `...default_temperature`;
  assertion updated to `temperature == 0.2`.

---

## [0.7.0] - 2026-02-23

### Added

- **`glm_ocr` MCP tool**: document and image OCR via `POST /api/paas/v4/layout_parsing`
  (non-OpenAI-compatible endpoint, implemented with `urllib.request`).
  - Parameters: `file` (HTTP/HTTPS URL, Base64 string, `data:` URI, or local file path),
    `model` (default `"glm-ocr"`), `start_page_id`, `end_page_id`.
  - Returns extracted text as Markdown (`md_results`).
  - Local files auto-encoded as Base64; bare Base64 strings prefixed with
    `data:application/pdf;base64,`; MIME auto-detected from extension.
  - No fallback (only one OCR model available).
- **`client.get_api_config()`** helper: returns `(api_key, base_url)` tuple for tools
  that make direct HTTP calls outside the OpenAI SDK.

### Changed

- `glm_vision` default model: `"glm-4v-plus"` → `"glm-4.6v"` (106B MoE flagship vision model).
- `glm_vision` default fallback model: `"glm-4v"` → `"glm-4.6v-flash"` (same-generation free variant).
- `_core._execute_chat_call` and `_core._do_fallback`: empty-string content check changed from
  `if content is None` to `if not content` — prevents silent failure with reasoning models
  that return `""` when `max_tokens` is too low.

### Tests

- 15 new unit tests (UT-OCR-01 ~ UT-OCR-15) covering URL/base64/local-file inputs, pagination,
  usage logging, error handling, and server registration.
- Updated `test_vision.py` spec constants: `_SPEC_DEFAULT_MODEL = "glm-4.6v"`,
  `_SPEC_DEFAULT_FALLBACK = "glm-4.6v-flash"`.
- New UT-VIS-22: verifies empty-string content (`""`) triggers RuntimeError.

---

## [0.6.0] - 2026-02-23

### Added

- **`glm_vision` MCP tool**: multimodal image analysis via GLM vision API.
  - Parameters: `image_url` (HTTP/HTTPS URL or Base64 string), `prompt`,
    `model` (default `"glm-4v-plus"`), `max_tokens` (default `2048`),
    `detail` (`"auto"` / `"low"` / `"high"`, default `"auto"`),
    `fallback_model` (default `"glm-4v"`), `avoid_peak_hours` (bool),
    `auto_fallback` (bool, default `True`).
  - Bare Base64 strings (no `data:` prefix) are automatically prefixed with
    `data:image/png;base64,`.
  - Auto-fallback on 429 / 503 / timeout / connection errors; `avoid_peak_hours`
    pre-emptively switches during UTC+8 14:00–18:00.
  - Input validation: raises `ValueError` when `image_url` or `prompt` is empty,
    or when `max_tokens` ≤ 0.

### Tests

- 21 new unit tests (UT-VIS-01 ~ UT-VIS-21) covering: basic call, URL passthrough,
  Base64 prefix injection, fallback trigger, peak-hours pre-emption, `detail`
  parameter passthrough, max_tokens default, empty image_url/prompt ValueError,
  non-positive max_tokens ValueError, server registration, model/fallback defaults,
  and `_DEFAULT_VISION_MODEL` / `_DEFAULT_VISION_FALLBACK_MODEL` / `_VISION_TEMPERATURE`
  constant values.
- Total: **103 UT, 100% coverage**.

---

## [0.5.0] - 2026-02-22

### Added

- **`glm_translate` MCP tool**: pure single-language translation that eliminates the
  Chinese–Japanese mixed-output problem common with general-purpose LLMs.
  - Parameters: `text`, `target_lang` (`"ja"` / `"zh"` / `"en"`), `source_lang`
    (default `"auto"`), `style` (`"formal"` / `"casual"`, default `"formal"`),
    `model` (default `"glm-4.7"`), `fallback_model`.
  - System prompt enforces `"Output ONLY the translated text in pure {lang}. Do NOT mix any other language."`.
  - Shares fallback/client/logging infrastructure with `glm_chat` via new `_core.py`.
- **`tools/_core.py`** shared module: extracted `_execute_chat_call(tool_name, ...)`,
  `_do_fallback(tool_name, ...)`, and `_is_peak_hours()` from `chat.py` into a
  shared internal module, eliminating code duplication across tools.

### Changed

- `tools/chat.py` refactored to a thin wrapper that delegates to `_core._execute_chat_call`.
  Public API (signature and behavior) is unchanged.

### Tests

- 12 new unit tests (UT-TRN-01 ~ UT-TRN-12) covering: basic translation, formal/casual
  style prompts, language constraint enforcement, language names in system prompt,
  fallback passthrough, usage logging with tool="glm_translate", 429 auto-fallback,
  OpenAI non-instantiation, server registration, and default model.
- Updated `test_chat.py`: mock patch paths migrated from `glm_mcp.tools.chat.*` to
  `glm_mcp.tools._core.*` to match the refactored module structure.
- Total: **82 UT, 100% coverage**.

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
