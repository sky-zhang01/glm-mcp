"""GLM OCR tool using the layout_parsing endpoint."""
import base64
import json
import os
import urllib.error
import urllib.request

from glm_mcp.client import get_api_config
from glm_mcp.usage_log import log_usage

_DEFAULT_OCR_MODEL = "glm-ocr"
_OCR_ENDPOINT = "layout_parsing"
_REQUEST_TIMEOUT = 60

_MIME_MAP: dict[str, str] = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}


def glm_ocr(
    file: str,
    model: str = _DEFAULT_OCR_MODEL,
    start_page_id: int | None = None,
    end_page_id: int | None = None,
) -> str:
    """Extract text from a document or image using the GLM OCR API.

    Accepts HTTP/HTTPS document URLs, Base64-encoded file data, ``data:`` URIs,
    or local file paths. Local files are automatically read and encoded as
    Base64. Bare Base64 strings (without a ``data:`` prefix) are automatically
    prefixed with ``data:application/pdf;base64,``.

    Args:
        file: HTTP/HTTPS URL, Base64 string, ``data:`` URI, or local file path.
        model: OCR model to use. Defaults to ``glm-ocr``.
        start_page_id: First page to process for multi-page PDFs (1-based).
            If ``None``, the API default is used.
        end_page_id: Last page to process for multi-page PDFs (1-based).
            If ``None``, the API default is used.

    Returns:
        Extracted document text formatted as Markdown.

    Raises:
        ValueError: If ``file`` is empty.
        RuntimeError: If the API call fails or returns no OCR content.
    """
    if not file:
        raise ValueError("'file' must not be empty.")

    file = _prepare_file(file)

    api_key, base_url = get_api_config()

    body: dict[str, object] = {"model": model, "file": file}
    if start_page_id is not None:
        body["start_page_id"] = start_page_id
    if end_page_id is not None:
        body["end_page_id"] = end_page_id

    url = base_url.rstrip("/") + "/" + _OCR_ENDPOINT
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        resp = urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT)
        raw = resp.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"GLM OCR API returned HTTP {exc.code}: {exc.reason}"
        ) from exc
    except TimeoutError as exc:
        raise RuntimeError("GLM OCR API request timed out. Please retry.") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not reach GLM OCR API: {exc.reason}"
        ) from exc

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"GLM OCR API returned invalid JSON: {exc}"
        ) from exc

    md_results: str = payload.get("md_results", "")
    if not md_results:
        raise RuntimeError("GLM OCR API returned no OCR content.")

    usage = payload.get("usage", {})
    prompt_tokens: int = usage.get("prompt_tokens", 0)
    completion_tokens: int = usage.get("completion_tokens", 0)
    log_usage("glm_ocr", model, prompt_tokens, completion_tokens)

    return md_results


def _prepare_file(file: str) -> str:
    """Normalise the file input to a format accepted by the GLM OCR API.

    - HTTP/HTTPS URL → returned as-is.
    - ``data:`` URI → returned as-is.
    - Local file path (``os.path.isfile`` is True) → read, base64-encoded,
      MIME-prefixed ``data:`` URI.
    - Everything else (bare Base64) → prefixed with
      ``data:application/pdf;base64,``.
    """
    if file.startswith(("http://", "https://", "data:")):
        return file

    if os.path.isfile(file):
        ext = os.path.splitext(file)[1].lower()
        mime = _MIME_MAP.get(ext, "application/pdf")
        with open(file, "rb") as fh:
            encoded = base64.b64encode(fh.read()).decode()
        return f"data:{mime};base64,{encoded}"

    # Bare Base64 — assume PDF
    return f"data:application/pdf;base64,{file}"
