"""Tests for glm_ocr tool."""
import base64
import json
import urllib.error
from unittest.mock import MagicMock, mock_open, patch

import pytest

_API_KEY = "test-api-key"
_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
_SAMPLE_MD = "# Document\n\nParagraph text here."


def _make_urlopen_mock(
    md_results: str = _SAMPLE_MD,
    prompt_tokens: int = 50,
    completion_tokens: int = 100,
) -> MagicMock:
    """Return a mock for urllib.request.urlopen with a canned OCR response."""
    response_data = json.dumps({
        "md_results": md_results,
        "layout_details": [],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
    }).encode()

    mock_resp = MagicMock()
    mock_resp.read.return_value = response_data
    return MagicMock(return_value=mock_resp)


def test_ocr_01_url_returns_str():
    """UT-OCR-01: HTTPS URL input returns the markdown string from md_results."""
    mock_urlopen = _make_urlopen_mock()

    with patch("glm_mcp.tools.ocr.get_api_config", return_value=(_API_KEY, _BASE_URL)), \
         patch("urllib.request.urlopen", mock_urlopen):
        from glm_mcp.tools.ocr import glm_ocr
        result = glm_ocr("https://example.com/doc.pdf")

    assert isinstance(result, str)
    assert result == _SAMPLE_MD


def test_ocr_02_base64_returns_str():
    """UT-OCR-02: data: URI input returns the markdown string from md_results."""
    b64_data = base64.b64encode(b"fake pdf content").decode()
    data_uri = f"data:application/pdf;base64,{b64_data}"
    mock_urlopen = _make_urlopen_mock()

    with patch("glm_mcp.tools.ocr.get_api_config", return_value=(_API_KEY, _BASE_URL)), \
         patch("urllib.request.urlopen", mock_urlopen):
        from glm_mcp.tools.ocr import glm_ocr
        result = glm_ocr(data_uri)

    assert isinstance(result, str)
    assert result == _SAMPLE_MD


def test_ocr_03_local_file_auto_encodes():
    """UT-OCR-03: Local file path is read, base64-encoded, and sent with data: prefix."""
    fake_content = b"PDF content here"
    mock_urlopen = _make_urlopen_mock()

    with patch("glm_mcp.tools.ocr.get_api_config", return_value=(_API_KEY, _BASE_URL)), \
         patch("urllib.request.urlopen", mock_urlopen), \
         patch("os.path.isfile", return_value=True), \
         patch("builtins.open", mock_open(read_data=fake_content)):
        from glm_mcp.tools.ocr import glm_ocr
        result = glm_ocr("/tmp/test.pdf")

    assert result == _SAMPLE_MD
    req = mock_urlopen.call_args[0][0]
    body = json.loads(req.data)
    expected_b64 = base64.b64encode(fake_content).decode()
    assert body["file"] == f"data:application/pdf;base64,{expected_b64}"


def test_ocr_04_default_model():
    """UT-OCR-04: Default model sent to API is 'glm-ocr'."""
    mock_urlopen = _make_urlopen_mock()

    with patch("glm_mcp.tools.ocr.get_api_config", return_value=(_API_KEY, _BASE_URL)), \
         patch("urllib.request.urlopen", mock_urlopen):
        from glm_mcp.tools.ocr import glm_ocr
        glm_ocr("https://example.com/doc.pdf")

    req = mock_urlopen.call_args[0][0]
    body = json.loads(req.data)
    assert body["model"] == "glm-ocr"


def test_ocr_05_custom_model():
    """UT-OCR-05: Custom model parameter is passed to the API."""
    mock_urlopen = _make_urlopen_mock()

    with patch("glm_mcp.tools.ocr.get_api_config", return_value=(_API_KEY, _BASE_URL)), \
         patch("urllib.request.urlopen", mock_urlopen):
        from glm_mcp.tools.ocr import glm_ocr
        glm_ocr("https://example.com/doc.pdf", model="glm-ocr-v2")

    req = mock_urlopen.call_args[0][0]
    body = json.loads(req.data)
    assert body["model"] == "glm-ocr-v2"


def test_ocr_06_usage_logged():
    """UT-OCR-06: log_usage called with tool='glm_ocr' and correct token counts."""
    mock_urlopen = _make_urlopen_mock(prompt_tokens=50, completion_tokens=100)

    with patch("glm_mcp.tools.ocr.get_api_config", return_value=(_API_KEY, _BASE_URL)), \
         patch("urllib.request.urlopen", mock_urlopen), \
         patch("glm_mcp.tools.ocr.log_usage") as mock_log:
        from glm_mcp.tools.ocr import glm_ocr
        glm_ocr("https://example.com/doc.pdf", model="glm-ocr")

    mock_log.assert_called_once_with("glm_ocr", "glm-ocr", 50, 100)


def test_ocr_07_pagination_params():
    """UT-OCR-07: start_page_id and end_page_id are included in the request body."""
    mock_urlopen = _make_urlopen_mock()

    with patch("glm_mcp.tools.ocr.get_api_config", return_value=(_API_KEY, _BASE_URL)), \
         patch("urllib.request.urlopen", mock_urlopen):
        from glm_mcp.tools.ocr import glm_ocr
        glm_ocr("https://example.com/doc.pdf", start_page_id=2, end_page_id=5)

    req = mock_urlopen.call_args[0][0]
    body = json.loads(req.data)
    assert body["start_page_id"] == 2
    assert body["end_page_id"] == 5


def test_ocr_08_empty_file_raises():
    """UT-OCR-08: Empty file string raises ValueError."""
    with patch("glm_mcp.tools.ocr.get_api_config", return_value=(_API_KEY, _BASE_URL)):
        from glm_mcp.tools.ocr import glm_ocr
        with pytest.raises(ValueError, match="file"):
            glm_ocr("")


def test_ocr_09_empty_md_results_raises():
    """UT-OCR-09: Response with empty md_results raises RuntimeError."""
    mock_urlopen = _make_urlopen_mock(md_results="")

    with patch("glm_mcp.tools.ocr.get_api_config", return_value=(_API_KEY, _BASE_URL)), \
         patch("urllib.request.urlopen", mock_urlopen):
        from glm_mcp.tools.ocr import glm_ocr
        with pytest.raises(RuntimeError, match="no OCR content"):
            glm_ocr("https://example.com/doc.pdf")


def test_ocr_10_http_error_raises():
    """UT-OCR-10: HTTP error (e.g. 429) raises RuntimeError mentioning the status code."""
    mock_urlopen = MagicMock()
    mock_urlopen.side_effect = urllib.error.HTTPError(
        url="https://open.bigmodel.cn/api/paas/v4/layout_parsing",
        code=429,
        msg="Too Many Requests",
        hdrs=None,  # type: ignore[arg-type]
        fp=None,
    )

    with patch("glm_mcp.tools.ocr.get_api_config", return_value=(_API_KEY, _BASE_URL)), \
         patch("urllib.request.urlopen", mock_urlopen):
        from glm_mcp.tools.ocr import glm_ocr
        with pytest.raises(RuntimeError, match="429"):
            glm_ocr("https://example.com/doc.pdf")


def test_ocr_11_timeout_raises():
    """UT-OCR-11: Request timeout raises RuntimeError."""
    mock_urlopen = MagicMock()
    mock_urlopen.side_effect = TimeoutError("timed out")

    with patch("glm_mcp.tools.ocr.get_api_config", return_value=(_API_KEY, _BASE_URL)), \
         patch("urllib.request.urlopen", mock_urlopen):
        from glm_mcp.tools.ocr import glm_ocr
        with pytest.raises(RuntimeError, match="timed out"):
            glm_ocr("https://example.com/doc.pdf")


def test_ocr_12_connection_error_raises():
    """UT-OCR-12: Network unreachable raises RuntimeError."""
    mock_urlopen = MagicMock()
    mock_urlopen.side_effect = urllib.error.URLError(reason="Name or service not known")

    with patch("glm_mcp.tools.ocr.get_api_config", return_value=(_API_KEY, _BASE_URL)), \
         patch("urllib.request.urlopen", mock_urlopen):
        from glm_mcp.tools.ocr import glm_ocr
        with pytest.raises(RuntimeError, match="Could not reach"):
            glm_ocr("https://example.com/doc.pdf")


def test_ocr_13_registered_in_server():
    """UT-OCR-13: glm_ocr is registered in the MCP server."""
    import asyncio

    from glm_mcp.server import mcp

    assert asyncio.run(mcp.get_tool("glm_ocr")) is not None


def test_ocr_14_bare_base64_auto_prefix():
    """UT-OCR-14: Bare base64 string (no data: prefix) gets data:application/pdf;base64, prefix."""
    b64_data = base64.b64encode(b"fake pdf content").decode()
    mock_urlopen = _make_urlopen_mock()

    with patch("glm_mcp.tools.ocr.get_api_config", return_value=(_API_KEY, _BASE_URL)), \
         patch("urllib.request.urlopen", mock_urlopen):
        from glm_mcp.tools.ocr import glm_ocr
        glm_ocr(b64_data)

    req = mock_urlopen.call_args[0][0]
    body = json.loads(req.data)
    assert body["file"] == f"data:application/pdf;base64,{b64_data}"


def test_ocr_15_data_uri_passthrough():
    """UT-OCR-15: data: URI is passed through to the API unchanged."""
    b64_data = base64.b64encode(b"fake png content").decode()
    data_uri = f"data:image/png;base64,{b64_data}"
    mock_urlopen = _make_urlopen_mock()

    with patch("glm_mcp.tools.ocr.get_api_config", return_value=(_API_KEY, _BASE_URL)), \
         patch("urllib.request.urlopen", mock_urlopen):
        from glm_mcp.tools.ocr import glm_ocr
        glm_ocr(data_uri)

    req = mock_urlopen.call_args[0][0]
    body = json.loads(req.data)
    assert body["file"] == data_uri
