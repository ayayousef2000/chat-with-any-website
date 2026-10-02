import socket
from typing import Any

import pytest

import app.extraction as extraction
from app.errors import ExtractionError, FetchError, InvalidURLError
from app.extraction import ExtractedPage, extract_content, load_page, normalize_url

ARTICLE = b"""<html><head><title>Sample Article</title></head><body>
<nav><a href="/">Home</a><a href="/about">About</a></nav>
<article><h1>Sample Article</h1>
<p>Retrieval systems find relevant passages before a model writes an answer. This paragraph is long enough
for the extractor to treat it as real content rather than boilerplate on the page.</p>
<p>A second paragraph continues the article with more detail about how chunking and embeddings work together
in a pipeline that answers questions from web pages.</p></article>
<footer>Copyright 2026 Example Corp. All rights reserved.</footer></body></html>"""


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("example.com", "https://example.com/"),
        ("  https://Example.COM/Path?q=1#frag ", "https://example.com/Path?q=1"),
        ("http://example.com", "http://example.com/"),
    ],
)
def test_normalize_url(raw: str, expected: str) -> None:
    assert normalize_url(raw) == expected


@pytest.mark.parametrize("raw", ["ftp://example.com", "javascript:alert(1)", "https://", "", "http://host:notaport/"])
def test_normalize_url_rejects_invalid(raw: str) -> None:
    with pytest.raises(InvalidURLError):
        normalize_url(raw)


def _fake_dns(*addresses: str) -> Any:
    def getaddrinfo(host: str, port: Any) -> list[tuple[Any, ...]]:
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 0)) for address in addresses]

    return getaddrinfo


@pytest.mark.parametrize("address", ["127.0.0.1", "10.0.0.5", "192.168.1.1", "169.254.169.254", "::1"])
def test_private_addresses_are_blocked(monkeypatch: pytest.MonkeyPatch, address: str) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", _fake_dns(address))
    with pytest.raises(InvalidURLError, match="private or local"):
        extraction._assert_public_host("https://internal.example/")


def test_public_address_is_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", _fake_dns("93.184.216.34"))
    extraction._assert_public_host("https://example.com/")


def test_host_with_any_private_address_is_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", _fake_dns("93.184.216.34", "10.0.0.1"))
    with pytest.raises(InvalidURLError):
        extraction._assert_public_host("https://mixed.example/")


def test_unresolvable_host_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    def getaddrinfo(host: str, port: Any) -> list[tuple[Any, ...]]:
        raise socket.gaierror("no such host")

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    with pytest.raises(InvalidURLError, match="resolve"):
        extraction._assert_public_host("https://nope.invalid/")


def test_extract_content_keeps_article_and_drops_boilerplate() -> None:
    page = extract_content(ARTICLE, "https://example.com/a")
    assert page.title == "Sample Article"
    assert "Retrieval systems find relevant passages" in page.text
    assert "second paragraph" in page.text
    assert "Copyright 2026" not in page.text


def test_extract_content_without_text_raises() -> None:
    with pytest.raises(ExtractionError):
        extract_content(b"<html><body><script>var x = 1;</script></body></html>", "https://example.com/")


def _stub(monkeypatch: pytest.MonkeyPatch, static_len: int | None, rendered_len: int | None) -> list[str]:
    """Stub fetching and extraction; ``None`` means the stage yields no text. Returns a log of browser calls."""
    browser_calls: list[str] = []

    def fake_extract(html: bytes, url: str) -> ExtractedPage:
        length = static_len if html == b"static" else rendered_len
        if length is None:
            raise ExtractionError("no text")
        return ExtractedPage(url=url, title="t", text="x" * length)

    def fake_render(url: str, timeout: float) -> bytes:
        browser_calls.append(url)
        return b"rendered"

    monkeypatch.setattr(extraction, "fetch_html", lambda *args: b"static")
    monkeypatch.setattr(extraction, "extract_content", fake_extract)
    monkeypatch.setattr(extraction, "render_html", fake_render)
    return browser_calls


def test_load_page_skips_browser_when_text_is_sufficient(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub(monkeypatch, 900, 5000)
    assert len(load_page("u", 1, 1, True, 500).text) == 900
    assert calls == []


def test_load_page_uses_browser_for_thin_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub(monkeypatch, 100, 5000)
    assert len(load_page("u", 1, 1, True, 500).text) == 5000
    assert calls == ["u"]


def test_load_page_uses_browser_when_static_has_no_text(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub(monkeypatch, None, 5000)
    assert len(load_page("u", 1, 1, True, 500).text) == 5000


def test_load_page_keeps_static_text_if_browser_is_worse(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub(monkeypatch, 300, 100)
    assert len(load_page("u", 1, 1, True, 500).text) == 300


def test_load_page_without_fallback_raises_for_empty_page(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub(monkeypatch, None, 5000)
    with pytest.raises(ExtractionError):
        load_page("u", 1, 1, False, 500)
    assert calls == []


def test_load_page_raises_when_browser_finds_nothing_either(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub(monkeypatch, None, None)
    with pytest.raises(ExtractionError):
        load_page("u", 1, 1, True, 500)


def test_fetch_errors_do_not_trigger_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    def failing_fetch(*args: Any) -> bytes:
        raise FetchError("HTTP 500")

    monkeypatch.setattr(extraction, "fetch_html", failing_fetch)
    with pytest.raises(FetchError):
        load_page("u", 1, 1, True, 500)
