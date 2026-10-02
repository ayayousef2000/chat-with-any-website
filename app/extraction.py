"""Steps 1-2: fetch a website and extract its main content."""

import asyncio
import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

import httpx
import trafilatura

from app.errors import ExtractionError, FetchError, InvalidURLError

USER_AGENT = "Mozilla/5.0 (compatible; ChatWithAnyWebsite/0.1)"


@dataclass(frozen=True)
class ExtractedPage:
    """The main content of a web page."""

    url: str
    title: str
    text: str


def normalize_url(url: str) -> str:
    """Validate a user-supplied URL and drop its fragment so it can be used as a stable key.

    Args:
        url: The address as typed by the user. A missing scheme is treated as https.

    Returns:
        The URL with a lowercase host, a path of at least ``/`` and no fragment.

    Raises:
        InvalidURLError: If the URL is not a valid http(s) address.
    """
    url = url.strip()
    if "://" not in url:
        url = f"https://{url}"
    try:
        parts = urlsplit(url)
        # Accessing .port validates it; a non-numeric port raises ValueError.
        valid = parts.scheme in {"http", "https"} and bool(parts.hostname) and parts.port != 0
    except ValueError:
        valid = False
    if not valid:
        raise InvalidURLError("Please enter a valid http(s) URL.")
    return urlunsplit((parts.scheme, parts.netloc.lower(), parts.path or "/", parts.query, ""))


def _assert_public_host(url: str) -> None:
    """Refuse URLs that resolve to private, loopback or otherwise non-public addresses."""
    host = urlsplit(url).hostname
    if not host:
        raise InvalidURLError("Please enter a valid http(s) URL.")
    try:
        addresses = {str(info[4][0]) for info in socket.getaddrinfo(host, None)}
    except socket.gaierror as exc:
        raise InvalidURLError(f"Could not resolve host '{host}'.") from exc
    for address in addresses:
        if not ipaddress.ip_address(address.split("%")[0]).is_global:
            raise InvalidURLError("URLs pointing to private or local network addresses are not allowed.")


def _check_request(request: httpx.Request) -> None:
    # Runs for the first request and for every redirect hop.
    _assert_public_host(str(request.url))


def fetch_html(url: str, timeout: float, max_bytes: int) -> bytes:
    """Download a page's HTML, enforcing a size limit and an HTML content type."""
    client = httpx.Client(
        follow_redirects=True,
        timeout=timeout,
        headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"},
        event_hooks={"request": [_check_request]},
    )
    try:
        with client, client.stream("GET", url) as response:
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            if "html" not in content_type:
                raise ExtractionError(f"Expected an HTML page but got '{content_type or 'unknown'}'.")
            body = bytearray()
            for part in response.iter_bytes():
                body.extend(part)
                if len(body) > max_bytes:
                    raise FetchError("The page is too large to process.")
    except httpx.HTTPStatusError as exc:
        raise FetchError(f"The website responded with HTTP {exc.response.status_code}.") from exc
    except httpx.HTTPError as exc:
        raise FetchError(f"Could not fetch the page: {exc}") from exc
    return bytes(body)


def extract_content(html: bytes, url: str) -> ExtractedPage:
    """Keep the main readable content (as Markdown) and drop navigation, ads and footers."""
    tree = trafilatura.load_html(html)
    if tree is None:
        raise ExtractionError("Could not parse the page's HTML.")

    metadata = trafilatura.extract_metadata(tree, default_url=url)
    text = trafilatura.extract(
        tree,
        url=url,
        output_format="markdown",
        include_tables=True,
        include_comments=False,
        include_links=False,
        favor_recall=True,
    )
    if not text or not text.strip():
        raise ExtractionError(
            "No readable text was found on this page. It may require JavaScript or block automated access."
        )

    title = metadata.title if metadata and metadata.title else url
    return ExtractedPage(url=url, title=title, text=text)


def render_html(url: str, timeout: float) -> bytes:
    """Load a page in a headless browser so JavaScript-generated content is included (needs the "browser" extra)."""
    try:
        from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
    except ImportError as exc:
        raise FetchError(
            "Browser fallback is enabled but crawl4ai is not installed. "
            "Run `uv sync --extra browser` and then `uv run crawl4ai-setup`."
        ) from exc

    async def _render() -> str | None:
        run_config = CrawlerRunConfig(wait_until="networkidle", page_timeout=int(timeout * 1000))
        async with AsyncWebCrawler(config=BrowserConfig(headless=True, verbose=False)) as crawler:
            result = await crawler.arun(url=url, config=run_config)
        return result.html if result.success else None

    # Called from a worker thread, which has no running event loop.
    html = asyncio.run(_render())
    if not html:
        raise FetchError("The browser could not load the page.")
    return html.encode("utf-8")


def load_page(
    url: str,
    timeout: float,
    max_bytes: int,
    browser_fallback: bool,
    min_text_chars: int,
) -> ExtractedPage:
    """Fetch and extract a page, retrying in a headless browser when the static HTML has too little text."""
    page: ExtractedPage | None = None
    error: ExtractionError | None = None
    try:
        page = extract_content(fetch_html(url, timeout, max_bytes), url)
    except ExtractionError as exc:
        error = exc

    if browser_fallback and (page is None or len(page.text) < min_text_chars):
        rendered = extract_content_or_none(render_html(url, timeout), url)
        if rendered and (page is None or len(rendered.text) > len(page.text)):
            page = rendered

    if page is None:
        raise error or ExtractionError("No readable text was found on this page.")
    return page


def extract_content_or_none(html: bytes, url: str) -> ExtractedPage | None:
    """Extract the main content of a page, returning ``None`` when it has no readable text.

    Args:
        html: The page's HTML.
        url: The page's URL, used as a hint for extraction.

    Returns:
        The extracted page, or ``None`` if nothing readable was found.
    """
    try:
        return extract_content(html, url)
    except ExtractionError:
        return None
