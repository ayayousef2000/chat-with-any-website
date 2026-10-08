"""Steps 1-2: fetch a website and extract its main content."""

import asyncio
import ipaddress
import logging
import socket
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

import httpx
import trafilatura

from app.errors import ExtractionError, FetchError, InvalidURLError

logger = logging.getLogger(__name__)

USER_AGENT = "Mozilla/5.0 (compatible; ChatWithAnyWebsite/0.1)"

# Messages shown to the user: plain language, what happened, and what to do next.
INVALID_URL_MESSAGE = "That doesn't look like a web address. Enter a link that starts with https://."
PRIVATE_ADDRESS_MESSAGE = (
    "That address is on a private network, so it can't be loaded. Enter the address of a public web page."
)
NO_TEXT_MESSAGE = (
    "We couldn't find readable text on this page. It may load its content with JavaScript or block automated "
    "tools. Try a different page."
)
TIMEOUT_MESSAGE = "The website took too long to respond. Try again in a moment."
UNREACHABLE_MESSAGE = "We couldn't reach that website. Check the address and try again."
TOO_LARGE_MESSAGE = "This page is too large to load. Try a shorter page."
UNREADABLE_MESSAGE = "We couldn't read this page. Try a different one."
BROWSER_FAILED_MESSAGE = "We couldn't load this page. Try again in a moment."
BROWSER_UNAVAILABLE_MESSAGE = "This page needs a feature that isn't available right now. Try a different page."

# A page with less text than this has nothing to answer from. It is typically an empty shell that fills in its
# content with JavaScript: such pages measure a few dozen characters, while the smallest real pages (like
# example.com) still have well over this.
MIN_READABLE_CHARS = 100

_FILE_KINDS = {"image": "an image", "video": "a video", "audio": "an audio file"}


def _status_message(status_code: int) -> str:
    """Describe a failing HTTP status of the requested website in plain words.

    The number is left out when the wording already explains the cause; it is only shown for statuses that have
    no wording of their own, so that a visitor has something to quote when asking for help.
    """
    if status_code in {401, 403}:
        return "The website doesn't allow access to this page. Try a different page."
    if status_code in {404, 410}:
        return "We couldn't find that page. Check the address and try again."
    if status_code == 429:
        return "The website is limiting requests right now. Try again in a moment."
    if status_code >= 500:
        return "The website is having problems right now. Try again later."
    return f"The website couldn't load this page (error {status_code}). Try a different page."


def _content_type_message(content_type: str) -> str:
    """Explain that a link leads to something other than a web page, naming what it is when known."""
    mime = content_type.split(";")[0].strip().lower()
    if mime == "application/pdf":
        return "That link is a PDF, and PDF files aren't supported yet. Enter the address of a web page."
    kind = _FILE_KINDS.get(mime.split("/")[0], "a file")
    return f"That link is {kind}, not a web page. Enter the address of an article or documentation page."


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
        raise InvalidURLError(INVALID_URL_MESSAGE)
    return urlunsplit((parts.scheme, parts.netloc.lower(), parts.path or "/", parts.query, ""))


def _assert_public_host(url: str) -> None:
    """Refuse URLs that resolve to private, loopback or otherwise non-public addresses."""
    host = urlsplit(url).hostname
    if not host:
        raise InvalidURLError(INVALID_URL_MESSAGE)
    try:
        addresses = {str(info[4][0]) for info in socket.getaddrinfo(host, None)}
    except socket.gaierror as exc:
        raise InvalidURLError(f"We couldn't find a website at {host}. Check the spelling and try again.") from exc
    for address in addresses:
        if not ipaddress.ip_address(address.split("%")[0]).is_global:
            raise InvalidURLError(PRIVATE_ADDRESS_MESSAGE)


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
                raise ExtractionError(_content_type_message(content_type))
            body = bytearray()
            for part in response.iter_bytes():
                body.extend(part)
                if len(body) > max_bytes:
                    raise FetchError(TOO_LARGE_MESSAGE)
    except httpx.HTTPStatusError as exc:
        logger.info("%s answered with HTTP %s", url, exc.response.status_code)
        raise FetchError(_status_message(exc.response.status_code)) from exc
    except httpx.TimeoutException as exc:
        raise FetchError(TIMEOUT_MESSAGE) from exc
    except httpx.HTTPError as exc:
        # The library's own wording is for developers; it stays out of the message shown to the user.
        logger.warning("Could not fetch %s: %s", url, exc)
        raise FetchError(UNREACHABLE_MESSAGE) from exc
    return bytes(body)


def extract_content(html: bytes, url: str) -> ExtractedPage:
    """Keep the main readable content (as Markdown) and drop navigation, ads and footers."""
    tree = trafilatura.load_html(html)
    if tree is None:
        raise ExtractionError(UNREADABLE_MESSAGE)

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
        raise ExtractionError(NO_TEXT_MESSAGE)

    title = metadata.title if metadata and metadata.title else url
    return ExtractedPage(url=url, title=title, text=text)


def render_html(url: str, timeout: float) -> bytes:
    """Load a page in a headless browser so JavaScript-generated content is included (needs the "browser" extra)."""
    try:
        from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
    except ImportError as exc:
        # The setup hint is for whoever runs the server, so it goes to the log, not to the user.
        logger.error(
            "Browser fallback is enabled but crawl4ai is not installed. "
            "Run `uv sync --extra browser` and then `uv run crawl4ai-setup`."
        )
        raise FetchError(BROWSER_UNAVAILABLE_MESSAGE) from exc

    async def _render() -> str | None:
        run_config = CrawlerRunConfig(wait_until="networkidle", page_timeout=int(timeout * 1000))
        async with AsyncWebCrawler(config=BrowserConfig(headless=True, verbose=False)) as crawler:
            result = await crawler.arun(url=url, config=run_config)
        return result.html if result.success else None

    # Called from a worker thread, which has no running event loop.
    html = asyncio.run(_render())
    if not html:
        raise FetchError(BROWSER_FAILED_MESSAGE)
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
        raise error or ExtractionError(NO_TEXT_MESSAGE)
    if len(page.text.strip()) < MIN_READABLE_CHARS:
        raise ExtractionError(NO_TEXT_MESSAGE)
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
