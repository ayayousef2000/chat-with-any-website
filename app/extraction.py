"""Steps 1-2: fetch a website and extract its main content."""

import asyncio
import ipaddress
import logging
import socket
import time
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
TOO_MANY_REDIRECTS_MESSAGE = "That address sends you around in circles. Try the final address of the page."
UNREADABLE_MESSAGE = "We couldn't read this page. Try a different one."
BROWSER_FAILED_MESSAGE = "We couldn't load this page. Try again in a moment."
BROWSER_UNAVAILABLE_MESSAGE = "This page needs a feature that isn't available right now. Try a different page."

# A page with less text than this has nothing to answer from. It is typically an empty shell that fills in its
# content with JavaScript: such pages measure a few dozen characters, while the smallest real pages (like
# example.com) still have well over this.
MIN_READABLE_CHARS = 100

# Limits of one download: redirects followed, addresses tried for one name, and the whole time as a multiple of the
# timeout of a single step (which a site sending a few bytes at a time would otherwise stretch without end).
MAX_REDIRECTS = 5
MAX_ADDRESSES_TRIED = 4
TOTAL_TIME_FACTOR = 2

_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
# Addresses in this range stand for IPv4 addresses (RFC 6052); 64:ff9b::a00:1 reaches 10.0.0.1 through a NAT64 gateway.
_NAT64_PREFIX = ipaddress.ip_network("64:ff9b::/96")

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


def _is_public(address: str) -> bool:
    """Tell whether an address belongs to the public internet, also when it wraps an IPv4 address.

    An IPv6 address can carry an IPv4 one (IPv4-mapped, 6to4 and the NAT64 prefix); the wrapped address decides.
    """
    ip = ipaddress.ip_address(address.split("%")[0])
    if isinstance(ip, ipaddress.IPv6Address):
        embedded = ip.ipv4_mapped or ip.sixtofour
        if embedded is None and ip in _NAT64_PREFIX:
            embedded = ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
        if embedded is not None and not embedded.is_global:
            return False
    return ip.is_global


def _resolve_public_addresses(host: str) -> list[str]:
    """Look the host up once and return its addresses, refusing hosts that have any non-public address."""
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise InvalidURLError(f"We couldn't find a website at {host}. Check the spelling and try again.") from exc
    addresses = list(dict.fromkeys(str(info[4][0]) for info in infos))
    if not addresses or not all(_is_public(address) for address in addresses):
        raise InvalidURLError(PRIVATE_ADDRESS_MESSAGE)
    return addresses


def _check_deadline(deadline: float) -> None:
    if time.monotonic() > deadline:
        raise FetchError(TIMEOUT_MESSAGE)


def _send_pinned(client: httpx.Client, target: httpx.URL, addresses: list[str], deadline: float) -> httpx.Response:
    """Connect to the address that was checked, not to a fresh lookup of the name.

    The name is only used for the Host header and for the TLS check, so a second answer of the name server
    (DNS rebinding) cannot lead the connection to a private address.
    """
    host = target.raw_host.decode("ascii")
    failure: httpx.TransportError | None = None
    for address in addresses[:MAX_ADDRESSES_TRIED]:
        _check_deadline(deadline)
        request = client.build_request(
            "GET",
            target.copy_with(host=address),
            headers={"Host": target.netloc.decode("ascii")},
            extensions={"sni_hostname": host},
        )
        try:
            return client.send(request, stream=True)
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            failure = exc
    if failure is None:  # pragma: no cover - the caller passes at least one address
        raise AssertionError("no address to connect to")
    raise failure


def _redirect_target(response: httpx.Response, current: httpx.URL) -> httpx.URL | None:
    """The address a redirect points to, or ``None`` when the response is not a redirect."""
    location = response.headers.get("location")
    if response.status_code not in _REDIRECT_STATUSES or not location:
        return None
    target = current.join(location)
    return target.copy_with(fragment=None)


def _read_body(response: httpx.Response, max_bytes: int, deadline: float) -> bytes:
    """Read an HTML response within the size limit and the deadline."""
    response.raise_for_status()
    content_type = response.headers.get("content-type", "")
    if "html" not in content_type:
        raise ExtractionError(_content_type_message(content_type))
    body = bytearray()
    for part in response.iter_bytes():
        body.extend(part)
        if len(body) > max_bytes:
            raise FetchError(TOO_LARGE_MESSAGE)
        _check_deadline(deadline)  # a site that sends a few bytes at a time cannot keep the request open
    return bytes(body)


def fetch_html(url: str, timeout: float, max_bytes: int) -> bytes:
    """Download a page's HTML, enforcing a size limit, an HTML content type and a total time limit.

    Every address, including each redirect, is looked up once, checked and then connected to as checked.
    """
    deadline = time.monotonic() + timeout * TOTAL_TIME_FACTOR
    current = url
    try:
        for _ in range(MAX_REDIRECTS + 1):
            _check_deadline(deadline)
            target = httpx.URL(current)
            if target.scheme not in {"http", "https"}:
                raise FetchError(UNREACHABLE_MESSAGE)
            addresses = _resolve_public_addresses(target.raw_host.decode("ascii"))
            # One client per address: a pooled connection is never reused for another name; the environment's proxy
            # settings are ignored because a proxy would do the connecting.
            with httpx.Client(
                timeout=timeout,
                headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"},
                trust_env=False,
            ) as client:
                response = _send_pinned(client, target, addresses, deadline)
                try:
                    redirect = _redirect_target(response, target)
                    if redirect is None:
                        return _read_body(response, max_bytes, deadline)
                finally:
                    response.close()
            current = str(redirect)
        raise FetchError(TOO_MANY_REDIRECTS_MESSAGE)
    except httpx.HTTPStatusError as exc:
        logger.info("%s answered with HTTP %s", url, exc.response.status_code)
        raise FetchError(_status_message(exc.response.status_code)) from exc
    except httpx.TimeoutException as exc:
        raise FetchError(TIMEOUT_MESSAGE) from exc
    except (httpx.HTTPError, httpx.InvalidURL) as exc:
        # The library's own wording is for developers; it stays out of the message shown to the user.
        logger.warning("Could not fetch %s: %s", url, exc)
        raise FetchError(UNREACHABLE_MESSAGE) from exc


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
