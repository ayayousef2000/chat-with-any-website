"""Steps 1-2: fetch a website and extract its main content."""

import asyncio
import ipaddress
import logging
import re
import socket
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from io import BytesIO
from itertools import islice
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit, urlunsplit

import httpx
import trafilatura
from pypdf import PdfReader

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

PDF_NO_TEXT_MESSAGE = (
    "This PDF has no text that can be selected. It may be a scan or a set of images, which can't be read yet. "
    "Try a different document."
)
PDF_PROTECTED_MESSAGE = "This PDF is protected with a password, so it can't be read. Try a different document."
PDF_UNREADABLE_MESSAGE = "We couldn't read this PDF. The file may be damaged. Try a different document."

_PDF_TYPES = frozenset({"application/pdf", "application/x-pdf"})

# A PDF without a title of its own gets the biggest text of its first page, when it is this much bigger than the body
# text and reads like a title: words, and between these lengths.
TITLE_SIZE_FACTOR = 1.15
MIN_TITLE_CHARS = 8
MAX_TITLE_CHARS = 200

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

# Contact details that the text extraction throws away: they sit in icon buttons, cards and footers, and the number
# or address of a link is only in its ``href``. Every pattern has bounded lengths so a page cannot make it slow.
_CONTACT_LIMIT = 10  # per kind
_CONTACT_SCAN_CHARS = 2_000_000
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]{1,64}@[A-Za-z0-9.-]{1,255}\.[A-Za-z]{2,24}")
_NOT_EMAIL_ENDINGS = frozenset({"png", "jpg", "jpeg", "gif", "svg", "webp", "avif", "ico", "css", "js"})  # logo@2x.png
_LABELLED_PHONE = re.compile(
    r"\b(?P<label>phone|tel|telephone|call|mobile|cell|whatsapp)\b[^\d+(]{0,12}(?P<number>\+?\(?\d[\d\s().-]{5,17}\d)",
    re.IGNORECASE,
)
_PHONE_CHARS = re.compile(r"[+\d\s().-]+")
_DATE = re.compile(r"\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[./-]\d{1,2}[./-]\d{2,4}")
_WHATSAPP_HOSTS = frozenset({"wa.me", "api.whatsapp.com", "web.whatsapp.com"})

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
    contacts: str = ""  # phone, WhatsApp and email lines; added to the text once the page is known to be readable


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


def _read_body(response: httpx.Response, max_bytes: int, pdf_max_bytes: int, deadline: float) -> tuple[str, bytes]:
    """Read an HTML or PDF response within its size limit and the deadline.

    Returns:
        ``("html", body)`` or ``("pdf", body)``. A PDF is only accepted when ``pdf_max_bytes`` is above zero.
    """
    response.raise_for_status()
    content_type = response.headers.get("content-type", "")
    mime = content_type.split(";")[0].strip().lower()
    if "html" in mime:
        kind, limit = "html", max_bytes
    elif mime in _PDF_TYPES and pdf_max_bytes > 0:
        kind, limit = "pdf", pdf_max_bytes
    else:
        raise ExtractionError(_content_type_message(content_type))
    declared = response.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > limit:
        raise FetchError(TOO_LARGE_MESSAGE)  # no need to download what is too large
    body = bytearray()
    for part in response.iter_bytes():
        body.extend(part)
        if len(body) > limit:
            raise FetchError(TOO_LARGE_MESSAGE)
        _check_deadline(deadline)  # a site that sends a few bytes at a time cannot keep the request open
    return kind, bytes(body)


def _host_of(url: str) -> str:
    """The host of an address, for the log: the path and the query of a page can hold private things."""
    return urlsplit(url).hostname or "an unknown host"


def fetch_html(url: str, timeout: float, max_bytes: int) -> bytes:
    """Download a page's HTML, enforcing a size limit, an HTML content type and a total time limit."""
    return fetch_document(url, timeout, max_bytes)[1]


def fetch_document(url: str, timeout: float, max_bytes: int, pdf_max_bytes: int = 0) -> tuple[str, bytes]:
    """Download a page or, when ``pdf_max_bytes`` is above zero, a PDF file.

    Every address, including each redirect, is looked up once, checked and then connected to as checked. The size
    limits and a total time limit apply to the bytes that arrive, whatever the server says.

    Returns:
        ``("html", body)`` or ``("pdf", body)``.
    """
    accept = "text/html,application/xhtml+xml" + (",application/pdf;q=0.9" if pdf_max_bytes > 0 else "")
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
                headers={"User-Agent": USER_AGENT, "Accept": accept},
                trust_env=False,
            ) as client:
                response = _send_pinned(client, target, addresses, deadline)
                try:
                    redirect = _redirect_target(response, target)
                    if redirect is None:
                        return _read_body(response, max_bytes, pdf_max_bytes, deadline)
                finally:
                    response.close()
            current = str(redirect)
        raise FetchError(TOO_MANY_REDIRECTS_MESSAGE)
    except httpx.HTTPStatusError as exc:
        logger.info("%s answered with HTTP %s", _host_of(url), exc.response.status_code)
        raise FetchError(_status_message(exc.response.status_code)) from exc
    except httpx.TimeoutException as exc:
        raise FetchError(TIMEOUT_MESSAGE) from exc
    except (httpx.HTTPError, httpx.InvalidURL) as exc:
        # The library's own wording is for developers; it stays out of the message shown to the user.
        logger.warning("Could not fetch a page from %s: %s", _host_of(url), type(exc).__name__)
        raise FetchError(UNREACHABLE_MESSAGE) from exc


def _phone_digits(value: str) -> str | None:
    """The digits of a phone number, or ``None`` when there are too few or too many for a real number."""
    digits = re.sub(r"\D", "", value)
    return digits if 7 <= len(digits) <= 15 else None


def _link_contact(href: str, shown: str) -> tuple[str, str, str] | None:
    """Read a ``tel:``, ``mailto:`` or WhatsApp link as ``(kind, key, text)``, or ``None`` for any other link."""
    href = href.strip()
    scheme = href.split(":", 1)[0].lower()
    if scheme == "tel":
        number = unquote(href[4:]).split(";")[0].strip()
        digits = _phone_digits(number) if _PHONE_CHARS.fullmatch(number) else None
        return ("Phone", digits, number) if digits else None
    if scheme == "mailto":
        address = unquote(href[7:]).split("?")[0].split(",")[0].strip()
        return ("Email", address.lower(), address) if _EMAIL.fullmatch(address) else None
    parts = urlsplit(href)
    host = (parts.hostname or "").lower()
    if parts.scheme in {"http", "https"} and host in _WHATSAPP_HOSTS:
        candidate = parts.path.strip("/") if host == "wa.me" else (parse_qs(parts.query).get("phone") or [""])[0]
    elif scheme == "whatsapp":
        candidate = (parse_qs(parts.query).get("phone") or [""])[0]
    else:
        return None
    digits = _phone_digits(candidate) if re.fullmatch(r"\+?\d+", candidate) else None
    if not digits:
        return None
    # The number as the page writes it, when the link text is that number; otherwise plain digits.
    written = _PHONE_CHARS.fullmatch(shown) and _phone_digits(shown) == digits
    return "WhatsApp", digits, shown if written else f"+{digits}"


def _contact_details(tree: Any) -> str:
    """Collect phone numbers, WhatsApp numbers and email addresses from links and from the visible text.

    Returns:
        A short Markdown section, or an empty string when the page shows none of them.
    """
    found: dict[str, dict[str, str]] = {"Phone": {}, "WhatsApp": {}, "Email": {}}

    def add(kind: str, key: str, text: str) -> None:
        entries = found[kind]
        if key in entries:
            if entries[key] == f"+{key}":  # only digits so far; prefer the number as the page writes it
                entries[key] = text
        elif len(entries) < _CONTACT_LIMIT:
            entries[key] = text

    for anchor in tree.iter("a"):
        contact = _link_contact(anchor.get("href") or "", " ".join(anchor.text_content().split()))
        if contact:
            add(*contact)

    words = tree.xpath("//text()[not(ancestor::script or ancestor::style or ancestor::noscript)]")
    visible = " ".join(word.strip() for word in words if word.strip())[:_CONTACT_SCAN_CHARS]
    for match in _EMAIL.finditer(visible):
        if match.group().rsplit(".", 1)[-1].lower() not in _NOT_EMAIL_ENDINGS:
            add("Email", match.group().lower(), match.group())
    for match in _LABELLED_PHONE.finditer(visible):
        digits = _phone_digits(match.group("number"))
        if digits and not _DATE.fullmatch(match.group("number").strip()):
            add(
                "WhatsApp" if match.group("label").lower() == "whatsapp" else "Phone",
                digits,
                match.group("number").strip(),
            )

    lines = [f"- {kind}: {text}" for kind, entries in found.items() for text in entries.values()]
    return "## Contact details\n\n" + "\n".join(lines) if lines else ""


def extract_content(html: bytes, url: str) -> ExtractedPage:
    """Keep the main readable content (as Markdown) and drop navigation, ads and footers.

    Contact details are collected separately, because the extraction drops them from icon buttons, cards and footers.
    """
    tree = trafilatura.load_html(html)
    if tree is None:
        raise ExtractionError(UNREADABLE_MESSAGE)

    contacts = _contact_details(tree)  # before the extraction, which changes the tree
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
    return ExtractedPage(url=url, title=title, text=text, contacts=contacts)


def _reflow(text: str) -> str:
    """Join the lines of a PDF page into paragraphs.

    A PDF stores text line by line as it is drawn, so a sentence is cut at the end of every line. Lines are joined
    with a space, a word that was split with a hyphen at the end of a line is joined without one, and a blank line
    stays a paragraph break.
    """
    paragraphs: list[str] = []
    current = ""
    for raw in text.replace("\r", "").split("\n"):
        line = raw.strip()
        if not line:
            if current:
                paragraphs.append(current)
                current = ""
        elif current.endswith("-") and line[:1].islower() and current[-2:-1].isalpha():
            current = current[:-1] + line
        else:
            current = f"{current} {line}" if current else line
    if current:
        paragraphs.append(current)
    return "\n\n".join(paragraphs)


def extract_pdf(data: bytes, url: str, max_pages: int, max_seconds: float = 30.0) -> ExtractedPage:
    """Read the text of a PDF file, a page at a time, each page under its own ``## Page N`` heading.

    The headings let an answer cite a page ("section: Page 12"). Only the first ``max_pages`` pages are read, and
    reading stops after ``max_seconds`` on a slow host. The text says how many pages were read when the document is
    longer, so an answer about a later page is not mistaken for a gap in the page.

    Raises:
        ExtractionError: If the file is not a readable PDF, is protected by a password or has no selectable text.
    """
    if not data[:1024].lstrip().startswith(b"%PDF-"):
        raise ExtractionError(PDF_UNREADABLE_MESSAGE)
    try:
        reader = PdfReader(BytesIO(data), strict=False)
        if reader.is_encrypted and not reader.decrypt(""):
            raise ExtractionError(PDF_PROTECTED_MESSAGE)
        total = len(reader.pages)
        started = time.monotonic()
        read = 0
        sections = []
        first_page: list[tuple[float, str]] = []  # the text of page one with its font sizes, to find a title
        for number, page in enumerate(islice(reader.pages, max_pages), start=1):
            if time.monotonic() - started > max_seconds:
                break
            if number == 1:
                text = _reflow(page.extract_text(visitor_text=_collect_sizes(first_page)) or "")
            else:
                text = _reflow(page.extract_text() or "")
            read = number
            if text:
                sections.append(f"## Page {number}\n\n{text}")
        title = _pdf_title(reader, url, first_page)
    except ExtractionError:
        raise
    except Exception as exc:
        logger.warning("Could not read a PDF from %s: %s", _host_of(url), type(exc).__name__)
        raise ExtractionError(PDF_UNREADABLE_MESSAGE) from exc
    if not sections:
        raise ExtractionError(PDF_NO_TEXT_MESSAGE)
    if read < total:
        sections.append(f"## Note\n\nOnly the first {read} of the {total} pages of this document were read.")
    return ExtractedPage(url=url, title=title, text="\n\n".join(sections))


def _collect_sizes(into: list[tuple[float, str]]) -> Callable[..., None]:
    """A reader of text pieces for pypdf that records every piece of a page with its font size."""

    def visit(text: str, cm: Any, tm: Any, font_dict: Any, font_size: Any) -> None:
        if text.strip():
            into.append((round(float(font_size or 0), 1), text.strip()))

    return visit


def _title_from_text(pieces: list[tuple[float, str]]) -> str:
    """The title of a paper or report: the biggest text on the first page that reads like words.

    The body is the font size with most text. A title is bigger than the body, so sizes are tried from the biggest
    down, and a size is skipped when its text is not made of words, such as a stamp of numbers and brackets
    (``arXiv:1706.03762v7 [cs.CL] 2 Aug 2023``), which an archive prints bigger than the title itself.
    """
    if not pieces:
        return ""
    characters: dict[float, int] = {}
    for size, text in pieces:
        characters[size] = characters.get(size, 0) + len(text)
    body = max(characters, key=lambda size: characters[size])
    for size in sorted((size for size in characters if size > body * TITLE_SIZE_FACTOR), reverse=True):
        candidate = " ".join(" ".join(text for piece_size, text in pieces if piece_size == size).split())
        letters = sum(character.isalpha() for character in candidate)
        words = len(candidate.split())
        if MIN_TITLE_CHARS <= len(candidate) <= MAX_TITLE_CHARS and words >= 2 and letters >= len(candidate) * 0.75:
            return candidate
    return ""


def _pdf_title(reader: PdfReader, url: str, first_page: list[tuple[float, str]]) -> str:
    """The title written into the PDF, else the biggest text of its first page, else the name of the file."""
    title = (reader.metadata.title if reader.metadata else None) or ""
    title = " ".join(str(title).split())
    if title:
        return title
    title = _title_from_text(first_page)
    if title:
        return title
    name = unquote(urlsplit(url).path.rsplit("/", 1)[-1])
    return name or url


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
    *,
    pdf_max_bytes: int = 0,
    pdf_max_pages: int = 100,
    pdf_max_seconds: float = 30.0,
) -> ExtractedPage:
    """Fetch and extract a page, retrying in a headless browser when the static HTML has too little text.

    A link to a PDF file is read as a document when ``pdf_max_bytes`` is above zero.
    """
    page: ExtractedPage | None = None
    error: ExtractionError | None = None
    try:
        kind, body = fetch_document(url, timeout, max_bytes, pdf_max_bytes)
        if kind == "pdf":
            return extract_pdf(
                body, url, pdf_max_pages, pdf_max_seconds
            )  # a document has no JavaScript to run, and no contacts
        page = extract_content(body, url)
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
    # Added only now: a page of nothing but a contact link must still count as having no readable text.
    return replace(page, text=f"{page.text}\n\n{page.contacts}", contacts="") if page.contacts else page


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
