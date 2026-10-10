"""Tests that the downloader connects to the address it checked, on the first request and on every redirect."""

import socket
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

import app.extraction as extraction
from app.errors import FetchError, InvalidURLError

PUBLIC = "93.184.216.34"
PAGE_BODY = b"<html>ok</html>"


def _page() -> httpx.Response:
    return httpx.Response(200, headers={"content-type": "text/html"}, content=PAGE_BODY)


def _answers(monkeypatch: pytest.MonkeyPatch, by_host: dict[str, list[str]]) -> list[str]:
    """Make each name resolve to the listed addresses, in order, and record every lookup."""
    lookups: list[str] = []

    def getaddrinfo(host: str, port: Any) -> list[tuple[Any, ...]]:
        lookups.append(host)
        addresses = by_host[host]
        family = socket.AF_INET6 if ":" in addresses[0] else socket.AF_INET
        return [(family, socket.SOCK_STREAM, 6, "", (address, 0)) for address in addresses]

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    return lookups


def _network(
    monkeypatch: pytest.MonkeyPatch, handler: Callable[[httpx.Request], httpx.Response]
) -> list[httpx.Request]:
    """Make the downloader talk to ``handler``; returns the requests it received."""
    received: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        received.append(request)
        return handler(request)

    real_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: real_client(transport=httpx.MockTransport(record), **kwargs))
    return received


def test_the_connection_uses_the_address_that_was_checked(monkeypatch: pytest.MonkeyPatch) -> None:
    lookups = _answers(monkeypatch, {"example.com": [PUBLIC]})
    received = _network(monkeypatch, lambda request: _page())

    assert extraction.fetch_html("https://example.com/a?b=1", 5, 1000) == PAGE_BODY

    assert lookups == ["example.com"]  # looked up once; the connection does not look the name up again
    assert received[0].url.host == PUBLIC
    assert received[0].headers["host"] == "example.com"
    assert received[0].extensions["sni_hostname"] == "example.com"  # the certificate is checked for the name
    assert received[0].url.path == "/a"
    assert received[0].url.query == b"b=1"


def test_a_name_that_changes_its_answer_cannot_reach_a_private_address(monkeypatch: pytest.MonkeyPatch) -> None:
    """DNS rebinding: a public answer for the check and a private one for the connection."""
    answers = iter([[PUBLIC], ["127.0.0.1"]])

    def getaddrinfo(host: str, port: Any) -> list[tuple[Any, ...]]:
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 0)) for address in next(answers)]

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    received = _network(monkeypatch, lambda request: _page())

    extraction.fetch_html("http://attacker.example/", 5, 1000)

    assert [request.url.host for request in received] == [PUBLIC]


def test_ipv6_addresses_are_connected_to_with_the_port_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    _answers(monkeypatch, {"v6.example": ["2606:4700::1111"]})
    received = _network(monkeypatch, lambda request: _page())

    extraction.fetch_html("https://v6.example:8443/", 5, 1000)

    assert received[0].url.host == "2606:4700::1111"
    assert received[0].url.port == 8443
    assert received[0].headers["host"] == "v6.example:8443"


def test_a_redirect_is_followed_and_checked_on_its_own(monkeypatch: pytest.MonkeyPatch) -> None:
    lookups = _answers(monkeypatch, {"old.example": [PUBLIC], "new.example": ["93.184.216.35"]})

    def handler(request: httpx.Request) -> httpx.Response:
        if request.headers["host"] == "old.example":
            return httpx.Response(301, headers={"location": "https://new.example/moved#part"})
        return _page()

    received = _network(monkeypatch, handler)

    extraction.fetch_html("https://old.example/", 5, 1000)

    assert [(r.url.host, r.headers["host"], r.url.path) for r in received] == [
        (PUBLIC, "old.example", "/"),
        ("93.184.216.35", "new.example", "/moved"),
    ]
    assert lookups == ["old.example", "new.example"]


def test_a_relative_redirect_stays_on_the_same_host(monkeypatch: pytest.MonkeyPatch) -> None:
    _answers(monkeypatch, {"site.example": [PUBLIC]})

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "/next"}) if request.url.path == "/" else _page()

    received = _network(monkeypatch, handler)

    extraction.fetch_html("https://site.example/", 5, 1000)

    assert [(r.headers["host"], r.url.path) for r in received] == [("site.example", "/"), ("site.example", "/next")]


@pytest.mark.parametrize(
    ("target", "resolves_to"),
    [
        ("http://127.0.0.1/admin", "127.0.0.1"),
        ("http://localhost/", "127.0.0.1"),
        ("http://[::1]/", "::1"),
        ("http://10.0.0.1/", "10.0.0.1"),
        ("http://metadata.example/", "169.254.169.254"),
        ("http://[64:ff9b::a00:1]/", "64:ff9b::a00:1"),
    ],
)
def test_a_redirect_to_a_private_address_is_refused(
    monkeypatch: pytest.MonkeyPatch, target: str, resolves_to: str
) -> None:
    host = httpx.URL(target).host
    _answers(monkeypatch, {"start.example": [PUBLIC], host: [resolves_to]})
    received = _network(monkeypatch, lambda request: httpx.Response(302, headers={"location": target}))

    with pytest.raises(InvalidURLError, match="private network"):
        extraction.fetch_html("https://start.example/", 5, 1000)

    assert [request.headers["host"] for request in received] == ["start.example"]  # the private address got nothing


def test_a_redirect_to_another_protocol_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    _answers(monkeypatch, {"start.example": [PUBLIC]})
    _network(monkeypatch, lambda request: httpx.Response(302, headers={"location": "file:///etc/passwd"}))

    with pytest.raises(FetchError) as caught:
        extraction.fetch_html("https://start.example/", 5, 1000)

    assert str(caught.value) == extraction.UNREACHABLE_MESSAGE


def test_too_many_redirects_end_with_a_plain_message(monkeypatch: pytest.MonkeyPatch) -> None:
    _answers(monkeypatch, {"loop.example": [PUBLIC]})
    received = _network(monkeypatch, lambda request: httpx.Response(302, headers={"location": "/again"}))

    with pytest.raises(FetchError) as caught:
        extraction.fetch_html("https://loop.example/", 5, 1000)

    assert str(caught.value) == extraction.TOO_MANY_REDIRECTS_MESSAGE
    assert len(received) == extraction.MAX_REDIRECTS + 1


def test_the_next_address_is_tried_when_one_cannot_be_reached(monkeypatch: pytest.MonkeyPatch) -> None:
    _answers(monkeypatch, {"two.example": ["2606:4700::1111", PUBLIC]})

    def handler(request: httpx.Request) -> httpx.Response:
        if ":" in request.url.host:
            raise httpx.ConnectError("no route", request=request)
        return _page()

    received = _network(monkeypatch, handler)

    extraction.fetch_html("https://two.example/", 5, 1000)

    assert [request.url.host for request in received] == ["2606:4700::1111", PUBLIC]


def test_a_site_that_sends_a_few_bytes_at_a_time_is_stopped_by_the_total_time(monkeypatch: pytest.MonkeyPatch) -> None:
    _answers(monkeypatch, {"slow.example": [PUBLIC]})
    clock = iter(range(0, 1000, 5))  # every look at the clock is five seconds later
    monkeypatch.setattr(extraction, "time", SimpleNamespace(monotonic=lambda: float(next(clock))))
    _network(
        monkeypatch,
        lambda request: httpx.Response(200, headers={"content-type": "text/html"}, content=iter([b"a"] * 100)),
    )

    with pytest.raises(FetchError) as caught:
        extraction.fetch_html("https://slow.example/", 5, 1_000_000)

    assert str(caught.value) == extraction.TIMEOUT_MESSAGE


def test_proxy_settings_of_the_environment_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[dict[str, Any]] = []
    real_client = httpx.Client

    def client(**kwargs: Any) -> httpx.Client:
        created.append(kwargs)
        return real_client(transport=httpx.MockTransport(lambda request: _page()), **kwargs)

    _answers(monkeypatch, {"example.com": [PUBLIC]})
    monkeypatch.setattr(httpx, "Client", client)

    extraction.fetch_html("https://example.com/", 5, 1000)

    assert created[0]["trust_env"] is False
