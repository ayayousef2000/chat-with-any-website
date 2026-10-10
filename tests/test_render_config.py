"""Tests of the trusted proxy addresses in render.yaml, run with uvicorn's own proxy header code.

Behind Render the app is reached through a local proxy (127.0.0.1) and the forwarded header ends with a Cloudflare
address and one of Render's internal addresses. Uvicorn reads that header from the right, skips the addresses it
trusts and takes the first other one as the visitor. These tests check that the list in render.yaml makes that give the
visitor, and never an address that a caller wrote into the header.
"""

import asyncio
import ipaddress
from pathlib import Path
from typing import Any

import pytest
import yaml
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

RENDER_YAML = Path(__file__).parent.parent / "render.yaml"

VISITOR = "203.0.113.50"
VISITOR_V6 = "2001:db8:aaaa:bbbb::7"
CLOUDFLARE = "172.71.122.82"
INTERNAL = "10.29.113.53"
LOCAL_PROXY = "127.0.0.1"


def _services() -> dict[str, dict[str, Any]]:
    document = yaml.safe_load(RENDER_YAML.read_text(encoding="utf-8"))
    return {service["name"]: service for service in document["services"]}


def _trusted(service: dict[str, Any]) -> str:
    values = [entry.get("value") for entry in service["envVars"] if entry["key"] == "FORWARDED_ALLOW_IPS"]
    assert len(values) == 1, "FORWARDED_ALLOW_IPS must be set once"
    return str(values[0])


def _client_taken(trusted: str, peer: str, forwarded: str | None) -> str:
    """The address the app would use for a request, as uvicorn decides it."""
    seen: dict[str, str] = {}

    async def inner(scope: Any, receive: Any, send: Any) -> None:
        seen["client"] = scope["client"][0]

    async def nothing(*_: Any) -> None:
        return None

    headers = [(b"x-forwarded-for", forwarded.encode())] if forwarded is not None else []
    scope: Any = {"type": "http", "client": (peer, 1234), "headers": headers, "scheme": "http"}
    asyncio.run(ProxyHeadersMiddleware(inner, trusted_hosts=trusted)(scope, nothing, nothing))  # type: ignore[arg-type]
    return seen["client"]


@pytest.fixture(params=sorted(_services()))
def trusted(request: pytest.FixtureRequest) -> str:
    return _trusted(_services()[request.param])


# --- the setting ----------------------------------------------------------------------------------------------------


def test_both_services_trust_the_same_addresses() -> None:
    values = {name: _trusted(service) for name, service in _services().items()}

    assert len(values) == 2
    assert len(set(values.values())) == 1


def test_every_entry_is_a_valid_address_or_range_and_the_local_proxy_and_render_are_in_it(trusted: str) -> None:
    entries = [entry.strip() for entry in trusted.split(",")]

    for entry in entries:
        ipaddress.ip_network(entry)  # raises for anything that uvicorn would silently take as a plain word
    assert "127.0.0.1" in entries  # the connection really comes from a local proxy inside the container
    assert "10.0.0.0/8" in entries  # Render's internal addresses
    assert any(":" in entry and "/" in entry for entry in entries)  # Cloudflare's IPv6 ranges
    assert len(entries) == len(set(entries))


@pytest.mark.parametrize(
    "address",
    [
        # Seen on Render in the answers of /health on 2026-10-10: Render's internal addresses ...
        "10.27.133.5",
        "10.29.113.53",
        "10.30.207.13",
        "10.200.25.98",
        # ... and Cloudflare's addresses in the middle of the forwarded header.
        "141.101.97.104",
        "172.68.151.85",
        "172.70.247.42",
        "172.71.122.82",
        "172.71.232.67",
    ],
)
def test_the_addresses_seen_on_render_are_trusted(trusted: str, address: str) -> None:
    assert _client_taken(trusted, LOCAL_PROXY, f"{VISITOR}, {address}") == VISITOR


def test_production_does_not_switch_on_the_diagnosis_of_the_caller_address() -> None:
    production = _services()["chat-with-any-website"]

    assert "DEBUG_CLIENT_ADDRESS" not in [entry["key"] for entry in production["envVars"]]


# --- who the visitor is ---------------------------------------------------------------------------------------------


def test_the_visitor_is_the_address_before_cloudflare_and_render(trusted: str) -> None:
    assert _client_taken(trusted, LOCAL_PROXY, f"{VISITOR}, {CLOUDFLARE}, {INTERNAL}") == VISITOR


def test_a_visitor_on_ipv6_is_found_too(trusted: str) -> None:
    assert _client_taken(trusted, LOCAL_PROXY, f"{VISITOR_V6}, {CLOUDFLARE}, {INTERNAL}") == VISITOR_V6


@pytest.mark.parametrize(
    "forged",
    [
        "1.2.3.4",
        "6.6.6.6, 7.7.7.7, 8.8.8.8",
        "10.1.1.1",  # looks like one of Render's own addresses
        CLOUDFLARE,  # looks like Cloudflare
        "127.0.0.1",  # looks like the local proxy
        "198.51.100.99",  # the address of another visitor
    ],
)
def test_what_a_caller_writes_into_the_header_never_becomes_the_visitor(trusted: str, forged: str) -> None:
    chain = f"{forged}, {VISITOR}, {CLOUDFLARE}, {INTERNAL}"  # the proxies add their entries to the right

    assert _client_taken(trusted, LOCAL_PROXY, chain) == VISITOR


def test_a_request_without_a_forwarded_header_keeps_the_connection_address(trusted: str) -> None:
    # Render's own health checks come this way.
    assert _client_taken(trusted, "10.200.25.98", None) == "10.200.25.98"


def test_a_header_from_an_address_that_is_not_a_proxy_is_ignored(trusted: str) -> None:
    # Someone reaching the app directly cannot choose the address that the limits are counted for.
    assert _client_taken(trusted, "198.51.100.77", "1.2.3.4") == "198.51.100.77"


def test_without_the_local_proxy_in_the_list_the_visitor_would_not_be_found(trusted: str) -> None:
    # The first list tried on Render left it out, and the app then used 127.0.0.1 for every visitor.
    without = ",".join(entry for entry in trusted.split(",") if entry not in {"127.0.0.1", "::1"})

    assert _client_taken(without, LOCAL_PROXY, f"{VISITOR}, {CLOUDFLARE}, {INTERNAL}") == LOCAL_PROXY
