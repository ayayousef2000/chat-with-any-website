from collections.abc import Callable
from typing import Any

import cohere.errors as cohere_errors
import groq
import httpx
import pytest
from fastapi.testclient import TestClient
from weaviate import exceptions as weaviate_exceptions

from app.errors import FetchError, NotIngestedError
from app.main import app
from app.pipeline import Answer, CitedSource, IngestResult
from app.ratelimit import ClientLimits
from app.vector_store import RetrievedChunk

_REQUEST = httpx.Request("POST", "https://api.example.invalid/v1")
UPSTREAM_ERRORS: dict[str, Callable[[], Exception]] = {
    "groq-rate-limit": lambda: groq.RateLimitError("limit", response=httpx.Response(429, request=_REQUEST), body=None),
    "cohere-rate-limit": lambda: cohere_errors.TooManyRequestsError(body={"message": "limit"}),
    "groq-rate-limit-23": lambda: groq.RateLimitError(
        "limit", response=httpx.Response(429, request=_REQUEST, headers={"retry-after": "23"}), body=None
    ),
    "groq-down": lambda: groq.APIConnectionError(request=_REQUEST),
    "cohere-down": lambda: cohere_errors.ServiceUnavailableError(body={"message": "down"}),
    "weaviate-down": lambda: weaviate_exceptions.WeaviateConnectionError("down"),
}


class FakePipeline:
    def __init__(self) -> None:
        self.ingest_calls: list[tuple[str, bool]] = []
        self.forgotten: list[str] = []

    def ingest(self, url: str, *, refresh: bool = False) -> IngestResult:
        self.ingest_calls.append((url, refresh))
        if "down" in url:
            raise FetchError("The website responded with HTTP 503.")
        if "saved" in url:
            return IngestResult(
                url="https://example.com/", title="Example", chunk_count=4, reused=True, age_seconds=240
            )
        return IngestResult(url="https://example.com/", title="Example", chunk_count=4)

    def forget(self, url: str) -> bool:
        self.forgotten.append(url)
        return "known" in url

    def ask(self, url: str, question: str) -> Answer:
        if "unloaded" in url:
            raise NotIngestedError("This URL has not been loaded yet.")
        if "boom" in question:
            raise RuntimeError("secret internal detail")
        if question in UPSTREAM_ERRORS:
            raise UPSTREAM_ERRORS[question]()
        chunk = RetrievedChunk(title="Example", heading="Intro", text="Some text.", chunk_index=2, score=0.8)
        return Answer(answer="42 [1]", sources=[CitedSource(number=1, chunk=chunk)], retrieved=[chunk])


def _client(limits: ClientLimits | None = None) -> TestClient:
    # Not used as a context manager, so the real startup (which connects to external services) is skipped.
    app.state.pipeline = FakePipeline()
    # Generous limits, so that tests that are not about the limits never reach them.
    app.state.limits = limits or ClientLimits(asks_per_minute=1000, asks_per_day=1000, loads_per_hour=1000)
    return TestClient(app, raise_server_exceptions=False)


def _pipeline() -> FakePipeline:
    pipeline = app.state.pipeline
    assert isinstance(pipeline, FakePipeline)
    return pipeline


def test_ingest_returns_summary() -> None:
    response = _client().post("/api/ingest", json={"url": "example.com"})
    assert response.status_code == 200
    assert response.json() == {
        "url": "https://example.com/",
        "title": "Example",
        "chunk_count": 4,
        "reused": False,
        "age_seconds": 0,
    }


def test_ingest_reports_user_facing_errors() -> None:
    response = _client().post("/api/ingest", json={"url": "https://down.example"})
    assert response.status_code == 502
    assert response.json() == {"detail": "The website responded with HTTP 503."}


def test_ask_returns_answer_and_sources() -> None:
    response = _client().post("/api/ask", json={"url": "https://example.com/", "question": "Why?"})
    assert response.status_code == 200
    assert response.json() == {
        "answer": "42 [1]",
        "sources": [{"number": 1, "chunk_index": 2, "heading": "Intro", "text": "Some text.", "score": 0.8}],
    }


def test_ask_for_unloaded_page_is_not_found() -> None:
    response = _client().post("/api/ask", json={"url": "https://unloaded.example", "question": "Why?"})
    assert response.status_code == 404


def test_unexpected_errors_do_not_leak_details() -> None:
    response = _client().post("/api/ask", json={"url": "https://example.com/", "question": "boom"})
    assert response.status_code == 500
    assert "secret" not in response.text


def test_request_validation() -> None:
    client = _client()
    assert client.post("/api/ask", json={"url": "https://example.com/", "question": ""}).status_code == 422
    assert client.post("/api/ingest", json={}).status_code == 422


def test_index_page_is_served() -> None:
    response = _client().get("/")
    assert response.status_code == 200
    assert "Chat With Any Website" in response.text


@pytest.mark.parametrize("question", ["groq-rate-limit", "cohere-rate-limit"])
def test_rate_limits_get_a_friendly_message_and_retry_hint(question: str) -> None:
    response = _client().post("/api/ask", json={"url": "https://example.com/", "question": question})
    assert response.status_code == 429
    assert response.headers["Retry-After"] == "10"
    assert "try again" in response.json()["detail"]
    assert "groq" not in response.json()["detail"].lower()  # no provider internals in the message


@pytest.mark.parametrize("question", ["groq-down", "cohere-down", "weaviate-down"])
def test_unreachable_services_get_a_friendly_message(question: str) -> None:
    response = _client().post("/api/ask", json={"url": "https://example.com/", "question": question})
    assert response.status_code == 503
    assert response.json() == {"detail": "An external service did not respond. Try again in a moment."}


def test_page_assets_are_served_from_static_files() -> None:
    client = _client()
    html = client.get("/").text
    assert 'src="/static/app.js"' in html
    assert 'href="/static/styles.css"' in html
    assert '<dialog id="confirm-dialog"' in html
    assert 'id="page-link" dir="auto"' in html  # a title in another script keeps its own word order
    assert "<script>" not in html  # no inline script, which the content security policy would block
    assert "javascript" in client.get("/static/app.js").headers["content-type"]
    # The theme switch is loaded in the head, so that a saved theme is applied before the page is first drawn.
    assert '<script src="/static/theme.js"></script>' in html
    assert "javascript" in client.get("/static/theme.js").headers["content-type"]
    assert 'id="theme-switch"' in html
    assert client.get("/static/styles.css").status_code == 200
    assert client.get("/static/favicon.svg").status_code == 200
    # Browsers cache tab icons hard; the version in the address makes them fetch a changed icon.
    assert 'href="/static/favicon.svg?v=' in html
    assert client.get("/static/favicon.svg?v=2").status_code == 200


def test_security_headers_are_set() -> None:
    headers = _client().get("/").headers
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["Referrer-Policy"] == "no-referrer"
    policy = headers["Content-Security-Policy"]
    assert "script-src 'self'" in policy
    assert "frame-ancestors 'none'" in policy
    assert "unsafe-inline" not in policy


def test_api_docs_are_not_restricted_by_the_content_security_policy() -> None:
    response = _client().get("/docs")
    assert response.status_code == 200
    assert "Content-Security-Policy" not in response.headers
    assert response.headers["X-Content-Type-Options"] == "nosniff"


def test_ingest_reports_when_a_saved_copy_was_used() -> None:
    response = _client().post("/api/ingest", json={"url": "https://saved.example"})
    assert response.json()["reused"] is True
    assert response.json()["age_seconds"] == 240


def test_ingest_passes_the_refresh_option_on() -> None:
    client = _client()
    client.post("/api/ingest", json={"url": "https://a.example"})
    client.post("/api/ingest", json={"url": "https://a.example", "refresh": True})
    assert _pipeline().ingest_calls == [("https://a.example", False), ("https://a.example", True)]


def test_a_stored_page_can_be_deleted() -> None:
    client = _client()
    assert client.delete("/api/page", params={"url": "https://known.example"}).json() == {"deleted": True}
    assert client.delete("/api/page", params={"url": "https://other.example"}).json() == {"deleted": False}
    assert _pipeline().forgotten == ["https://known.example", "https://other.example"]


def test_deleting_requires_an_address() -> None:
    assert _client().delete("/api/page").status_code == 422


def test_the_wait_the_service_asked_for_is_shown_and_sent_as_a_header() -> None:
    response = _client().post("/api/ask", json={"url": "https://example.com/", "question": "groq-rate-limit-23"})
    assert response.status_code == 429
    assert response.headers["Retry-After"] == "23"
    assert "Please try again in about 23 seconds" in response.json()["detail"]


def test_without_a_named_wait_the_message_says_a_few_seconds() -> None:
    response = _client().post("/api/ask", json={"url": "https://example.com/", "question": "groq-rate-limit"})
    assert response.headers["Retry-After"] == "10"
    assert "Please try again in a few seconds" in response.json()["detail"]


# --- limits per visitor ------------------------------------------------------------------------------------------


def _ask(client: TestClient, **headers: str) -> Any:
    return client.post("/api/ask", json={"url": "https://example.com/", "question": "Why?"}, headers=headers)


def test_a_visitor_over_the_question_limit_is_told_to_wait() -> None:
    client = _client(ClientLimits(asks_per_minute=2, asks_per_day=100, loads_per_hour=100))

    assert [_ask(client).status_code for _ in range(2)] == [200, 200]
    response = _ask(client)

    assert response.status_code == 429
    assert response.json()["detail"].startswith("You're asking questions too quickly. Please try again in about ")
    assert 1 <= int(response.headers["Retry-After"]) <= 60


def test_the_daily_question_limit_is_explained_in_plain_words() -> None:
    client = _client(ClientLimits(asks_per_minute=100, asks_per_day=1, loads_per_hour=100))
    _ask(client)

    response = _ask(client)

    assert response.status_code == 429
    assert "today's limit of questions" in response.json()["detail"]
    assert int(response.headers["Retry-After"]) > 3600


def test_questions_over_the_limit_never_reach_the_pipeline() -> None:
    client = _client(ClientLimits(asks_per_minute=1, asks_per_day=100, loads_per_hour=100))
    _ask(client)
    unloaded = client.post("/api/ask", json={"url": "https://unloaded.example/", "question": "Why?"})

    assert unloaded.status_code == 429  # it was refused before the pipeline could say the page is not loaded


def test_a_made_up_forwarded_address_does_not_get_a_fresh_limit() -> None:
    client = _client(ClientLimits(asks_per_minute=1, asks_per_day=100, loads_per_hour=100))

    assert _ask(client, **{"X-Forwarded-For": "198.51.100.1"}).status_code == 200
    assert _ask(client, **{"X-Forwarded-For": "198.51.100.2"}).status_code == 429


def test_loading_and_deleting_pages_share_one_limit_that_questions_do_not_use() -> None:
    client = _client(ClientLimits(asks_per_minute=100, asks_per_day=100, loads_per_hour=2))

    assert client.post("/api/ingest", json={"url": "example.com"}).status_code == 200
    assert client.delete("/api/page", params={"url": "known.example"}).status_code == 200
    blocked = client.post("/api/ingest", json={"url": "example.com"})
    blocked_delete = client.delete("/api/page", params={"url": "known.example"})

    assert blocked.status_code == 429
    assert "loaded a lot of pages" in blocked.json()["detail"]
    assert blocked_delete.status_code == 429
    assert len(_pipeline().ingest_calls) == 1  # the refused load never reached the pipeline
    assert _ask(client).status_code == 200


def test_limits_that_are_switched_off_let_everything_through() -> None:
    client = _client(ClientLimits(asks_per_minute=1, asks_per_day=1, loads_per_hour=1, enabled=False))

    assert all(_ask(client).status_code == 200 for _ in range(5))


# --- health check ------------------------------------------------------------------------------------------------


def test_health_answers_without_touching_any_service() -> None:
    client = _client()
    app.state.pipeline = None  # a health check must not need the pipeline or any outside service

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_is_not_counted_against_the_limits_per_visitor() -> None:
    client = _client(ClientLimits(asks_per_minute=1, asks_per_day=1, loads_per_hour=1))

    assert all(client.get("/health").status_code == 200 for _ in range(20))
    assert _ask(client).status_code == 200  # the one allowed question is still available


def test_health_is_not_listed_in_the_api_documentation() -> None:
    assert "/health" not in _client().get("/openapi.json").json()["paths"]


# --- caching of the page and its files -----------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["/", "/static/styles.css", "/static/app.js", "/static/theme.js"])
def test_the_page_and_its_files_must_be_checked_again_before_a_browser_reuses_them(path: str) -> None:
    # Without this, a browser may keep an old stylesheet for hours and show a new page with old styles.
    assert _client().get(path).headers["cache-control"] == "no-cache"


def test_an_unchanged_file_costs_only_a_not_modified_answer() -> None:
    client = _client()
    first = client.get("/static/styles.css")

    again = client.get("/static/styles.css", headers={"If-None-Match": first.headers["etag"]})

    assert again.status_code == 304
    assert again.content == b""


def test_the_interface_answers_are_not_marked_as_cacheable_pages() -> None:
    assert "cache-control" not in _client().get("/health").headers


# --- the diagnostic headers of /health -------------------------------------------------------------------------------


def test_health_shows_nothing_about_the_caller_unless_the_diagnosis_is_switched_on() -> None:
    client = _client()
    app.state.debug_client_address = False

    response = client.get("/health", headers={"X-Forwarded-For": "198.51.100.9"})

    assert not [name for name in response.headers if name.lower().startswith("x-debug")]


def test_health_shows_the_peer_the_forwarded_header_and_the_address_used_when_the_diagnosis_is_on() -> None:
    client = _client()
    app.state.debug_client_address = True
    try:
        response = client.get("/health", headers={"X-Forwarded-For": "198.51.100.9, 10.1.2.3"})
    finally:
        app.state.debug_client_address = False

    assert response.headers["x-debug-peer"] == "testclient"
    assert response.headers["x-debug-forwarded-for"] == "198.51.100.9, 10.1.2.3"
    assert response.headers["x-debug-client-id"] == "testclient"  # the forwarded header is never what is used


def test_only_health_shows_the_diagnostic_headers() -> None:
    client = _client()
    app.state.debug_client_address = True
    try:
        other = client.get("/", headers={"X-Forwarded-For": "198.51.100.9"})
    finally:
        app.state.debug_client_address = False

    assert not [name for name in other.headers if name.lower().startswith("x-debug")]
