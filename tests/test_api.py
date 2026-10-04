from collections.abc import Callable

import cohere.errors as cohere_errors
import groq
import httpx
import pytest
from fastapi.testclient import TestClient
from weaviate import exceptions as weaviate_exceptions

from app.errors import FetchError, NotIngestedError
from app.main import app
from app.pipeline import Answer, CitedSource, IngestResult
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


def _client() -> TestClient:
    # Not used as a context manager, so the real startup (which connects to external services) is skipped.
    app.state.pipeline = FakePipeline()
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
    assert response.headers["Retry-After"] == "60"
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
    assert "Wait about 23 seconds and try again" in response.json()["detail"]


def test_without_a_named_wait_the_message_says_about_a_minute() -> None:
    response = _client().post("/api/ask", json={"url": "https://example.com/", "question": "groq-rate-limit"})
    assert response.headers["Retry-After"] == "60"
    assert "Wait about a minute and try again" in response.json()["detail"]
