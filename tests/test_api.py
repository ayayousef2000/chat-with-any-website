from fastapi.testclient import TestClient

from app.errors import FetchError, NotIngestedError
from app.main import app
from app.pipeline import Answer, CitedSource, IngestResult
from app.vector_store import RetrievedChunk


class FakePipeline:
    def ingest(self, url: str) -> IngestResult:
        if "down" in url:
            raise FetchError("The website responded with HTTP 503.")
        return IngestResult(url="https://example.com/", title="Example", chunk_count=4)

    def ask(self, url: str, question: str) -> Answer:
        if "unloaded" in url:
            raise NotIngestedError("This URL has not been loaded yet.")
        if "boom" in question:
            raise RuntimeError("secret internal detail")
        chunk = RetrievedChunk(title="Example", heading="Intro", text="Some text.", chunk_index=2, score=0.8)
        return Answer(answer="42 [1]", sources=[CitedSource(number=1, chunk=chunk)], retrieved=[chunk])


def _client() -> TestClient:
    # Not used as a context manager, so the real startup (which connects to external services) is skipped.
    app.state.pipeline = FakePipeline()
    return TestClient(app, raise_server_exceptions=False)


def test_ingest_returns_summary() -> None:
    response = _client().post("/api/ingest", json={"url": "example.com"})
    assert response.status_code == 200
    assert response.json() == {"url": "https://example.com/", "title": "Example", "chunk_count": 4}


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
