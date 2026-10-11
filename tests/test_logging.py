"""Tests of structured logging: the JSON lines, the request id, and that nothing private is logged."""

import io
import json
import logging
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.config import Settings
from app.logging_setup import JsonFormatter, RequestIdFilter, configure_logging, request_id_var
from app.main import app
from app.pipeline import Answer
from app.ratelimit import ClientLimits
from tests.fakes import make_settings


def _record(message: str = "hello", level: int = logging.INFO, **extra: Any) -> logging.LogRecord:
    record = logging.LogRecord("app.test", level, __file__, 1, message, None, None)
    for key, value in extra.items():
        setattr(record, key, value)
    return record


# --- the JSON format --------------------------------------------------------------------------------------------


def test_a_json_line_has_the_fields_a_log_collector_filters_on() -> None:
    line = JsonFormatter().format(
        _record("Loaded a page", method="POST", path="/api/ingest", status=200, duration_ms=42)
    )

    payload = json.loads(line)
    assert "\n" not in line
    assert payload["level"] == "INFO"
    assert payload["logger"] == "app.test"
    assert payload["message"] == "Loaded a page"
    assert payload["request_id"] == "-"
    assert (payload["method"], payload["path"], payload["status"], payload["duration_ms"]) == (
        "POST",
        "/api/ingest",
        200,
        42,
    )
    assert payload["time"].endswith("+00:00")  # UTC, with milliseconds


def test_a_record_without_extra_fields_has_none_of_them() -> None:
    payload = json.loads(JsonFormatter().format(_record()))

    assert not {"method", "path", "status", "duration_ms"} & payload.keys()


def test_non_ascii_text_stays_readable_and_the_message_is_filled_in() -> None:
    record = logging.LogRecord("app.test", logging.WARNING, __file__, 1, "%s answered %d", ("المضيف", 503), None)

    line = JsonFormatter().format(record)

    assert "المضيف answered 503" in line
    assert json.loads(line)["level"] == "WARNING"


def test_an_exception_is_kept_in_one_field_and_the_line_stays_one_line() -> None:
    try:
        raise ValueError("broken")
    except ValueError:
        import sys

        record = logging.LogRecord("app.test", logging.ERROR, __file__, 1, "failed", None, sys.exc_info())

    line = JsonFormatter().format(record)

    assert "\n" not in line
    assert "ValueError: broken" in json.loads(line)["exception"]


def test_the_request_id_filter_adds_the_current_id() -> None:
    token = request_id_var.set("abc123")
    try:
        record = _record()
        assert RequestIdFilter().filter(record) is True
    finally:
        request_id_var.reset(token)

    assert record.request_id == "abc123"  # type: ignore[attr-defined]


# --- setting it up ----------------------------------------------------------------------------------------------


@pytest.fixture
def clean_logging() -> Iterator[None]:
    root = logging.getLogger()
    before = (root.handlers[:], root.level)
    yield
    root.handlers, root.level = before[0], before[1]


def test_configuring_sends_logs_to_one_handler_in_the_chosen_format(
    clean_logging: None, capsys: pytest.CaptureFixture[str]
) -> None:
    configure_logging("INFO", "json")
    configure_logging("INFO", "json")  # again: the earlier setup is replaced, lines are not doubled

    logging.getLogger("app.test").info("one line")

    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["message"] == "one line"
    assert len(logging.getLogger().handlers) == 1


def test_text_format_is_readable_and_has_the_request_id(
    clean_logging: None, capsys: pytest.CaptureFixture[str]
) -> None:
    configure_logging("INFO", "text")
    token = request_id_var.set("r-42")
    try:
        logging.getLogger("app.test").info("hello there")
    finally:
        request_id_var.reset(token)

    out = capsys.readouterr().out
    assert "INFO [r-42] app.test: hello there" in out
    assert not out.lstrip().startswith("{")


def test_the_level_decides_what_is_kept(clean_logging: None, capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("WARNING", "text")

    logging.getLogger("app.test").info("not kept")
    logging.getLogger("app.test").warning("kept")

    out = capsys.readouterr().out
    assert "kept" in out
    assert "not kept" not in out


def test_the_servers_access_log_is_silent_and_its_other_logs_use_the_same_format(
    clean_logging: None, capsys: pytest.CaptureFixture[str]
) -> None:
    server = logging.getLogger("uvicorn.error")
    server.handlers = [logging.StreamHandler(io.StringIO())]
    configure_logging("INFO", "json")

    logging.getLogger("uvicorn.access").info("GET / 200")
    server.info("Application startup complete.")

    lines = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    assert [line["message"] for line in lines] == ["Application startup complete."]


def test_libraries_that_log_every_web_address_only_show_warnings(
    clean_logging: None, capsys: pytest.CaptureFixture[str]
) -> None:
    """The address of every request would include the pages that visitors load."""
    configure_logging("INFO", "json")

    logging.getLogger("httpx").info('HTTP Request: GET https://private.example/secret-page "HTTP/1.1 200 OK"')
    logging.getLogger("httpcore.connection").info("connect_tcp.started host='private.example'")
    logging.getLogger("weaviate").info("a query")

    assert capsys.readouterr().out == ""
    assert logging.getLogger("httpx").level == logging.WARNING


# --- the settings -----------------------------------------------------------------------------------------------


def test_defaults_are_readable_text_at_info(tmp_path: Any) -> None:
    settings = make_settings(tmp_path)

    assert (settings.log_level, settings.log_format) == ("INFO", "text")


@pytest.mark.parametrize(("name", "value"), [("log_level", "LOUD"), ("log_format", "xml"), ("log_level", "info")])
def test_unknown_values_are_refused(tmp_path: Any, name: str, value: str) -> None:
    with pytest.raises(ValidationError):
        make_settings(tmp_path, **{name: value})


def test_both_render_services_log_json() -> None:
    from pathlib import Path

    import yaml

    document = yaml.safe_load((Path(__file__).parent.parent / "render.yaml").read_text(encoding="utf-8"))

    for service in document["services"]:
        values = [entry.get("value") for entry in service["envVars"] if entry["key"] == "LOG_FORMAT"]
        assert values == ["json"], service["name"]
        assert Settings.model_fields["log_format"].annotation is not None


# --- the request line and the request id ------------------------------------------------------------------------


class _Pipeline:
    def ask(self, url: str, question: str, history: Any = ()) -> Answer:
        if "boom" in question:
            raise RuntimeError("secret internal detail")
        return Answer(answer="42", sources=[], retrieved=[])

    def forget(self, url: str) -> bool:
        return True


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.addFilter(RequestIdFilter())
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def served() -> Iterator[tuple[TestClient, _Capture]]:
    app.state.pipeline = _Pipeline()
    app.state.limits = ClientLimits(asks_per_minute=1000, asks_per_day=1000, loads_per_hour=1000)
    capture = _Capture()
    root = logging.getLogger()
    before = (root.handlers[:], root.level)
    root.handlers, root.level = [capture], logging.DEBUG
    try:
        yield TestClient(app, raise_server_exceptions=False), capture
    finally:
        root.handlers, root.level = before


def _lines(capture: _Capture) -> list[logging.LogRecord]:
    return [record for record in capture.records if record.name == "app.access"]


def test_every_response_carries_a_request_id_and_each_is_new(served: tuple[TestClient, _Capture]) -> None:
    client, _ = served

    ids = {
        client.post("/api/ask", json={"url": "https://example.com/", "question": "Why?"}).headers["X-Request-ID"]
        for _ in range(5)
    }

    assert len(ids) == 5
    assert all(len(value) == 16 for value in ids)


def test_a_request_id_sent_by_the_caller_is_not_used(served: tuple[TestClient, _Capture]) -> None:
    client, capture = served

    response = client.post(
        "/api/ask",
        json={"url": "https://example.com/", "question": "Why?"},
        headers={"X-Request-ID": "forged-by-caller"},
    )

    assert response.headers["X-Request-ID"] != "forged-by-caller"
    assert _lines(capture)[0].request_id != "forged-by-caller"  # type: ignore[attr-defined]


def test_one_line_is_logged_per_request_with_method_path_status_and_time(served: tuple[TestClient, _Capture]) -> None:
    client, capture = served

    response = client.post("/api/ask", json={"url": "https://example.com/", "question": "Why?"})

    (line,) = _lines(capture)
    assert line.getMessage().startswith("POST /api/ask 200 ")
    assert (line.method, line.path, line.status) == ("POST", "/api/ask", 200)  # type: ignore[attr-defined]
    assert line.duration_ms >= 0  # type: ignore[attr-defined]
    assert line.request_id == response.headers["X-Request-ID"]  # type: ignore[attr-defined]
    assert line.levelno == logging.INFO


def test_nothing_private_is_logged(served: tuple[TestClient, _Capture]) -> None:
    client, capture = served

    client.post(
        "/api/ask",
        json={"url": "https://secret-site.example/private/page?token=abc", "question": "my private question"},
    )
    client.delete("/api/page", params={"url": "https://secret-site.example/private/page?token=abc"})

    own = [record for record in capture.records if not record.name.startswith("httpx")]  # not the test client's own
    everything = " ".join(record.getMessage() + json.dumps(record.__dict__, default=str) for record in own)
    for private in ("my private question", "secret-site.example", "private/page", "token=abc", "testclient"):
        assert private not in everything, private


def test_the_path_is_logged_without_its_query_string(served: tuple[TestClient, _Capture]) -> None:
    client, capture = served

    client.delete("/api/page", params={"url": "https://example.com/a"})

    (line,) = _lines(capture)
    assert line.path == "/api/page"  # type: ignore[attr-defined]


def test_a_successful_health_check_is_not_logged_at_info(served: tuple[TestClient, _Capture]) -> None:
    client, capture = served

    assert client.get("/health").status_code == 200

    (line,) = _lines(capture)
    assert line.levelno == logging.DEBUG  # Render asks every few seconds; this would fill the log


def test_an_unexpected_error_is_logged_with_the_same_request_id_and_status_500(
    served: tuple[TestClient, _Capture],
) -> None:
    client, capture = served

    response = client.post("/api/ask", json={"url": "https://example.com/", "question": "boom"})

    assert response.status_code == 500
    assert "secret internal detail" not in response.text
    errors = [record for record in capture.records if record.levelno >= logging.ERROR]
    (line,) = _lines(capture)
    assert line.status == 500  # type: ignore[attr-defined]
    assert errors
    assert {getattr(record, "request_id", None) for record in [*errors, line]} == {line.request_id}  # type: ignore[attr-defined]


def test_the_id_is_reset_when_the_request_is_done(served: tuple[TestClient, _Capture]) -> None:
    client, _ = served

    client.post("/api/ask", json={"url": "https://example.com/", "question": "Why?"})

    assert request_id_var.get() == "-"
