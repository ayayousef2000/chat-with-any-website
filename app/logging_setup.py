"""Logging: readable lines on a developer's computer, one JSON object per line where logs are collected.

Every log line carries the id of the request that produced it, so one failing request can be followed through the
log. Nothing private is logged by this module: the access line holds the method, the path without its query string,
the status and the time taken, and never a question, a request body or a visitor's address.
"""

import json
import logging
import sys
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

# The request being handled. Worker threads of the server copy the context, so the id is also seen in the logs of
# the code that a request runs in a thread.
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

# Fields that a log call adds through ``extra`` and that the JSON format keeps.
_EXTRA_FIELDS = ("method", "path", "status", "duration_ms")

TEXT_FORMAT = "%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s"

# Libraries that log the address of every request at INFO. Those addresses include the pages that visitors load, so
# only their warnings are kept.
_QUIET_LIBRARIES = ("httpx", "httpcore", "urllib3", "weaviate", "cohere", "groq", "trafilatura", "asyncio")


class RequestIdFilter(logging.Filter):
    """Put the id of the current request on every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Add ``request_id`` to the record; the record is always kept."""
        record.request_id = request_id_var.get()
        return True


class JsonFormatter(logging.Formatter):
    """Write each record as one line of JSON, with the fields a log collector can filter on."""

    def format(self, record: logging.LogRecord) -> str:
        """Turn a record into a JSON object on one line."""
        payload: dict[str, Any] = {
            "time": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", request_id_var.get()),
        }
        for field in _EXTRA_FIELDS:
            if hasattr(record, field):
                payload[field] = getattr(record, field)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str, log_format: str) -> None:
    """Send all logs to standard output in the chosen format. Calling it again replaces the earlier setup.

    Args:
        level: The lowest level that is kept, for example ``"INFO"``.
        log_format: ``"json"`` for one JSON object per line, anything else for readable text.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if log_format == "json" else logging.Formatter(TEXT_FORMAT))
    handler.addFilter(RequestIdFilter())

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)

    # The server's own loggers write through the root handler, so that every line has the same format. Its access
    # log is replaced by the request line of the application, which also has the request id and the time taken.
    for name in ("uvicorn", "uvicorn.error"):
        server = logging.getLogger(name)
        server.handlers = []
        server.propagate = True
    access = logging.getLogger("uvicorn.access")
    access.handlers = []
    access.propagate = False

    for name in _QUIET_LIBRARIES:
        logging.getLogger(name).setLevel(logging.WARNING)
