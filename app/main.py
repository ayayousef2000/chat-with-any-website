"""FastAPI application exposing the ingest and ask endpoints."""

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Query, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import get_settings
from app.errors import AppError, TooManyRequestsError
from app.janitor import Janitor
from app.pipeline import Pipeline
from app.ratelimit import ClientLimits
from app.upstream import RATE_LIMIT_ERRORS, UNAVAILABLE_ERRORS, public_error_details

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

RATE_LIMIT_MESSAGE = "We're busy right now. Please try again in {wait}."
UNAVAILABLE_MESSAGE = "An external service did not respond. Try again in a moment."

# Scripts and styles are served from this origin only, so injected markup could not run or load anything.
CONTENT_SECURITY_POLICY = (
    "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self'; connect-src 'self'; "
    "base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
)
# The generated API docs load their assets from a CDN, so the policy does not apply to them.
DOCS_PATHS = ("/docs", "/redoc", "/openapi.json")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create the pipeline on startup and close its connections on shutdown."""
    settings = get_settings()
    pipeline = Pipeline(settings)
    try:
        pipeline.reconcile()
    except Exception:
        logger.exception("Could not compare stored pages with their usage records")
    janitor = Janitor(pipeline.cleanup, settings.cleanup_interval_seconds)
    janitor.start()
    app.state.pipeline = pipeline
    app.state.limits = ClientLimits.from_settings(settings)
    try:
        yield
    finally:
        janitor.stop()
        pipeline.close()


app = FastAPI(title="Chat With Any Website", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class IngestRequest(BaseModel):
    """Request body for loading a page."""

    url: str = Field(min_length=1, max_length=2048)
    refresh: bool = False


class IngestResponse(BaseModel):
    """Result of loading a page. ``reused`` means a stored copy, ``age_seconds`` old, was used."""

    url: str
    title: str
    chunk_count: int
    reused: bool
    age_seconds: int


class ForgetResponse(BaseModel):
    """Result of deleting a stored page."""

    deleted: bool


class AskRequest(BaseModel):
    """Request body for asking a question about a loaded page."""

    url: str = Field(min_length=1, max_length=2048)
    question: str = Field(min_length=1, max_length=2000)


class Source(BaseModel):
    """A page excerpt that the answer cites, with the number used in the answer text."""

    number: int
    chunk_index: int
    heading: str
    text: str
    score: float | None


class AskResponse(BaseModel):
    """An answer with the excerpts it is based on."""

    answer: str
    sources: list[Source]


@app.exception_handler(AppError)
async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
    """Return a user-facing error response for expected failures."""
    return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})


async def handle_too_many_requests(request: Request, exc: Exception) -> JSONResponse:
    """Tell a caller who is over the limits of the API how long to wait."""
    wait = exc.retry_after if isinstance(exc, TooManyRequestsError) else 60.0
    logger.info("A client was over the limit on %s", request.url.path)
    return JSONResponse(
        status_code=429,
        content={"detail": str(exc)},
        headers={"Retry-After": str(max(1, round(wait)))},
    )


app.add_exception_handler(TooManyRequestsError, handle_too_many_requests)


async def handle_rate_limit(request: Request, exc: Exception) -> JSONResponse:
    """Tell the user an upstream service is rate limiting, and how long to wait."""
    details = public_error_details(exc)
    logger.warning("Rate limited by an upstream service on %s: %s", request.url.path, type(exc).__name__)
    return JSONResponse(
        status_code=429,
        content={"detail": RATE_LIMIT_MESSAGE.format(wait=details["wait_text"])},
        headers={"Retry-After": details["retry_after"]},
    )


async def handle_unavailable(request: Request, exc: Exception) -> JSONResponse:
    """Tell the user an upstream service is unreachable."""
    logger.warning("Upstream service unavailable on %s: %s", request.url.path, type(exc).__name__)
    return JSONResponse(status_code=503, content={"detail": UNAVAILABLE_MESSAGE})


for _rate_limit_error in RATE_LIMIT_ERRORS:
    app.add_exception_handler(_rate_limit_error, handle_rate_limit)
for _unavailable_error in UNAVAILABLE_ERRORS:
    app.add_exception_handler(_unavailable_error, handle_unavailable)


@app.middleware("http")
async def add_security_headers(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    """Add browser security headers to every response."""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    if not request.url.path.startswith(DOCS_PATHS):
        response.headers["Content-Security-Policy"] = CONTENT_SECURITY_POLICY
    return response


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    """Log an unexpected failure and return a generic error response."""
    logger.exception("Unhandled error while processing %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Something went wrong. Please try again."})


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Serve the web UI."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", include_in_schema=False)
async def health() -> dict[str, str]:
    """Say that the server is running, for the health check of the hosting platform.

    It answers at once and calls no outside service, so that a slow Cohere, Groq or Weaviate never makes the
    platform think the app is down and restart it. It is not counted against the limits per visitor.
    """
    return {"status": "ok"}


def limit_asks(request: Request) -> None:
    """Count a question against the limits of the caller, or refuse it."""
    limits: ClientLimits | None = getattr(request.app.state, "limits", None)  # set at startup
    if limits is not None:
        limits.enforce(request, "ask")


def limit_loads(request: Request) -> None:
    """Count a page load or delete against the limits of the caller, or refuse it."""
    limits: ClientLimits | None = getattr(request.app.state, "limits", None)  # set at startup
    if limits is not None:
        limits.enforce(request, "load")


# Plain `def` endpoints run in a worker thread, so the blocking SDK calls don't stall the event loop.
@app.post("/api/ingest", dependencies=[Depends(limit_loads)])
def ingest(body: IngestRequest, request: Request) -> IngestResponse:
    """Fetch, clean, chunk, embed and store a page."""
    result = request.app.state.pipeline.ingest(body.url, refresh=body.refresh)
    return IngestResponse(
        url=result.url,
        title=result.title,
        chunk_count=result.chunk_count,
        reused=result.reused,
        age_seconds=result.age_seconds,
    )


@app.delete("/api/page", dependencies=[Depends(limit_loads)])
def forget(request: Request, url: str = Query(min_length=1, max_length=2048)) -> ForgetResponse:
    """Delete the stored copy of a page."""
    return ForgetResponse(deleted=request.app.state.pipeline.forget(url))


@app.post("/api/ask", dependencies=[Depends(limit_asks)])
def ask(body: AskRequest, request: Request) -> AskResponse:
    """Answer a question about a previously loaded page."""
    result = request.app.state.pipeline.ask(body.url, body.question)
    return AskResponse(
        answer=result.answer,
        sources=[
            Source(
                number=source.number,
                chunk_index=source.chunk.chunk_index,
                heading=source.chunk.heading,
                text=source.chunk.text,
                score=source.chunk.score,
            )
            for source in result.sources
        ],
    )
