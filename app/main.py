"""FastAPI application exposing the ingest and ask endpoints."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from app.config import get_settings
from app.errors import AppError
from app.pipeline import Pipeline

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create the pipeline on startup and close its connections on shutdown."""
    pipeline = Pipeline(get_settings())
    app.state.pipeline = pipeline
    try:
        yield
    finally:
        pipeline.close()


app = FastAPI(title="Chat With Any Website", lifespan=lifespan)


class IngestRequest(BaseModel):
    """Request body for loading a page."""

    url: str = Field(min_length=1, max_length=2048)


class IngestResponse(BaseModel):
    """Result of loading a page."""

    url: str
    title: str
    chunk_count: int


class AskRequest(BaseModel):
    """Request body for asking a question about a loaded page."""

    url: str = Field(min_length=1, max_length=2048)
    question: str = Field(min_length=1, max_length=2000)


class Source(BaseModel):
    """A page excerpt that supports an answer."""

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


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    """Log an unexpected failure and return a generic error response."""
    logger.exception("Unhandled error while processing %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Something went wrong. Please try again."})


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Serve the web UI."""
    return FileResponse(STATIC_DIR / "index.html")


# Plain `def` endpoints run in a worker thread, so the blocking SDK calls don't stall the event loop.
@app.post("/api/ingest")
def ingest(body: IngestRequest, request: Request) -> IngestResponse:
    """Fetch, clean, chunk, embed and store a page."""
    result = request.app.state.pipeline.ingest(body.url)
    return IngestResponse(url=result.url, title=result.title, chunk_count=result.chunk_count)


@app.post("/api/ask")
def ask(body: AskRequest, request: Request) -> AskResponse:
    """Answer a question about a previously loaded page."""
    result = request.app.state.pipeline.ask(body.url, body.question)
    return AskResponse(
        answer=result.answer,
        sources=[
            Source(chunk_index=chunk.chunk_index, heading=chunk.heading, text=chunk.text, score=chunk.score)
            for chunk in result.sources
        ],
    )
