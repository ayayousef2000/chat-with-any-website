"""Wires the steps together: ingest a URL, then answer questions about it."""

import logging
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from app.chunking import chunk_text, contextualize
from app.citations import cited_numbers
from app.cleaning import clean_text
from app.config import Settings
from app.embeddings import CohereEmbedder
from app.errors import ExtractionError, NotIngestedError
from app.extraction import load_page, normalize_url
from app.llm import GroqChat
from app.registry import PageRecord, PageRegistry, is_expired
from app.reranking import CohereReranker
from app.upstream import RETRYABLE_ERRORS, Retrier, RetryPolicy
from app.vector_store import RetrievedChunk, StoredChunk, WeaviateStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestResult:
    """Summary of a page that was loaded.

    ``reused`` is true when a copy that was already stored was used instead of fetching the page again;
    ``age_seconds`` is then how long ago that copy was loaded.
    """

    url: str
    title: str
    chunk_count: int
    reused: bool = False
    age_seconds: int = 0


@dataclass(frozen=True)
class CitedSource:
    """An excerpt that the answer cites, with the number it was cited by."""

    number: int
    chunk: RetrievedChunk


@dataclass(frozen=True)
class Answer:
    """An answer, the excerpts it cites, and every excerpt that was given to the model.

    ``sources`` holds only cited excerpts, so an answer that says the page does not cover the question has none.
    """

    answer: str
    sources: list[CitedSource]
    retrieved: list[RetrievedChunk]


class Embedder(Protocol):
    """Creates embeddings for documents and queries."""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts to be stored."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a question."""
        ...


class Store(Protocol):
    """Stores chunks and searches them."""

    def replace_source(self, url: str, title: str, chunks: Sequence[StoredChunk]) -> None:
        """Replace the stored chunks of a URL."""
        ...

    def has_source(self, url: str) -> bool:
        """Check whether a URL has stored chunks."""
        ...

    def delete_source(self, url: str) -> None:
        """Delete every chunk stored for a URL."""
        ...

    def list_sources(self) -> dict[str, int]:
        """Count the stored chunks per URL."""
        ...

    def search(self, url: str, query: str, vector: Sequence[float], limit: int, alpha: float) -> list[RetrievedChunk]:
        """Retrieve candidate chunks for a question."""
        ...

    def close(self) -> None:
        """Release resources."""
        ...


class Reranker(Protocol):
    """Reorders candidate chunks by relevance."""

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_n: int) -> list[RetrievedChunk]:
        """Keep the most relevant chunks."""
        ...


class Chat(Protocol):
    """Generates an answer from excerpts."""

    def answer(self, question: str, title: str, chunks: Sequence[RetrievedChunk]) -> str:
        """Answer a question from the excerpts."""
        ...


class Pipeline:
    """Runs the full flow: extraction, cleaning, chunking, embedding, storage, retrieval and answering.

    Components are created from the settings unless passed in, which lets tests supply fakes.

    Loaded pages are shared: everyone who asks about the same address uses the same stored copy. Each copy is
    deleted after it has gone unused for a while, after a maximum age, or when the store is over its size limit.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        embedder: Embedder | None = None,
        store: Store | None = None,
        reranker: Reranker | None = None,
        llm: Chat | None = None,
        registry: PageRegistry | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._settings = settings
        self._clock = clock
        retrier = Retrier(
            RetryPolicy(
                attempts=settings.retry_attempts,
                max_delay=settings.retry_max_wait_seconds,
                budget=settings.retry_budget_seconds,
            )
        )
        self._registry = registry or PageRegistry(settings.state_db_path)
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()
        self._embedder: Embedder = embedder or CohereEmbedder(
            api_key=settings.cohere_api_key,
            model=settings.cohere_embed_model,
            dimension=settings.cohere_embed_dimension,
            retrier=retrier,
        )
        self._store: Store = store or WeaviateStore(
            url=settings.weaviate_url,
            api_key=settings.weaviate_api_key,
            collection_name=settings.weaviate_collection,
            init_timeout=settings.weaviate_init_timeout_seconds,
        )
        self._reranker: Reranker | None = reranker
        if self._reranker is None and settings.rerank_enabled:
            # Reranking only improves the order of the excerpts, so it is not worth a long wait: when it fails,
            # the search order is used (see _best_chunks).
            quick_retrier = Retrier(
                RetryPolicy(attempts=2, max_delay=2.0, budget=3.0), sleep=retrier.sleep, jitter=retrier.jitter
            )
            self._reranker = CohereReranker(
                api_key=settings.cohere_api_key, model=settings.cohere_rerank_model, retrier=quick_retrier
            )
        self._llm: Chat = llm or GroqChat(
            api_key=settings.groq_api_key,
            model=settings.groq_model,
            retrier=retrier,
            backup_keys=settings.groq_backup_keys,
        )

    def _lock_for(self, url: str) -> threading.Lock:
        """Return the lock that lets only one load, delete or cleanup of a URL run at a time."""
        with self._locks_guard:
            return self._locks.setdefault(url, threading.Lock())

    def _max_age_seconds(self) -> float:
        return self._settings.page_max_age_hours * 3600

    def _idle_seconds(self) -> float:
        return self._settings.page_idle_minutes * 60

    def ingest(self, url: str, *, refresh: bool = False) -> IngestResult:
        """Make a page ready for questions.

        If a copy of the page is already stored and is younger than the maximum age, it is reused, which
        saves the fetching and the embedding. Otherwise the page is fetched and stored, replacing any old copy.
        Loads of the same address never run at the same time: a second caller waits and then reuses the result.

        Args:
            url: The page address. A missing scheme is treated as https.
            refresh: Fetch the page again even if a stored copy could be reused.

        Returns:
            The normalized URL, the page title, the number of chunks, and whether a stored copy was reused.

        Raises:
            AppError: If the URL is invalid, the page cannot be fetched, or it has no readable text.
        """
        url = normalize_url(url)
        with self._lock_for(url):
            now = self._clock()
            record = self._registry.get(url)
            reusable = (
                record is not None
                and not refresh
                and now - record.loaded_at < self._max_age_seconds()
                and self._store.has_source(url)
            )
            if record is not None and reusable:
                self._registry.touch(url, now)
                return IngestResult(
                    url=url,
                    title=record.title,
                    chunk_count=record.chunk_count,
                    reused=True,
                    age_seconds=int(now - record.loaded_at),
                )
            result = self._load(url)
            self._registry.record_load(url, result.title, result.chunk_count, self._clock())
            return result

    def _load(self, url: str) -> IngestResult:
        """Fetch, clean, chunk and embed a page, and store it in place of any earlier copy."""
        page = load_page(
            url,
            timeout=self._settings.fetch_timeout_seconds,
            max_bytes=self._settings.fetch_max_bytes,
            browser_fallback=self._settings.browser_fallback,
            min_text_chars=self._settings.min_text_chars,
        )
        text = clean_text(page.text)
        chunks = chunk_text(text, self._settings.chunk_size, self._settings.chunk_overlap)
        if not chunks:
            raise ExtractionError("This page doesn't have enough text to answer questions about. Try a different page.")

        vectors = self._embedder.embed_documents(
            [contextualize(page.title, chunk.heading, chunk.text) for chunk in chunks]
        )
        stored = [
            StoredChunk(text=chunk.text, heading=chunk.heading, vector=vector)
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self._store.replace_source(url, page.title, stored)
        return IngestResult(url=url, title=page.title, chunk_count=len(chunks))

    def ask(self, url: str, question: str) -> Answer:
        """Answer a question about a page that was loaded with :meth:`ingest`.

        Args:
            url: The page address.
            question: The user's question.

        Returns:
            The answer and the excerpts it cites.

        Raises:
            NotIngestedError: If the page has not been loaded.
        """
        url = normalize_url(url)
        if self._registry.get(url) is None or not self._store.has_source(url):
            raise NotIngestedError("Load a page first, then ask your question.")
        self._registry.touch(url, self._clock())

        query_vector = self._embedder.embed_query(question)
        chunks = self._store.search(url, question, query_vector, self._settings.retrieve_k, self._settings.hybrid_alpha)
        if not chunks:
            return Answer(answer="I couldn't find anything relevant on this page.", sources=[], retrieved=[])

        chunks = self._best_chunks(question, chunks)

        answer = self._llm.answer(question, chunks[0].title, chunks)
        sources = [CitedSource(number=n, chunk=chunks[n - 1]) for n in cited_numbers(answer, len(chunks))]
        return Answer(answer=answer, sources=sources, retrieved=chunks)

    def _best_chunks(self, question: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Pick the excerpts to answer from: the reranked best ones, or the top of the search order.

        A reranking service that is limited or briefly down must not stop a question from being answered, so in
        that case the order the search gave is used.
        """
        if self._reranker:
            try:
                return self._reranker.rerank(question, chunks, self._settings.top_k)
            except RETRYABLE_ERRORS as error:
                logger.warning("Reranking failed with %s; using the search order instead", type(error).__name__)
        return chunks[: self._settings.top_k]

    def forget(self, url: str) -> bool:
        """Delete the stored copy of a page right away.

        Anyone who is still asking about the page will load it again on their next question.

        Args:
            url: The page address.

        Returns:
            ``True`` if a stored copy existed.
        """
        url = normalize_url(url)
        with self._lock_for(url):
            known = self._registry.get(url) is not None
            self._store.delete_source(url)
            self._registry.remove(url)
        return known

    def cleanup(self) -> list[str]:
        """Delete pages that were unused for too long, are too old, or exceed the size limit.

        A page that is used between the check and the deletion is kept. Oldest-used pages go first when the
        store is over its limit.

        Returns:
            The addresses of the deleted pages.
        """
        removed: list[str] = []
        for candidate in self._registry.expired(self._clock(), self._idle_seconds(), self._max_age_seconds()):
            with self._lock_for(candidate.url):
                record = self._registry.get(candidate.url)
                if record and is_expired(record, self._clock(), self._idle_seconds(), self._max_age_seconds()):
                    self._delete(record)
                    removed.append(record.url)
        for record in self._registry.over_capacity(self._settings.max_stored_chunks):
            with self._lock_for(record.url):
                self._delete(record)
                removed.append(record.url)
        if removed:
            logger.info("Deleted %d stored page(s)", len(removed))
        return removed

    def _delete(self, record: PageRecord) -> None:
        self._store.delete_source(record.url)
        self._registry.remove(record.url)

    def reconcile(self) -> None:
        """Make the usage records match what is really stored.

        Pages found in the store that have no record (for example from an earlier version) are registered as
        just used, so that they expire normally. Records of pages that are no longer stored are dropped.
        """
        stored = self._store.list_sources()
        now = self._clock()
        known = {record.url for record in self._registry.all()}
        for url, chunk_count in stored.items():
            if url not in known:
                self._registry.record_load(url, url, chunk_count, now)
        for url in known - stored.keys():
            self._registry.remove(url)

    def close(self) -> None:
        """Close the vector database connection."""
        self._store.close()
