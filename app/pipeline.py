"""Wires the steps together: ingest a URL, then answer questions about it."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from app.chunking import chunk_text, contextualize
from app.cleaning import clean_text
from app.config import Settings
from app.embeddings import CohereEmbedder
from app.errors import ExtractionError, NotIngestedError
from app.extraction import load_page, normalize_url
from app.llm import GroqChat
from app.reranking import CohereReranker
from app.vector_store import RetrievedChunk, StoredChunk, WeaviateStore


@dataclass(frozen=True)
class IngestResult:
    """Summary of a page that was loaded."""

    url: str
    title: str
    chunk_count: int


@dataclass(frozen=True)
class Answer:
    """An answer together with the chunks it was generated from."""

    answer: str
    sources: list[RetrievedChunk]


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
    """

    def __init__(
        self,
        settings: Settings,
        *,
        embedder: Embedder | None = None,
        store: Store | None = None,
        reranker: Reranker | None = None,
        llm: Chat | None = None,
    ) -> None:
        self._settings = settings
        self._embedder: Embedder = embedder or CohereEmbedder(
            api_key=settings.cohere_api_key,
            model=settings.cohere_embed_model,
            dimension=settings.cohere_embed_dimension,
        )
        self._store: Store = store or WeaviateStore(
            url=settings.weaviate_url,
            api_key=settings.weaviate_api_key,
            collection_name=settings.weaviate_collection,
        )
        self._reranker: Reranker | None = reranker
        if self._reranker is None and settings.rerank_enabled:
            self._reranker = CohereReranker(api_key=settings.cohere_api_key, model=settings.cohere_rerank_model)
        self._llm: Chat = llm or GroqChat(api_key=settings.groq_api_key, model=settings.groq_model)

    def ingest(self, url: str) -> IngestResult:
        """Load a page so questions can be asked about it.

        Any chunks previously stored for the same URL are replaced.

        Args:
            url: The page address. A missing scheme is treated as https.

        Returns:
            The normalized URL, the page title and the number of chunks stored.

        Raises:
            AppError: If the URL is invalid, the page cannot be fetched, or it has no readable text.
        """
        url = normalize_url(url)
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
            raise ExtractionError("The page did not contain enough text to index.")

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
            The answer and the excerpts it is based on.

        Raises:
            NotIngestedError: If the page has not been loaded.
        """
        url = normalize_url(url)
        if not self._store.has_source(url):
            raise NotIngestedError("This URL has not been loaded yet. Load it first, then ask your question.")

        query_vector = self._embedder.embed_query(question)
        chunks = self._store.search(url, question, query_vector, self._settings.retrieve_k, self._settings.hybrid_alpha)
        if not chunks:
            return Answer(answer="I could not find anything relevant on this page.", sources=[])

        if self._reranker:
            chunks = self._reranker.rerank(question, chunks, self._settings.top_k)
        else:
            chunks = chunks[: self._settings.top_k]

        answer = self._llm.answer(question, chunks[0].title, chunks)
        return Answer(answer=answer, sources=chunks)

    def close(self) -> None:
        """Close the vector database connection."""
        self._store.close()
