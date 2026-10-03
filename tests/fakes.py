"""Stand-ins for the external services and the clock, shared by the pipeline tests."""

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from app.config import Settings
from app.vector_store import RetrievedChunk, StoredChunk

URL = "https://example.com/"
OTHER_URL = "https://example.org/other"
TEXT = "# Guide\n\n" + "\n\n".join(f"Paragraph {i} about retrieval. " + "word " * 60 for i in range(12))


class FakeClock:
    """A clock that only moves when told to."""

    def __init__(self, start: float = 1_000_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float = 0, minutes: float = 0, hours: float = 0) -> None:
        self.now += seconds + minutes * 60 + hours * 3600


class FakeEmbedder:
    def __init__(self) -> None:
        self.documents: list[str] = []
        self.queries: list[str] = []

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.documents.extend(texts)
        return [[float(len(text)), 1.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return [1.0, 1.0]


class FakeStore:
    def __init__(self) -> None:
        self.sources: dict[str, tuple[str, list[StoredChunk]]] = {}
        self.search_args: tuple[str, str, int, float] | None = None
        self.closed = False

    def replace_source(self, url: str, title: str, chunks: Sequence[StoredChunk]) -> None:
        self.sources[url] = (title, list(chunks))

    def has_source(self, url: str) -> bool:
        return url in self.sources

    def delete_source(self, url: str) -> None:
        self.sources.pop(url, None)

    def list_sources(self) -> dict[str, int]:
        return {url: len(chunks) for url, (_, chunks) in self.sources.items()}

    def search(self, url: str, query: str, vector: Sequence[float], limit: int, alpha: float) -> list[RetrievedChunk]:
        self.search_args = (url, query, limit, alpha)
        title, chunks = self.sources[url]
        return [
            RetrievedChunk(title=title, heading=c.heading, text=c.text, chunk_index=i, score=1.0 / (i + 1))
            for i, c in enumerate(chunks[:limit])
        ]

    def close(self) -> None:
        self.closed = True


class FakeReranker:
    def rerank(self, query: str, chunks: list[RetrievedChunk], top_n: int) -> list[RetrievedChunk]:
        return [replace(c, score=0.9) for c in reversed(chunks)][:top_n]


class FakeChat:
    def __init__(self, reply: str = "the answer [1][3]") -> None:
        self.reply = reply
        self.calls: list[tuple[str, str, list[RetrievedChunk]]] = []

    def answer(self, question: str, title: str, chunks: Sequence[RetrievedChunk]) -> str:
        self.calls.append((question, title, list(chunks)))
        return self.reply


def make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "cohere_api_key": "c",
        "weaviate_url": "https://example.weaviate.cloud",
        "weaviate_api_key": "w",
        "groq_api_key": "g",
        "chunk_size": 400,
        "chunk_overlap": 50,
        "retrieve_k": 8,
        "top_k": 3,
        "hybrid_alpha": 0.5,
        "state_db_path": str(tmp_path / "state.db"),
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[call-arg]
