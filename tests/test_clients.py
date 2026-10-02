"""Tests for the thin wrappers around external SDKs, using fake clients instead of network calls."""

from types import SimpleNamespace
from typing import Any

import pytest

from app.embeddings import CohereEmbedder
from app.llm import GroqChat
from app.reranking import CohereReranker
from app.vector_store import RetrievedChunk


def _chunk(index: int, heading: str = "Intro") -> RetrievedChunk:
    return RetrievedChunk(title="Page", heading=heading, text=f"text {index}", chunk_index=index, score=0.1)


class FakeCohere:
    def __init__(self) -> None:
        self.embed_calls: list[dict[str, Any]] = []
        self.rerank_calls: list[dict[str, Any]] = []

    def embed(self, **kwargs: Any) -> Any:
        self.embed_calls.append(kwargs)
        vectors = [[0.0, 1.0] for _ in kwargs["texts"]]
        return SimpleNamespace(embeddings=SimpleNamespace(float_=vectors))

    def rerank(self, **kwargs: Any) -> Any:
        self.rerank_calls.append(kwargs)
        results = [SimpleNamespace(index=2, relevance_score=0.9), SimpleNamespace(index=0, relevance_score=0.5)]
        return SimpleNamespace(results=results)


def test_embed_documents_batches_and_sets_input_type() -> None:
    fake = FakeCohere()
    embedder = CohereEmbedder(api_key="k", model="m", dimension=8)
    embedder._client = fake  # type: ignore[assignment]

    vectors = embedder.embed_documents([f"t{i}" for i in range(200)])

    assert len(vectors) == 200
    assert [len(call["texts"]) for call in fake.embed_calls] == [96, 96, 8]
    assert {call["input_type"] for call in fake.embed_calls} == {"search_document"}
    assert fake.embed_calls[0]["model"] == "m"
    assert fake.embed_calls[0]["output_dimension"] == 8
    assert fake.embed_calls[0]["embedding_types"] == ["float"]


def test_embed_query_uses_query_input_type() -> None:
    fake = FakeCohere()
    embedder = CohereEmbedder(api_key="k", model="m", dimension=8)
    embedder._client = fake  # type: ignore[assignment]

    assert embedder.embed_query("hello") == [0.0, 1.0]
    assert fake.embed_calls[0]["input_type"] == "search_query"


def test_embedder_detects_missing_vectors() -> None:
    class Broken(FakeCohere):
        def embed(self, **kwargs: Any) -> Any:
            return SimpleNamespace(embeddings=SimpleNamespace(float_=[[1.0]]))

    embedder = CohereEmbedder(api_key="k", model="m", dimension=8)
    embedder._client = Broken()  # type: ignore[assignment]
    with pytest.raises(RuntimeError, match="unexpected number"):
        embedder.embed_documents(["a", "b"])


def test_rerank_returns_chunks_in_reranked_order_with_new_scores() -> None:
    fake = FakeCohere()
    reranker = CohereReranker(api_key="k", model="rerank-model")
    reranker._client = fake  # type: ignore[assignment]
    chunks = [_chunk(0), _chunk(1), _chunk(2, heading="")]

    result = reranker.rerank("question", chunks, top_n=2)

    assert [(c.chunk_index, c.score) for c in result] == [(2, 0.9), (0, 0.5)]
    call = fake.rerank_calls[0]
    assert call["model"] == "rerank-model"
    assert call["top_n"] == 2
    assert call["documents"] == ["Intro\ntext 0", "Intro\ntext 1", "text 2"]


def test_rerank_skips_api_for_single_chunk() -> None:
    fake = FakeCohere()
    reranker = CohereReranker(api_key="k", model="m")
    reranker._client = fake  # type: ignore[assignment]
    chunks = [_chunk(0)]
    assert reranker.rerank("q", chunks, top_n=5) == chunks
    assert fake.rerank_calls == []


def test_groq_prompt_contains_numbered_excerpts_and_question() -> None:
    captured: dict[str, Any] = {}

    def create(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="  Answer [1]  "))])

    chat = GroqChat(api_key="k", model="gpt-model")
    chat._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))  # type: ignore[assignment]

    answer = chat.answer("What is it?", "Page", [_chunk(0), _chunk(1, heading="")])

    assert answer == "Answer [1]"
    assert captured["model"] == "gpt-model"
    system, user = (m["content"] for m in captured["messages"])
    assert "only the numbered excerpts" in system
    assert "[1] (section: Intro)\ntext 0" in user
    assert "[2]\ntext 1" in user
    assert user.endswith("Question: What is it?")
