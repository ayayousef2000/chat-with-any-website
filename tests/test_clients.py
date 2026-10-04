"""Tests for the thin wrappers around external SDKs, using fake clients instead of network calls."""

from types import SimpleNamespace
from typing import Any

import cohere.errors as cohere_errors
import groq
import httpx
import pytest

from app.embeddings import CohereEmbedder
from app.llm import SYSTEM_PROMPT, GroqChat
from app.reranking import CohereReranker
from app.upstream import Retrier, RetryPolicy
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


def test_groq_answer_citations_are_normalized() -> None:
    def create(**kwargs: Any) -> Any:
        message = SimpleNamespace(content="It was 2020 【1†L1-L3】 【2†L1-L2】.")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    chat = GroqChat(api_key="k", model="m")
    chat._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))  # type: ignore[assignment]

    assert chat.answer("When?", "Page", [_chunk(0), _chunk(1)]) == "It was 2020 [1][2]."


def test_groq_system_prompt_forbids_other_citation_styles() -> None:
    assert "square brackets" in SYSTEM_PROMPT
    assert "never mention line numbers" in SYSTEM_PROMPT


# --- trying again when a service limits or fails ----------------------------------------------------------------

_REQUEST = httpx.Request("POST", "https://api.example.invalid/v1")


def _quick_retrier() -> tuple[Retrier, list[float]]:
    pauses: list[float] = []
    return Retrier(RetryPolicy(), sleep=pauses.append, jitter=lambda _: 0.0), pauses


def _groq_limit() -> groq.RateLimitError:
    response = httpx.Response(429, request=_REQUEST, headers={"retry-after": "2"})
    return groq.RateLimitError("limit", response=response, body=None)


def test_embeddings_are_tried_again_after_a_rate_limit_and_the_client_does_not_retry_by_itself() -> None:
    retrier, pauses = _quick_retrier()
    fake = FakeCohere()
    original = fake.embed
    attempts: list[dict[str, Any]] = []

    def embed(**kwargs: Any) -> Any:
        attempts.append(kwargs)
        if len(attempts) == 1:
            raise cohere_errors.TooManyRequestsError(body={"message": "limit"}, headers={"retry-after": "4"})
        return original(**kwargs)

    fake.embed = embed  # type: ignore[method-assign]
    embedder = CohereEmbedder(api_key="k", model="m", dimension=8, retrier=retrier)
    embedder._client = fake  # type: ignore[assignment]

    assert embedder.embed_query("hello") == [0.0, 1.0]
    assert pauses == [4.0]
    assert all(call["request_options"] == {"max_retries": 0} for call in attempts)


def test_each_batch_of_documents_is_retried_on_its_own() -> None:
    retrier, pauses = _quick_retrier()
    fake = FakeCohere()
    original = fake.embed
    seen: list[int] = []

    def embed(**kwargs: Any) -> Any:
        seen.append(len(kwargs["texts"]))
        if len(seen) == 2:  # the second batch fails once
            raise cohere_errors.ServiceUnavailableError(body={"message": "down"})
        return original(**kwargs)

    fake.embed = embed  # type: ignore[method-assign]
    embedder = CohereEmbedder(api_key="k", model="m", dimension=8, retrier=retrier)
    embedder._client = fake  # type: ignore[assignment]

    assert len(embedder.embed_documents([f"t{i}" for i in range(150)])) == 150
    assert seen == [96, 54, 54]  # the failed second batch was sent again, the first was not
    assert len(pauses) == 1


def test_reranking_is_tried_again_after_a_rate_limit() -> None:
    retrier, pauses = _quick_retrier()
    fake = FakeCohere()
    original = fake.rerank
    calls: list[dict[str, Any]] = []

    def rerank(**kwargs: Any) -> Any:
        calls.append(kwargs)
        if len(calls) == 1:
            raise cohere_errors.TooManyRequestsError(body={"message": "limit"})
        return original(**kwargs)

    fake.rerank = rerank  # type: ignore[method-assign]
    reranker = CohereReranker(api_key="k", model="m", retrier=retrier)
    reranker._client = fake  # type: ignore[assignment]

    result = reranker.rerank("q", [_chunk(0), _chunk(1), _chunk(2)], top_n=2)

    assert [c.chunk_index for c in result] == [2, 0]
    assert pauses == [1.0]
    assert all(call["request_options"] == {"max_retries": 0} for call in calls)


def test_the_answer_is_requested_again_after_a_rate_limit() -> None:
    retrier, pauses = _quick_retrier()
    calls: list[int] = []

    def create(**kwargs: Any) -> Any:
        calls.append(1)
        if len(calls) == 1:
            raise _groq_limit()
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Answer [1]"))])

    chat = GroqChat(api_key="k", model="m", retrier=retrier)
    chat._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))  # type: ignore[assignment]

    assert chat.answer("Why?", "Page", [_chunk(0)]) == "Answer [1]"
    assert pauses == [2.0]


def test_the_answer_error_reaches_the_caller_when_every_attempt_is_limited() -> None:
    retrier, pauses = _quick_retrier()

    def create(**kwargs: Any) -> Any:
        raise _groq_limit()

    chat = GroqChat(api_key="k", model="m", retrier=retrier)
    chat._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))  # type: ignore[assignment]

    with pytest.raises(groq.RateLimitError):
        chat.answer("Why?", "Page", [_chunk(0)])
    assert len(pauses) == 3  # four attempts


def test_the_groq_client_leaves_retrying_to_the_retrier() -> None:
    assert GroqChat(api_key="k", model="m")._client.max_retries == 0
