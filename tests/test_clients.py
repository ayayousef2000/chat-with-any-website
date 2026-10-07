"""Tests for the thin wrappers around external SDKs, using fake clients instead of network calls."""

from types import SimpleNamespace
from typing import Any

import cohere.errors as cohere_errors
import groq
import httpx
import pytest
import weaviate

from app.embeddings import CohereEmbedder
from app.llm import SYSTEM_PROMPT, GroqChat
from app.reranking import CohereReranker
from app.upstream import Retrier, RetryPolicy, retry_after_seconds
from app.vector_store import RetrievedChunk, WeaviateStore


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


def _fake_groq(create: Any) -> Any:
    """A stand-in for a Groq client whose chat completions are made by ``create``."""
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def test_groq_prompt_contains_numbered_excerpts_and_question() -> None:
    captured: dict[str, Any] = {}

    def create(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="  Answer [1]  "))])

    chat = GroqChat(api_key="k", model="gpt-model")
    chat._clients = [_fake_groq(create)]

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
    chat._clients = [_fake_groq(create)]

    assert chat.answer("When?", "Page", [_chunk(0), _chunk(1)]) == "It was 2020 [1][2]."


def test_groq_system_prompt_forbids_adding_reasons_of_its_own() -> None:
    assert "do not add reasons, explanations or background of your" in SYSTEM_PROMPT


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
    chat._clients = [_fake_groq(create)]

    assert chat.answer("Why?", "Page", [_chunk(0)]) == "Answer [1]"
    assert pauses == [2.0]


def test_the_answer_error_reaches_the_caller_when_every_attempt_is_limited() -> None:
    retrier, pauses = _quick_retrier()

    def create(**kwargs: Any) -> Any:
        raise _groq_limit()

    chat = GroqChat(api_key="k", model="m", retrier=retrier)
    chat._clients = [_fake_groq(create)]

    with pytest.raises(groq.RateLimitError):
        chat.answer("Why?", "Page", [_chunk(0)])
    assert len(pauses) == 3  # four attempts


def test_the_groq_client_leaves_retrying_to_the_retrier() -> None:
    assert all(client.max_retries == 0 for client in GroqChat(api_key="k", model="m", backup_keys=["k2"])._clients)


# --- several Groq keys, used in turn ----------------------------------------------------------------------------


class Keys:
    """Six fake Groq clients that record their use; the ones named in ``limited`` answer "too many requests"."""

    def __init__(self, count: int = 6, retry_after: str | None = "30") -> None:
        self.used: list[int] = []
        self.limited: set[int] = set()
        self.retry_after = retry_after
        self.clients = [_fake_groq(self._create_for(number)) for number in range(count)]

    def _create_for(self, number: int) -> Any:
        def create(**kwargs: Any) -> Any:
            self.used.append(number)
            if number in self.limited:
                headers = {} if self.retry_after is None else {"retry-after": self.retry_after}
                response = httpx.Response(429, request=_REQUEST, headers=headers)
                raise groq.RateLimitError("limit", response=response, body=None)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=f"Answer {number} [1]"))])

        return create


def _chat_with(keys: Keys, retrier: Retrier | None = None, clock: Any = lambda: 0.0, count: int = 6) -> GroqChat:
    chat = GroqChat(
        api_key="k1", model="m", backup_keys=[f"k{n}" for n in range(2, count + 1)], retrier=retrier, clock=clock
    )
    chat._clients = keys.clients
    return chat


def _ask(chat: GroqChat) -> str:
    return chat.answer("Why?", "Page", [_chunk(0)])


def test_the_first_key_is_used_until_its_limit_is_reached() -> None:
    keys = Keys()
    chat = _chat_with(keys)

    assert [_ask(chat) for _ in range(3)] == ["Answer 0 [1]"] * 3
    assert keys.used == [0, 0, 0]


def test_each_key_takes_over_when_the_one_before_is_limited() -> None:
    keys = Keys()
    chat = _chat_with(keys)

    keys.limited = {0}
    assert _ask(chat) == "Answer 1 [1]"
    assert keys.used == [0, 1]  # the request that hit the limit moved on at once, with no waiting

    keys.used.clear()
    assert _ask(chat) == "Answer 1 [1]"
    assert keys.used == [1]  # the first key is resting, so it is not asked again

    keys.used.clear()
    keys.limited = {0, 1}
    assert _ask(chat) == "Answer 2 [1]"
    assert keys.used == [1, 2]

    keys.used.clear()
    keys.limited = {0, 1, 2, 3, 4}
    assert _ask(chat) == "Answer 5 [1]"
    assert keys.used == [2, 3, 4, 5]  # the third key is limited now, as are the fourth and fifth; the sixth answers

    keys.used.clear()
    assert _ask(chat) == "Answer 5 [1]"
    assert keys.used == [5]  # every earlier key is resting, so only the sixth is asked


def test_after_the_last_key_the_first_is_used_again_once_it_has_recovered() -> None:
    keys = Keys(count=3, retry_after="30")
    now = [0.0]
    chat = _chat_with(keys, clock=lambda: now[0], count=3)

    keys.limited = {0, 1}
    assert _ask(chat) == "Answer 2 [1]"
    assert keys.used == [0, 1, 2]

    keys.used.clear()
    keys.limited = {2}
    now[0] = 31.0  # the first and second keys have recovered
    assert _ask(chat) == "Answer 0 [1]"
    assert keys.used == [2, 0]  # the circle went on from the third key to the first


def test_a_key_that_has_recovered_is_not_returned_to_while_the_current_one_still_works() -> None:
    keys = Keys(count=3, retry_after="30")
    now = [0.0]
    chat = _chat_with(keys, clock=lambda: now[0], count=3)

    keys.limited = {0}
    assert _ask(chat) == "Answer 1 [1]"
    keys.limited = set()
    now[0] = 31.0  # the first key has recovered, but the second one is in use and not limited
    keys.used.clear()
    assert [_ask(chat) for _ in range(2)] == ["Answer 1 [1]"] * 2
    assert keys.used == [1, 1]


def test_a_key_that_is_still_resting_is_skipped_in_the_circle() -> None:
    keys = Keys(count=3, retry_after="30")
    now = [0.0]
    chat = _chat_with(keys, clock=lambda: now[0], count=3)

    keys.limited = {0, 1}
    assert _ask(chat) == "Answer 2 [1]"
    keys.used.clear()
    keys.limited = {2}
    now[0] = 10.0  # the first and second keys are still resting
    with pytest.raises(groq.RateLimitError):
        _ask(chat)
    assert keys.used == [2]


def test_a_key_with_no_stated_wait_rests_for_a_minute() -> None:
    keys = Keys(count=2, retry_after=None)
    now = [0.0]
    retrier = Retrier(RetryPolicy(attempts=1), sleep=lambda _: None, jitter=lambda _: 0.0)
    chat = _chat_with(keys, retrier=retrier, clock=lambda: now[0], count=2)

    keys.limited = {0}
    assert _ask(chat) == "Answer 1 [1]"
    keys.used.clear()
    keys.limited = {1}
    now[0] = 59.0
    with pytest.raises(groq.RateLimitError):
        _ask(chat)
    assert keys.used == [1]  # the first key had rested for less than a minute, so it was not asked

    keys.used.clear()
    keys.limited = set()
    now[0] = 61.0
    assert _ask(chat) == "Answer 0 [1]"  # a minute has passed for the first key, the second one still rests
    assert keys.used == [0]


def test_when_every_key_is_limited_the_shortest_wait_is_reported() -> None:
    keys = Keys(count=3, retry_after="40")
    retrier = Retrier(RetryPolicy(attempts=1), sleep=lambda _: None, jitter=lambda _: 0.0)
    chat = GroqChat(api_key="k1", model="m", backup_keys=["k2", "k3"], retrier=retrier, clock=lambda: 0.0)
    chat._clients = keys.clients
    keys.limited = {0, 1, 2}

    with pytest.raises(groq.RateLimitError) as raised:
        _ask(chat)

    assert keys.used == [0, 1, 2]  # all three were tried before giving up
    assert retry_after_seconds(raised.value) == 40.0


def test_a_request_that_waits_continues_with_a_key_that_has_recovered() -> None:
    keys = Keys(count=2, retry_after="10")
    now = [0.0]
    pauses: list[float] = []

    def sleep(seconds: float) -> None:
        pauses.append(seconds)
        now[0] += seconds

    retrier = Retrier(RetryPolicy(), sleep=sleep, jitter=lambda _: 0.0)
    chat = GroqChat(api_key="k1", model="m", backup_keys=["k2"], retrier=retrier, clock=lambda: now[0])
    chat._clients = keys.clients
    original = keys.clients[0].chat.completions.create
    calls = [0]

    def first_key(**kwargs: Any) -> Any:
        calls[0] += 1
        if calls[0] == 1:
            keys.limited = {0, 1}  # both keys are limited on the first try, then they recover
        elif calls[0] == 2:
            keys.limited = set()
        return original(**kwargs)

    keys.clients[0].chat.completions.create = first_key

    assert _ask(chat) == "Answer 0 [1]"
    assert pauses == [10.0]


def test_a_repeated_key_is_used_only_once() -> None:
    chat = GroqChat(api_key="k1", model="m", backup_keys=["k1", "k2", "k2"])
    assert len(chat._clients) == 2


# --- answers in the wrong language ------------------------------------------------------------------------------


def _chat_answering(*answers: str) -> tuple[GroqChat, list[int]]:
    calls: list[int] = []

    def create(**kwargs: Any) -> Any:
        calls.append(1)
        text = answers[min(len(calls), len(answers)) - 1]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])

    chat = GroqChat(api_key="k", model="m")
    chat._clients = [_fake_groq(create)]
    return chat, calls


def test_an_answer_in_chinese_to_an_english_question_is_asked_again() -> None:
    chat, calls = _chat_answering("控制 JSON 输出 [1]", "It controls the JSON output [1]")

    assert chat.answer("What does it do?", "Page", [_chunk(0)]) == "It controls the JSON output [1]"
    assert len(calls) == 2


def test_the_wrong_language_is_asked_again_only_a_few_times() -> None:
    chat, calls = _chat_answering("控制 JSON 输出 [1]")

    assert chat.answer("What does it do?", "Page", [_chunk(0)]) == "控制 JSON 输出 [1]"
    assert len(calls) == 3  # the first try and two more


def test_an_answer_in_chinese_to_a_chinese_question_is_kept() -> None:
    chat, calls = _chat_answering("它控制 JSON 输出 [1]")

    assert chat.answer("它有什么作用？", "Page", [_chunk(0)]) == "它控制 JSON 输出 [1]"
    assert len(calls) == 1


def test_an_english_answer_to_an_arabic_question_is_not_asked_again() -> None:
    chat, calls = _chat_answering("It controls the output [1]")

    chat.answer("ماذا يفعل؟", "Page", [_chunk(0)])
    assert len(calls) == 1


# --- the general request used by the evaluation judge -----------------------------------------------------------


def test_complete_sends_the_messages_and_options_and_returns_the_text() -> None:
    captured: dict[str, Any] = {}

    def create(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='  {"ok": true}  '))])

    chat = GroqChat(api_key="k", model="judge-model")
    chat._clients = [_fake_groq(create)]
    messages: list[Any] = [{"role": "user", "content": "hi"}]

    assert chat.complete(messages, temperature=0, response_format={"type": "json_object"}) == '{"ok": true}'
    assert captured["model"] == "judge-model"
    assert captured["messages"] == messages
    assert captured["temperature"] == 0
    assert captured["response_format"] == {"type": "json_object"}


def test_complete_also_moves_on_to_the_next_key_when_a_limit_is_reached() -> None:
    keys = Keys()
    chat = _chat_with(keys)
    keys.limited = {0}

    assert chat.complete([{"role": "user", "content": "hi"}]) == "Answer 1 [1]"
    assert keys.used == [0, 1]


# --- connecting to Weaviate ---------------------------------------------------------------------------------------


class _FakeCollections:
    def exists(self, name: str) -> bool:
        return True

    def use(self, name: str) -> str:
        return name


class _FakeWeaviate:
    collections = _FakeCollections()


def test_weaviate_gets_the_configured_time_for_its_startup_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def connect(**kwargs: Any) -> _FakeWeaviate:
        captured.update(kwargs)
        return _FakeWeaviate()

    monkeypatch.setattr(weaviate, "connect_to_weaviate_cloud", connect)

    WeaviateStore("https://cluster.example", "key", "Chunks", init_timeout=25)

    assert captured["additional_config"].timeout.init == 25
    assert captured["cluster_url"] == "https://cluster.example"


def test_the_startup_time_for_weaviate_is_30_seconds_unless_set() -> None:
    from app.config import Settings

    assert Settings.model_fields["weaviate_init_timeout_seconds"].default == 30
