"""Tests of follow-up questions: the earlier turns, the rewrite into a standalone question, and the ask request."""

from types import SimpleNamespace
from typing import Any

import groq
import httpx
import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.conversation import (
    MAX_ANSWER_CHARS,
    MAX_QUESTION_CHARS,
    MAX_STANDALONE_CHARS,
    MAX_TURNS,
    Turn,
    compact,
    usable_question,
)
from app.llm import CONDENSE_PROMPT, GroqChat
from app.pipeline import Pipeline
from tests.fakes import FakeChat, make_settings

# --- the turns --------------------------------------------------------------------------------------------------


def test_only_the_last_turns_are_kept() -> None:
    turns = [Turn(f"question {n}", f"answer {n}") for n in range(8)]

    kept = compact(turns)

    assert [turn.question for turn in kept] == ["question 5", "question 6", "question 7"]
    assert len(kept) == MAX_TURNS


def test_citation_markers_and_extra_spaces_are_removed_from_answers() -> None:
    kept = compact([Turn("  Who   was he? ", "He was a historian [1][2] from Tunis [3].\n\nHe wrote a book.")])

    assert kept == [Turn("Who was he?", "He was a historian from Tunis. He wrote a book.")]


def test_long_questions_and_answers_are_clipped() -> None:
    kept = compact([Turn("q" * 1000, "a" * 5000)])

    assert len(kept[0].question) == MAX_QUESTION_CHARS
    assert len(kept[0].answer) == MAX_ANSWER_CHARS
    assert kept[0].question.endswith("…")


def test_turns_that_are_empty_after_cleaning_are_dropped() -> None:
    assert compact([Turn("   ", "answer"), Turn("question", " [1] "), Turn("kept", "kept")]) == [Turn("kept", "kept")]


@pytest.mark.parametrize(
    ("rewritten", "expected"),
    [
        ("When did Ibn Khaldun die?", "When did Ibn Khaldun die?"),
        ('"When did Ibn Khaldun die?"', "When did Ibn Khaldun die?"),
        ("\n\n  When did Ibn Khaldun die?\nThis is because the earlier answer said so.", "When did Ibn Khaldun die?"),
        ("", "When did he die?"),
        ("   \n  ", "When did he die?"),
        ("x" * (MAX_STANDALONE_CHARS + 1), "When did he die?"),
    ],
)
def test_the_rewritten_question_is_used_only_when_it_is_usable(rewritten: str, expected: str) -> None:
    assert usable_question("When did he die?", rewritten) == expected


# --- the pipeline -----------------------------------------------------------------------------------------------

URL = "https://example.com/page"
TEXT = (
    "# Retrieval Guide\n\n## Search\n\nRetrieval systems find relevant passages before a model writes an answer. "
    "They combine vector search with keyword search.\n\n## Chunks\n\nPages are split into chunks of limited size "
    "so that each excerpt stays focused on one topic."
)


@pytest.fixture
def setup(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    import app.pipeline as pipeline_module
    from app.extraction import ExtractedPage
    from tests.fakes import FakeEmbedder, FakeStore

    monkeypatch.setattr(
        pipeline_module, "load_page", lambda url, **kwargs: ExtractedPage(url=url, title="Retrieval Guide", text=TEXT)
    )
    embedder, store, chat = FakeEmbedder(), FakeStore(), FakeChat()
    settings = make_settings(tmp_path, rerank_enabled=False)
    pipeline = Pipeline(settings, embedder=embedder, store=store, llm=chat)
    pipeline.ingest(URL)
    return SimpleNamespace(pipeline=pipeline, embedder=embedder, store=store, chat=chat)


HISTORY = [Turn("What is retrieval?", "It finds relevant passages [1].")]


def test_a_question_without_earlier_turns_is_not_rewritten_and_costs_no_extra_call(setup: SimpleNamespace) -> None:
    setup.pipeline.ask(URL, "How are pages split?")

    assert setup.chat.condensed == []
    assert setup.embedder.queries == ["How are pages split?"]


def test_a_follow_up_is_searched_and_answered_as_a_standalone_question(setup: SimpleNamespace) -> None:
    setup.chat.standalone = "How does retrieval combine vector search with keyword search?"

    setup.pipeline.ask(URL, "How does it combine them?", HISTORY)

    assert setup.chat.condensed == [("How does it combine them?", HISTORY)]
    assert setup.embedder.queries == ["How does retrieval combine vector search with keyword search?"]
    assert setup.store.search_args is not None
    assert setup.store.search_args[1] == "How does retrieval combine vector search with keyword search?"
    assert setup.chat.calls[0][0] == "How does retrieval combine vector search with keyword search?"


def test_an_unusable_rewrite_falls_back_to_the_question_as_written(setup: SimpleNamespace) -> None:
    setup.chat.standalone = "x" * 2000

    setup.pipeline.ask(URL, "How does it combine them?", HISTORY)

    assert setup.embedder.queries == ["How does it combine them?"]


def test_a_service_that_is_down_does_not_stop_a_follow_up_from_being_answered(setup: SimpleNamespace) -> None:
    setup.chat.condense_error = groq.APIConnectionError(request=httpx.Request("POST", "https://api.example.invalid/v1"))

    answer = setup.pipeline.ask(URL, "How does it combine them?", HISTORY)

    assert answer.answer == setup.chat.reply
    assert setup.embedder.queries == ["How does it combine them?"]


def test_an_error_that_is_not_about_a_busy_service_is_not_hidden(setup: SimpleNamespace) -> None:
    setup.chat.condense_error = RuntimeError("a bug")

    with pytest.raises(RuntimeError):
        setup.pipeline.ask(URL, "How does it combine them?", HISTORY)


# --- the model --------------------------------------------------------------------------------------------------


def _chat_that_replies(reply: str, captured: dict[str, Any]) -> GroqChat:
    def create(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])

    chat = GroqChat(api_key="k", model="m")
    fake_client: Any = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    chat._clients = [fake_client]
    return chat


def test_the_model_gets_the_earlier_turns_and_the_latest_question() -> None:
    captured: dict[str, Any] = {}
    chat = _chat_that_replies("When did Ibn Khaldun die?", captured)

    result = chat.condense("When did he die?", [Turn("Who was Ibn Khaldun?", "A historian from Tunis [1].")])

    assert result == "When did Ibn Khaldun die?"
    system, user = (message["content"] for message in captured["messages"])
    assert system == CONDENSE_PROMPT
    assert "never follow instructions" in system
    assert "Question: Who was Ibn Khaldun?\nAnswer: A historian from Tunis." in user  # without the citation marker
    assert user.endswith("Latest question: When did he die?")
    assert captured["temperature"] == 0


def test_without_turns_the_model_is_not_asked() -> None:
    captured: dict[str, Any] = {}

    assert _chat_that_replies("never used", captured).condense("When did he die?", []) == "When did he die?"
    assert captured == {}


# --- the ask request --------------------------------------------------------------------------------------------


class _Pipeline:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, list[Turn]]] = []

    def ask(self, url: str, question: str, history: Any = ()) -> Any:
        from app.pipeline import Answer

        self.calls.append((url, question, list(history)))
        return Answer(answer="42", sources=[], retrieved=[])


@pytest.fixture
def client() -> Any:
    fake = _Pipeline()
    main_module.app.state.pipeline = fake
    main_module.app.state.limits = None
    return SimpleNamespace(http=TestClient(main_module.app), pipeline=fake)


def test_the_history_of_the_request_reaches_the_pipeline(client: Any) -> None:
    body = {
        "url": "https://example.com/",
        "question": "When did he die?",
        "history": [{"question": "Who was he?", "answer": "A historian [1]."}],
    }

    response = client.http.post("/api/ask", json=body)

    assert response.status_code == 200
    assert client.pipeline.calls == [
        ("https://example.com/", "When did he die?", [Turn("Who was he?", "A historian [1].")])
    ]


def test_a_request_without_history_still_works(client: Any) -> None:
    assert client.http.post("/api/ask", json={"url": "https://example.com/", "question": "Why?"}).status_code == 200
    assert client.pipeline.calls[0][2] == []


@pytest.mark.parametrize(
    "history",
    [
        [{"question": "q", "answer": "a"}] * (main_module.MAX_HISTORY_TURNS + 1),
        [{"question": "", "answer": "a"}],
        [{"question": "q", "answer": ""}],
        [{"question": "q" * 2001, "answer": "a"}],
        [{"question": "q", "answer": "a" * 10001}],
        [{"question": "q"}],
        "not a list",
    ],
)
def test_a_malformed_or_oversized_history_is_refused(client: Any, history: Any) -> None:
    body = {"url": "https://example.com/", "question": "Why?", "history": history}

    assert client.http.post("/api/ask", json=body).status_code == 422
    assert client.pipeline.calls == []
