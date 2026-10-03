from pathlib import Path

import pytest

import app.pipeline as pipeline_module
from app.config import Settings
from app.embeddings import CohereEmbedder
from app.errors import ExtractionError, NotIngestedError
from app.extraction import ExtractedPage
from app.pipeline import Pipeline
from app.reranking import CohereReranker
from tests.fakes import TEXT, URL, FakeChat, FakeEmbedder, FakeReranker, FakeStore, make_settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return make_settings(tmp_path)


@pytest.fixture(autouse=True)
def stub_page(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        pipeline_module,
        "load_page",
        lambda url, **kwargs: ExtractedPage(url=url, title="Retrieval Guide", text=TEXT),
    )


def _pipeline(
    settings: Settings, reranker: FakeReranker | None = None
) -> tuple[Pipeline, FakeEmbedder, FakeStore, FakeChat]:
    embedder, store, chat = FakeEmbedder(), FakeStore(), FakeChat()
    return Pipeline(settings, embedder=embedder, store=store, reranker=reranker, llm=chat), embedder, store, chat


def test_ingest_stores_chunks_with_contextual_embeddings(settings: Settings) -> None:
    pipeline, embedder, store, _ = _pipeline(settings)

    result = pipeline.ingest("example.com")

    assert result.url == URL
    assert result.title == "Retrieval Guide"
    title, stored = store.sources[URL]
    assert title == "Retrieval Guide"
    assert result.chunk_count == len(stored) > 1
    assert all(len(c.text) <= 400 for c in stored)
    # Embeddings are computed from text prefixed with the title and section, but the stored text stays plain.
    assert all(text.startswith("Retrieval Guide") for text in embedder.documents)
    assert all(not c.text.startswith("Retrieval Guide") for c in stored)


def test_refreshing_replaces_previous_chunks(settings: Settings) -> None:
    pipeline, embedder, store, _ = _pipeline(settings)
    pipeline.ingest(URL)
    first = store.sources[URL][1]
    calls_after_first_load = len(embedder.documents)

    result = pipeline.ingest(URL, refresh=True)

    assert result.reused is False
    assert len(embedder.documents) == 2 * calls_after_first_load
    assert len(store.sources) == 1
    assert store.sources[URL][1] == first


def test_ingest_rejects_page_without_chunks(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        pipeline_module, "load_page", lambda url, **kwargs: ExtractedPage(url=url, title="Empty", text="  ")
    )
    pipeline, _, store, _ = _pipeline(settings)
    with pytest.raises(ExtractionError):
        pipeline.ingest(URL)
    assert store.sources == {}


def test_ask_requires_a_loaded_page(settings: Settings) -> None:
    pipeline, *_ = _pipeline(settings)
    with pytest.raises(NotIngestedError):
        pipeline.ask(URL, "What is this?")


def test_ask_without_reranker_keeps_top_k(settings: Settings) -> None:
    pipeline, embedder, store, chat = _pipeline(settings.model_copy(update={"rerank_enabled": False}))
    pipeline.ingest(URL)

    answer = pipeline.ask(URL, "What is retrieval?")

    assert answer.answer == "the answer [1][3]"
    assert embedder.queries == ["What is retrieval?"]
    assert store.search_args == (URL, "What is retrieval?", 8, 0.5)
    assert [c.chunk_index for c in answer.retrieved] == [0, 1, 2]
    assert chat.calls[0][1] == "Retrieval Guide"


def test_ask_with_reranker_uses_reranked_order(settings: Settings) -> None:
    pipeline, _, _, chat = _pipeline(settings, reranker=FakeReranker())
    pipeline.ingest(URL)

    answer = pipeline.ask(URL, "What is retrieval?")

    # The fake store returns the first 8 chunks; the fake reranker reverses them and keeps 3.
    assert [c.chunk_index for c in answer.retrieved] == [7, 6, 5]
    assert all(c.score == 0.9 for c in answer.retrieved)
    assert chat.calls[0][2] == answer.retrieved


def test_ask_returns_only_cited_sources_with_their_numbers(settings: Settings) -> None:
    pipeline, *_ = _pipeline(settings.model_copy(update={"rerank_enabled": False}))
    pipeline.ingest(URL)

    answer = pipeline.ask(URL, "What is retrieval?")

    # The answer cites [1] and [3]; excerpt [2] was retrieved but not used.
    assert [(s.number, s.chunk.chunk_index) for s in answer.sources] == [(1, 0), (3, 2)]
    assert len(answer.retrieved) == 3


def test_ask_has_no_sources_when_the_answer_cites_nothing(settings: Settings) -> None:
    store, chat = FakeStore(), FakeChat("The page does not seem to cover that.")
    pipeline = Pipeline(
        settings.model_copy(update={"rerank_enabled": False}), embedder=FakeEmbedder(), store=store, llm=chat
    )
    pipeline.ingest(URL)

    answer = pipeline.ask(URL, "Unrelated question?")

    assert answer.sources == []
    assert answer.answer == "The page does not seem to cover that."
    assert len(answer.retrieved) == 3


def test_ask_ignores_citations_to_missing_excerpts(settings: Settings) -> None:
    store, chat = FakeStore(), FakeChat("Made up [9] and real [2].")
    pipeline = Pipeline(
        settings.model_copy(update={"rerank_enabled": False}), embedder=FakeEmbedder(), store=store, llm=chat
    )
    pipeline.ingest(URL)

    answer = pipeline.ask(URL, "Question?")

    assert [s.number for s in answer.sources] == [2]


def test_default_components_are_created_from_settings(settings: Settings) -> None:
    store = FakeStore()
    pipeline = Pipeline(settings, store=store, llm=FakeChat())
    assert isinstance(pipeline._embedder, CohereEmbedder)
    assert isinstance(pipeline._reranker, CohereReranker)
    pipeline.close()
    assert store.closed
