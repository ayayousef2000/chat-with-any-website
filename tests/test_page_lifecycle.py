"""How stored pages are shared, reused and deleted."""

import threading
import time
from pathlib import Path

import pytest

import app.pipeline as pipeline_module
from app.errors import NotIngestedError
from app.extraction import ExtractedPage
from app.janitor import Janitor
from app.pipeline import IngestResult, Pipeline
from app.registry import PageRecord, PageRegistry, is_expired
from tests.fakes import OTHER_URL, TEXT, URL, FakeChat, FakeClock, FakeEmbedder, FakeStore, make_settings

IDLE = 15 * 60
MAX_AGE = 12 * 3600


@pytest.fixture(autouse=True)
def stub_page(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        pipeline_module,
        "load_page",
        lambda url, **kwargs: ExtractedPage(url=url, title=f"Title of {url}", text=TEXT),
    )


class Setup:
    """A pipeline wired to fakes, with a clock that is moved by hand."""

    def __init__(self, tmp_path: Path, **overrides: object) -> None:
        self.clock = FakeClock()
        self.embedder = FakeEmbedder()
        self.store = FakeStore()
        self.settings = make_settings(tmp_path, rerank_enabled=False, **overrides)
        self.registry = PageRegistry(self.settings.state_db_path)
        self.pipeline = Pipeline(
            self.settings,
            embedder=self.embedder,
            store=self.store,
            llm=FakeChat("answer [1]"),
            registry=self.registry,
            clock=self.clock,
        )


@pytest.fixture
def setup(tmp_path: Path) -> Setup:
    return Setup(tmp_path)


# --- registry --------------------------------------------------------------------------------------------------


def _record(url: str = URL, loaded_at: float = 100, last_used_at: float = 100, chunks: int = 10) -> PageRecord:
    return PageRecord(url=url, title="t", chunk_count=chunks, loaded_at=loaded_at, last_used_at=last_used_at)


def test_page_is_expired_after_the_idle_time_or_the_maximum_age() -> None:
    record = _record(loaded_at=0, last_used_at=0)
    assert not is_expired(record, now=IDLE - 1, idle_seconds=IDLE, max_age_seconds=MAX_AGE)
    assert is_expired(record, now=IDLE, idle_seconds=IDLE, max_age_seconds=MAX_AGE)
    # Still in use, but older than the maximum age.
    used = _record(loaded_at=0, last_used_at=MAX_AGE - 1)
    assert not is_expired(used, now=MAX_AGE - 1, idle_seconds=IDLE, max_age_seconds=MAX_AGE)
    assert is_expired(used, now=MAX_AGE, idle_seconds=IDLE, max_age_seconds=MAX_AGE)


def test_registry_stores_and_updates_records(tmp_path: Path) -> None:
    registry = PageRegistry(str(tmp_path / "nested" / "state.db"))
    assert registry.get(URL) is None

    registry.record_load(URL, "Title", 7, now=50)
    assert registry.get(URL) == PageRecord(URL, "Title", 7, loaded_at=50, last_used_at=50)

    registry.touch(URL, now=80)
    record = registry.get(URL)
    assert record is not None
    assert (record.loaded_at, record.last_used_at) == (50, 80)

    registry.record_load(URL, "New title", 9, now=120)
    assert registry.get(URL) == PageRecord(URL, "New title", 9, loaded_at=120, last_used_at=120)

    assert registry.remove(URL) is True
    assert registry.remove(URL) is False
    assert registry.all() == []


def test_registry_survives_a_restart(tmp_path: Path) -> None:
    path = str(tmp_path / "state.db")
    PageRegistry(path).record_load(URL, "Title", 3, now=10)
    assert PageRegistry(path).get(URL) is not None


def test_registry_lists_expired_pages(tmp_path: Path) -> None:
    registry = PageRegistry(str(tmp_path / "state.db"))
    registry.record_load(URL, "old", 1, now=0)
    registry.record_load(OTHER_URL, "recent", 1, now=IDLE)
    assert [r.url for r in registry.expired(now=IDLE + 1, idle_seconds=IDLE, max_age_seconds=MAX_AGE)] == [URL]


def test_registry_picks_least_recently_used_pages_over_capacity(tmp_path: Path) -> None:
    registry = PageRegistry(str(tmp_path / "state.db"))
    registry.record_load("https://a/", "a", 40, now=1)
    registry.record_load("https://b/", "b", 40, now=2)
    registry.record_load("https://c/", "c", 40, now=3)
    registry.touch("https://a/", now=4)  # a is now the most recently used

    assert [r.url for r in registry.over_capacity(max_chunks=120)] == []
    assert [r.url for r in registry.over_capacity(max_chunks=100)] == ["https://b/"]
    assert [r.url for r in registry.over_capacity(max_chunks=50)] == ["https://b/", "https://c/"]


# --- sharing and reuse -----------------------------------------------------------------------------------------


def test_a_second_guest_reuses_the_stored_copy(setup: Setup) -> None:
    first = setup.pipeline.ingest(URL)
    embedded = len(setup.embedder.documents)
    setup.clock.advance(minutes=4)

    second = setup.pipeline.ingest(URL)

    assert first.reused is False
    assert second == IngestResult(
        url=URL, title=first.title, chunk_count=first.chunk_count, reused=True, age_seconds=240
    )
    assert len(setup.embedder.documents) == embedded  # nothing was fetched or embedded again


def test_different_urls_are_stored_separately(setup: Setup) -> None:
    setup.pipeline.ingest(URL)
    setup.pipeline.ingest(OTHER_URL)

    assert set(setup.store.sources) == {URL, OTHER_URL}
    assert setup.store.sources[URL][0] != setup.store.sources[OTHER_URL][0]
    assert setup.pipeline.forget(URL) is True
    assert set(setup.store.sources) == {OTHER_URL}  # the other page is untouched


def test_refresh_fetches_the_page_again(setup: Setup) -> None:
    setup.pipeline.ingest(URL)
    embedded = len(setup.embedder.documents)

    result = setup.pipeline.ingest(URL, refresh=True)

    assert result.reused is False
    assert len(setup.embedder.documents) == 2 * embedded


def test_a_copy_older_than_the_maximum_age_is_fetched_again(setup: Setup) -> None:
    setup.pipeline.ingest(URL)
    setup.clock.advance(hours=12)
    assert setup.pipeline.ingest(URL).reused is False


def test_a_copy_missing_from_the_store_is_fetched_again(setup: Setup) -> None:
    setup.pipeline.ingest(URL)
    setup.store.sources.clear()  # for example, the vector database was reset

    assert setup.pipeline.ingest(URL).reused is False
    assert URL in setup.store.sources


def test_two_guests_loading_the_same_url_at_once_load_it_once(setup: Setup, monkeypatch: pytest.MonkeyPatch) -> None:
    loads: list[str] = []

    def slow_load(url: str, **kwargs: object) -> ExtractedPage:
        loads.append(url)
        time.sleep(0.2)
        return ExtractedPage(url=url, title="Title", text=TEXT)

    monkeypatch.setattr(pipeline_module, "load_page", slow_load)
    results: list[IngestResult] = []
    threads = [threading.Thread(target=lambda: results.append(setup.pipeline.ingest(URL))) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(loads) == 1
    assert sorted(r.reused for r in results) == [False, True]
    assert len(setup.store.sources[URL][1]) == results[0].chunk_count  # no duplicate chunks


# --- deleting ---------------------------------------------------------------------------------------------------


def test_asking_about_an_unknown_or_deleted_page_is_refused(setup: Setup) -> None:
    with pytest.raises(NotIngestedError):
        setup.pipeline.ask(URL, "Question?")
    setup.pipeline.ingest(URL)
    setup.pipeline.forget(URL)
    with pytest.raises(NotIngestedError):
        setup.pipeline.ask(URL, "Question?")


def test_a_page_nobody_uses_is_deleted_after_the_idle_time(setup: Setup) -> None:
    setup.pipeline.ingest(URL)
    setup.clock.advance(seconds=IDLE - 1)
    assert setup.pipeline.cleanup() == []
    setup.clock.advance(seconds=1)

    assert setup.pipeline.cleanup() == [URL]
    assert URL not in setup.store.sources
    assert setup.registry.get(URL) is None


def test_the_page_stays_while_guests_keep_asking(setup: Setup) -> None:
    """Guest 1 leaves, guest 3 keeps asking: the page stays; when guest 3 stops too, it expires."""
    setup.pipeline.ingest(URL)  # guest 1
    for _ in range(3):  # guest 3 asks every 10 minutes
        setup.clock.advance(minutes=10)
        setup.pipeline.ask(URL, "Question?")
        assert setup.pipeline.cleanup() == []
    assert URL in setup.store.sources

    setup.clock.advance(minutes=15)  # everyone has left
    assert setup.pipeline.cleanup() == [URL]


def test_each_page_expires_on_its_own_schedule(setup: Setup) -> None:
    setup.pipeline.ingest(URL)  # guest 1
    setup.clock.advance(minutes=10)
    setup.pipeline.ingest(OTHER_URL)  # guest 2, ten minutes later
    setup.clock.advance(minutes=6)

    assert setup.pipeline.cleanup() == [URL]
    assert set(setup.store.sources) == {OTHER_URL}
    setup.clock.advance(minutes=10)
    assert setup.pipeline.cleanup() == [OTHER_URL]
    assert setup.store.sources == {}


def test_a_page_in_constant_use_is_still_deleted_at_the_maximum_age(setup: Setup) -> None:
    setup.pipeline.ingest(URL)
    for _ in range(12):  # an hour at a time, always in use
        setup.clock.advance(hours=1)
        if setup.registry.get(URL):
            setup.pipeline.ask(URL, "Question?")
        assert setup.pipeline.cleanup() == ([URL] if setup.clock.now >= 1_000_000 + MAX_AGE else [])
    assert URL not in setup.store.sources


def test_a_page_used_while_cleanup_runs_is_kept(setup: Setup, monkeypatch: pytest.MonkeyPatch) -> None:
    setup.pipeline.ingest(URL)
    setup.clock.advance(minutes=20)
    original = setup.registry.expired

    def expired_then_used(*args: float) -> list[PageRecord]:
        found = original(*args)
        setup.registry.touch(URL, setup.clock.now)  # someone asks right after the list was made
        return found

    monkeypatch.setattr(setup.registry, "expired", expired_then_used)

    assert setup.pipeline.cleanup() == []
    assert URL in setup.store.sources


def test_the_least_recently_used_pages_go_first_when_the_store_is_full(setup: Setup) -> None:
    pages = setup.pipeline.ingest(URL).chunk_count
    setup.clock.advance(minutes=1)
    setup.pipeline.ingest(OTHER_URL)
    setup.clock.advance(minutes=1)
    setup.pipeline.ask(URL, "Question?")  # the first page is now the most recently used
    assert setup.pipeline.cleanup() == []  # room for both
    setup.settings.max_stored_chunks = pages + 1  # room for one page only

    removed = setup.pipeline.cleanup()

    assert removed == [OTHER_URL]
    assert set(setup.store.sources) == {URL}


def test_forgetting_an_unknown_page_reports_that_nothing_was_deleted(setup: Setup) -> None:
    assert setup.pipeline.forget(URL) is False


def test_reconcile_adopts_unknown_pages_and_drops_missing_ones(setup: Setup) -> None:
    setup.store.replace_source("https://old.example/", "Old", [])  # stored by an earlier version, no record
    setup.registry.record_load("https://gone.example/", "Gone", 5, now=setup.clock.now)  # record, no chunks

    setup.pipeline.reconcile()

    assert setup.registry.get("https://old.example/") is not None
    assert setup.registry.get("https://gone.example/") is None
    setup.clock.advance(minutes=15)
    assert setup.pipeline.cleanup() == ["https://old.example/"]  # adopted pages expire like any other


# --- background cleaner ----------------------------------------------------------------------------------------


def test_janitor_runs_the_task_repeatedly_and_survives_errors() -> None:
    calls: list[int] = []

    def task() -> None:
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("first run fails")

    janitor = Janitor(task, interval_seconds=0.02)
    janitor.start()
    deadline = time.time() + 3
    while len(calls) < 3 and time.time() < deadline:
        time.sleep(0.01)
    janitor.stop()

    assert len(calls) >= 3
    count = len(calls)
    time.sleep(0.1)
    assert len(calls) == count  # stopped
