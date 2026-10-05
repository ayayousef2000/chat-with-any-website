import pytest
from pydantic import ValidationError

from app.config import Settings

REQUIRED = {
    "COHERE_API_KEY": "c",
    "WEAVIATE_URL": "https://example.weaviate.cloud",
    "WEAVIATE_API_KEY": "w",
    "GROQ_API_KEY": "g",
}


def _settings(monkeypatch: pytest.MonkeyPatch, **overrides: str) -> Settings:
    for key, value in {**REQUIRED, **overrides}.items():
        monkeypatch.setenv(key, value)
    return Settings(_env_file=None)  # type: ignore[call-arg]


def test_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch)
    assert settings.chunk_size == 2000
    assert settings.chunk_overlap == 300
    assert settings.retrieve_k == 25
    assert settings.top_k == 5
    assert settings.rerank_enabled is True
    assert settings.browser_fallback is False


def test_overrides_come_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch, TOP_K="3", HYBRID_ALPHA="0.4", RERANK_ENABLED="false")
    assert (settings.top_k, settings.hybrid_alpha, settings.rerank_enabled) == (3, 0.4, False)


def test_missing_required_value_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in REQUIRED:
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_overlap_must_be_smaller_than_chunk_size(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValidationError, match="CHUNK_OVERLAP"):
        _settings(monkeypatch, CHUNK_SIZE="500", CHUNK_OVERLAP="500")


def test_top_k_must_not_exceed_retrieve_k(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValidationError, match="TOP_K"):
        _settings(monkeypatch, RETRIEVE_K="3", TOP_K="5")


def test_alpha_must_be_between_zero_and_one(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValidationError):
        _settings(monkeypatch, HYBRID_ALPHA="1.5")


def test_page_lifetime_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch)
    assert settings.page_idle_minutes == 15
    assert settings.page_max_age_hours == 12
    assert settings.max_stored_chunks == 60_000
    assert settings.cleanup_interval_seconds == 60
    assert settings.state_db_path == "data/state.db"


@pytest.mark.parametrize(
    "name", ["PAGE_IDLE_MINUTES", "PAGE_MAX_AGE_HOURS", "MAX_STORED_CHUNKS", "CLEANUP_INTERVAL_SECONDS"]
)
def test_page_lifetime_values_must_be_positive(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    with pytest.raises(ValidationError):
        _settings(monkeypatch, **{name: "0"})


def test_retry_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch)
    assert (settings.retry_attempts, settings.retry_max_wait_seconds, settings.retry_budget_seconds) == (4, 20, 45)


@pytest.mark.parametrize("name", ["RETRY_ATTEMPTS", "RETRY_MAX_WAIT_SECONDS", "RETRY_BUDGET_SECONDS"])
def test_retry_values_must_be_positive(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    with pytest.raises(ValidationError):
        _settings(monkeypatch, **{name: "0"})


def test_backup_groq_keys_are_read_in_order_without_blanks(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch, GROQ_BACKUP_API_KEYS=" k2, k3 ,,k4,")
    assert settings.groq_backup_keys == ["k2", "k3", "k4"]


def test_there_are_no_backup_groq_keys_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_BACKUP_API_KEYS", raising=False)
    assert _settings(monkeypatch).groq_backup_keys == []
