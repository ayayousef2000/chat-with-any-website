"""Tests for how the Weaviate store opens its collection and tenant, with a fake Weaviate client."""

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
import weaviate
from weaviate.exceptions import UnexpectedStatusCodeError

from app.vector_store import NOT_MULTI_TENANT_MESSAGE, WeaviateStore


def _status_error(status: int) -> UnexpectedStatusCodeError:
    return UnexpectedStatusCodeError("Create collection", httpx.Response(status, text="refused"))


class FakeTenants:
    def __init__(self, existing: set[str]) -> None:
        self.existing = existing
        self.created: list[str] = []
        self.create_error: Exception | None = None
        self.someone_else_made_it = False  # the error is the answer to a race that another start won

    def exists(self, tenant: str) -> bool:
        return tenant in self.existing

    def create(self, tenant: str) -> None:
        self.created.append(tenant)
        if self.create_error is not None:
            if self.someone_else_made_it:
                self.existing.add(tenant)
            raise self.create_error
        self.existing.add(tenant)


class FakeCollection:
    def __init__(self, *, multi_tenant: bool = True, tenants: set[str] | None = None) -> None:
        self.config = SimpleNamespace(
            get=lambda: SimpleNamespace(multi_tenancy_config=SimpleNamespace(enabled=multi_tenant))
        )
        self.tenants = FakeTenants(set(tenants or ()))
        self.used_tenant: str | None = None

    def with_tenant(self, tenant: str) -> str:
        self.used_tenant = tenant
        return f"collection-for-{tenant}"


class FakeCollections:
    def __init__(self, existing: dict[str, FakeCollection] | None = None) -> None:
        self.existing = dict(existing or {})
        self.created: list[dict[str, Any]] = []
        self.create_error: Callable[[FakeCollections], Exception] | None = None
        self.new_collection = FakeCollection()

    def exists(self, name: str) -> bool:
        return name in self.existing

    def use(self, name: str) -> FakeCollection:
        return self.existing[name]

    def create(self, name: str, **kwargs: Any) -> FakeCollection:
        self.created.append({"name": name, **kwargs})
        if self.create_error is not None:
            raise self.create_error(self)
        self.existing[name] = self.new_collection
        return self.new_collection


class FakeWeaviate:
    def __init__(self, collections: FakeCollections) -> None:
        self.collections = collections
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _store(
    monkeypatch: pytest.MonkeyPatch, collections: FakeCollections, tenant: str = "staging"
) -> tuple[WeaviateStore, FakeWeaviate, dict[str, Any]]:
    client = FakeWeaviate(collections)
    captured: dict[str, Any] = {}

    def connect(**kwargs: Any) -> FakeWeaviate:
        captured.update(kwargs)
        return client

    monkeypatch.setattr(weaviate, "connect_to_weaviate_cloud", connect)
    store = WeaviateStore("https://cluster.example", "key", "Chunks", init_timeout=25, tenant=tenant)
    return store, client, captured


def _opened(store: WeaviateStore) -> Any:
    """What the store reads and writes through: with the fake client, a marker naming the tenant."""
    return store._collection


# --- connecting ----------------------------------------------------------------------------------------------------


def test_weaviate_gets_the_configured_time_for_its_startup_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    _, _, captured = _store(monkeypatch, FakeCollections())

    assert captured["additional_config"].timeout.init == 25
    assert captured["cluster_url"] == "https://cluster.example"


# --- the collection and the tenant ------------------------------------------------------------------------------------


def test_a_new_collection_is_created_with_multi_tenancy_and_the_tenant_is_made(monkeypatch: pytest.MonkeyPatch) -> None:
    collections = FakeCollections()

    store, _, _ = _store(monkeypatch, collections, tenant="staging")

    assert len(collections.created) == 1
    assert collections.created[0]["name"] == "Chunks"
    assert collections.created[0]["multi_tenancy_config"].enabled is True
    assert collections.new_collection.tenants.created == ["staging"]
    assert _opened(store) == "collection-for-staging"  # every read and write goes through the tenant


def test_an_existing_multi_tenant_collection_is_reused_and_only_the_missing_tenant_is_made(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing = FakeCollection(tenants={"production"})
    collections = FakeCollections({"Chunks": existing})

    store, _, _ = _store(monkeypatch, collections, tenant="staging")

    assert collections.created == []  # the collection is not made again
    assert existing.tenants.created == ["staging"]
    assert existing.tenants.existing == {"production", "staging"}  # production's tenant is left alone
    assert _opened(store) == "collection-for-staging"


def test_a_tenant_that_already_exists_is_not_made_again(monkeypatch: pytest.MonkeyPatch) -> None:
    existing = FakeCollection(tenants={"production"})

    _store(monkeypatch, FakeCollections({"Chunks": existing}), tenant="production")

    assert existing.tenants.created == []
    assert existing.used_tenant == "production"


def test_two_environments_use_the_same_collection_with_their_own_tenants(monkeypatch: pytest.MonkeyPatch) -> None:
    collections = FakeCollections()

    first, _, _ = _store(monkeypatch, collections, tenant="production")
    second, _, _ = _store(monkeypatch, collections, tenant="staging")

    assert len(collections.created) == 1  # one collection, as the free plan allows
    assert (_opened(first), _opened(second)) == ("collection-for-production", "collection-for-staging")
    assert collections.new_collection.tenants.existing == {"production", "staging"}


# --- things that go wrong ------------------------------------------------------------


def test_a_collection_without_multi_tenancy_stops_the_start_with_a_plain_message_and_closes_the_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collections = FakeCollections({"Chunks": FakeCollection(multi_tenant=False)})
    client = FakeWeaviate(collections)
    monkeypatch.setattr(weaviate, "connect_to_weaviate_cloud", lambda **_: client)

    with pytest.raises(RuntimeError, match="Chunks exists but is not multi-tenant") as stopped:
        WeaviateStore("https://cluster.example", "key", "Chunks", tenant="staging")

    assert str(stopped.value) == NOT_MULTI_TENANT_MESSAGE.format(name="Chunks")
    assert client.closed is True


def test_the_limit_of_one_collection_is_reported_and_the_connection_is_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    collections = FakeCollections()
    collections.create_error = lambda _: _status_error(429)
    client = FakeWeaviate(collections)
    monkeypatch.setattr(weaviate, "connect_to_weaviate_cloud", lambda **_: client)

    with pytest.raises(UnexpectedStatusCodeError):
        WeaviateStore("https://cluster.example", "key", "Chunks", tenant="staging")

    assert client.closed is True


def test_a_collection_made_by_another_service_a_moment_ago_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    collections = FakeCollections()
    made_elsewhere = FakeCollection()

    def another_service_won(current: FakeCollections) -> Exception:
        current.existing["Chunks"] = made_elsewhere  # it exists by the time the error is looked at
        return _status_error(422)

    collections.create_error = another_service_won

    store, _, _ = _store(monkeypatch, collections, tenant="staging")

    assert _opened(store) == "collection-for-staging"
    assert made_elsewhere.tenants.created == ["staging"]


def test_a_tenant_made_by_another_start_a_moment_ago_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    existing = FakeCollection()
    existing.tenants.create_error = _status_error(422)
    existing.tenants.someone_else_made_it = True

    store, _, _ = _store(monkeypatch, FakeCollections({"Chunks": existing}), tenant="staging")

    assert _opened(store) == "collection-for-staging"


def test_a_tenant_that_cannot_be_made_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    existing = FakeCollection()
    existing.tenants.create_error = _status_error(422)
    client = FakeWeaviate(FakeCollections({"Chunks": existing}))
    monkeypatch.setattr(weaviate, "connect_to_weaviate_cloud", lambda **_: client)

    with pytest.raises(UnexpectedStatusCodeError):
        WeaviateStore("https://cluster.example", "key", "Chunks", tenant="staging")

    assert client.closed is True
