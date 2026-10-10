"""Steps 6-7: store chunk vectors in Weaviate and retrieve candidates with hybrid (keyword + vector) search."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import weaviate
from weaviate.classes.aggregate import GroupByAggregate
from weaviate.classes.config import Configure, DataType, Property, Tokenization
from weaviate.classes.data import DataObject
from weaviate.classes.init import AdditionalConfig, Auth, Timeout
from weaviate.classes.query import Filter, HybridFusion, MetadataQuery
from weaviate.collections import Collection
from weaviate.exceptions import UnexpectedStatusCodeError


@dataclass(frozen=True)
class StoredChunk:
    """A chunk and its embedding, ready to be stored."""

    text: str
    heading: str
    vector: Sequence[float]


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned by a search."""

    title: str
    heading: str
    text: str
    chunk_index: int
    score: float | None


NOT_MULTI_TENANT_MESSAGE = (
    "The Weaviate collection {name} exists but is not multi-tenant, so it cannot keep the pages of several "
    "environments apart. Delete the collection in the Weaviate console and start the app again: it is created "
    "again with multi-tenancy, and the pages in it are only temporary copies. See the README."
)


class WeaviateStore:
    """Stores chunk vectors in Weaviate and searches them.

    Everything is kept in one multi-tenant collection with one tenant for each environment (for example
    ``production``, ``staging`` and ``local``). Weaviate keeps the tenants apart, so one environment never sees or
    deletes the pages of another, even though a free Weaviate Cloud plan allows only one collection.
    """

    def __init__(
        self, url: str, api_key: str, collection_name: str, init_timeout: int = 30, tenant: str = "local"
    ) -> None:
        # The client allows only 2 seconds for its startup checks, which a slow connection can miss.
        self._client = weaviate.connect_to_weaviate_cloud(
            cluster_url=url,
            auth_credentials=Auth.api_key(api_key),
            additional_config=AdditionalConfig(timeout=Timeout(init=init_timeout)),
        )
        try:
            self._collection = self._open_tenant(collection_name, tenant)
        except Exception:
            self._client.close()  # a failed start must not leave the connection open
            raise

    def _open_tenant(self, name: str, tenant: str) -> Collection[Any, Any]:
        """Open the collection for one tenant, creating the collection and the tenant if they are missing."""
        collection = self._get_or_create_collection(name)
        if not collection.config.get().multi_tenancy_config.enabled:
            raise RuntimeError(NOT_MULTI_TENANT_MESSAGE.format(name=name))
        if not collection.tenants.exists(tenant):
            try:
                collection.tenants.create(tenant)
            except UnexpectedStatusCodeError:
                if not collection.tenants.exists(tenant):  # the same tenant may have been created a moment ago
                    raise
        return collection.with_tenant(tenant)

    def _get_or_create_collection(self, name: str) -> Collection[Any, Any]:
        if self._client.collections.exists(name):
            return self._client.collections.use(name)
        try:
            return self._client.collections.create(
                name,
                properties=[
                    # FIELD tokenization keeps the whole URL as one token so filters match it exactly.
                    Property(name="url", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
                    Property(name="title", data_type=DataType.TEXT, index_searchable=False),
                    Property(name="heading", data_type=DataType.TEXT),
                    Property(name="chunk_index", data_type=DataType.INT),
                    Property(name="text", data_type=DataType.TEXT),
                ],
                # Vectors are computed by the application (Cohere), not by Weaviate.
                vector_config=Configure.Vectors.self_provided(),
                multi_tenancy_config=Configure.multi_tenancy(enabled=True),
            )
        except UnexpectedStatusCodeError:
            # Another service may have created the collection while this one was starting.
            if self._client.collections.exists(name):
                return self._client.collections.use(name)
            raise

    def replace_source(self, url: str, title: str, chunks: Sequence[StoredChunk]) -> None:
        """Remove any chunks previously stored for this URL and insert the new ones."""
        self._collection.data.delete_many(where=Filter.by_property("url").equal(url))
        objects = [
            DataObject(
                properties={
                    "url": url,
                    "title": title,
                    "heading": chunk.heading,
                    "chunk_index": index,
                    "text": chunk.text,
                },
                vector=list(chunk.vector),
            )
            for index, chunk in enumerate(chunks)
        ]
        result = self._collection.data.insert_many(objects)
        if result.has_errors:
            first_error = next(iter(result.errors.values()))
            raise RuntimeError(f"Failed to store {len(result.errors)} chunk(s) in Weaviate: {first_error.message}")

    def delete_source(self, url: str) -> None:
        """Delete every chunk stored for a URL.

        Args:
            url: The normalized page address.
        """
        self._collection.data.delete_many(where=Filter.by_property("url").equal(url))

    def list_sources(self) -> dict[str, int]:
        """Count the stored chunks per URL.

        Returns:
            A mapping from page address to the number of chunks stored for it.
        """
        result = self._collection.aggregate.over_all(group_by=GroupByAggregate(prop="url"))
        return {str(group.grouped_by.value): group.total_count or 0 for group in result.groups}

    def has_source(self, url: str) -> bool:
        """Check whether any chunks are stored for a URL.

        Args:
            url: The normalized page address.

        Returns:
            ``True`` if at least one chunk exists.
        """
        response = self._collection.query.fetch_objects(
            filters=Filter.by_property("url").equal(url),
            limit=1,
            return_properties=[],
        )
        return bool(response.objects)

    def search(
        self,
        url: str,
        query: str,
        vector: Sequence[float],
        limit: int,
        alpha: float,
    ) -> list[RetrievedChunk]:
        """Hybrid search: BM25 on the text and heading blended with vector similarity (alpha=1 is vector only)."""
        response = self._collection.query.hybrid(
            query=query,
            vector=list(vector),
            alpha=alpha,
            fusion_type=HybridFusion.RELATIVE_SCORE,
            query_properties=["text", "heading"],
            filters=Filter.by_property("url").equal(url),
            limit=limit,
            return_properties=["title", "heading", "text", "chunk_index"],
            return_metadata=MetadataQuery(score=True),
        )
        return [
            RetrievedChunk(
                title=str(obj.properties["title"]),
                heading=str(obj.properties.get("heading") or ""),
                text=str(obj.properties["text"]),
                chunk_index=int(obj.properties["chunk_index"]),
                score=obj.metadata.score,
            )
            for obj in response.objects
        ]

    def close(self) -> None:
        """Close the connection to Weaviate."""
        self._client.close()
