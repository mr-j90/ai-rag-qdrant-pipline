"""Qdrant store — LlamaIndex `QdrantVectorStore` plus a few admin helpers.

LlamaIndex owns the read/write path (upsert during ingestion, similarity_search
during retrieval). The raw `QdrantClient` is only used for collection-level
admin (count, reset, list distinct sources) that LlamaIndex doesn't expose.
"""
from __future__ import annotations

from functools import lru_cache

from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from src.config import get_settings


@lru_cache
def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=get_settings().qdrant_url)


@lru_cache
def get_vector_store() -> QdrantVectorStore:
    s = get_settings()
    return QdrantVectorStore(
        client=get_qdrant_client(),
        collection_name=s.qdrant_collection,
    )


def collection_exists() -> bool:
    s = get_settings()
    names = {c.name for c in get_qdrant_client().get_collections().collections}
    return s.qdrant_collection in names


def count() -> int:
    """Total points in the collection. 0 if the collection doesn't exist yet."""
    if not collection_exists():
        return 0
    s = get_settings()
    return get_qdrant_client().count(collection_name=s.qdrant_collection, exact=True).count


def list_sources() -> list[str]:
    """Distinct `file_name` values in the collection.

    LlamaIndex stores its node metadata nested under the `metadata` payload key,
    so we look for `metadata.file_name`. Uses scroll() rather than facets to
    avoid requiring a payload index — fine for playground-sized collections.
    """
    if not collection_exists():
        return []
    s = get_settings()
    client = get_qdrant_client()
    sources: set[str] = set()
    next_offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=s.qdrant_collection,
            with_payload=["metadata"],
            with_vectors=False,
            limit=256,
            offset=next_offset,
        )
        for p in points:
            meta = (p.payload or {}).get("metadata") or {}
            name = meta.get("file_name")
            if name:
                sources.add(name)
        if next_offset is None:
            break
    return sorted(sources)


def reset() -> None:
    """Delete the collection so the next write recreates it. Useful in dev."""
    s = get_settings()
    client = get_qdrant_client()
    if collection_exists():
        client.delete_collection(s.qdrant_collection)
    # Bust the cached vector store — it holds onto a reference to the (now-gone) collection.
    get_vector_store.cache_clear()
