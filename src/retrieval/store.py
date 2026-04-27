"""Qdrant store via LangChain `QdrantVectorStore` plus a few admin helpers.

LangChain's `QdrantVectorStore` owns the read/write path (add_documents,
similarity_search, as_retriever). The raw `QdrantClient` is only used for
collection-level admin (count, reset, list distinct sources) that LangChain
doesn't expose.
"""
from __future__ import annotations

from functools import lru_cache

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from src.config import get_settings
from src.retrieval.embeddings import get_embeddings


@lru_cache
def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=get_settings().qdrant_url)


def _ensure_collection() -> None:
    s = get_settings()
    client = get_qdrant_client()
    if s.qdrant_collection not in {c.name for c in client.get_collections().collections}:
        client.create_collection(
            collection_name=s.qdrant_collection,
            vectors_config=VectorParams(size=s.embedding_dim, distance=Distance.COSINE),
        )


@lru_cache
def get_vector_store() -> QdrantVectorStore:
    s = get_settings()
    _ensure_collection()
    return QdrantVectorStore(
        client=get_qdrant_client(),
        collection_name=s.qdrant_collection,
        embedding=get_embeddings(),
    )


def collection_exists() -> bool:
    s = get_settings()
    return s.qdrant_collection in {
        c.name for c in get_qdrant_client().get_collections().collections
    }


def count() -> int:
    """Total points in the collection. 0 if the collection doesn't exist yet."""
    if not collection_exists():
        return 0
    s = get_settings()
    return get_qdrant_client().count(collection_name=s.qdrant_collection, exact=True).count


def list_sources() -> list[str]:
    """Distinct `source` values in the collection.

    LangChain's QdrantVectorStore stores Document.metadata nested under the
    `metadata` payload key, so we look for `metadata.source`. Uses scroll()
    rather than facets to avoid requiring a payload index — fine for
    playground-sized collections.
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
            name = meta.get("source")
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
    # Bust caches — the dropped collection is referenced inside.
    get_vector_store.cache_clear()
