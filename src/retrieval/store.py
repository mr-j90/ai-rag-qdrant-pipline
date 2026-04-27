"""Qdrant client wrapper — collection lifecycle, upsert, search."""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from src.config import get_settings


@dataclass
class SearchResult:
    id: str
    score: float
    text: str
    metadata: dict[str, Any]


class QdrantStore:
    def __init__(self) -> None:
        settings = get_settings()
        self.client = QdrantClient(url=settings.qdrant_url)
        self.collection = settings.qdrant_collection
        self.dim = settings.embedding_dim

    def ensure_collection(self) -> None:
        """Create the collection if it doesn't already exist."""
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection not in existing:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
            )

    def upsert(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> list[str]:
        """Insert or update points. Returns the assigned IDs."""
        assert len(texts) == len(embeddings) == len(metadatas), "length mismatch"
        ids = [str(uuid.uuid4()) for _ in texts]
        points = [
            PointStruct(
                id=pid,
                vector=vec,
                payload={"text": txt, **meta},
            )
            for pid, vec, txt, meta in zip(ids, embeddings, texts, metadatas)
        ]
        self.client.upsert(collection_name=self.collection, points=points)
        return ids

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        source_filter: str | None = None,
    ) -> list[SearchResult]:
        """Vector search with optional source filename filter."""
        qfilter = None
        if source_filter:
            qfilter = Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=source_filter))]
            )
        hits = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=top_k,
            query_filter=qfilter,
            with_payload=True,
        ).points
        return [
            SearchResult(
                id=str(h.id),
                score=h.score,
                text=h.payload.get("text", ""),
                metadata={k: v for k, v in h.payload.items() if k != "text"},
            )
            for h in hits
        ]

    def count(self) -> int:
        return self.client.count(collection_name=self.collection, exact=True).count

    def list_sources(self) -> list[str]:
        """Distinct source filenames currently in the collection.

        Uses scroll() rather than facets to avoid requiring a payload index.
        Fine for playground-sized collections; for large ones, switch to
        client.facet() with a keyword index on `source`.
        """
        sources: set[str] = set()
        next_offset = None
        while True:
            points, next_offset = self.client.scroll(
                collection_name=self.collection,
                with_payload=["source"],
                with_vectors=False,
                limit=256,
                offset=next_offset,
            )
            for p in points:
                src = (p.payload or {}).get("source")
                if src:
                    sources.add(src)
            if next_offset is None:
                break
        return sorted(sources)

    def reset(self) -> None:
        """Delete and recreate the collection. Useful during development."""
        if self.collection in {c.name for c in self.client.get_collections().collections}:
            self.client.delete_collection(self.collection)
        self.ensure_collection()
