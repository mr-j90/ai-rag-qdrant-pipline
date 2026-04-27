"""Voyage AI embeddings wrapper.

Voyage distinguishes between document and query embeddings via input_type.
Always use input_type='document' when indexing and 'query' when searching —
this gives noticeably better retrieval quality.
"""
from __future__ import annotations

import voyageai

from src.config import get_settings


class VoyageEmbedder:
    def __init__(self) -> None:
        settings = get_settings()
        self.client = voyageai.Client(api_key=settings.voyage_api_key)
        self.model = settings.voyage_model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed chunks for storage in the vector DB."""
        if not texts:
            return []
        result = self.client.embed(texts, model=self.model, input_type="document")
        return result.embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed a single user query for retrieval."""
        result = self.client.embed([text], model=self.model, input_type="query")
        return result.embeddings[0]
