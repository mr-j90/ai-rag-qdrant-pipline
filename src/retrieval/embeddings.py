"""Voyage embeddings — LlamaIndex wrapper.

LlamaIndex's VoyageEmbedding handles the document/query input_type asymmetry
internally (uses 'document' for embed_documents, 'query' for get_query_embedding),
so we don't have to manage it here.
"""
from __future__ import annotations

from functools import lru_cache

from llama_index.embeddings.voyageai import VoyageEmbedding

from src.config import get_settings


@lru_cache
def get_embed_model() -> VoyageEmbedding:
    s = get_settings()
    return VoyageEmbedding(
        model_name=s.voyage_model,
        voyage_api_key=s.voyage_api_key,
    )
