"""Voyage embeddings via LangChain.

`VoyageAIEmbeddings` handles the document/query input_type asymmetry under
the hood (uses 'document' for `embed_documents`, 'query' for `embed_query`),
so we don't manage it here.
"""
from __future__ import annotations

from functools import lru_cache

from langchain_voyageai import VoyageAIEmbeddings

from src.config import get_settings


@lru_cache
def get_embeddings() -> VoyageAIEmbeddings:
    s = get_settings()
    return VoyageAIEmbeddings(
        model=s.voyage_model,
        api_key=s.voyage_api_key,
    )
