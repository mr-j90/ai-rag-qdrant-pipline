"""Chunk LangChain Documents into overlapping windows."""
from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import get_settings


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split page-level Documents into chunk-level Documents.

    `split_documents` carries each parent Document's metadata onto every child
    chunk, so `source` and `page` propagate down for free. We layer
    `chunk_index` on top so chunks within a page are addressable.
    """
    s = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=s.chunk_size,
        chunk_overlap=s.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    # Tag chunks with a per-page running index for traceability.
    page_counters: dict[tuple, int] = {}
    for c in chunks:
        key = (c.metadata.get("source"), c.metadata.get("page"))
        idx = page_counters.get(key, 0)
        c.metadata["chunk_index"] = idx
        page_counters[key] = idx + 1
    return chunks
