"""Orchestrate load -> chunk -> embed -> upsert via the LangChain stack.

`QdrantVectorStore.add_documents(...)` handles batching, embedding (via the
embedding function bound to the store), and upsert in one call.
"""
from __future__ import annotations

from pathlib import Path

from src.ingest.chunker import chunk_documents
from src.ingest.loaders import load_pdf, load_pdfs_from_dir
from src.retrieval.store import get_vector_store


def ingest_path(path: str | Path) -> dict:
    """Ingest a single PDF or a directory of PDFs."""
    p = Path(path)
    if p.is_dir():
        pages = load_pdfs_from_dir(p)
    elif p.suffix.lower() == ".pdf":
        pages = load_pdf(p)
    else:
        raise ValueError(f"Unsupported path: {p}")

    if not pages:
        return {"pages": 0, "chunks": 0, "ids": []}

    chunks = chunk_documents(pages)
    ids = get_vector_store().add_documents(chunks)
    return {"pages": len(pages), "chunks": len(chunks), "ids": ids}
