"""Orchestrate load -> chunk -> embed -> upsert."""
from __future__ import annotations

from pathlib import Path

from src.ingest.chunker import chunk_pages
from src.ingest.loaders import load_pdf, load_pdfs_from_dir
from src.retrieval.embeddings import VoyageEmbedder
from src.retrieval.store import QdrantStore


def _embed_in_batches(
    embedder: VoyageEmbedder, texts: list[str], batch_size: int = 64
) -> list[list[float]]:
    """Voyage has per-request token limits; batching keeps us safe and parallelizable later."""
    all_vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        all_vectors.extend(embedder.embed_documents(batch))
    return all_vectors


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

    chunks = chunk_pages(pages)
    texts = [c.text for c in chunks]
    metadatas = [c.metadata for c in chunks]

    embedder = VoyageEmbedder()
    vectors = _embed_in_batches(embedder, texts)

    store = QdrantStore()
    store.ensure_collection()
    ids = store.upsert(texts=texts, embeddings=vectors, metadatas=metadatas)

    return {"pages": len(pages), "chunks": len(chunks), "ids": ids}
