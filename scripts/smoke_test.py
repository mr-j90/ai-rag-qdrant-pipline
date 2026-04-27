"""Phase 1 / Step 3 — hardcoded smoke test.

Verifies your Voyage + Qdrant pipeline works end-to-end before you touch any PDFs.

Run AFTER: docker compose up -d  (and uv sync, and .env populated)

Usage:
    uv run python -m scripts.smoke_test
"""
from __future__ import annotations

from rich import print

from src.retrieval.embeddings import VoyageEmbedder
from src.retrieval.store import QdrantStore

DOCS = [
    "Qdrant is an open-source vector database written in Rust.",
    "FastAPI is a modern Python web framework built on Starlette and Pydantic.",
    "Voyage AI provides high-quality embedding models, often paired with Claude.",
    "Claude is an AI assistant created by Anthropic.",
    "RAG (Retrieval-Augmented Generation) combines vector search with LLMs.",
]
QUERY = "What database should I use to store embeddings?"


def main() -> None:
    print("[bold cyan]== Smoke test ==[/bold cyan]")

    embedder = VoyageEmbedder()
    store = QdrantStore()
    store.reset()  # fresh collection for the smoke test

    print(f"Embedding {len(DOCS)} docs with Voyage...")
    vectors = embedder.embed_documents(DOCS)
    print(f"  vector dim = {len(vectors[0])}")

    metadatas = [{"source": "smoke_test", "page": 0, "chunk_index": i} for i in range(len(DOCS))]
    ids = store.upsert(texts=DOCS, embeddings=vectors, metadatas=metadatas)
    print(f"Upserted {len(ids)} points. Collection count = {store.count()}")

    print(f"\n[bold]Query:[/bold] {QUERY}")
    qvec = embedder.embed_query(QUERY)
    hits = store.search(query_vector=qvec, top_k=3)

    for h in hits:
        print(f"  [green]{h.score:.4f}[/green]  {h.text}")

    print("\n[bold green]✓ Pipeline works end-to-end.[/bold green]")


if __name__ == "__main__":
    main()
