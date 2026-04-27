"""Phase 1 / Step 3 — hardcoded smoke test.

Verifies your Voyage + Qdrant + LlamaIndex pipeline works end-to-end before
you touch any PDFs.

Run AFTER: docker compose up -d  (and uv sync, and .env populated)

Usage:
    uv run python -m scripts.smoke_test
"""
from __future__ import annotations

from llama_index.core import Document, VectorStoreIndex, StorageContext
from rich import print

from src.retrieval import store
from src.retrieval.embeddings import get_embed_model
from src.retrieval.store import get_vector_store

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

    # Fresh collection so the smoke test is reproducible.
    store.reset()

    documents = [Document(text=t, metadata={"file_name": "smoke_test"}) for t in DOCS]
    vs = get_vector_store()
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=StorageContext.from_defaults(vector_store=vs),
        embed_model=get_embed_model(),
    )
    print(f"Indexed {len(documents)} docs. Collection count = {store.count()}")

    print(f"\n[bold]Query:[/bold] {QUERY}")
    retriever = index.as_retriever(similarity_top_k=3)
    for h in retriever.retrieve(QUERY):
        print(f"  [green]{h.score:.4f}[/green]  {h.node.get_content()}")

    print("\n[bold green]✓ Pipeline works end-to-end.[/bold green]")


if __name__ == "__main__":
    main()
