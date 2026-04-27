"""Phase 1 / Step 3 — hardcoded smoke test.

Verifies your Voyage + Qdrant + LangChain pipeline works end-to-end before
you touch any PDFs.

Run AFTER: docker compose up -d  (and uv sync, and .env populated)

Usage:
    uv run python -m scripts.smoke_test
"""
from __future__ import annotations

from langchain_core.documents import Document
from rich import print

from src.retrieval import store
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

    store.reset()  # fresh collection for the smoke test

    documents = [Document(page_content=t, metadata={"source": "smoke_test"}) for t in DOCS]
    vs = get_vector_store()
    ids = vs.add_documents(documents)
    print(f"Upserted {len(ids)} points. Collection count = {store.count()}")

    print(f"\n[bold]Query:[/bold] {QUERY}")
    for doc, score in vs.similarity_search_with_score(QUERY, k=3):
        print(f"  [green]{score:.4f}[/green]  {doc.page_content}")

    print("\n[bold green]✓ Pipeline works end-to-end.[/bold green]")


if __name__ == "__main__":
    main()
