"""Ingest PDFs into Qdrant.

Usage:
    uv run python -m scripts.ingest_cli ./data/pdfs
    uv run python -m scripts.ingest_cli ./data/pdfs/some_file.pdf
    uv run python -m scripts.ingest_cli ./data/pdfs --reset
"""
from __future__ import annotations

import typer
from rich import print

from src.ingest.pipeline import ingest_path
from src.retrieval.store import QdrantStore

app = typer.Typer(add_completion=False)


@app.command()
def main(
    path: str = typer.Argument(..., help="PDF file or directory of PDFs."),
    reset: bool = typer.Option(False, "--reset", help="Wipe the collection before ingesting."),
) -> None:
    if reset:
        print("[yellow]Resetting collection...[/yellow]")
        QdrantStore().reset()

    print(f"Ingesting [bold]{path}[/bold]...")
    result = ingest_path(path)
    print(f"  pages:  {result['pages']}")
    print(f"  chunks: {result['chunks']}")
    print(f"  total in collection: {QdrantStore().count()}")
    print("[bold green]✓ Done.[/bold green]")


if __name__ == "__main__":
    app()
