"""Chunk page records into overlapping text windows."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import get_settings
from src.ingest.loaders import PageRecord


@dataclass
class Chunk:
    text: str
    metadata: dict[str, Any]


def chunk_pages(pages: list[PageRecord]) -> list[Chunk]:
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Chunk] = []
    for page in pages:
        for i, piece in enumerate(splitter.split_text(page.text)):
            chunks.append(
                Chunk(
                    text=piece,
                    metadata={
                        "source": page.source,
                        "page": page.page,
                        "chunk_index": i,
                    },
                )
            )
    return chunks
