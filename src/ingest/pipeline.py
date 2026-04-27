"""LlamaIndex ingestion: load PDFs -> split -> embed -> upsert into Qdrant.

`IngestionPipeline` runs the transformations sequentially, threading metadata
(file_name, page_label) through from the reader down to each chunk node.
"""
from __future__ import annotations

from pathlib import Path

from llama_index.core import SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter

from src.config import get_settings
from src.retrieval.embeddings import get_embed_model
from src.retrieval.store import get_vector_store


def _load_documents(path: Path) -> list:
    """Load PDFs into LlamaIndex Documents — one per page (PDFReader default)."""
    if path.is_dir():
        reader = SimpleDirectoryReader(input_dir=str(path), required_exts=[".pdf"])
    elif path.suffix.lower() == ".pdf":
        reader = SimpleDirectoryReader(input_files=[str(path)])
    else:
        raise ValueError(f"Unsupported path: {path}")
    return reader.load_data()


def ingest_path(path: str | Path) -> dict:
    """Ingest a single PDF or a directory of PDFs."""
    p = Path(path)
    docs = _load_documents(p)
    if not docs:
        return {"pages": 0, "chunks": 0, "ids": []}

    s = get_settings()
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=s.chunk_size, chunk_overlap=s.chunk_overlap),
            get_embed_model(),
        ],
        vector_store=get_vector_store(),
    )
    nodes = pipeline.run(documents=docs, show_progress=False)

    return {
        "pages": len(docs),
        "chunks": len(nodes),
        "ids": [n.node_id for n in nodes],
    }
