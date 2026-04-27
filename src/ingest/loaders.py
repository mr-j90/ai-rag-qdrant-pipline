"""PDF loaders via LangChain.

`PyPDFLoader.load()` returns one `Document` per page with `source` (file path)
and `page` (0-indexed) in metadata. We normalize `source` to the basename so
filtering and display are stable regardless of where the file lives on disk.
"""
from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def load_pdf(path: str | Path) -> list[Document]:
    path = Path(path)
    docs = PyPDFLoader(str(path)).load()
    for d in docs:
        # PyPDFLoader uses 0-indexed pages; bump to 1-indexed for display.
        d.metadata["source"] = path.name
        d.metadata["page"] = d.metadata.get("page", 0) + 1
    return [d for d in docs if d.page_content.strip()]


def load_pdfs_from_dir(dir_path: str | Path) -> list[Document]:
    dir_path = Path(dir_path)
    out: list[Document] = []
    for pdf_path in sorted(dir_path.glob("*.pdf")):
        out.extend(load_pdf(pdf_path))
    return out
