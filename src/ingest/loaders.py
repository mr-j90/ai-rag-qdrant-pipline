"""PDF loader. Returns one record per page so we can attribute chunks to pages."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pdfplumber


@dataclass
class PageRecord:
    source: str       # filename (e.g. "anthropic-handbook.pdf")
    page: int         # 1-indexed page number
    text: str


def load_pdf(path: str | Path) -> list[PageRecord]:
    path = Path(path)
    records: list[PageRecord] = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                records.append(PageRecord(source=path.name, page=i, text=text))
    return records


def load_pdfs_from_dir(dir_path: str | Path) -> list[PageRecord]:
    dir_path = Path(dir_path)
    out: list[PageRecord] = []
    for pdf_path in sorted(dir_path.glob("*.pdf")):
        out.extend(load_pdf(pdf_path))
    return out
