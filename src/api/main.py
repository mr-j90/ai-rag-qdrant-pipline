"""FastAPI app: /health, /query, /stats, /sources, /upload."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.config import get_settings
from src.generation.llm import answer as generate_answer
from src.ingest.pipeline import ingest_path
from src.retrieval import store

app = FastAPI(title="RAG-Qdrant", version="0.1.0")

# Where uploaded PDFs land. Same directory the CLI ingester reads from,
# so anything uploaded via the API is also picked up by `ingest_cli ./data/pdfs`.
UPLOAD_DIR = Path("data/pdfs")


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int | None = None
    source_filter: str | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]


class UploadResponse(BaseModel):
    filename: str
    pages: int
    chunks: int


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/stats")
def stats() -> dict:
    s = get_settings()
    return {
        "collection": s.qdrant_collection,
        "vector_count": store.count(),
        "dim": s.embedding_dim,
    }


@app.get("/sources")
def sources() -> dict:
    """Distinct source filenames currently indexed. Powers the UI filter dropdown."""
    return {"sources": store.list_sources()}


@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)) -> UploadResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are accepted.")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    # Use only the basename to guard against path-traversal in the upload name.
    dest = UPLOAD_DIR / Path(file.filename).name
    dest.write_bytes(await file.read())

    result = ingest_path(dest)
    if result["chunks"] == 0:
        raise HTTPException(
            status_code=400,
            detail=f"No extractable text in {dest.name}. Scanned PDFs need OCR first.",
        )

    return UploadResponse(filename=dest.name, pages=result["pages"], chunks=result["chunks"])


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    settings = get_settings()
    top_k = req.top_k or settings.top_k

    if store.count() == 0:
        raise HTTPException(
            status_code=400,
            detail="Collection is empty. Ingest some PDFs first via scripts/ingest_cli.py.",
        )

    result, nodes = generate_answer(req.question, top_k=top_k, source_filter=req.source_filter)
    if not nodes:
        raise HTTPException(status_code=404, detail="No matching documents found.")

    return QueryResponse(answer=result.answer, sources=result.sources)
