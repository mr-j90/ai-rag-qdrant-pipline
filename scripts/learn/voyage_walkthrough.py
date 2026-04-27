"""
LangChain × Voyage walkthrough — embeddings through the LangChain lens.

What changes when you go from "raw voyageai client" to LangChain's
`VoyageAIEmbeddings`:

    - You stop calling vo.embed(...) directly. LangChain calls it for you,
      via embed_documents(texts) and embed_query(text).
    - The document/query input_type asymmetry is handled automatically based
      on which method you call. You almost never have to think about it again.
    - Embeddings are computed inside QdrantVectorStore.add_documents() /
      .similarity_search() transparently — no orchestration step.

This walkthrough:
    1. Generate one embedding the LangChain way
    2. Cosine similarity (still useful to understand)
    3. Confirm the doc-vs-query asymmetry IS still happening, just hidden
    4. Build a QdrantVectorStore with real embeddings
    5. Run semantic queries
    6. Contrast with random vectors so you remember why this works

Prereqs:
    - docker compose up -d
    - VOYAGE_API_KEY in .env

Run:
    uv run python -m scripts.learn.voyage_walkthrough
"""
from __future__ import annotations

import math
import random

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_voyageai import VoyageAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from rich import print
from rich.rule import Rule
from rich.table import Table

from src.config import get_settings

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
COLLECTION = "voyage_walkthrough"
settings = get_settings()

embeddings = VoyageAIEmbeddings(
    model=settings.voyage_model,
    api_key=settings.voyage_api_key,
)
qdrant = QdrantClient(url=settings.qdrant_url)


# ---------------------------------------------------------------------------
# 1. One embedding, the LangChain way
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]1. Generate one embedding via VoyageAIEmbeddings"))

text = "Qdrant is a vector database written in Rust."
vec = embeddings.embed_query(text)

print(f"Input text:    '{text}'")
print(f"Model:         {settings.voyage_model}")
print(f"Dimensions:    {len(vec)}")
print(f"First 8 dims:  {[round(x, 4) for x in vec[:8]]}")
print(f"Vector type:   {type(vec[0]).__name__}")
print("[dim]No more passing input_type='document'/'query' yourself — pick the right method.[/dim]")


# ---------------------------------------------------------------------------
# 2. Cosine similarity, by hand (still worth seeing)
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]2. Cosine similarity, demystified"))


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b)


pairs = [
    ("A dog is barking loudly", "A puppy is making lots of noise"),
    ("A dog is barking loudly", "The stock market closed lower today"),
    ("A dog is barking loudly", "A dog is barking loudly"),
]
flat = [t for pair in pairs for t in pair]
flat_vecs = embeddings.embed_documents(flat)

table = Table(show_lines=True)
table.add_column("Text A", overflow="fold", max_width=35)
table.add_column("Text B", overflow="fold", max_width=35)
table.add_column("Cosine", justify="right")
table.add_column("Interpretation")
labels = ["semantically similar", "unrelated", "identical"]
for i, (a, b) in enumerate(pairs):
    sim = cosine(flat_vecs[i * 2], flat_vecs[i * 2 + 1])
    table.add_row(a, b, f"{sim:.4f}", labels[i])
print(table)


# ---------------------------------------------------------------------------
# 3. The doc-vs-query asymmetry IS still happening
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]3. Doc vs query — LangChain picks input_type for you"))

doc_text = "Paris is the capital and most populous city of France."
query_text = "What is the capital of France?"

doc_emb = embeddings.embed_documents([doc_text])[0]   # uses input_type='document'
query_emb = embeddings.embed_query(query_text)         # uses input_type='query'
both_doc = embeddings.embed_documents([query_text])[0] # accidentally treat the query as a doc

print(f"  embed_documents(doc) vs embed_query(question)   "
      f"[bold green]{cosine(doc_emb, query_emb):.4f}[/bold green]  ← correct path")
print(f"  embed_documents(doc) vs embed_documents(question)  "
      f"{cosine(doc_emb, both_doc):.4f}  ← still works but suboptimal")
print("[dim]Pick the right method (`embed_documents` for the corpus, `embed_query` for queries) "
      "and you get the asymmetry for free.[/dim]")


# ---------------------------------------------------------------------------
# 4. Build a tiny corpus into a QdrantVectorStore
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]4. Build a QdrantVectorStore"))

corpus = [
    "Qdrant is an open-source vector database written in Rust, optimized for similarity search.",
    "FastAPI is a modern Python web framework built on Starlette and Pydantic.",
    "Voyage AI provides high-quality text embedding models, often paired with Claude.",
    "Claude is a family of large language models created by Anthropic.",
    "RAG (Retrieval-Augmented Generation) combines vector search with LLMs for grounded answers.",
    "Cosine similarity measures the angle between two vectors, ignoring their magnitude.",
    "PyTorch and TensorFlow are the two dominant deep learning frameworks.",
    "Docker containers let you package an application with all its dependencies.",
    "PostgreSQL is a relational database with strong support for JSON and full-text search.",
    "HNSW is the graph algorithm Qdrant uses for fast vector search.",
]

if COLLECTION in {c.name for c in qdrant.get_collections().collections}:
    qdrant.delete_collection(COLLECTION)
qdrant.create_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=len(vec), distance=Distance.COSINE),
)

vector_store = QdrantVectorStore(
    client=qdrant, collection_name=COLLECTION, embedding=embeddings
)
documents = [Document(page_content=t) for t in corpus]
vector_store.add_documents(documents)
print(f"Indexed {len(documents)} documents into '{COLLECTION}'.")


# ---------------------------------------------------------------------------
# 5. Watch retrieval work
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]5. Run semantic queries"))

queries = [
    "What database should I use for embeddings?",
    "How do I serve a Python API?",
    "Tell me about LLMs",
    "What's the math behind similarity?",
]
for q in queries:
    print(f"\n[bold]Query:[/bold] {q}")
    for doc, score in vector_store.similarity_search_with_score(q, k=3):
        text = doc.page_content
        if len(text) > 80:
            text = text[:77] + "..."
        print(f"  [green]{score:.4f}[/green]  {text}")


# ---------------------------------------------------------------------------
# 6. The contrast — random vectors give garbage
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]6. Random query vector = nonsense results"))

random.seed(0)
random_qvec = [random.uniform(-1, 1) for _ in range(len(vec))]

# Drop down to the raw vector store to inject our random query vector — the
# LangChain `similarity_search` path always re-embeds the query string.
hits = vector_store.similarity_search_by_vector(random_qvec, k=3)
for doc in hits:
    text = doc.page_content
    if len(text) > 80:
        text = text[:77] + "..."
    print(f"  [yellow]{text}[/yellow]")
print("[dim]The 'nearest neighbours' to a random direction in 1024-D space are meaningless. "
      "Embeddings (Voyage's contribution) are what makes search work.[/dim]")


# ---------------------------------------------------------------------------
# 7. Done
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]7. Done"))
print(f"Collection '[bold]{COLLECTION}[/bold]' preserved. Visit "
      f"http://localhost:6333/dashboard to inspect.")
print("[bold green]✓ Walkthrough complete.[/bold green]")
