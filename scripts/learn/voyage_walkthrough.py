"""
LlamaIndex × Voyage walkthrough — embeddings through the LlamaIndex lens.

What changes when you go from "raw voyageai client" to LlamaIndex's
VoyageEmbedding wrapper:

    - You stop calling vo.embed(...) directly. LlamaIndex calls it for you,
      via get_text_embedding (one doc), get_text_embedding_batch (many docs),
      and get_query_embedding (a query — uses input_type='query' internally).
    - The document/query asymmetry is handled automatically based on which
      method you call. You almost never have to think about input_type again.
    - Embeddings are computed inside the IngestionPipeline / VectorStoreIndex
      transparently, instead of as an explicit step you orchestrate.

This walkthrough:
    1. Generate one embedding the LlamaIndex way
    2. See cosine similarity (still useful to understand)
    3. Confirm the doc-vs-query asymmetry IS still happening, just hidden
    4. Build a VectorStoreIndex with real embeddings
    5. Run semantic queries through the retriever
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

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.vector_stores.types import VectorStoreQuery
from llama_index.embeddings.voyageai import VoyageEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from rich import print
from rich.rule import Rule
from rich.table import Table

from src.config import get_settings

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
COLLECTION = "voyage_walkthrough"
settings = get_settings()

embed_model = VoyageEmbedding(
    model_name=settings.voyage_model,
    voyage_api_key=settings.voyage_api_key,
)
qdrant = QdrantClient(url=settings.qdrant_url)


# ---------------------------------------------------------------------------
# 1. One embedding, the LlamaIndex way
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]1. Generate one embedding via VoyageEmbedding"))

text = "Qdrant is a vector database written in Rust."
vec = embed_model.get_text_embedding(text)

print(f"Input text:    '{text}'")
print(f"Model:         {settings.voyage_model}")
print(f"Dimensions:    {len(vec)}")
print(f"First 8 dims:  {[round(x, 4) for x in vec[:8]]}")
print(f"Vector type:   {type(vec[0]).__name__}")
print("[dim]No more passing input_type='document' yourself — it's set for you.[/dim]")


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
texts_flat = [t for pair in pairs for t in pair]
flat_vecs = embed_model.get_text_embedding_batch(texts_flat)

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
print(Rule("[bold cyan]3. Doc vs query — LlamaIndex picks input_type for you"))

doc_text = "Paris is the capital and most populous city of France."
query_text = "What is the capital of France?"

doc_emb = embed_model.get_text_embedding(doc_text)        # uses input_type='document'
query_emb = embed_model.get_query_embedding(query_text)    # uses input_type='query'
both_doc = embed_model.get_text_embedding(query_text)      # what if you accidentally use the doc method on a query?

print(f"  doc(text) vs query(question)   [bold green]{cosine(doc_emb, query_emb):.4f}[/bold green]  ← correct path")
print(f"  doc(text) vs doc(question)     {cosine(doc_emb, both_doc):.4f}  ← still works but suboptimal")
print("[dim]Pick the right method (`get_text_embedding` for docs, `get_query_embedding` for queries) "
      "and you get the asymmetry for free.[/dim]")


# ---------------------------------------------------------------------------
# 4. Build a tiny corpus into a VectorStoreIndex
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]4. Build a VectorStoreIndex"))

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

vector_store = QdrantVectorStore(client=qdrant, collection_name=COLLECTION)
documents = [Document(text=t) for t in corpus]
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=StorageContext.from_defaults(vector_store=vector_store),
    embed_model=embed_model,
)
print(f"Indexed {len(documents)} documents into '{COLLECTION}'.")


# ---------------------------------------------------------------------------
# 5. Watch retrieval work
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]5. Run semantic queries via the retriever"))

retriever = index.as_retriever(similarity_top_k=3)
queries = [
    "What database should I use for embeddings?",
    "How do I serve a Python API?",
    "Tell me about LLMs",
    "What's the math behind similarity?",
]
for q in queries:
    print(f"\n[bold]Query:[/bold] {q}")
    for h in retriever.retrieve(q):
        text = h.node.get_content()
        if len(text) > 80:
            text = text[:77] + "..."
        print(f"  [green]{h.score:.4f}[/green]  {text}")


# ---------------------------------------------------------------------------
# 6. The contrast — random vectors give garbage
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]6. Random query vector = nonsense results"))

random.seed(0)
random_qvec = [random.uniform(-1, 1) for _ in range(len(vec))]

# Drop down to the raw vector store to inject our random query vector
# (the index/retriever path always re-embeds the query string, so we go around it here).
result = vector_store.query(VectorStoreQuery(query_embedding=random_qvec, similarity_top_k=3))
for node, score in zip(result.nodes or [], result.similarities or []):
    text = node.get_content()
    if len(text) > 80:
        text = text[:77] + "..."
    print(f"  [yellow]{score:+.4f}[/yellow]  {text}")
print("[dim]Scores hover near zero — proximity to a random direction in 1024-D space "
      "is meaningless. Embeddings (Voyage's contribution) are what makes search work.[/dim]")


# ---------------------------------------------------------------------------
# 7. Done
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]7. Done"))
print(f"Collection '[bold]{COLLECTION}[/bold]' preserved. Visit "
      f"http://localhost:6333/dashboard to inspect.")
print("[bold green]✓ Walkthrough complete.[/bold green]")
