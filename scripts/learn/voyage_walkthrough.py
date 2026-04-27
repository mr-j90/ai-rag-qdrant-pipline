"""
Voyage embeddings walkthrough — see semantic search actually work.

This script picks up where the Qdrant walkthrough left off. We're going to:

    1. Generate REAL embeddings for some text
    2. Inspect what an embedding actually looks like
    3. Compute cosine similarity by hand (to demystify it)
    4. See the document-vs-query asymmetry in action
    5. Index a small corpus into Qdrant with REAL vectors
    6. Run semantic queries and watch retrieval actually work
    7. Compare voyage-3 embeddings vs random vectors (apples-to-oranges)

Prereqs:
    - docker compose up -d  (Qdrant running)
    - VOYAGE_API_KEY in your environment (export VOYAGE_API_KEY=...)

Run:
    uv run python -m scripts.learn.voyage_walkthrough
"""
from __future__ import annotations

import math
import os

import voyageai
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from rich import print
from rich.rule import Rule
from rich.table import Table

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
COLLECTION = "voyage_walkthrough"
MODEL = "voyage-3"
DIM = 1024  # voyage-3 outputs 1024 dimensions

api_key = os.getenv("VOYAGE_API_KEY")
if not api_key:
    raise SystemExit(
        "VOYAGE_API_KEY not set. Get one at https://www.voyageai.com/ and:\n"
        "  export VOYAGE_API_KEY=pa-..."
    )

vo = voyageai.Client(api_key=api_key)
qdrant = QdrantClient(url="http://localhost:6333")


# ---------------------------------------------------------------------------
# 1. What does an embedding actually look like?
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]1. Generate one embedding"))

text = "Qdrant is a vector database written in Rust."
result = vo.embed([text], model=MODEL, input_type="document")
vec = result.embeddings[0]

print(f"Input text:    '{text}'")
print(f"Model:         {MODEL}")
print(f"Dimensions:    {len(vec)}")
print(f"First 8 dims:  {[round(x, 4) for x in vec[:8]]}")
print(f"Last 4 dims:   {[round(x, 4) for x in vec[-4:]]}")
print(f"Tokens billed: {result.total_tokens}")
print(f"Vector type:   {type(vec[0]).__name__}  (it's just floats!)")


# ---------------------------------------------------------------------------
# 2. Cosine similarity — by hand
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]2. Cosine similarity, demystified"))

def cosine(a: list[float], b: list[float]) -> float:
    """cos(θ) = (a · b) / (||a|| ||b||)"""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b)

pairs = [
    ("A dog is barking loudly", "A puppy is making lots of noise"),  # similar meaning
    ("A dog is barking loudly", "The stock market closed lower today"),  # unrelated
    ("A dog is barking loudly", "A dog is barking loudly"),  # identical
]

texts_flat = [t for pair in pairs for t in pair]
vecs = vo.embed(texts_flat, model=MODEL, input_type="document").embeddings

table = Table(show_lines=True)
table.add_column("Text A", overflow="fold", max_width=35)
table.add_column("Text B", overflow="fold", max_width=35)
table.add_column("Cosine", justify="right")
table.add_column("Interpretation")

interpretations = ["semantically similar", "unrelated", "identical"]
for i, (a, b) in enumerate(pairs):
    sim = cosine(vecs[i * 2], vecs[i * 2 + 1])
    table.add_row(a, b, f"{sim:.4f}", interpretations[i])

print(table)
print("[dim]Note the spread: similar text scores ~0.7+, unrelated ~0.3-0.5, identical = 1.0[/dim]")


# ---------------------------------------------------------------------------
# 3. The document vs query asymmetry
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]3. Document vs query input_type — the secret sauce"))

doc_text = "Paris is the capital and most populous city of France, with over 2 million residents."
query_text = "What is the capital of France?"

doc_as_doc = vo.embed([doc_text], model=MODEL, input_type="document").embeddings[0]
doc_as_query = vo.embed([doc_text], model=MODEL, input_type="query").embeddings[0]
query_as_doc = vo.embed([query_text], model=MODEL, input_type="document").embeddings[0]
query_as_query = vo.embed([query_text], model=MODEL, input_type="query").embeddings[0]

right_way = cosine(doc_as_doc, query_as_query)
wrong_way_both_docs = cosine(doc_as_doc, query_as_doc)
wrong_way_both_queries = cosine(doc_as_query, query_as_query)

print(f"  Doc embedded as 'document' vs query embedded as 'query':   [bold green]{right_way:.4f}[/bold green]  ← correct")
print(f"  Both embedded as 'document':                              {wrong_way_both_docs:.4f}  ← wrong")
print(f"  Both embedded as 'query':                                 {wrong_way_both_queries:.4f}  ← wrong")
print()
print(
    "[dim]The 'right way' lands the query closer to its answer. This is small here because "
    "voyage-3's projection is already strong, but it compounds at scale and on harder retrievals.[/dim]"
)


# ---------------------------------------------------------------------------
# 4. Build a small searchable corpus
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]4. Index a small corpus with REAL embeddings"))

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
    "HNSW (Hierarchical Navigable Small World) is the graph algorithm Qdrant uses for fast vector search.",
]

print(f"Embedding {len(corpus)} documents in one batch...")
doc_embeddings = vo.embed(corpus, model=MODEL, input_type="document").embeddings
print(f"  ✓ Got {len(doc_embeddings)} vectors of dim {len(doc_embeddings[0])}")

if COLLECTION in {c.name for c in qdrant.get_collections().collections}:
    qdrant.delete_collection(COLLECTION)
qdrant.create_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=DIM, distance=Distance.COSINE),
)

points = [
    PointStruct(id=i, vector=vec, payload={"text": txt})
    for i, (txt, vec) in enumerate(zip(corpus, doc_embeddings))
]
qdrant.upsert(collection_name=COLLECTION, points=points)
print(f"  ✓ Upserted {len(points)} points into '{COLLECTION}'")


# ---------------------------------------------------------------------------
# 5. Watch semantic search actually work
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]5. Run semantic queries"))

queries = [
    "What database should I use for embeddings?",
    "How do I serve a Python API?",
    "Tell me about LLMs",
    "What's the math behind similarity?",
]

for q in queries:
    qvec = vo.embed([q], model=MODEL, input_type="query").embeddings[0]
    hits = qdrant.query_points(
        collection_name=COLLECTION,
        query=qvec,
        limit=3,
        with_payload=True,
    ).points

    print(f"\n[bold]Query:[/bold] {q}")
    for h in hits:
        text = h.payload["text"]
        if len(text) > 80:
            text = text[:77] + "..."
        print(f"  [green]{h.score:.4f}[/green]  {text}")


# ---------------------------------------------------------------------------
# 6. The contrast: random vectors give garbage results
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]6. Why this works — contrast with random"))

import random
random.seed(0)
random_qvec = [random.uniform(-1, 1) for _ in range(DIM)]

print("Querying with a RANDOM vector (no semantic meaning):")
hits = qdrant.query_points(
    collection_name=COLLECTION,
    query=random_qvec,
    limit=3,
    with_payload=True,
).points
for h in hits:
    text = h.payload["text"]
    if len(text) > 80:
        text = text[:77] + "..."
    print(f"  [yellow]{h.score:+.4f}[/yellow]  {text}")

print(
    "\n[dim]Notice the scores are near zero (no semantic alignment). The 'top' results "
    "are essentially noise — proximity to a random direction in 1024D space. "
    "This is why embeddings ARE the magic. Qdrant just finds neighbors fast; "
    "Voyage decides what 'neighbor' means.[/dim]"
)


# ---------------------------------------------------------------------------
# 7. Wrap up
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]7. Done"))
print(f"Collection '[bold]{COLLECTION}[/bold]' is preserved.")
print("Visit http://localhost:6333/dashboard to inspect it visually.")
print("[bold green]✓ Walkthrough complete.[/bold green]")
