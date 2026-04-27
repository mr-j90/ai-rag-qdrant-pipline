"""
Qdrant walkthrough — learn the vector DB in isolation.

This script uses FAKE (random) vectors so you can focus on Qdrant's mechanics
without worrying about embeddings yet. We'll run the same sequence of operations
you'd do in a real RAG pipeline:

    1. Connect to Qdrant
    2. Create a collection
    3. Insert ("upsert") points
    4. Inspect what's stored
    5. Run a vector search
    6. Run a FILTERED vector search
    7. Add a payload index
    8. Clean up

Run it section by section. Each block prints what just happened.

Prereq:  docker compose up -d
Run:     uv run python -m scripts.learn.qdrant_walkthrough
"""
from __future__ import annotations

import random
from rich import print
from rich.rule import Rule

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    Range,
    VectorParams,
)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
COLLECTION = "qdrant_walkthrough"
DIM = 8  # tiny dimension so we can actually print vectors and read them
random.seed(42)


def random_vector() -> list[float]:
    """Pretend this is what an embedding model returns."""
    return [round(random.uniform(-1, 1), 3) for _ in range(DIM)]


client = QdrantClient(url="http://localhost:6333")


# ---------------------------------------------------------------------------
# 1. Connect & inspect what's already there
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]1. Connect & inspect"))
existing = [c.name for c in client.get_collections().collections]
print(f"Existing collections: {existing}")

# Clean slate for this walkthrough
if COLLECTION in existing:
    client.delete_collection(COLLECTION)
    print(f"Deleted old '{COLLECTION}' collection.")


# ---------------------------------------------------------------------------
# 2. Create a collection
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]2. Create collection"))
client.create_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=DIM, distance=Distance.COSINE),
)
info = client.get_collection(COLLECTION)
print(f"Created '{COLLECTION}'")
print(f"  vector size:  {info.config.params.vectors.size}")
print(f"  distance:     {info.config.params.vectors.distance}")
print(f"  point count:  {info.points_count}")


# ---------------------------------------------------------------------------
# 3. Upsert some points
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]3. Upsert points"))
docs = [
    {"id": 1, "text": "Qdrant is a vector database written in Rust.",      "source": "wiki.pdf",     "page": 1, "topic": "infra"},
    {"id": 2, "text": "FastAPI is a Python web framework.",                 "source": "fastapi.pdf",  "page": 1, "topic": "web"},
    {"id": 3, "text": "Voyage AI provides text embeddings.",                "source": "voyage.pdf",   "page": 3, "topic": "ml"},
    {"id": 4, "text": "Claude is Anthropic's family of LLMs.",              "source": "anthropic.pdf","page": 2, "topic": "ml"},
    {"id": 5, "text": "RAG combines retrieval with generation.",            "source": "rag.pdf",      "page": 1, "topic": "ml"},
    {"id": 6, "text": "Cosine similarity measures angle between vectors.",  "source": "math.pdf",     "page": 7, "topic": "math"},
]

points = [
    PointStruct(
        id=d["id"],
        vector=random_vector(),
        payload={k: v for k, v in d.items() if k != "id"},
    )
    for d in docs
]

client.upsert(collection_name=COLLECTION, points=points)
print(f"Upserted {len(points)} points.")
print(f"Collection count now: {client.count(COLLECTION, exact=True).count}")


# ---------------------------------------------------------------------------
# 4. Inspect what's stored
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]4. Inspect stored points"))
fetched = client.retrieve(
    collection_name=COLLECTION,
    ids=[1, 3, 5],
    with_payload=True,
    with_vectors=True,
)
for p in fetched:
    print(f"  id={p.id}")
    print(f"    payload: {p.payload}")
    print(f"    vector:  {p.vector[:4]}... (truncated)")


# ---------------------------------------------------------------------------
# 5. Vector search (no filter)
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]5. Vector search"))
query_vec = random_vector()
print(f"Query vector: {query_vec}")

hits = client.query_points(
    collection_name=COLLECTION,
    query=query_vec,
    limit=3,
    with_payload=True,
).points

print(f"\nTop {len(hits)} results:")
for h in hits:
    print(f"  score={h.score:+.4f}  id={h.id}  text='{h.payload['text']}'")


# ---------------------------------------------------------------------------
# 6. Filtered search — the killer feature
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]6. Filtered vector search"))
ml_only = Filter(must=[FieldCondition(key="topic", match=MatchValue(value="ml"))])

hits = client.query_points(
    collection_name=COLLECTION,
    query=query_vec,
    query_filter=ml_only,
    limit=3,
    with_payload=True,
).points

print("Filter: topic == 'ml'")
for h in hits:
    print(f"  score={h.score:+.4f}  id={h.id}  topic={h.payload['topic']}  text='{h.payload['text']}'")

print("\nFilter: topic == 'ml' AND page <= 2")
complex_filter = Filter(
    must=[
        FieldCondition(key="topic", match=MatchValue(value="ml")),
        FieldCondition(key="page", range=Range(lte=2)),
    ]
)
hits = client.query_points(
    collection_name=COLLECTION,
    query=query_vec,
    query_filter=complex_filter,
    limit=3,
    with_payload=True,
).points
for h in hits:
    print(f"  score={h.score:+.4f}  id={h.id}  page={h.payload['page']}  text='{h.payload['text']}'")


# ---------------------------------------------------------------------------
# 7. Payload index — speeds up filters at scale
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]7. Create a payload index"))
client.create_payload_index(
    collection_name=COLLECTION,
    field_name="topic",
    field_schema=PayloadSchemaType.KEYWORD,
)
client.create_payload_index(
    collection_name=COLLECTION,
    field_name="page",
    field_schema=PayloadSchemaType.INTEGER,
)
print("Indexed 'topic' (keyword) and 'page' (integer).")

info = client.get_collection(COLLECTION)
print(f"Payload schema now: {info.payload_schema}")


# ---------------------------------------------------------------------------
# 8. Cleanup
# ---------------------------------------------------------------------------
print(Rule("[bold cyan]8. Cleanup"))
print(f"Collection '{COLLECTION}' kept around. Visit http://localhost:6333/dashboard to explore.")
print("[bold green]✓ Walkthrough complete.[/bold green]")
